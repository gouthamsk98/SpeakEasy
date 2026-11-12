// SPDX-License-Identifier: GPL-3.0-or-later
//! Local storage for consent records and calibration sessions.
//!
//! ```text
//! data/consent/<speaker>.json
//! data/raw/<speaker>/calib-YYYYMMDD-HHMMSS/{01.wav, 01.txt, ..., session.json}
//! ```

use crate::align::Score;
use crate::prompts::{all_prompts, session_mix, Prompt};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const CONSENT_STATEMENT: &str = "I agree that my voice recordings and the sentences I read \
are stored on this computer only, to measure and improve how well this app recognises my \
speech. They are not sent anywhere. I can delete them at any time by deleting my folder \
in data/raw and data/consent.";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Consent {
    pub consent_id: String,
    pub speaker: String,
    pub statement: String,
    pub accepted_at: String,
}

impl Consent {
    fn path(data_root: &Path, speaker: &str) -> PathBuf {
        data_root.join("consent").join(format!("{speaker}.json"))
    }

    pub fn load(data_root: &Path, speaker: &str) -> Option<Consent> {
        let f = std::fs::File::open(Self::path(data_root, speaker)).ok()?;
        serde_json::from_reader(f).ok()
    }

    /// Record that `speaker` accepted [`CONSENT_STATEMENT`] now.
    pub fn record(data_root: &Path, speaker: &str) -> Result<Consent> {
        let now = chrono::Local::now();
        let c = Consent {
            consent_id: format!("local-{speaker}-{}", now.format("%Y%m%d%H%M%S")),
            speaker: speaker.into(),
            statement: CONSENT_STATEMENT.into(),
            accepted_at: now.to_rfc3339(),
        };
        let p = Self::path(data_root, speaker);
        std::fs::create_dir_all(p.parent().unwrap())?;
        std::fs::write(&p, serde_json::to_vec_pretty(&c)?)?;
        Ok(c)
    }
}

/// Speaker IDs become folder names: keep them boring.
pub fn valid_speaker_id(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 32
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TakeResult {
    pub index: usize,
    pub audio: String,
    pub prompt: String,
    pub hypothesis: String,
    pub seconds: f32,
    pub score: Score,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub speaker: String,
    pub consent_id: String,
    pub model: String,
    pub started_at: String,
    pub prompt_set: String,
    pub takes: Vec<TakeResult>,
    pub ref_words: usize,
    pub word_errors: usize,
    pub ref_chars: usize,
    pub char_errors: usize,
}

impl SessionSummary {
    pub fn wer(&self) -> f64 {
        self.word_errors as f64 / self.ref_words.max(1) as f64
    }
    pub fn cer(&self) -> f64 {
        self.char_errors as f64 / self.ref_chars.max(1) as f64
    }
}

pub struct SessionWriter {
    pub dir: PathBuf,
    summary: SessionSummary,
}

impl SessionWriter {
    pub fn new(data_root: &Path, consent: &Consent, model: &str, prompt_set: &str) -> Result<Self> {
        if !valid_speaker_id(&consent.speaker) {
            bail!("invalid speaker id {:?}", consent.speaker);
        }
        let now = chrono::Local::now();
        let dir = data_root
            .join("raw")
            .join(&consent.speaker)
            .join(format!("calib-{}", now.format("%Y%m%d-%H%M%S")));
        std::fs::create_dir_all(&dir).with_context(|| format!("create {dir:?}"))?;
        Ok(Self {
            dir,
            summary: SessionSummary {
                speaker: consent.speaker.clone(),
                consent_id: consent.consent_id.clone(),
                model: model.into(),
                started_at: now.to_rfc3339(),
                prompt_set: prompt_set.into(),
                takes: Vec::new(),
                ref_words: 0,
                word_errors: 0,
                ref_chars: 0,
                char_errors: 0,
            },
        })
    }

    /// Save one accepted take (16 kHz mono f32 audio) and its score.
    pub fn add_take(
        &mut self,
        index: usize,
        prompt: &str,
        audio_16k: &[f32],
        hypothesis: &str,
        score: Score,
    ) -> Result<()> {
        let stem = format!("{:02}", index + 1);
        let wav = self.dir.join(format!("{stem}.wav"));
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&wav, spec)?;
        for &s in audio_16k {
            w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
        }
        w.finalize()?;
        std::fs::write(self.dir.join(format!("{stem}.txt")), format!("{prompt}\n"))?;

        let s = &mut self.summary;
        s.takes.retain(|t| t.index != index);
        s.takes.push(TakeResult {
            index,
            audio: format!("{stem}.wav"),
            prompt: prompt.into(),
            hypothesis: hypothesis.into(),
            seconds: audio_16k.len() as f32 / 16_000.0,
            score,
        });
        s.takes.sort_by_key(|t| t.index);
        s.ref_words = s.takes.iter().map(|t| t.score.ref_words).sum();
        s.word_errors = s.takes.iter().map(|t| t.score.word_errors).sum();
        s.ref_chars = s.takes.iter().map(|t| t.score.ref_chars).sum();
        s.char_errors = s.takes.iter().map(|t| t.score.char_errors).sum();
        self.write_summary()
    }

    fn write_summary(&self) -> Result<()> {
        std::fs::write(
            self.dir.join("session.json"),
            serde_json::to_vec_pretty(&self.summary)?,
        )?;
        Ok(())
    }

    pub fn summary(&self) -> &SessionSummary {
        &self.summary
    }
}

/// All finished sessions of a speaker, oldest first.
pub fn sessions(data_root: &Path, speaker: &str) -> Vec<(PathBuf, SessionSummary)> {
    let mut v: Vec<_> = std::fs::read_dir(data_root.join("raw").join(speaker))
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with("calib-"))
        })
        .filter_map(|p| {
            let s: SessionSummary =
                serde_json::from_reader(std::fs::File::open(p.join("session.json")).ok()?).ok()?;
            (!s.takes.is_empty()).then_some((p, s))
        })
        .collect();
    v.sort_by(|a, b| a.0.cmp(&b.0));
    v
}

/// How many times this speaker has recorded each prompt text.
pub fn recorded_counts(
    data_root: &Path,
    speaker: &str,
) -> std::collections::HashMap<String, usize> {
    let mut m = std::collections::HashMap::new();
    for (_, s) in sessions(data_root, speaker) {
        for t in s.takes {
            *m.entry(t.prompt).or_insert(0) += 1;
        }
    }
    m
}

/// Build the next session of `n` items: the [`session_mix`] of kinds, and within each
/// kind the items this speaker has recorded least (ties: list order, so stories continue
/// where they stopped).
pub fn next_session(data_root: &Path, speaker: &str, n: usize) -> Vec<Prompt> {
    let counts = recorded_counts(data_root, speaker);
    let all = all_prompts();
    let mut out = Vec::new();
    for (kind, want) in session_mix(n) {
        let mut pool: Vec<(usize, usize, &Prompt)> = all
            .iter()
            .enumerate()
            .filter(|(_, p)| p.kind == kind)
            .map(|(i, p)| (counts.get(&p.text).copied().unwrap_or(0), i, p))
            .collect();
        pool.sort_by_key(|(c, i, _)| (*c, *i));
        out.extend(pool.into_iter().take(want).map(|(_, _, p)| p.clone()));
    }
    out
}

/// The baseline = the speaker's first session with the same prompt set.
pub fn baseline(
    data_root: &Path,
    speaker: &str,
    prompt_set: &str,
) -> Option<(PathBuf, SessionSummary)> {
    sessions(data_root, speaker)
        .into_iter()
        .find(|(_, s)| s.prompt_set == prompt_set)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::align::score;
    use crate::prompts::Kind;

    #[test]
    fn session_roundtrip_and_baseline() {
        let root = std::env::temp_dir().join(format!("asr-calib-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        assert!(Consent::load(&root, "spk01").is_none());
        let c = Consent::record(&root, "spk01").unwrap();
        assert_eq!(
            Consent::load(&root, "spk01").unwrap().consent_id,
            c.consent_id
        );

        let mut w = SessionWriter::new(&root, &c, "test-model", "en-baseline-5").unwrap();
        let audio = vec![0.1f32; 16_000];
        w.add_take(
            0,
            "thank you",
            &audio,
            "thank you",
            score("thank you", "thank you"),
        )
        .unwrap();
        w.add_take(
            1,
            "call my daughter",
            &audio,
            "tall my daughter",
            score("call my daughter", "tall my daughter"),
        )
        .unwrap();
        assert!(w.dir.join("01.wav").exists() && w.dir.join("02.txt").exists());
        assert_eq!(w.summary().word_errors, 1);
        assert_eq!(w.summary().ref_words, 5);

        let (_, b) = baseline(&root, "spk01", "en-baseline-5").unwrap();
        assert_eq!(b.takes.len(), 2);
        assert!(baseline(&root, "spk01", "other-set").is_none());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn next_session_mixes_kinds_and_continues() {
        let root = std::env::temp_dir().join(format!("asr-next-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let first = next_session(&root, "s", 12);
        assert_eq!(first.len(), 12);
        for k in Kind::ALL {
            assert!(first.iter().any(|p| p.kind == k), "missing {k:?}");
        }
        let story1 = first
            .iter()
            .find(|p| p.kind == Kind::Story)
            .unwrap()
            .clone();
        assert!(story1.context.as_deref().unwrap().contains("part 1 of"));

        // record the story part and the first word -> next session moves on
        let c = Consent::record(&root, "s").unwrap();
        let mut w = SessionWriter::new(&root, &c, "m", "set").unwrap();
        w.add_take(0, &story1.text, &[0.0; 100], "", score(&story1.text, ""))
            .unwrap();
        w.add_take(
            1,
            &first[0].text,
            &[0.0; 100],
            "",
            score(&first[0].text, ""),
        )
        .unwrap();
        let next = next_session(&root, "s", 12);
        let story2 = next.iter().find(|p| p.kind == Kind::Story).unwrap();
        assert!(story2.context.as_deref().unwrap().contains("part 2 of"));
        assert!(!next.iter().any(|p| p.text == first[0].text));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn speaker_ids() {
        assert!(valid_speaker_id("spk_01"));
        assert!(!valid_speaker_id("../etc"));
        assert!(!valid_speaker_id(""));
    }
}
