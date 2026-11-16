// SPDX-License-Identifier: GPL-3.0-or-later
//! Moving data between the user's app and the trainer's machine.
//!
//! * **Dataset export** (user -> trainer): one zip with every recorded take of a speaker,
//!   a JSONL manifest, the consent record and a summary. Nothing else from the machine.
//! * **Model import** (trainer -> user): a zip containing `model.onnx` + `model.json`
//!   (optionally `model-card.json`), installed into the models folder.

use crate::store::{sessions, valid_speaker_id, Consent};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};

pub const DATASET_FORMAT: &str = "speakeasy-dataset";
pub const DATASET_VERSION: u32 = 1;

#[derive(Debug, Serialize, Deserialize)]
pub struct DatasetInfo {
    pub format: String,
    pub version: u32,
    pub speaker: String,
    pub created_at: String,
    pub app_version: String,
    pub utterances: usize,
    pub total_seconds: f32,
    pub sessions: usize,
    pub consent: Consent,
}

/// One line of manifest.jsonl (also what `research/` reads).
#[derive(Debug, Serialize, Deserialize)]
pub struct ManifestRow {
    pub audio: String,
    pub text: String,
    pub speaker: String,
    pub duration: f32,
    pub consent_id: String,
    pub session: String,
    /// What the app's model recognised at recording time, and its word error.
    pub heard: String,
    pub wer: f64,
    pub model: String,
    /// Word / Phrase / Sentence / TongueTwister / Story (None for imported recordings).
    #[serde(default)]
    pub kind: Option<crate::Kind>,
    /// How the user was asked to say it: Normal, Far, Close, PauseBefore, ...
    #[serde(default)]
    pub condition: crate::Condition,
}

pub struct ExportResult {
    pub path: PathBuf,
    pub utterances: usize,
    pub total_seconds: f32,
}

/// Write every saved take of `speaker` into `out_zip`.
pub fn export_dataset(data_root: &Path, speaker: &str, out_zip: &Path) -> Result<ExportResult> {
    if !valid_speaker_id(speaker) {
        bail!("invalid speaker id {speaker:?}");
    }
    let consent =
        Consent::load(data_root, speaker).context("no consent record for this speaker")?;
    let sess = sessions(data_root, speaker);
    if sess.is_empty() {
        bail!("no recordings for {speaker:?} yet");
    }

    if let Some(dir) = out_zip.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = out_zip.with_extension("zip.part");
    let mut zw = zip::ZipWriter::new(std::fs::File::create(&tmp)?);
    // WAV barely compresses; store it to keep export instant.
    let stored =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    let deflated = zip::write::SimpleFileOptions::default();

    let mut manifest = String::new();
    let (mut n, mut secs) = (0usize, 0f32);
    for (dir, s) in &sess {
        let sname = dir.file_name().unwrap().to_string_lossy().to_string();
        for t in &s.takes {
            let rel = format!("audio/{sname}/{}", t.audio);
            let mut bytes = Vec::new();
            std::fs::File::open(dir.join(&t.audio))
                .with_context(|| format!("read {:?}", dir.join(&t.audio)))?
                .read_to_end(&mut bytes)?;
            zw.start_file(&rel, stored)?;
            zw.write_all(&bytes)?;
            let row = ManifestRow {
                audio: rel,
                text: t.prompt.clone(),
                speaker: speaker.into(),
                duration: t.seconds,
                consent_id: consent.consent_id.clone(),
                session: sname.clone(),
                heard: t.hypothesis.clone(),
                wer: t.score.wer(),
                model: s.model.clone(),
                kind: crate::kind_of(&t.prompt),
                condition: t.condition,
            };
            manifest.push_str(&serde_json::to_string(&row)?);
            manifest.push('\n');
            n += 1;
            secs += t.seconds;
        }
    }
    let info = DatasetInfo {
        format: DATASET_FORMAT.into(),
        version: DATASET_VERSION,
        speaker: speaker.into(),
        created_at: chrono::Local::now().to_rfc3339(),
        app_version: env!("CARGO_PKG_VERSION").into(),
        utterances: n,
        total_seconds: secs,
        sessions: sess.len(),
        consent,
    };
    zw.start_file("manifest.jsonl", deflated)?;
    zw.write_all(manifest.as_bytes())?;
    zw.start_file("dataset.json", deflated)?;
    zw.write_all(&serde_json::to_vec_pretty(&info)?)?;
    zw.finish()?;
    std::fs::rename(&tmp, out_zip)?;
    Ok(ExportResult {
        path: out_zip.to_path_buf(),
        utterances: n,
        total_seconds: secs,
    })
}

/// Default file name for an export, e.g. `speakeasy-dataset-user01-20261112-1830.zip`.
pub fn dataset_file_name(speaker: &str) -> String {
    format!(
        "speakeasy-dataset-{speaker}-{}.zip",
        chrono::Local::now().format("%Y%m%d-%H%M")
    )
}

/// Install a model package zip into `models_dir`. Accepts files at the zip root or inside a
/// single top-level folder. Returns the installed model directory (never overwrites).
pub fn import_model(zip_path: &Path, models_dir: &Path) -> Result<PathBuf> {
    let mut za = zip::ZipArchive::new(std::fs::File::open(zip_path)?)
        .with_context(|| format!("{zip_path:?} is not a zip file"))?;

    // Collect files, rejecting anything that could escape the target directory.
    let mut files: Vec<(usize, PathBuf)> = Vec::new();
    for i in 0..za.len() {
        let f = za.by_index(i)?;
        if f.is_dir() {
            continue;
        }
        let name = f.name()?.to_string();
        let p = PathBuf::from(&name);
        if p.components().any(|c| !matches!(c, Component::Normal(_))) {
            bail!("unsafe path in model package: {name:?}");
        }
        files.push((i, p));
    }
    let json = files
        .iter()
        .find(|(_, p)| p.file_name().is_some_and(|n| n == "model.json"))
        .map(|(_, p)| p.clone())
        .context("model package has no model.json")?;
    let prefix = json.parent().map(Path::to_path_buf).unwrap_or_default();
    if prefix.components().count() > 1 {
        bail!("model.json is nested too deep in the package");
    }
    if !files.iter().any(|(_, p)| *p == prefix.join("model.onnx")) {
        bail!("model package has no model.onnx next to model.json");
    }

    // Name: the folder inside the zip, else the zip's file stem.
    let base = match prefix.file_name() {
        Some(n) => n.to_string_lossy().to_string(),
        None => zip_path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string(),
    };
    let base: String = base
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || "-_.".contains(c) {
                c
            } else {
                '-'
            }
        })
        .collect();
    let mut dest = models_dir.join(&base);
    let mut k = 2;
    while dest.exists() {
        dest = models_dir.join(format!("{base}-{k}"));
        k += 1;
    }
    let tmp = models_dir.join(format!(".importing-{base}"));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp)?;
    for (i, p) in &files {
        let Ok(rel) = p.strip_prefix(&prefix) else {
            continue;
        };
        if rel.components().count() != 1 {
            continue; // only top-level files of the model folder
        }
        let mut f = za.by_index(*i)?;
        let mut out = std::fs::File::create(tmp.join(rel))?;
        std::io::copy(&mut f, &mut out)?;
    }
    // Must parse as our model metadata before we accept it.
    let meta: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(tmp.join("model.json"))?)
            .context("model.json is not valid JSON")?;
    if meta.get("format_version").and_then(|v| v.as_u64()) != Some(1) {
        let _ = std::fs::remove_dir_all(&tmp);
        bail!("model.json: unsupported or missing format_version");
    }
    std::fs::rename(&tmp, &dest)?;
    Ok(dest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{score, SessionWriter};

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("asr-pkg-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn export_dataset_contains_audio_and_manifest() {
        let root = tmp("export");
        let c = Consent::record(&root, "spk").unwrap();
        let mut w = SessionWriter::new(&root, &c, "m", "set").unwrap();
        w.add_take(
            0,
            "hello there",
            &[0.1; 8000],
            "hello",
            score("hello there", "hello"),
            crate::Condition::Normal,
        )
        .unwrap();
        w.add_take(
            1,
            "good night",
            &[0.1; 16000],
            "good night",
            score("good night", "good night"),
            crate::Condition::Far,
        )
        .unwrap();
        let out = root.join("out/ds.zip");
        let r = export_dataset(&root, "spk", &out).unwrap();
        assert_eq!(r.utterances, 2);
        assert!((r.total_seconds - 1.5).abs() < 1e-3);

        let mut za = zip::ZipArchive::new(std::fs::File::open(&out).unwrap()).unwrap();
        let mut m = String::new();
        za.by_name("manifest.jsonl")
            .unwrap()
            .read_to_string(&mut m)
            .unwrap();
        let rows: Vec<ManifestRow> = m
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(rows[0].text, "hello there");
        assert_eq!(rows[0].heard, "hello");
        assert!(za.by_name(&rows[1].audio).is_ok());
        let info: DatasetInfo =
            serde_json::from_reader(za.by_name("dataset.json").unwrap()).unwrap();
        assert_eq!(info.consent.consent_id, c.consent_id);
        let _ = std::fs::remove_dir_all(&root);
    }

    fn make_zip(path: &Path, entries: &[(&str, &[u8])]) {
        let mut zw = zip::ZipWriter::new(std::fs::File::create(path).unwrap());
        for (name, data) in entries {
            zw.start_file(*name, zip::write::SimpleFileOptions::default())
                .unwrap();
            zw.write_all(data).unwrap();
        }
        zw.finish().unwrap();
    }

    #[test]
    fn import_model_package() {
        let root = tmp("import");
        let models = root.join("models");
        std::fs::create_dir_all(&models).unwrap();
        let z = root.join("pkg.zip");
        make_zip(
            &z,
            &[
                ("user01-v1/model.onnx", b"onnx"),
                ("user01-v1/model.json", br#"{"format_version": 1}"#),
                ("user01-v1/model-card.json", b"{}"),
            ],
        );
        let d = import_model(&z, &models).unwrap();
        assert_eq!(d, models.join("user01-v1"));
        assert!(d.join("model.onnx").exists() && d.join("model-card.json").exists());
        // second import does not overwrite
        assert_eq!(
            import_model(&z, &models).unwrap(),
            models.join("user01-v1-2")
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn import_rejects_bad_packages() {
        let root = tmp("bad");
        let models = root.join("models");
        std::fs::create_dir_all(&models).unwrap();
        let z = root.join("evil.zip");
        make_zip(&z, &[("../model.json", b"{}"), ("model.onnx", b"x")]);
        assert!(import_model(&z, &models).is_err());
        make_zip(&z, &[("m/model.json", br#"{"format_version": 1}"#)]);
        assert!(import_model(&z, &models)
            .unwrap_err()
            .to_string()
            .contains("model.onnx"));
        make_zip(&z, &[("m/model.json", b"{}"), ("m/model.onnx", b"x")]);
        assert!(import_model(&z, &models).is_err());
        assert_eq!(
            std::fs::read_dir(&models)
                .unwrap()
                .filter_map(|e| e.ok())
                .filter(|e| !e.file_name().to_string_lossy().starts_with('.'))
                .count(),
            0
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}
