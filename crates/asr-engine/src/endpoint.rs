// SPDX-License-Identifier: GPL-3.0-or-later
//! Energy-based endpointing on log-mel frames.
//!
//! Glossectomy speech can have long intra-word pauses, so the trailing-silence
//! threshold is deliberately configurable per user. A neural VAD (e.g. Silero) or
//! model-based endpointing (blank-run length in CTC output) is on the roadmap.

#[derive(Debug, Clone)]
pub struct EndpointConfig {
    /// Frame is speech when mean log-mel exceeds noise floor by this many nats (~4.3 dB each).
    pub speech_margin: f32,
    /// Silence needed after speech to close the utterance.
    pub trailing_silence_ms: f32,
    /// Minimum speech before a silence can end the utterance.
    pub min_speech_ms: f32,
}

impl Default for EndpointConfig {
    fn default() -> Self {
        Self {
            speech_margin: 2.5,
            trailing_silence_ms: 700.0,
            min_speech_ms: 150.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    Silence,
    Speech,
    EndOfUtterance,
}

pub struct Endpointer {
    cfg: EndpointConfig,
    hop_ms: f32,
    floor: Option<f32>,
    speech_ms: f32,
    silence_ms: f32,
}

impl Endpointer {
    pub fn new(cfg: EndpointConfig, hop_secs: f64) -> Self {
        Self {
            cfg,
            hop_ms: (hop_secs * 1e3) as f32,
            floor: None,
            speech_ms: 0.0,
            silence_ms: 0.0,
        }
    }

    pub fn update(&mut self, frame: &[f32]) -> State {
        let e = frame.iter().sum::<f32>() / frame.len() as f32;
        let floor = *self.floor.get_or_insert(e);
        // Floor tracks down fast, up slowly (so speech doesn't drag it up).
        self.floor = Some(if e < floor {
            0.9 * floor + 0.1 * e
        } else {
            0.999 * floor + 0.001 * e
        });

        if e > floor + self.cfg.speech_margin {
            self.speech_ms += self.hop_ms;
            self.silence_ms = 0.0;
            State::Speech
        } else {
            self.silence_ms += self.hop_ms;
            if self.speech_ms >= self.cfg.min_speech_ms
                && self.silence_ms >= self.cfg.trailing_silence_ms
            {
                self.speech_ms = 0.0;
                State::EndOfUtterance
            } else {
                State::Silence
            }
        }
    }
}
