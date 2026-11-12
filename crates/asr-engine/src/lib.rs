// SPDX-License-Identifier: GPL-3.0-or-later
//! Streaming recognition runtime.
//!
//! ```text
//!  16 kHz f32 ──► FbankStream ──► chunk of N frames ──► Recognizer ──► partial text
//!                     │                                     ▲
//!                     └──► Endpointer (energy) ── silence ──┘ finalize ──► final text
//! ```
//!
//! Everything here is backend-agnostic; models plug in through [`Recognizer`].

pub mod dummy;
mod endpoint;
#[cfg(feature = "onnx")]
pub mod onnx;

pub use endpoint::{EndpointConfig, Endpointer};

use anyhow::Result;
use asr_features::{FbankConfig, FbankStream};
use std::time::{Duration, Instant};

/// A streaming acoustic model + decoder.
///
/// Contract: `accept_chunk` receives exactly `chunk_frames() * n_mels` features
/// (time-major). It returns `Some(text)` when the running hypothesis changed.
pub trait Recognizer: Send {
    fn name(&self) -> String;
    fn feature_config(&self) -> FbankConfig;
    /// Frames per inference call. Smaller = lower latency, more overhead per frame.
    fn chunk_frames(&self) -> usize;
    fn accept_chunk(&mut self, feats: &[f32]) -> Result<Option<String>>;
    /// End of utterance: flush and return the final text. Must leave the
    /// recognizer ready for a new utterance.
    fn finalize(&mut self) -> Result<String>;
    fn reset(&mut self);
}

#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// Running hypothesis for the current utterance (replace what's on screen).
    Partial(String),
    /// Utterance finished; commit this text.
    Final(String),
}

/// Rolling latency stats, shown in the app's debug overlay.
#[derive(Debug, Clone, Default)]
pub struct Metrics {
    pub chunks: u64,
    pub last_compute: Duration,
    pub max_compute: Duration,
    pub ema_compute_ms: f64,
    /// Audio duration covered by one chunk.
    pub chunk_audio: Duration,
}

impl Metrics {
    fn record(&mut self, d: Duration) {
        self.chunks += 1;
        self.last_compute = d;
        self.max_compute = self.max_compute.max(d);
        let ms = d.as_secs_f64() * 1e3;
        self.ema_compute_ms = if self.chunks == 1 {
            ms
        } else {
            0.9 * self.ema_compute_ms + 0.1 * ms
        };
    }
    /// Real-time factor: compute / audio. Must be << 1.
    pub fn rtf(&self) -> f64 {
        if self.chunk_audio.is_zero() {
            0.0
        } else {
            self.ema_compute_ms / (self.chunk_audio.as_secs_f64() * 1e3)
        }
    }
}

pub struct Pipeline {
    fbank: FbankStream,
    rec: Box<dyn Recognizer>,
    endpointer: Endpointer,
    feats: Vec<f32>,
    n_mels: usize,
    chunk: usize,
    in_utterance: bool,
    pub metrics: Metrics,
}

impl Pipeline {
    pub fn new(rec: Box<dyn Recognizer>, ep: EndpointConfig) -> Self {
        let fcfg = rec.feature_config();
        let chunk = rec.chunk_frames();
        let hop = fcfg.frame_shift as f64 / fcfg.sample_rate as f64;
        Self {
            n_mels: fcfg.n_mels,
            fbank: FbankStream::new(fcfg),
            endpointer: Endpointer::new(ep, hop),
            feats: Vec::with_capacity(chunk * 80 * 4),
            chunk,
            in_utterance: false,
            metrics: Metrics {
                chunk_audio: Duration::from_secs_f64(hop * chunk as f64),
                ..Default::default()
            },
            rec,
        }
    }

    pub fn recognizer_name(&self) -> String {
        self.rec.name()
    }

    /// Feed 16 kHz mono audio; returns any events produced.
    pub fn push_audio(&mut self, samples: &[f32], events: &mut Vec<Event>) -> Result<()> {
        let start = self.feats.len();
        let n_new = self.fbank.push(samples, &mut self.feats);
        let mut end_of_utt = false;
        for f in 0..n_new {
            let frame = &self.feats[start + f * self.n_mels..start + (f + 1) * self.n_mels];
            match self.endpointer.update(frame) {
                endpoint::State::Speech => self.in_utterance = true,
                endpoint::State::EndOfUtterance if self.in_utterance => end_of_utt = true,
                _ => {}
            }
        }

        let step = self.chunk * self.n_mels;
        while self.feats.len() >= step {
            let t0 = Instant::now();
            let out = self.rec.accept_chunk(&self.feats[..step])?;
            self.metrics.record(t0.elapsed());
            self.feats.drain(..step);
            if let Some(text) = out {
                events.push(Event::Partial(text));
            }
        }

        if end_of_utt {
            let text = self.rec.finalize()?;
            if !text.trim().is_empty() {
                events.push(Event::Final(text));
            }
            self.in_utterance = false;
        }
        Ok(())
    }

    /// Force end of utterance (e.g. push-to-talk released).
    pub fn flush(&mut self, events: &mut Vec<Event>) -> Result<()> {
        let text = self.rec.finalize()?;
        if !text.trim().is_empty() {
            events.push(Event::Final(text));
        }
        self.feats.clear();
        self.fbank.reset();
        self.in_utterance = false;
        Ok(())
    }
}

/// Which compute device to run the model on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    /// Best available GPU provider compiled in, then CPU.
    #[default]
    Auto,
    Cpu,
    Cuda,
    TensorRt,
    DirectMl,
}

impl std::str::FromStr for Device {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "auto" => Self::Auto,
            "cpu" => Self::Cpu,
            "cuda" => Self::Cuda,
            "tensorrt" | "trt" => Self::TensorRt,
            "directml" | "dml" => Self::DirectMl,
            _ => anyhow::bail!("unknown device {s:?} (auto|cpu|cuda|tensorrt|directml)"),
        })
    }
}

/// Load a recognizer: a model directory (ONNX) or `"dummy"`.
pub fn load_recognizer(model: &str, device: Device) -> Result<Box<dyn Recognizer>> {
    if model == "dummy" {
        return Ok(Box::new(dummy::DummyRecognizer::default()));
    }
    #[cfg(feature = "onnx")]
    {
        Ok(Box::new(onnx::OnnxCtcRecognizer::load(
            std::path::Path::new(model),
            device,
        )?))
    }
    #[cfg(not(feature = "onnx"))]
    {
        let _ = device;
        anyhow::bail!("built without the `onnx` feature; rebuild with --features cuda (or onnx)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dummy_pipeline_emits_partial_and_final() {
        let mut p = Pipeline::new(
            Box::new(dummy::DummyRecognizer::default()),
            EndpointConfig {
                trailing_silence_ms: 300.0,
                ..Default::default()
            },
        );
        let mut ev = Vec::new();
        // deterministic white-ish noise: quiet room, then loud "speech", then quiet again
        let mut seed = 12345u32;
        let mut noise = |amp: f32, n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    ((seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * amp
                })
                .collect()
        };
        p.push_audio(&noise(0.002, 8000), &mut ev).unwrap();
        p.push_audio(&noise(0.5, 16000), &mut ev).unwrap();
        p.push_audio(&noise(0.002, 16000), &mut ev).unwrap();
        assert!(ev.iter().any(|e| matches!(e, Event::Partial(_))), "{ev:?}");
        assert!(ev.iter().any(|e| matches!(e, Event::Final(_))), "{ev:?}");
        assert!(p.metrics.rtf() < 1.0);
    }
}
