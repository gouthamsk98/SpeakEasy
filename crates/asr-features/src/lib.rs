// SPDX-License-Identifier: GPL-3.0-or-later
//! Streaming log-mel filterbank ("fbank") features.
//!
//! This crate is the **single source of truth** for acoustic features. The Python
//! training code calls it through `bindings/python` (module `asr_native`), so the
//! model is trained on bit-identical features to what the desktop app feeds it.
//! Train/inference feature mismatch is one of the most common silent ASR bugs —
//! never re-implement this in Python.
//!
//! Streaming: push any number of samples with [`FbankStream::push`]; every complete
//! frame (25 ms window, 10 ms hop by default) is appended to the output. No lookahead,
//! so the features add zero algorithmic latency beyond one window.

use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use std::sync::Arc;

/// Feature configuration. Must match `research/configs/*.yaml` → stored in model metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct FbankConfig {
    pub sample_rate: u32,
    /// Window length in samples (400 = 25 ms @ 16 kHz).
    pub frame_length: usize,
    /// Hop in samples (160 = 10 ms @ 16 kHz).
    pub frame_shift: usize,
    pub n_fft: usize,
    pub n_mels: usize,
    pub f_min: f32,
    pub f_max: f32,
    pub preemphasis: f32,
    pub remove_dc: bool,
    pub log_floor: f32,
}

impl Default for FbankConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            frame_length: 400,
            frame_shift: 160,
            n_fft: 512,
            n_mels: 80,
            f_min: 20.0,
            f_max: 7_600.0,
            preemphasis: 0.97,
            remove_dc: true,
            log_floor: 1e-10,
        }
    }
}

fn hz_to_mel(hz: f32) -> f32 {
    1127.0 * (1.0 + hz / 700.0).ln()
}
#[cfg(test)]
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * ((mel / 1127.0).exp() - 1.0)
}

/// Sparse triangular mel filter: (first FFT bin, weights).
struct MelFilter {
    start: usize,
    weights: Vec<f32>,
}

fn build_mel_bank(cfg: &FbankConfig) -> Vec<MelFilter> {
    let n_bins = cfg.n_fft / 2 + 1;
    let bin_hz = cfg.sample_rate as f32 / cfg.n_fft as f32;
    let (mlo, mhi) = (hz_to_mel(cfg.f_min), hz_to_mel(cfg.f_max));
    let centers: Vec<f32> = (0..cfg.n_mels + 2)
        .map(|i| mlo + (mhi - mlo) * i as f32 / (cfg.n_mels + 1) as f32)
        .collect();
    (0..cfg.n_mels)
        .map(|m| {
            let (l, c, r) = (centers[m], centers[m + 1], centers[m + 2]);
            let mut start = usize::MAX;
            let mut weights = Vec::new();
            for b in 0..n_bins {
                let mel = hz_to_mel(b as f32 * bin_hz);
                let w = if mel > l && mel <= c {
                    (mel - l) / (c - l)
                } else if mel > c && mel < r {
                    (r - mel) / (r - c)
                } else {
                    0.0
                };
                if w > 0.0 {
                    if start == usize::MAX {
                        start = b;
                    }
                    // fill gaps (shouldn't happen for triangles, but stay contiguous)
                    weights.resize(b - start, 0.0);
                    weights.push(w);
                }
            }
            MelFilter {
                start: start.min(n_bins),
                weights,
            }
        })
        .collect()
}

/// Stateful streaming fbank extractor. Allocation-free after construction
/// (except output growth), safe to run on a real-time-ish worker thread.
pub struct FbankStream {
    cfg: FbankConfig,
    window: Vec<f32>,
    mel: Vec<MelFilter>,
    fft: Arc<dyn Fft<f32>>,
    pending: Vec<f32>,
    frame: Vec<f32>,
    spec: Vec<Complex32>,
    scratch: Vec<Complex32>,
    power: Vec<f32>,
}

impl FbankStream {
    pub fn new(cfg: FbankConfig) -> Self {
        assert!(
            cfg.n_fft >= cfg.frame_length,
            "n_fft must be >= frame_length"
        );
        let fft = FftPlanner::<f32>::new().plan_fft_forward(cfg.n_fft);
        // Povey-ish window (Hann^0.85) is Kaldi's default; plain Hann is fine & standard.
        let window = (0..cfg.frame_length)
            .map(|i| {
                0.5 - 0.5
                    * (2.0 * std::f32::consts::PI * i as f32 / (cfg.frame_length - 1) as f32).cos()
            })
            .collect();
        let scratch = vec![Complex32::default(); fft.get_inplace_scratch_len()];
        Self {
            mel: build_mel_bank(&cfg),
            window,
            fft,
            pending: Vec::with_capacity(cfg.frame_length * 4),
            frame: vec![0.0; cfg.frame_length],
            spec: vec![Complex32::default(); cfg.n_fft],
            scratch,
            power: vec![0.0; cfg.n_fft / 2 + 1],
            cfg,
        }
    }

    pub fn config(&self) -> &FbankConfig {
        &self.cfg
    }

    pub fn n_mels(&self) -> usize {
        self.cfg.n_mels
    }

    /// Clear buffered audio (e.g. at utterance boundary).
    pub fn reset(&mut self) {
        self.pending.clear();
    }

    /// Push mono f32 samples in [-1, 1] at `cfg.sample_rate`. Appends
    /// `n_new_frames * n_mels` values (row-major, time-major) to `out`.
    /// Returns number of new frames.
    pub fn push(&mut self, samples: &[f32], out: &mut Vec<f32>) -> usize {
        self.pending.extend_from_slice(samples);
        let (len, hop) = (self.cfg.frame_length, self.cfg.frame_shift);
        let mut produced = 0;
        let mut offset = 0;
        while offset + len <= self.pending.len() {
            self.frame
                .copy_from_slice(&self.pending[offset..offset + len]);
            self.compute_frame(out);
            produced += 1;
            offset += hop;
        }
        self.pending.drain(..offset);
        produced
    }

    fn compute_frame(&mut self, out: &mut Vec<f32>) {
        let cfg = &self.cfg;
        let f = &mut self.frame;
        if cfg.remove_dc {
            let mean = f.iter().sum::<f32>() / f.len() as f32;
            f.iter_mut().for_each(|x| *x -= mean);
        }
        if cfg.preemphasis > 0.0 {
            for i in (1..f.len()).rev() {
                f[i] -= cfg.preemphasis * f[i - 1];
            }
            f[0] *= 1.0 - cfg.preemphasis;
        }
        for (i, s) in self.spec.iter_mut().enumerate() {
            *s = if i < f.len() {
                Complex32::new(f[i] * self.window[i], 0.0)
            } else {
                Complex32::default()
            };
        }
        self.fft
            .process_with_scratch(&mut self.spec, &mut self.scratch);
        for (p, c) in self.power.iter_mut().zip(&self.spec) {
            *p = c.norm_sqr();
        }
        for filt in &self.mel {
            let e: f32 = filt
                .weights
                .iter()
                .zip(&self.power[filt.start..])
                .map(|(w, p)| w * p)
                .sum();
            out.push(e.max(cfg.log_floor).ln());
        }
    }
}

/// Convenience: whole-utterance features (used by Python bindings / offline eval).
pub fn fbank(samples: &[f32], cfg: &FbankConfig) -> (Vec<f32>, usize) {
    let mut s = FbankStream::new(cfg.clone());
    let mut out = Vec::new();
    let n = s.push(samples, &mut out);
    (out, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(hz: f32, secs: f32) -> Vec<f32> {
        let sr = 16_000.0;
        (0..(sr * secs) as usize)
            .map(|i| (2.0 * std::f32::consts::PI * hz * i as f32 / sr).sin() * 0.5)
            .collect()
    }

    #[test]
    fn frame_count() {
        let cfg = FbankConfig::default();
        let (feats, n) = fbank(&vec![0.0; 16_000], &cfg);
        assert_eq!(n, (16_000 - 400) / 160 + 1);
        assert_eq!(feats.len(), n * 80);
    }

    #[test]
    fn streaming_equals_batch() {
        let cfg = FbankConfig::default();
        let audio = sine(440.0, 1.0);
        let (batch, nb) = fbank(&audio, &cfg);
        let mut s = FbankStream::new(cfg);
        let mut out = Vec::new();
        let mut ns = 0;
        for chunk in audio.chunks(137) {
            ns += s.push(chunk, &mut out); // odd chunk size on purpose
        }
        assert_eq!(nb, ns);
        for (a, b) in batch.iter().zip(&out) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn tone_peaks_in_right_band() {
        let cfg = FbankConfig::default();
        let (feats, n) = fbank(&sine(1000.0, 0.5), &cfg);
        let mid = &feats[(n / 2) * 80..(n / 2 + 1) * 80];
        let argmax = mid
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        let lo = mel_to_hz(
            hz_to_mel(20.0) + (hz_to_mel(7600.0) - hz_to_mel(20.0)) * argmax as f32 / 81.0,
        );
        let hi = mel_to_hz(
            hz_to_mel(20.0) + (hz_to_mel(7600.0) - hz_to_mel(20.0)) * (argmax + 2) as f32 / 81.0,
        );
        assert!(
            lo < 1000.0 && 1000.0 < hi,
            "peak bin {argmax} spans {lo}-{hi} Hz"
        );
    }
}
