// SPDX-License-Identifier: GPL-3.0-or-later
//! Minimal streaming resampler: windowed-sinc anti-alias low-pass + linear interpolation.
//! Good enough for ASR front-ends (speech energy < 7.6 kHz). A polyphase resampler is a
//! known TODO (see docs/ROADMAP.md) — measure WER impact before bothering.

pub struct StreamingResampler {
    step: f64, // input samples per output sample
    pos: f64,  // fractional read position into `hist`
    taps: Vec<f32>,
    hist: Vec<f32>, // filtered input not yet consumed
    fir_state: Vec<f32>,
    passthrough: bool,
}

impl StreamingResampler {
    pub fn new(in_rate: u32, out_rate: u32) -> Self {
        let passthrough = in_rate == out_rate;
        let ratio = out_rate as f64 / in_rate as f64;
        let taps = if ratio < 1.0 {
            lowpass(31, ratio as f32 * 0.475)
        } else {
            vec![1.0]
        };
        Self {
            step: in_rate as f64 / out_rate as f64,
            pos: 0.0,
            fir_state: vec![0.0; taps.len()],
            taps,
            hist: Vec::with_capacity(16384),
            passthrough,
        }
    }

    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        if self.passthrough {
            out.extend_from_slice(input);
            return;
        }
        let n = self.taps.len();
        for &x in input {
            self.fir_state.rotate_left(1);
            self.fir_state[n - 1] = x;
            let y: f32 = self
                .fir_state
                .iter()
                .zip(&self.taps)
                .map(|(a, b)| a * b)
                .sum();
            self.hist.push(y);
        }
        while self.pos + 1.0 < self.hist.len() as f64 {
            let i = self.pos as usize;
            let f = (self.pos - i as f64) as f32;
            out.push(self.hist[i] * (1.0 - f) + self.hist[i + 1] * f);
            self.pos += self.step;
        }
        // pos may overshoot the buffer (next output needs samples not yet received):
        // drop only what we have, keep the remainder in `pos`.
        let consumed = (self.pos as usize).min(self.hist.len());
        self.hist.drain(..consumed);
        self.pos -= consumed as f64;
    }
}

/// Hann-windowed sinc low-pass, cutoff as a fraction of the input sample rate.
fn lowpass(n: usize, cutoff: f32) -> Vec<f32> {
    let m = (n - 1) as f32 / 2.0;
    let mut h: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 - m;
            let sinc = if t == 0.0 {
                2.0 * cutoff
            } else {
                (2.0 * std::f32::consts::PI * cutoff * t).sin() / (std::f32::consts::PI * t)
            };
            let w = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos();
            sinc * w
        })
        .collect();
    let s: f32 = h.iter().sum();
    h.iter_mut().for_each(|x| *x /= s);
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_48k_to_16k() {
        let mut r = StreamingResampler::new(48_000, 16_000);
        let mut out = Vec::new();
        for _ in 0..100 {
            r.process(&[0.1; 480], &mut out); // 100 x 10 ms
        }
        assert!((out.len() as i64 - 16_000).abs() <= 2, "{}", out.len());
    }

    #[test]
    fn odd_block_sizes_48k_and_44k1() {
        for (rate, blocks) in [
            (48_000u32, [161usize, 7, 1023, 1, 480]),
            (44_100, [441, 13, 2, 999, 100]),
        ] {
            let mut r = StreamingResampler::new(rate, 16_000);
            let mut out = Vec::new();
            let mut fed = 0;
            for _ in 0..200 {
                for &n in &blocks {
                    r.process(&vec![0.1; n], &mut out);
                    fed += n;
                }
            }
            let expect = fed as f64 * 16_000.0 / rate as f64;
            assert!(
                (out.len() as f64 - expect).abs() <= 3.0,
                "{rate}: {} vs {expect}",
                out.len()
            );
        }
    }

    #[test]
    fn passthrough() {
        let mut r = StreamingResampler::new(16_000, 16_000);
        let mut out = Vec::new();
        r.process(&[1.0, 2.0], &mut out);
        assert_eq!(out, vec![1.0, 2.0]);
    }
}
