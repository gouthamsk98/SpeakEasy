// SPDX-License-Identifier: GPL-3.0-or-later
//! A model-free recognizer. Shows a growing bar while you speak so the entire
//! mic -> features -> UI path (and its latency) can be tested before any model exists.

use crate::Recognizer;
use anyhow::Result;
use asr_features::FbankConfig;

#[derive(Default)]
pub struct DummyRecognizer {
    level: Vec<char>,
    floor: Option<f32>,
}

impl Recognizer for DummyRecognizer {
    fn name(&self) -> String {
        "dummy (no model: shows voice activity)".into()
    }
    fn feature_config(&self) -> FbankConfig {
        FbankConfig::default()
    }
    fn chunk_frames(&self) -> usize {
        4 // 40 ms
    }
    fn accept_chunk(&mut self, feats: &[f32]) -> Result<Option<String>> {
        let e = feats.iter().sum::<f32>() / feats.len() as f32;
        // Adaptive noise floor: voiced if clearly above it.
        let floor = *self.floor.get_or_insert(e);
        self.floor = Some(if e < floor {
            e
        } else {
            0.995 * floor + 0.005 * e
        });
        if e > floor + 2.5 {
            self.level.push('▮');
            if self.level.len() > 60 {
                self.level.remove(0);
            }
            return Ok(Some(self.level.iter().collect()));
        }
        Ok(None)
    }
    fn finalize(&mut self) -> Result<String> {
        let n = self.level.len();
        self.level.clear();
        Ok(if n > 0 {
            format!("[speech {:.2} s]", n as f32 * 0.04)
        } else {
            String::new()
        })
    }
    fn reset(&mut self) {
        self.level.clear();
    }
}
