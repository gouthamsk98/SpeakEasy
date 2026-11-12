// SPDX-License-Identifier: GPL-3.0-or-later
//! ONNX Runtime backend for streaming CTC models exported by `research/`.
//!
//! Model directory layout (see docs/MODEL_CONTRACT.md):
//! ```text
//! my-model/
//!   model.onnx   inputs : features [1, chunk_frames, n_mels] f32, state_in_*  (any shapes)
//!                outputs: log_probs [1, T', vocab] f32,           state_out_* (same as in)
//!   model.json   feature config, chunk_frames, blank_id, vocab
//! ```
//! Feature normalisation (CMVN) is baked into the ONNX graph, so the runtime stays dumb.

use crate::{Device, Recognizer};
use anyhow::{bail, Context, Result};
use asr_features::FbankConfig;
use ort::{session::Session, value::Tensor};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct FeatureMeta {
    pub sample_rate: u32,
    pub frame_length: usize,
    pub frame_shift: usize,
    pub n_fft: usize,
    pub n_mels: usize,
    pub f_min: f32,
    pub f_max: f32,
    pub preemphasis: f32,
    pub remove_dc: bool,
}

#[derive(Debug, Deserialize)]
pub struct ModelMeta {
    pub format_version: u32,
    pub name: String,
    pub feature: FeatureMeta,
    pub chunk_frames: usize,
    pub blank_id: usize,
    /// Token strings. SentencePiece-style "▁" marks a word start; "<space>" or " " is a literal space.
    pub vocab: Vec<String>,
}

struct State {
    in_name: String,
    out_name: String,
    shape: Vec<i64>,
    data: Vec<f32>,
}

pub struct OnnxCtcRecognizer {
    session: Session,
    meta: ModelMeta,
    states: Vec<State>,
    tokens: Vec<usize>,
    last_token: Option<usize>,
    device_used: &'static str,
}

/// ONNX Runtime is loaded at runtime (`load-dynamic`) so we can ship Microsoft's
/// official builds: CUDA build on Linux, DirectML build on Windows.
/// Search order: $ORT_DYLIB_PATH, next to the executable, third_party/onnxruntime/lib.
pub fn find_onnxruntime() -> Option<std::path::PathBuf> {
    let name = if cfg!(windows) {
        "onnxruntime.dll"
    } else {
        "libonnxruntime.so"
    };
    let mut candidates = Vec::new();
    if let Ok(p) = std::env::var("ORT_DYLIB_PATH") {
        candidates.push(std::path::PathBuf::from(p));
    }
    if let Some(dir) = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(Path::to_path_buf))
    {
        candidates.push(dir.join(name));
        candidates.push(dir.join("lib").join(name));
    }
    candidates.push(Path::new("third_party/onnxruntime/lib").join(name));
    candidates.into_iter().find(|p| p.exists())
}

fn ort_builder() -> Result<ort::environment::EnvironmentBuilder> {
    let path = find_onnxruntime().context(
        "ONNX Runtime library not found: run scripts/fetch-onnxruntime.sh or set ORT_DYLIB_PATH",
    )?;
    log::info!("using ONNX Runtime at {path:?}");
    Ok(ort::init_from(&path)?)
}

fn build_session(path: &Path, device: Device) -> Result<(Session, &'static str)> {
    use ort::ep;
    // Try providers in order; ORT silently falls back to CPU for unsupported ops/EPs.
    let mut eps: Vec<(ep::ExecutionProviderDispatch, &'static str)> = Vec::new();
    let want = |d: Device| device == Device::Auto || device == d;
    #[cfg(feature = "tensorrt")]
    if want(Device::TensorRt) {
        eps.push((
            ep::TensorRT::default()
                .with_fp16(true)
                .with_engine_cache(true)
                .with_engine_cache_path("models/.trt-cache")
                .build(),
            "tensorrt",
        ));
    }
    #[cfg(feature = "cuda")]
    if want(Device::Cuda) || device == Device::TensorRt {
        eps.push((ep::CUDA::default().build(), "cuda"));
    }
    #[cfg(feature = "directml")]
    if want(Device::DirectMl) {
        eps.push((ep::DirectML::default().build(), "directml"));
    }
    if device != Device::Cpu && device != Device::Auto && eps.is_empty() {
        bail!("device {device:?} requested but its cargo feature is not enabled");
    }
    let label = eps.first().map(|e| e.1).unwrap_or("cpu");
    // Explicitly requested device: fail loudly instead of silently running on CPU.
    let strict = device != Device::Auto;
    let eps: Vec<_> = eps
        .into_iter()
        .map(|(e, l)| (if strict { e.error_on_failure() } else { e }, l))
        .collect();
    // Telemetry off: this app processes medical speech; nothing leaves the machine.
    let env = ort_builder()?
        .with_name("asr")
        .with_telemetry(false)
        .build()?;
    let session = Session::builder(&env)?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
        .map_err(ort::Error::<()>::from)?
        // Single-stream, tiny batches: extra CPU threads just add sync overhead.
        .with_intra_threads(2)
        .map_err(ort::Error::<()>::from)?
        .with_execution_providers(eps.into_iter().map(|e| e.0).collect::<Vec<_>>())
        .map_err(ort::Error::<()>::from)?
        .commit_from_file(path)
        .with_context(|| format!("loading {path:?}"))?;
    Ok((session, label))
}

impl OnnxCtcRecognizer {
    pub fn load(dir: &Path, device: Device) -> Result<Self> {
        let meta: ModelMeta = serde_json::from_reader(
            std::fs::File::open(dir.join("model.json")).context("open model.json")?,
        )?;
        if meta.format_version != 1 {
            bail!("unsupported model format_version {}", meta.format_version);
        }
        let (session, device_used) = build_session(&dir.join("model.onnx"), device)?;

        let mut states = Vec::new();
        for input in session.inputs() {
            let name = input.name();
            if let Some(suffix) = name.strip_prefix("state_in_") {
                let shape: Vec<i64> = input
                    .dtype()
                    .tensor_shape()
                    .context("state input must be a tensor")?
                    .iter()
                    .map(|&d| if d < 0 { 1 } else { d }) // dynamic batch -> 1
                    .collect();
                let n = shape.iter().product::<i64>() as usize;
                states.push(State {
                    in_name: name.to_string(),
                    out_name: format!("state_out_{suffix}"),
                    shape,
                    data: vec![0.0; n],
                });
            }
        }
        log::info!(
            "loaded {:?} on {device_used}: {} states, chunk {} frames, vocab {}",
            meta.name,
            states.len(),
            meta.chunk_frames,
            meta.vocab.len()
        );
        let mut me = Self {
            session,
            meta,
            states,
            tokens: Vec::new(),
            last_token: None,
            device_used,
        };
        me.warm_up()?;
        Ok(me)
    }

    /// First GPU run pays for kernel compilation / memory allocation (~100 ms).
    /// Pay it at load time, not on the user's first word.
    fn warm_up(&mut self) -> Result<()> {
        let n = self.meta.chunk_frames * self.meta.feature.n_mels;
        let t0 = std::time::Instant::now();
        for _ in 0..3 {
            self.accept_chunk(&vec![-10.0; n])?;
        }
        self.reset();
        log::info!("warm-up took {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
        Ok(())
    }

    fn text(&self) -> String {
        let mut s = String::new();
        for &t in &self.tokens {
            let piece = &self.meta.vocab[t];
            match piece.as_str() {
                "<space>" | " " => s.push(' '),
                p if p.starts_with('▁') => {
                    if !s.is_empty() {
                        s.push(' ');
                    }
                    s.push_str(&p['▁'.len_utf8()..]);
                }
                p => s.push_str(p),
            }
        }
        s
    }
}

impl Recognizer for OnnxCtcRecognizer {
    fn name(&self) -> String {
        format!("{} [{}]", self.meta.name, self.device_used)
    }

    fn feature_config(&self) -> FbankConfig {
        let f = &self.meta.feature;
        FbankConfig {
            sample_rate: f.sample_rate,
            frame_length: f.frame_length,
            frame_shift: f.frame_shift,
            n_fft: f.n_fft,
            n_mels: f.n_mels,
            f_min: f.f_min,
            f_max: f.f_max,
            preemphasis: f.preemphasis,
            remove_dc: f.remove_dc,
            ..FbankConfig::default()
        }
    }

    fn chunk_frames(&self) -> usize {
        self.meta.chunk_frames
    }

    fn accept_chunk(&mut self, feats: &[f32]) -> Result<Option<String>> {
        let n_mels = self.meta.feature.n_mels;
        let t = feats.len() / n_mels;
        let mut inputs: Vec<(String, ort::session::SessionInputValue)> = Vec::new();
        inputs.push((
            "features".into(),
            Tensor::from_array((vec![1i64, t as i64, n_mels as i64], feats.to_vec()))?.into(),
        ));
        for s in &self.states {
            inputs.push((
                s.in_name.clone(),
                Tensor::from_array((s.shape.clone(), s.data.clone()))?.into(),
            ));
        }
        let outputs = self.session.run(inputs)?;

        let (shape, logp) = outputs["log_probs"].try_extract_tensor::<f32>()?;
        let (frames, vocab) = (shape[1] as usize, shape[2] as usize);
        let before = self.tokens.len();
        // Greedy CTC: argmax per frame, collapse repeats, drop blanks. State carries across chunks.
        for f in 0..frames {
            let row = &logp[f * vocab..(f + 1) * vocab];
            let best = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap_or(self.meta.blank_id);
            if best != self.meta.blank_id && Some(best) != self.last_token {
                self.tokens.push(best);
            }
            self.last_token = Some(best);
        }
        for s in &mut self.states {
            let (_, d) = outputs[s.out_name.as_str()].try_extract_tensor::<f32>()?;
            s.data.copy_from_slice(d);
        }
        drop(outputs);
        Ok((self.tokens.len() != before).then(|| self.text()))
    }

    fn finalize(&mut self) -> Result<String> {
        let text = self.text();
        self.reset();
        Ok(text)
    }

    fn reset(&mut self) {
        self.tokens.clear();
        self.last_token = None;
        for s in &mut self.states {
            s.data.iter_mut().for_each(|x| *x = 0.0);
        }
    }
}
