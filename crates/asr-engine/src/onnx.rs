// SPDX-License-Identifier: GPL-3.0-or-later
//! ONNX Runtime backend for streaming CTC models.
//!
//! Model directory layout (see docs/MODEL_CONTRACT.md):
//! ```text
//! my-model/
//!   model.onnx   one streaming step: features (+ optional length) + states -> log-probs + states
//!   model.json   feature config, chunking, vocab, and (optionally) tensor names/layout
//! ```
//! Defaults match models exported by `research/` (`features`, `log_probs`,
//! `state_in_*`/`state_out_*`, time-major, blank 0). Third-party models such as
//! NVIDIA NeMo streaming FastConformer are described via the optional `io` block.

use crate::{Device, Recognizer};
use anyhow::{bail, Context, Result};
use asr_features::{FbankConfig, MelScale};
use ort::{
    session::{Session, SessionInputValue},
    value::{Tensor, TensorElementType},
};
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
    /// "htk" (default) or "slaney"
    #[serde(default)]
    pub mel_scale: Option<String>,
    #[serde(default)]
    pub log_floor: Option<f32>,
}

#[derive(Debug, Deserialize)]
pub struct StateSpec {
    pub input: String,
    pub output: String,
    /// Required when the ONNX graph declares dynamic dims for this state.
    #[serde(default)]
    pub shape: Option<Vec<i64>>,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct IoSpec {
    pub features: String,
    /// "time_major" = [1, T, n_mels] (default), "mel_major" = [1, n_mels, T]
    pub layout: String,
    /// Optional int64 [1] input holding the number of frames in the window.
    pub length: Option<String>,
    pub log_probs: String,
    /// Explicit state pairs. If empty, `state_in_*` / `state_out_*` are discovered.
    pub states: Vec<StateSpec>,
}

impl Default for IoSpec {
    fn default() -> Self {
        Self {
            features: "features".into(),
            layout: "time_major".into(),
            length: None,
            log_probs: "log_probs".into(),
            states: Vec::new(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ModelMeta {
    pub format_version: u32,
    pub name: String,
    pub feature: FeatureMeta,
    /// Frames per call (window).
    pub chunk_frames: usize,
    /// Frames to advance per call; defaults to `chunk_frames` (no overlap).
    #[serde(default)]
    pub chunk_shift: Option<usize>,
    pub blank_id: usize,
    /// Token strings. SentencePiece-style "▁" marks a word start; "<space>" or " " is a literal space.
    pub vocab: Vec<String>,
    #[serde(default)]
    pub io: IoSpec,
}

enum StateData {
    F32(Vec<f32>),
    I64(Vec<i64>),
}

struct State {
    in_name: String,
    out_name: String,
    shape: Vec<i64>,
    data: StateData,
}

impl State {
    fn zero(&mut self) {
        match &mut self.data {
            StateData::F32(v) => v.fill(0.0),
            StateData::I64(v) => v.fill(0),
        }
    }
}

pub struct OnnxCtcRecognizer {
    session: Session,
    meta: ModelMeta,
    mel_major: bool,
    states: Vec<State>,
    scratch: Vec<f32>,
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
    candidates.extend(crate::resolve_path(
        Path::new("third_party/onnxruntime/lib").join(name),
    ));
    candidates.into_iter().find(|p| p.exists())
}

/// cuDNN is needed by the CUDA provider for convolutions. Users should not have to set
/// LD_LIBRARY_PATH, so if a bundled copy exists (`third_party/cudnn/lib` or `cudnn/` next to
/// the executable) we load it globally first; ORT's later dlopen then finds it by soname.
#[cfg(feature = "cuda")]
fn preload_cudnn() {
    use std::sync::OnceLock;
    static LIBS: OnceLock<Vec<libloading::Library>> = OnceLock::new();
    LIBS.get_or_init(|| {
        let mut dirs: Vec<std::path::PathBuf> = crate::resolve_path("third_party/cudnn/lib")
            .into_iter()
            .collect();
        if let Some(d) = std::env::current_exe()
            .ok()
            .and_then(|e| e.parent().map(Path::to_path_buf))
        {
            dirs.insert(0, d.join("cudnn"));
        }
        let Some(dir) = dirs.into_iter().find(|d| d.exists()) else {
            return Vec::new(); // rely on system cuDNN
        };
        let names: &[&str] = if cfg!(windows) {
            &["cudnn64_9.dll"]
        } else {
            // sub-libraries first so the main library resolves against them
            &[
                "libcudnn_graph.so.9",
                "libcudnn_engines_precompiled.so.9",
                "libcudnn_engines_runtime_compiled.so.9",
                "libcudnn_heuristic.so.9",
                "libcudnn_ops.so.9",
                "libcudnn_cnn.so.9",
                "libcudnn_adv.so.9",
                "libcudnn.so.9",
            ]
        };
        let mut libs = Vec::new();
        for n in names {
            let p = dir.join(n);
            #[cfg(unix)]
            let lib = unsafe {
                libloading::os::unix::Library::open(
                    Some(&p),
                    libloading::os::unix::RTLD_NOW | libloading::os::unix::RTLD_GLOBAL,
                )
                .map(libloading::Library::from)
            };
            #[cfg(windows)]
            let lib = unsafe { libloading::Library::new(&p) };
            match lib {
                Ok(l) => libs.push(l),
                Err(e) => log::warn!("could not preload {p:?}: {e}"),
            }
        }
        log::info!("preloaded {} cuDNN libraries from {dir:?}", libs.len());
        libs
    });
}

/// ONNX Runtime allows one environment per process; every session (model switches, CPU
/// fallback) must share it.
fn shared_env() -> Result<ort::environment::Environment> {
    static ENV: std::sync::Mutex<Option<ort::environment::Environment>> =
        std::sync::Mutex::new(None);
    let mut g = ENV.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(e) = g.as_ref() {
        return Ok(e.clone());
    }
    // Telemetry off: this app processes medical speech; nothing leaves the machine.
    let env = ort_builder()?
        .with_name("asr")
        .with_telemetry(false)
        .build()?;
    *g = Some(env.clone());
    Ok(env)
}

fn ort_builder() -> Result<ort::environment::EnvironmentBuilder> {
    #[cfg(feature = "cuda")]
    preload_cudnn();
    let path = find_onnxruntime().context(
        "ONNX Runtime library not found: run scripts/fetch-onnxruntime.sh or set ORT_DYLIB_PATH",
    )?;
    log::info!("using ONNX Runtime at {path:?}");
    Ok(ort::init_from(&path)?)
}

fn build_session(path: &Path, device: Device) -> Result<(Session, &'static str)> {
    use ort::ep;
    // Try providers in order; ORT falls back to CPU for unsupported ops.
    let mut eps: Vec<(ep::ExecutionProviderDispatch, &'static str)> = Vec::new();
    let want = |d: Device| device == Device::Auto || device == d;
    #[cfg(feature = "tensorrt")]
    if want(Device::TensorRt) {
        eps.push((
            ep::TensorRT::default()
                .with_fp16(true)
                .with_engine_cache(true)
                .with_engine_cache_path(
                    crate::resolve_path("models")
                        .unwrap_or_else(|| "models".into())
                        .join(".trt-cache")
                        .to_string_lossy(),
                )
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
        .map(|(e, _)| if strict { e.error_on_failure() } else { e })
        .collect();
    let env = shared_env()?;
    let session = Session::builder(&env)?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
        .map_err(ort::Error::<()>::from)?
        // Single-stream, tiny batches: extra CPU threads just add sync overhead.
        .with_intra_threads(2)
        .map_err(ort::Error::<()>::from)?
        .with_execution_providers(eps)
        .map_err(ort::Error::<()>::from)?
        .commit_from_file(path)
        .with_context(|| format!("loading {path:?}"))?;
    Ok((session, label))
}

impl OnnxCtcRecognizer {
    pub fn load(dir: &Path, device: Device) -> Result<Self> {
        match Self::load_on(dir, device) {
            // Auto must work on any machine: if the GPU path fails (no NVIDIA GPU, driver
            // missing, out of memory...), run on the CPU instead of refusing to start.
            Err(e) if device == Device::Auto => {
                log::warn!("GPU unavailable ({e:#}); falling back to CPU");
                Self::load_on(dir, Device::Cpu)
            }
            r => r,
        }
    }

    fn load_on(dir: &Path, device: Device) -> Result<Self> {
        let meta: ModelMeta = serde_json::from_reader(
            std::fs::File::open(dir.join("model.json")).context("open model.json")?,
        )?;
        if meta.format_version != 1 {
            bail!("unsupported model format_version {}", meta.format_version);
        }
        let mel_major = match meta.io.layout.as_str() {
            "time_major" => false,
            "mel_major" => true,
            other => bail!("unknown io.layout {other:?}"),
        };
        let (session, device_used) = build_session(&dir.join("model.onnx"), device)?;

        // State pairs: explicit, or discovered by naming convention.
        let specs: Vec<StateSpec> = if meta.io.states.is_empty() {
            session
                .inputs()
                .iter()
                .filter_map(|i| i.name().strip_prefix("state_in_"))
                .map(|k| StateSpec {
                    input: format!("state_in_{k}"),
                    output: format!("state_out_{k}"),
                    shape: None,
                })
                .collect()
        } else {
            meta.io.states.iter().map(clone_spec).collect()
        };
        let mut states = Vec::new();
        for spec in specs {
            let outlet = session
                .inputs()
                .iter()
                .find(|i| i.name() == spec.input)
                .with_context(|| format!("model has no input {:?}", spec.input))?;
            let declared: Vec<i64> = outlet
                .dtype()
                .tensor_shape()
                .context("state must be a tensor")?
                .to_vec();
            let shape = match spec.shape {
                Some(s) => s,
                None if declared.iter().skip(1).all(|&d| d >= 0) => {
                    declared.iter().map(|&d| d.max(1)).collect() // dynamic batch -> 1
                }
                None => bail!(
                    "state {:?} has dynamic dims {declared:?}; give its shape in model.json",
                    spec.input
                ),
            };
            let n = shape.iter().product::<i64>() as usize;
            let data = match outlet.dtype().tensor_type() {
                Some(TensorElementType::Float32) => StateData::F32(vec![0.0; n]),
                Some(TensorElementType::Int64) => StateData::I64(vec![0; n]),
                t => bail!("state {:?}: unsupported dtype {t:?}", spec.input),
            };
            states.push(State {
                in_name: spec.input,
                out_name: spec.output,
                shape,
                data,
            });
        }
        log::info!(
            "loaded {:?} on {device_used}: {} states, window {} / shift {} frames, vocab {}",
            meta.name,
            states.len(),
            meta.chunk_frames,
            meta.chunk_shift.unwrap_or(meta.chunk_frames),
            meta.vocab.len()
        );
        let mut me = Self {
            session,
            meta,
            mel_major,
            states,
            scratch: Vec::new(),
            tokens: Vec::new(),
            last_token: None,
            device_used,
        };
        me.warm_up()?;
        Ok(me)
    }

    /// First GPU run pays for kernel selection / memory allocation (~100 ms).
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
                "<unk>" => {}
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

fn clone_spec(s: &StateSpec) -> StateSpec {
    StateSpec {
        input: s.input.clone(),
        output: s.output.clone(),
        shape: s.shape.clone(),
    }
}

impl Recognizer for OnnxCtcRecognizer {
    fn name(&self) -> String {
        format!("{} [{}]", self.meta.name, self.device_used)
    }

    fn feature_config(&self) -> FbankConfig {
        let f = &self.meta.feature;
        let d = FbankConfig::default();
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
            log_floor: f.log_floor.unwrap_or(d.log_floor),
            mel_scale: match f.mel_scale.as_deref() {
                Some("slaney") => MelScale::Slaney,
                _ => MelScale::Htk,
            },
        }
    }

    fn chunk_frames(&self) -> usize {
        self.meta.chunk_frames
    }

    fn chunk_shift(&self) -> usize {
        self.meta.chunk_shift.unwrap_or(self.meta.chunk_frames)
    }

    fn accept_chunk(&mut self, feats: &[f32]) -> Result<Option<String>> {
        let n_mels = self.meta.feature.n_mels;
        let t = feats.len() / n_mels;
        let (shape, data) = if self.mel_major {
            self.scratch.resize(feats.len(), 0.0);
            for (ti, frame) in feats.chunks_exact(n_mels).enumerate() {
                for (m, &v) in frame.iter().enumerate() {
                    self.scratch[m * t + ti] = v;
                }
            }
            (vec![1i64, n_mels as i64, t as i64], self.scratch.clone())
        } else {
            (vec![1i64, t as i64, n_mels as i64], feats.to_vec())
        };

        let io = &self.meta.io;
        let mut inputs: Vec<(String, SessionInputValue)> = Vec::new();
        inputs.push((
            io.features.clone(),
            Tensor::from_array((shape, data))?.into(),
        ));
        if let Some(len_name) = &io.length {
            inputs.push((
                len_name.clone(),
                Tensor::from_array((vec![1i64], vec![t as i64]))?.into(),
            ));
        }
        for s in &self.states {
            let v: SessionInputValue = match &s.data {
                StateData::F32(d) => Tensor::from_array((s.shape.clone(), d.clone()))?.into(),
                StateData::I64(d) => Tensor::from_array((s.shape.clone(), d.clone()))?.into(),
            };
            inputs.push((s.in_name.clone(), v));
        }
        let outputs = self.session.run(inputs)?;

        let (shape, logp) = outputs[io.log_probs.as_str()].try_extract_tensor::<f32>()?;
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
            let out = &outputs[s.out_name.as_str()];
            match &mut s.data {
                StateData::F32(d) => d.copy_from_slice(out.try_extract_tensor::<f32>()?.1),
                StateData::I64(d) => d.copy_from_slice(out.try_extract_tensor::<i64>()?.1),
            }
        }
        drop(outputs);
        Ok((self.tokens.len() != before).then(|| self.text()))
    }

    fn finalize(&mut self) -> Result<String> {
        let text = self.text();
        self.reset();
        Ok(text)
    }

    fn commit(&mut self) -> Result<String> {
        // Keep states and last_token: the stream continues, only the text line ends.
        let text = self.text();
        self.tokens.clear();
        Ok(text)
    }

    fn reset(&mut self) {
        self.tokens.clear();
        self.last_token = None;
        self.states.iter_mut().for_each(State::zero);
    }
}
