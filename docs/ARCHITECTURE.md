# Architecture

## Goals, in priority order

1. **Understand glossectomy speech.** Accuracy on *this* population beats everything else.
2. **Lowest possible latency** from the end of a sound to text on screen.
3. **Local and private.** Health data never leaves the device.
4. **Cross-platform desktop:** Linux first, Windows next (macOS possible later).
5. **Two languages, two jobs:** Rust for everything that ships, Python for experiments.

## Process / thread model

```
┌──────────── audio thread (cpal callback, real-time) ────────────┐
│ downmix -> anti-alias + resample to 16 kHz -> push into rtrb SPSC │  no alloc, no locks
└──────────────────────────────┬──────────────────────────────────┘
                               │ lock-free ring buffer (2 s)
┌──────────── engine thread ───▼──────────────────────────────────┐
│ pop every 2 ms -> FbankStream (10 ms frames) -> endpointer        │
│ every `chunk_frames` frames -> Recognizer::accept_chunk (GPU)    │
│ -> Event::Partial / Event::Final -> channel + request_repaint()  │
└──────────────────────────────┬──────────────────────────────────┘
┌──────────── UI thread (egui on wgpu: Vulkan/DX12/Metal) ─────────┐
│ drains events, draws final lines + grey partial line, next vsync │
└──────────────────────────────────────────────────────────────────┘
```

## Key decisions (and why)

| Decision | Choice | Why | Alternatives considered |
|---|---|---|---|
| Runtime language | **Rust** | C-level speed, no GC pauses, memory-safe, one toolchain for Linux and Windows | C++ (unsafe, painful build), C (same) |
| Inference engine | **ONNX Runtime** via `ort`, loaded at runtime | One model file runs on CUDA, TensorRT, DirectML (any Windows GPU) and CPU. Microsoft ships official builds. Runtime loading avoids the glibc problems of static linking and lets the installer pick the GPU flavour | `candle` (pure Rust, GPU support less mature for streaming RNN/attention), whisper.cpp (Whisper is not streaming), TensorRT directly (NVIDIA only) |
| Model family | **Streaming CTC** first, then transducer (RNN-T) | CTC with a causal encoder has no decoder loop, so it is the cheapest per chunk and exactly streamable. Transducers are usually more accurate at equal latency, so they are a planned upgrade | Whisper and other attention encoder-decoders (need the whole utterance, high latency) |
| Features | **Rust fbank, shared with training via PyO3** | Removes the most common silent bug: training on slightly different features than inference | torchaudio Kaldi fbank in training plus a Rust re-implementation (drifts over time) |
| Normalisation | CMVN baked into the ONNX graph | The runtime stays model-agnostic | Separate stats file |
| GUI | **egui / eframe (wgpu)** | Native Rust with no web runtime. GPU-rendered, repaints on demand within one frame. Same code on Linux and Windows | Tauri (adds a webview layer and IPC hop), Qt (C++/licensing friction), GTK (weak on Windows) |
| Audio I/O | **cpal** | ALSA/PulseAudio/PipeWire, WASAPI and CoreAudio behind one API | PortAudio (C dependency) |
| Python | Training, data, evaluation, export **only** | Fast iteration. Nothing in Python ships to users | |

## GPU strategy

Streaming ASR sends **small chunks** (40–160 ms of audio) to the model. For tiny models,
the fixed cost of a GPU call (kernel launch plus host↔device copies, about 0.05–1 ms) can
exceed the compute itself, so a well-optimised CPU path can match the GPU. The GPU wins
clearly as models grow, for example a 30–100M-parameter Conformer or Zipformer, which is
where accuracy on hard speech will come from. So:

- `Device::Auto` picks the best GPU provider compiled in: TensorRT > CUDA > DirectML > CPU.
- Choosing a device explicitly makes a failed GPU setup **an error, not a silent CPU fallback**.
- Every model is **warmed up at load time** (the first CUDA run costs about 100 ms and
  should never land on the user's first word).
- `asr file` and the in-app overlay report per-chunk compute time and RTF, so device
  choices are made from measurements.

Planned GPU work: keep recurrent/attention state **on the device** between chunks
(ORT IoBinding), use FP16 or INT8 TensorRT engines, and use CUDA graphs. See ROADMAP.

## Platform matrix

| | Linux | Windows |
|---|---|---|
| Audio | ALSA / PulseAudio / PipeWire (cpal) | WASAPI (cpal) |
| GUI backend | wgpu → Vulkan / GL | wgpu → DX12 / Vulkan |
| Inference | ORT CUDA / TensorRT build | ORT **DirectML** (any GPU) or CUDA build |
| ORT library | `libonnxruntime.so` next to the binary or in `third_party/` | `onnxruntime.dll` next to the `.exe` |
| Cargo feature | `--features cuda` (or `tensorrt`) | `--features directml` (or `cuda`) |

## Repository rules

- Anything both training and the runtime do (features, decoding) lives in Rust, or a test
  checks the Python version against the Rust one.
- Every exported model must pass `asr-export --verify` (ONNX output matches PyTorch).
- Every model change is judged by `asr-eval` **per speaker**, run in streaming mode.
- Patient audio, and weights trained on it, are never committed. See DATA_ETHICS.md.
