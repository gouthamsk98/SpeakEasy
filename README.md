# SpeakEasy — real-time speech recognition for people after glossectomy

> Working name. Rename freely.

People who have had part or all of their tongue removed (glossectomy, usually for oral
cancer) often find that mainstream speech recognition does not understand them. This
project trains speech models **on their own voices** and runs them **locally** on a
laptop, putting what they say on screen as fast as the hardware allows.

* **Private by design.** Audio never leaves the machine. No cloud, and telemetry is off.
* **Fast.** The Rust runtime runs on the GPU (CUDA on Linux, DirectML/CUDA on Windows)
  and streams partial text while the person is still speaking.
* **Personal.** A base model is fine-tuned on each user's speech. For atypical speech,
  personal adaptation helps far more than anything else.
* **Free software.** GPL-3.0-or-later. If you ship a modified version, you must publish
  your source. See [License](#license).

## Layout

```
crates/                 Rust runtime: the product (low-level, fast)
  asr-features/         log-mel features, the single source of truth (also used by training)
  asr-audio/            mic capture (cpal) -> 16 kHz mono -> lock-free ring buffer
  asr-engine/           streaming pipeline, endpointing, ONNX Runtime backend (GPU)
  asr-app/              desktop app (egui, GPU-rendered): binary `speak-easy`
  asr-cli/              headless tools: `asr file|mic|devices`, latency stats
bindings/python/        PyO3 module `asr_native`: Rust features callable from Python
research/               Python: data prep, training, evaluation, ONNX export (experiments)
docs/                   architecture, latency budget, model contract, roadmap, decisions
scripts/                ONNX Runtime download (Linux .sh / Windows .ps1)
data/  models/          local only, gitignored; patient data never goes in git
```

How it fits together:

```
research/ (PyTorch)  --train-->  checkpoint  --asr-export-->  models/<name>/{model.onnx, model.json}
        ^                                                                  |
        | features via asr_native (same Rust code)                         v
crates/asr-features  <--------------------------------  crates/asr-engine (ONNX Runtime, GPU)
                                                                           |
                                         mic -> asr-audio -> pipeline -> asr-app (screen)
```

## Quick start (Linux)

```bash
# 1. Rust runtime with no model yet: shows a voice-activity bar, proves mic -> screen latency
cargo run --release -p asr-app                        # GUI
cargo run --release -p asr-cli -- mic                 # terminal

# 2. GPU inference: fetch ONNX Runtime (CUDA 12 build), then build with the cuda feature
scripts/fetch-onnxruntime.sh                          # -> third_party/onnxruntime
cargo run --release -p asr-app --features cuda -- models/<name> cuda
cargo run --release -p asr-cli --features cuda -- --model models/<name> --device cuda file some.wav

# 3. Research environment (Python 3.10+)
cd research
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'                               # torch, onnx, onnxruntime-gpu, ...
(cd ../bindings/python && maturin develop --release)  # builds `asr_native`
pytest
```

The CUDA provider also needs **cuDNN 9**. `fetch-onnxruntime.sh` downloads it into
`third_party/cudnn/lib`, and the app preloads it from there, so you don't need to set
`LD_LIBRARY_PATH`.

### Demo: real-time recognition of normal English speech

Before we have glossectomy data, use a strong pretrained streaming model to check the whole
real-time path: NVIDIA's NeMo FastConformer (CTC, cache-aware streaming, 80 ms lookahead,
CC-BY-4.0), as exported by sherpa-onnx.

```bash
scripts/fetch-onnxruntime.sh                 # once: ORT + cuDNN
scripts/fetch-demo-model.sh                  # ~430 MB; writes model.json via the importer
cargo run --release -p asr-app --features cuda -- cuda             # speak into the mic
cargo run --release -p asr-cli --features cuda -- --device cuda file \
    models/sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-80ms/test_wavs/1.wav
```

With no model argument, the first model in `models/` is used. Paths resolve from the
project folder, so the binaries work from any directory. Arguments for the app:
`speak-easy [MODEL_DIR|auto|dummy] [auto|cpu|cuda|tensorrt|directml]`.

Measured on an RTX 4060 Laptop GPU: all three bundled test clips are transcribed
word-for-word. Each 160 ms step takes about 10 ms on CUDA (RTF 0.06) and about 36 ms on CPU
(RTF 0.23). This architecture is also the planned **base model for fine-tuning on
glossectomy speech**.

### Train → export → run

```bash
cd research
asr-prepare --audio-dir ../data/raw/spk01 --speaker spk01 --consent-id C-2026-001 --out ../data/manifests
asr-train  configs/baseline_ctc.yaml
asr-eval   runs/baseline/best.pt ../data/manifests/valid.jsonl      # streaming eval, per speaker
asr-export runs/baseline/best.pt ../models/baseline --verify       # ONNX == PyTorch check
cd .. && cargo run --release -p asr-app --features cuda -- models/baseline
```

## Windows

The codebase is cross-platform by construction: cpal uses WASAPI, egui/wgpu uses DX12,
and ONNX Runtime uses **DirectML**, which works on any DX12 GPU (NVIDIA, AMD or Intel).

```powershell
powershell -ExecutionPolicy Bypass -File scripts\fetch-onnxruntime.ps1   # DirectML build
cargo run --release -p asr-app --features directml -- models\baseline directml
```

CI builds and tests on both Linux and Windows (`.github/workflows/ci.yml`).

## Status

Scaffolding is done, the end-to-end runtime has been verified with a toy model on CPU and
CUDA, and the training pipeline is written. **There is no trained model yet.** The data
work comes next. See [docs/ROADMAP.md](docs/ROADMAP.md).

## Docs

* [Architecture & key decisions](docs/ARCHITECTURE.md)
* [Latency budget: what "instant" can and can't mean](docs/LATENCY.md)
* [Model contract: what the runtime expects from an exported model](docs/MODEL_CONTRACT.md)
* [Roadmap & research directions](docs/ROADMAP.md)
* [Data ethics & consent](DATA_ETHICS.md): read before touching any recording
* [Contributing](CONTRIBUTING.md) · [Donations](DONATIONS.md)

## License

Copyright (C) 2026 the SpeakEasy contributors.

This program is free software: you can redistribute it and/or modify it under the terms of
the **GNU General Public License, version 3 or (at your option) any later version**. See
[LICENSE](LICENSE).

In plain words (the license text is what governs):
- Anyone may use, study, share and modify this code, including commercially.
- If you **distribute** this program or a modified version, you must release the complete
  corresponding source under the same license. Closed-source forks are not allowed.
- GPLv3 rather than GPLv2 (Linux's license) because GPLv3 is compatible with the
  Apache-2.0 dependencies we use, and it adds patent and anti-tivoization protections.

Trained model weights and datasets are **not** covered automatically. See
[DATA_ETHICS.md](DATA_ETHICS.md).
