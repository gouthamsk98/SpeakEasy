# Project notes for Claude

Real-time, local ASR for people after glossectomy. Rust runtime ships; Python is for research only.

- Build/test: `cargo build`, `cargo test`, `cargo clippy --all-targets`. GPU: `--features cuda` (Linux) / `directml` (Windows).
- ONNX Runtime is loaded at runtime (`ort` `load-dynamic`): `scripts/fetch-onnxruntime.sh` -> `third_party/onnxruntime/lib`.
  The `ort` download-binaries build needs glibc 2.38+, which this machine (Pop!_OS 22.04, glibc 2.35) lacks. Do not switch back.
- Model interface: docs/MODEL_CONTRACT.md. Features must come from `crates/asr-features` (Python uses `asr_native` bindings).
- Never commit audio, patient transcripts or trained weights (DATA_ETHICS.md).
- Licence GPL-3.0-or-later; new files get `SPDX-License-Identifier: GPL-3.0-or-later`.
- Machine: RTX 4060 Laptop 8 GB, CUDA 12.8. Default `python3` on PATH is an ESP-IDF venv; use `/usr/bin/python3` or research/.venv.
