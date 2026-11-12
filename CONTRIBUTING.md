# Contributing

Thank you! This project exists to help people who lost part of their voice to cancer
surgery, and every improvement matters to them.

## Ground rules
- Read [DATA_ETHICS.md](DATA_ETHICS.md) before touching any recording.
- By contributing, you agree that your contribution is licensed under **GPL-3.0-or-later**.
- Sign off your commits (Developer Certificate of Origin, https://developercertificate.org):
  `git commit -s` adds `Signed-off-by: Your Name <you@example.com>`.

## Where to put code
- Anything that ships to users goes in **Rust** (`crates/`). Performance-critical code
  should be allocation-free on the audio and engine threads.
- Experiments go in **Python** (`research/`). If an experiment wins, its model reaches the
  app through the ONNX [model contract](docs/MODEL_CONTRACT.md). Logic that has to run at
  inference time is ported to Rust.
- Never re-implement feature extraction in Python. Use `asr_native`.

## Before opening a PR
```bash
cargo fmt --all && cargo clippy --all-targets -- -D warnings && cargo test
cd research && ruff check . && pytest
```
For model changes, include `asr-eval` output (per speaker, streaming) and latency numbers
from `asr file`.

## New source files
Start each source file with:
```
// SPDX-License-Identifier: GPL-3.0-or-later
```
(`#` comment in Python.)
