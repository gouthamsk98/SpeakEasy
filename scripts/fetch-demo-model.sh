#!/usr/bin/env bash
# Pretrained English streaming model for testing the real-time pipeline on normal speech.
# NVIDIA NeMo FastConformer hybrid (CTC branch), cache-aware streaming, 80 ms lookahead,
# exported to ONNX by the sherpa-onnx project. Licence: CC-BY-4.0 (NVIDIA NGC).
#   scripts/fetch-demo-model.sh          # fp32 (~430 MB) - best for GPU
#   scripts/fetch-demo-model.sh int8     # int8 (~100 MB) - best for CPU-only machines
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-80ms"
[[ "${1:-}" == "int8" ]] && NAME="$NAME-int8"
URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$NAME.tar.bz2"
mkdir -p "$ROOT/models"
echo "downloading $URL"
curl -fL --progress-bar "$URL" | tar -xj -C "$ROOT/models"
# int8 archives name the file model.int8.onnx; the runtime expects model.onnx
[[ -f "$ROOT/models/$NAME/model.int8.onnx" ]] && mv "$ROOT/models/$NAME/model.int8.onnx" "$ROOT/models/$NAME/model.onnx"

PY="$ROOT/research/.venv/bin/python"; [[ -x "$PY" ]] || PY=python3
"$PY" -c "import onnx" 2>/dev/null || { echo "need the 'onnx' python package: $PY -m pip install onnx" >&2; exit 1; }
PYTHONPATH="$ROOT/research/src" "$PY" -m asr_research.import_sherpa_nemo "$ROOT/models/$NAME"
echo
echo "try:  cargo run --release -p asr-app --features cuda -- models/$NAME cuda"
