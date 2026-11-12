#!/usr/bin/env bash
# Download Microsoft's official ONNX Runtime into third_party/onnxruntime.
#   scripts/fetch-onnxruntime.sh          # CUDA 12 GPU build (default, NVIDIA)
#   scripts/fetch-onnxruntime.sh cpu      # CPU-only build (11 MB)
# The CUDA build also needs cuDNN 9 for CUDA 12 on the library path
# (e.g. `pip install nvidia-cudnn-cu12` in research/.venv, or the system package).
set -euo pipefail
VERSION="${ORT_VERSION:-1.31.1}"
FLAVOR="${1:-gpu}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/onnxruntime"

case "$FLAVOR" in
  gpu) PKG="onnxruntime-linux-x64-gpu_cuda12-$VERSION" ;;
  cpu) PKG="onnxruntime-linux-x64-$VERSION" ;;
  *) echo "usage: $0 [gpu|cpu]" >&2; exit 1 ;;
esac

URL="https://github.com/microsoft/onnxruntime/releases/download/v$VERSION/$PKG.tgz"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
echo "downloading $URL"
curl -fL --progress-bar "$URL" -o "$TMP/ort.tgz"
tar -xzf "$TMP/ort.tgz" -C "$TMP"
rm -rf "$DEST"
mkdir -p "$DEST"
cp -r "$TMP/$PKG/." "$DEST/"
echo "installed ONNX Runtime $VERSION ($FLAVOR) -> $DEST"
ls "$DEST/lib"
