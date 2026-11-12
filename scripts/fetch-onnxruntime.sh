#!/usr/bin/env bash
# Download Microsoft's official ONNX Runtime into third_party/onnxruntime.
#   scripts/fetch-onnxruntime.sh          # CUDA 12 GPU build (default, NVIDIA)
#   scripts/fetch-onnxruntime.sh cpu      # CPU-only build (11 MB)
# The CUDA build also needs cuDNN 9 (for convolutions). The gpu flavour downloads it into
# third_party/cudnn/lib; the app preloads it from there, so no LD_LIBRARY_PATH is needed.
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

if [[ "$FLAVOR" == gpu && ! -f "$ROOT/third_party/cudnn/lib/libcudnn.so.9" ]]; then
  echo "downloading cuDNN 9 for CUDA 12 (~1 GB, from PyPI wheel nvidia-cudnn-cu12)"
  python3 -m pip download --quiet --no-deps --only-binary=:all: --platform manylinux_2_27_x86_64 \
    --python-version 3.10 "nvidia-cudnn-cu12>=9,<10" -d "$TMP/cudnn"
  python3 -m zipfile -e "$TMP"/cudnn/*.whl "$TMP/cudnn/x"
  mkdir -p "$ROOT/third_party/cudnn/lib"
  cp "$TMP"/cudnn/x/nvidia/cudnn/lib/*.so.9 "$ROOT/third_party/cudnn/lib/"
  ln -sf libcudnn.so.9 "$ROOT/third_party/cudnn/lib/libcudnn.so"
  echo "installed cuDNN -> third_party/cudnn/lib"
fi
