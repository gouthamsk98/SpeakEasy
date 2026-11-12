# models/ (local only, never committed)

Each model is a directory that follows ../docs/MODEL_CONTRACT.md:
```
models/<name>/model.onnx
models/<name>/model.json
```
Produced by `asr-export`. Personal models trained on a user's voice stay on that user's
machine (see ../DATA_ETHICS.md). TensorRT engine caches go to `models/.trt-cache/`.
