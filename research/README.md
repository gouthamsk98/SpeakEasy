# research/: experiments & training (Python)

Nothing here ships to users. Models reach the app through `asr-export` → ONNX
(see ../docs/MODEL_CONTRACT.md).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
(cd ../bindings/python && maturin develop --release)   # Rust features -> `import asr_native`
pytest
```

| Command | What it does |
|---|---|
| `asr-prepare` | wav+txt folder → JSONL manifest (with consent ID) + tokens.txt |
| `asr-train CONFIG [--init-from ckpt --freeze-layers N]` | train / personalise a model |
| `asr-eval CKPT MANIFEST` | **streaming** evaluation, CER/WER per speaker |
| `asr-export CKPT OUT_DIR --verify` | ONNX export + numerical check against PyTorch |

Put new model families in `src/asr_research/` next to `model.py`. Each one must provide a
`forward_chunk(features, *states)` step and pass the streaming-equivalence test.
