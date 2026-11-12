# Model contract (format_version 1)

Any model (baseline LSTM, Conformer, Zipformer, whatever research produces) ships to the
runtime as a directory:

```
models/<name>/
  model.onnx
  model.json
```

## model.onnx: one streaming step

| Name | Dir | Type / shape | Meaning |
|---|---|---|---|
| `features` | in | f32 `[1, chunk_frames, n_mels]` | raw log-mel from `asr-features` (no normalisation) |
| `state_in_<k>` | in | f32, any fixed shape | recurrent / attention-cache state; zeros at utterance start |
| `log_probs` | out | f32 `[1, T', vocab]` | per-output-frame log-probabilities (CTC) |
| `state_out_<k>` | out | same shape as `state_in_<k>` | fed back as `state_in_<k>` on the next call |

- Any number of `state_*` pairs is allowed. The runtime discovers them by name and
  allocates zeros from the declared shape (a dynamic batch dimension becomes 1).
- CMVN or any other normalisation must be **inside** the graph.
- Running N chunks one after another must equal running the whole utterance (to within
  1e-3). `asr-export --verify` and `research/tests/test_streaming.py` check this.

## model.json

```json
{
  "format_version": 1,
  "name": "baseline-ctc-lstm",
  "feature": {"sample_rate": 16000, "frame_length": 400, "frame_shift": 160, "n_fft": 512,
              "n_mels": 80, "f_min": 20.0, "f_max": 7600.0, "preemphasis": 0.97, "remove_dc": true},
  "chunk_frames": 8,
  "blank_id": 0,
  "vocab": ["<blk>", "<space>", "a", "b", "..."]
}
```

- `feature` is written by `asr-export` from `asr_native.default_feature_config()`, so it
  always matches the Rust extractor.
- Vocab decoding: `<space>` or `" "` becomes a space; a token starting with `▁`
  (SentencePiece) starts a new word; anything else is appended as is.

## Future versions

- v2: transducer (encoder / decoder / joiner graphs) for RNN-T models.
- Adapters: base model + per-user adapter weights (`adapter.onnx` or initializer
  overrides) so one base model serves many users.
