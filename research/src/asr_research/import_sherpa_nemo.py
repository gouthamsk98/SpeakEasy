# SPDX-License-Identifier: GPL-3.0-or-later
"""Write our model.json for a sherpa-onnx NeMo *streaming CTC* export, so the Rust runtime
can run it directly. Used to test the real-time pipeline with a strong pretrained English
model (and as the starting point for fine-tuning on glossectomy speech).

  asr-import-nemo ../models/sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-80ms

Feature recipe = what sherpa-onnx uses for NeMo CTC (kaldi-native-fbank, is_librosa):
Slaney mel 0-8 kHz, hann, no DC removal, pre-emphasis 0.97, log floor FLT_EPSILON,
no per-utterance normalisation (model metadata `normalize_type` must be empty).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    a = ap.parse_args(argv)

    m = onnx.load(str(a.model_dir / "model.onnx"), load_external_data=False)
    md = {p.key: p.value for p in m.metadata_props}
    if md.get("normalize_type"):
        raise SystemExit(f"normalize_type={md['normalize_type']!r} not supported yet")

    vocab: dict[int, str] = {}
    for line in (a.model_dir / "tokens.txt").read_text(encoding="utf-8").splitlines():
        piece, idx = line.rsplit(" ", 1)
        vocab[int(idx)] = piece
    tokens = [vocab[i] for i in range(len(vocab))]
    blank = tokens.index("<blk>")

    i = lambda k: int(md[k])  # noqa: E731
    meta = {
        "format_version": 1,
        "name": a.model_dir.name,
        "source": md.get("url", ""),
        "feature": {
            "sample_rate": 16000, "frame_length": 400, "frame_shift": 160, "n_fft": 512,
            "n_mels": 80, "f_min": 0.0, "f_max": 8000.0, "preemphasis": 0.97,
            "remove_dc": False, "mel_scale": "slaney", "log_floor": 1.1920929e-07,
        },
        "chunk_frames": i("window_size"),
        "chunk_shift": i("chunk_shift"),
        "blank_id": blank,
        "vocab": tokens,
        "io": {
            "features": "audio_signal",
            "layout": "mel_major",
            "length": "length",
            "log_probs": "logprobs",
            "states": [
                {"input": "cache_last_channel", "output": "cache_last_channel_next",
                 "shape": [1, i("cache_last_channel_dim1"), i("cache_last_channel_dim2"),
                           i("cache_last_channel_dim3")]},
                {"input": "cache_last_time", "output": "cache_last_time_next",
                 "shape": [1, i("cache_last_time_dim1"), i("cache_last_time_dim2"),
                           i("cache_last_time_dim3")]},
                {"input": "cache_last_channel_len", "output": "cache_last_channel_next_len",
                 "shape": [1]},
            ],
        },
    }
    (a.model_dir / "model.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"wrote {a.model_dir / 'model.json'}: window {meta['chunk_frames']}, "
          f"shift {meta['chunk_shift']}, vocab {len(tokens)}, blank {blank}")


if __name__ == "__main__":
    main()
