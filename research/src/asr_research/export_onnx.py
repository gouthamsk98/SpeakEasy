# SPDX-License-Identifier: GPL-3.0-or-later
"""Export a checkpoint to the runtime model contract (docs/MODEL_CONTRACT.md).

  asr-export runs/baseline/best.pt ../models/baseline [--chunk-frames 8] [--verify]

Produces model.onnx (one streaming step with explicit state) + model.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .model import build_model


class StepWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, features, state_in_h, state_in_c):
        logp, h, c = self.m.forward_chunk(features, state_in_h, state_in_c)
        return logp, h, c


def export(ckpt_path: Path, out_dir: Path, chunk_frames: int | None, opset=17) -> Path:
    from .features import feature_config  # needs asr_native

    ck = torch.load(ckpt_path, map_location="cpu")
    cfg, vocab = ck["config"], ck["vocab"]
    model = build_model(cfg, len(vocab))
    model.load_state_dict(ck["model"])
    model.eval()
    chunk = chunk_frames or cfg["stream"]["chunk_frames"]
    assert chunk % model.subsample == 0, "chunk_frames must be a multiple of subsample"

    out_dir.mkdir(parents=True, exist_ok=True)
    h, c = model.init_state()
    feats = torch.zeros(1, chunk, model.n_mels)
    torch.onnx.export(
        StepWrapper(model), (feats, h, c), str(out_dir / "model.onnx"),
        input_names=["features", "state_in_h", "state_in_c"],
        output_names=["log_probs", "state_out_h", "state_out_c"],
        opset_version=opset, dynamo=False,
    )
    meta = {
        "format_version": 1,
        "name": cfg["name"],
        "feature": feature_config(),
        "chunk_frames": chunk,
        "blank_id": 0,
        "vocab": vocab,
        "train": {"epoch": ck.get("epoch"), "cer": ck.get("cer"), "wer": ck.get("wer")},
    }
    (out_dir / "model.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return out_dir


def verify(ckpt_path: Path, out_dir: Path, n_chunks=20):
    """Exported ONNX == PyTorch streaming step, over several chunks with carried state."""
    import onnxruntime as ort

    ck = torch.load(ckpt_path, map_location="cpu")
    meta = json.loads((out_dir / "model.json").read_text())
    model = build_model(ck["config"], len(ck["vocab"]))
    model.load_state_dict(ck["model"])
    model.eval()
    sess = ort.InferenceSession(str(out_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    h, c = model.init_state()
    oh, oc = h.numpy(), c.numpy()
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(n_chunks):
        x = rng.standard_normal((1, meta["chunk_frames"], model.n_mels)).astype(np.float32) - 5
        with torch.no_grad():
            lp, h, c = model.forward_chunk(torch.from_numpy(x), h, c)
        olp, oh, oc = sess.run(None, {"features": x, "state_in_h": oh, "state_in_c": oc})
        worst = max(worst, float(np.abs(olp - lp.numpy()).max()))
    print(f"max |onnx - torch| over {n_chunks} chunks: {worst:.2e}")
    assert worst < 1e-3, "exported model diverges from PyTorch"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--chunk-frames", type=int)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args(argv)
    out = export(a.checkpoint, a.out_dir, a.chunk_frames)
    print(f"exported -> {out}")
    if a.verify:
        verify(a.checkpoint, out)


if __name__ == "__main__":
    main()
