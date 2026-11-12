# SPDX-License-Identifier: GPL-3.0-or-later
"""Evaluate *the way the app runs*: chunk-by-chunk streaming, greedy CTC.

  asr-eval runs/baseline/best.pt ../data/manifests/valid.jsonl [--chunk-frames 8]

Reports CER / WER and per-speaker breakdown — glossectomy outcomes vary hugely by
resection site, so always look per speaker, not just the average.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import jiwer
import torch

from .data import load_audio, read_manifest
from .features import fbank
from .model import build_model, greedy_ctc
from .tokenizer import CharTokenizer, normalize


@torch.no_grad()
def stream_decode(model, feats: torch.Tensor, chunk: int) -> list[int]:
    h, c = model.init_state()
    ids, last = [], None
    usable = feats.shape[0] - feats.shape[0] % chunk  # runtime only runs full chunks
    for s in range(0, usable, chunk):
        logp, h, c = model.forward_chunk(feats[None, s : s + chunk], h, c)
        new, last = greedy_ctc(logp[0], prev=last)
        ids += new
    return ids


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--chunk-frames", type=int)
    a = ap.parse_args(argv)

    ck = torch.load(a.checkpoint, map_location="cpu")
    tok = CharTokenizer(ck["vocab"])
    model = build_model(ck["config"], len(tok))
    model.load_state_dict(ck["model"])
    model.eval()
    chunk = a.chunk_frames or ck["config"]["stream"]["chunk_frames"]

    by_spk = defaultdict(lambda: ([], []))
    for r in read_manifest(a.manifest):
        feats = torch.from_numpy(fbank(load_audio(r["audio"])))
        hyp = tok.decode(stream_decode(model, feats, chunk))
        refs, hyps = by_spk[r.get("speaker", "?")]
        refs.append(normalize(r["text"]))
        hyps.append(hyp)

    all_r = [x for r, _ in by_spk.values() for x in r]
    all_h = [x for _, h in by_spk.values() for x in h]
    print(f"chunk {chunk} frames ({chunk * 10} ms)")
    for spk, (r, h) in sorted(by_spk.items()):
        print(f"  {spk:12s} n={len(r):4d}  CER {jiwer.cer(r, h):.3f}  WER {jiwer.wer(r, h):.3f}")
    print(f"  {'ALL':12s} n={len(all_r):4d}  CER {jiwer.cer(all_r, all_h):.3f}  "
          f"WER {jiwer.wer(all_r, all_h):.3f}")


if __name__ == "__main__":
    main()
