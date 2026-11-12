# SPDX-License-Identifier: GPL-3.0-or-later
"""Datasets from JSONL manifests.

One line per utterance:
  {"audio": "relative/or/absolute.wav", "text": "...", "speaker": "spk01",
   "duration": 3.2, "consent_id": "C-2026-001"}

`consent_id` is mandatory: no consent record, no training. See DATA_ETHICS.md.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .features import SAMPLE_RATE, fbank
from .tokenizer import CharTokenizer


def read_manifest(path: str | Path) -> list[dict]:
    path = Path(path)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("consent_id"):
            raise ValueError(f"{path}: utterance {r.get('audio')} has no consent_id")
        a = Path(r["audio"])
        r["audio"] = str(a if a.is_absolute() else (path.parent / a).resolve())
        rows.append(r)
    return rows


def load_audio(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import torchaudio.functional as AF  # only needed for non-16k data

        audio = AF.resample(torch.from_numpy(audio), sr, SAMPLE_RATE).numpy()
    return audio


def speed_perturb(audio: np.ndarray, factor: float) -> np.ndarray:
    if factor == 1.0:
        return audio
    n = int(len(audio) / factor)
    return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(
        np.float32
    )


class AsrDataset(Dataset):
    def __init__(self, manifest, tok: CharTokenizer, max_seconds=20.0, speeds=(1.0,)):
        self.rows = [r for r in read_manifest(manifest) if r.get("duration", 0) <= max_seconds]
        self.tok, self.speeds = tok, list(speeds)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        audio = speed_perturb(load_audio(r["audio"]), random.choice(self.speeds))
        feats = torch.from_numpy(fbank(audio))
        return feats, torch.tensor(self.tok.encode(r["text"]), dtype=torch.long), r["text"]


def collate(batch):
    feats, targets, texts = zip(*batch)
    flen = torch.tensor([f.shape[0] for f in feats])
    tlen = torch.tensor([t.shape[0] for t in targets])
    x = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True)
    y = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)
    return x, flen, y, tlen, list(texts)


def spec_augment(x, freq_masks=2, freq_width=15, time_masks=4, time_width=25):
    """In-place SpecAugment on [B, T, F]. Pads with the per-utterance mean."""
    b, t, f = x.shape
    fill = x.mean(dim=(1, 2), keepdim=True)
    for _ in range(freq_masks):
        w = torch.randint(0, freq_width + 1, (b,))
        s = (torch.rand(b) * (f - w)).long()
        for i in range(b):
            x[i, :, s[i] : s[i] + w[i]] = fill[i, 0, 0]
    for _ in range(time_masks):
        w = torch.randint(0, time_width + 1, (b,))
        s = (torch.rand(b) * (t - w).clamp(min=1)).long()
        for i in range(b):
            x[i, s[i] : s[i] + w[i], :] = fill[i, 0, 0]
    return x
