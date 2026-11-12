# SPDX-License-Identifier: GPL-3.0-or-later
"""Baseline streaming acoustic model: CMVN -> frame stacking -> causal LSTM -> CTC.

Exactly streamable: running it chunk-by-chunk with carried (h, c) gives the same output
as running it on the whole utterance (tests/test_streaming.py proves it). Zero lookahead,
so algorithmic latency = one chunk.
"""
from __future__ import annotations

import torch
from torch import nn


class StreamingCtcLstm(nn.Module):
    def __init__(self, vocab_size: int, n_mels=80, d_model=320, num_layers=3, subsample=2,
                 dropout=0.15):
        super().__init__()
        self.n_mels, self.subsample = n_mels, subsample
        self.d_model, self.num_layers = d_model, num_layers
        # CMVN lives in the graph so the runtime never has to know about it.
        self.register_buffer("feat_mean", torch.zeros(n_mels))
        self.register_buffer("feat_std", torch.ones(n_mels))
        self.inp = nn.Sequential(nn.Linear(n_mels * subsample, d_model), nn.LayerNorm(d_model))
        self.lstm = nn.LSTM(d_model, d_model, num_layers, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, vocab_size)

    def _front(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.feat_mean) / self.feat_std
        b, t, f = x.shape
        t = t - t % self.subsample
        x = x[:, :t].reshape(b, t // self.subsample, f * self.subsample)
        return self.inp(x)

    def forward(self, x, lengths):
        """Training/eval on padded batches. x [B,T,F] -> log-probs [B,T',V], lengths'."""
        h = self._front(x)
        out_len = (lengths // self.subsample).clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(h, out_len.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        y, _ = self.lstm(packed)
        y, _ = nn.utils.rnn.pad_packed_sequence(y, batch_first=True, total_length=h.shape[1])
        return self.out(self.drop(y)).log_softmax(-1), out_len

    def init_state(self, batch=1, device="cpu"):
        z = torch.zeros(self.num_layers, batch, self.d_model, device=device)
        return z, z.clone()

    def forward_chunk(self, feats, h, c):
        """Streaming step. feats [1, chunk, F]; chunk must be a multiple of `subsample`."""
        y, (h, c) = self.lstm(self._front(feats), (h, c))
        return self.out(y).log_softmax(-1), h, c


def greedy_ctc(logp: torch.Tensor, blank: int = 0, prev: int | None = None):
    """[T, V] -> token ids (collapsed), last token. Same algorithm as crates/asr-engine/src/onnx.rs."""
    ids, last = [], prev
    for t in logp.argmax(-1).tolist():
        if t != blank and t != last:
            ids.append(t)
        last = t
    return ids, last


def build_model(cfg: dict, vocab_size: int) -> StreamingCtcLstm:
    m = cfg["model"]
    return StreamingCtcLstm(vocab_size, d_model=m["d_model"], num_layers=m["num_layers"],
                            subsample=m["subsample"], dropout=m["dropout"])
