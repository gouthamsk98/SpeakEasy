# SPDX-License-Identifier: GPL-3.0-or-later
"""The model must give identical output streamed chunk-by-chunk vs. whole-utterance."""
import torch

from asr_research.model import StreamingCtcLstm


def test_chunked_equals_full():
    torch.manual_seed(0)
    m = StreamingCtcLstm(vocab_size=30, d_model=64, num_layers=2, dropout=0.0).eval()
    x = torch.randn(1, 96, 80)
    with torch.no_grad():
        full, _ = m(x, torch.tensor([96]))
        h, c = m.init_state()
        parts = []
        for s in range(0, 96, 8):
            lp, h, c = m.forward_chunk(x[:, s : s + 8], h, c)
            parts.append(lp)
    torch.testing.assert_close(torch.cat(parts, 1), full, atol=1e-5, rtol=1e-5)
