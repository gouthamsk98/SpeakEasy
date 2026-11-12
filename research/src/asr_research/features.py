# SPDX-License-Identifier: GPL-3.0-or-later
"""Acoustic features — delegated to the Rust implementation used by the app.

Build the binding once:  `cd bindings/python && maturin develop --release`
"""
from __future__ import annotations

import numpy as np

try:
    import asr_native
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "asr_native not built. Run: (cd bindings/python && maturin develop --release) "
        "inside research/.venv. Features MUST come from the Rust code so training and the "
        "app see identical inputs."
    ) from e

SAMPLE_RATE = 16_000


def fbank(audio: np.ndarray, n_mels: int = 80) -> np.ndarray:
    """[samples] float32 @16 kHz -> [frames, n_mels] float32 log-mel."""
    return asr_native.fbank(np.ascontiguousarray(audio, dtype=np.float32), n_mels, SAMPLE_RATE)


def feature_config() -> dict:
    return dict(asr_native.default_feature_config())
