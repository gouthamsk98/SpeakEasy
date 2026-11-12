# SPDX-License-Identifier: GPL-3.0-or-later
"""Character tokenizer. id 0 = CTC blank. Space is stored as "<space>" in tokens.txt."""
from __future__ import annotations

from pathlib import Path

BLANK = "<blk>"
SPACE = "<space>"


class CharTokenizer:
    def __init__(self, vocab: list[str]):
        assert vocab[0] == BLANK, "token 0 must be the CTC blank"
        self.vocab = vocab
        self.index = {t: i for i, t in enumerate(vocab)}

    @classmethod
    def build(cls, texts: list[str]) -> "CharTokenizer":
        chars = sorted({c for t in texts for c in normalize(t)} - {" "})
        return cls([BLANK, SPACE, *chars])

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        return cls(Path(path).read_text(encoding="utf-8").splitlines())

    def save(self, path: str | Path) -> None:
        Path(path).write_text("\n".join(self.vocab) + "\n", encoding="utf-8")

    def encode(self, text: str) -> list[int]:
        return [self.index[SPACE if c == " " else c] for c in normalize(text) if
                (SPACE if c == " " else c) in self.index]

    def decode(self, ids: list[int]) -> str:
        return "".join(" " if self.vocab[i] == SPACE else self.vocab[i] for i in ids if i != 0)

    def __len__(self) -> int:
        return len(self.vocab)


def normalize(text: str) -> str:
    return " ".join(text.lower().split())
