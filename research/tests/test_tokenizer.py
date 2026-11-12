# SPDX-License-Identifier: GPL-3.0-or-later
from asr_research.tokenizer import CharTokenizer


def test_roundtrip():
    tok = CharTokenizer.build(["Hello world", "help"])
    assert tok.vocab[0] == "<blk>"
    assert tok.decode(tok.encode("Hello  World")) == "hello world"
