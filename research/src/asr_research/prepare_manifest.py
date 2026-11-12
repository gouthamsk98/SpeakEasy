# SPDX-License-Identifier: GPL-3.0-or-later
"""Build JSONL manifests + tokens.txt from a folder of `utt.wav` + `utt.txt` pairs.

  asr-prepare --audio-dir ../data/raw/spk01 --speaker spk01 --consent-id C-2026-001 \
              --out ../data/manifests
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import soundfile as sf

from .tokenizer import CharTokenizer


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--audio-dir", required=True, type=Path)
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--consent-id", required=True,
                    help="ID of the signed consent record covering these recordings")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--valid-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rows = []
    for wav in sorted(args.audio_dir.rglob("*.wav")):
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            print(f"skip (no transcript): {wav}")
            continue
        rows.append({"audio": str(wav.resolve()), "text": txt.read_text(encoding="utf-8").strip(),
                     "speaker": args.speaker, "duration": round(sf.info(str(wav)).duration, 3),
                     "consent_id": args.consent_id})
    if not rows:
        raise SystemExit("no wav+txt pairs found")

    random.Random(args.seed).shuffle(rows)
    n_valid = max(1, int(len(rows) * args.valid_fraction))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, part in (("valid", rows[:n_valid]), ("train", rows[n_valid:])):
        with open(args.out / f"{name}.jsonl", "a", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in part)

    # Rebuild tokens over everything seen so far.
    texts = [json.loads(line)["text"] for p in ("train", "valid")
             for line in (args.out / f"{p}.jsonl").read_text(encoding="utf-8").splitlines()]
    CharTokenizer.build(texts).save(args.out / "tokens.txt")
    print(f"{len(rows) - n_valid} train / {n_valid} valid utterances -> {args.out}")


if __name__ == "__main__":
    main()
