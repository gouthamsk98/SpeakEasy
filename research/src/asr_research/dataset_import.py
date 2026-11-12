# SPDX-License-Identifier: GPL-3.0-or-later
"""Unpack a dataset zip exported by the app and prepare it for training.

  asr-dataset-import speakeasy-dataset-user01-20261112-1830.zip

Creates ../data/datasets/<speaker>/ (merging with earlier exports of the same speaker):
  audio/...                 the recordings
  all.jsonl                 every utterance (our manifest format)
  train.jsonl / test.jsonl  split BY SENTENCE: a sentence is either always train or always
                            test (stable hash), so test scores are never inflated by training
                            on the same sentence.
  nemo_train.json / nemo_test.json   NeMo manifests (absolute paths, normalised text)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

DATA = Path(__file__).resolve().parents[3] / "data"


def normalize(text: str) -> str:
    """What the NeMo English BPE vocab expects: lowercase letters, digits, apostrophes."""
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return " ".join(text.split())


def is_test(text: str, fraction: float) -> bool:
    h = int(hashlib.sha1(normalize(text).encode()).hexdigest()[:8], 16)
    return (h % 10_000) < fraction * 10_000


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("zip", type=Path)
    ap.add_argument("--out", type=Path, default=DATA / "datasets")
    ap.add_argument("--test-fraction", type=float, default=0.15)
    a = ap.parse_args(argv)

    with zipfile.ZipFile(a.zip) as z:
        info = json.loads(z.read("dataset.json"))
        if info.get("format") != "speakeasy-dataset" or info.get("version") != 1:
            raise SystemExit("not a SpeakEasy dataset (v1) zip")
        consent = info.get("consent") or {}
        if not consent.get("consent_id"):
            raise SystemExit("dataset has no consent record; refusing to use it")
        speaker = info["speaker"]
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", speaker):
            raise SystemExit(f"bad speaker id {speaker!r}")
        dest = a.out / speaker
        dest.mkdir(parents=True, exist_ok=True)
        for name in z.namelist():
            p = Path(name)
            if p.is_absolute() or ".." in p.parts:
                raise SystemExit(f"unsafe path in zip: {name}")
            if name.startswith("audio/") and not name.endswith("/"):
                target = dest / p
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        new_rows = [json.loads(l) for l in z.read("manifest.jsonl").decode().splitlines() if l.strip()]
        (dest / "consent.json").write_text(json.dumps(consent, indent=2))

    # merge with earlier imports (same audio path = same recording)
    all_path = dest / "all.jsonl"
    rows = {}
    if all_path.exists():
        for l in all_path.read_text().splitlines():
            r = json.loads(l)
            rows[r["audio"]] = r
    for r in new_rows:
        if r.get("consent_id") != consent["consent_id"]:
            raise SystemExit(f"{r['audio']}: consent id mismatch")
        if not (dest / r["audio"]).exists():
            raise SystemExit(f"{r['audio']} listed in manifest but missing from zip")
        rows[r["audio"]] = r
    rows = sorted(rows.values(), key=lambda r: r["audio"])
    rows = [r for r in rows if normalize(r["text"])]

    def write(name, subset):
        (dest / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in subset))
        with open(dest / f"nemo_{name}.json", "w") as f:
            for r in subset:
                f.write(json.dumps({"audio_filepath": str((dest / r["audio"]).resolve()),
                                    "duration": r["duration"], "text": normalize(r["text"])}) + "\n")

    test = [r for r in rows if is_test(r["text"], a.test_fraction)]
    train = [r for r in rows if not is_test(r["text"], a.test_fraction)]
    if not test:  # tiny datasets: hold out at least one sentence
        first = normalize(rows[0]["text"])
        test = [r for r in rows if normalize(r["text"]) == first]
        train = [r for r in rows if normalize(r["text"]) != first]
    write("all", rows)
    write("train", train)
    write("test", test)
    mins = lambda s: sum(r["duration"] for r in s) / 60
    print(f"{speaker}: {len(rows)} utterances ({mins(rows):.1f} min) -> train {len(train)} "
          f"({mins(train):.1f} min), test {len(test)} ({mins(test):.1f} min, "
          f"{len({normalize(r['text']) for r in test})} sentences never trained on)\n-> {dest}")


if __name__ == "__main__":
    main()
