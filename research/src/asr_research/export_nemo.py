# SPDX-License-Identifier: GPL-3.0-or-later
"""Export a fine-tuned FastConformer (.nemo) as an app model package.

  asr-export-nemo runs/user01-v1/model.nemo --name user01-v1

Writes ../models/<name>/{model.onnx, model.json, model-card.json} and
<out>/speakeasy-model-<name>.zip: the file to send back to the user, who installs it
with "Import model…" in the app (or drops it onto the window).

If the Rust CLI is built (target/release/asr) and the dataset has a test split, the
package's model card includes held-out WER of the base and the new model, measured by the
app's own streaming runtime.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def add_meta(filename: Path, meta: dict):
    import onnx

    m = onnx.load(str(filename))
    del m.metadata_props[:]
    for k, v in meta.items():
        p = m.metadata_props.add()
        p.key, p.value = k, str(v)
    onnx.save(m, str(filename))


def export(nemo_path: Path, out_dir: Path):
    import nemo.collections.asr as nemo_asr
    import torch

    model = nemo_asr.models.ASRModel.restore_from(str(nemo_path), map_location="cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab = list(model.joint.vocabulary) if hasattr(model, "joint") else list(model.decoder.vocabulary)
    with open(out_dir / "tokens.txt", "w", encoding="utf-8") as f:
        f.writelines(f"{s} {i}\n" for i, s in enumerate(vocab))
        f.write(f"<blk> {len(vocab)}\n")

    model.change_decoding_strategy(decoder_type="ctc")
    model.eval()
    sc = model.encoder.streaming_cfg
    pick = lambda v: v[1] if isinstance(v, list) else v
    chunk, pre = pick(sc.chunk_size), pick(sc.pre_encode_cache_size)
    model.set_export_config({"decoder_type": "ctc", "cache_support": True})
    with torch.no_grad():
        model.export(str(out_dir / "model.onnx"))
    normalize = model.cfg.preprocessor.normalize
    add_meta(out_dir / "model.onnx", {
        "vocab_size": model.decoder.vocab_size,
        "window_size": chunk + pre,
        "chunk_shift": chunk,
        "normalize_type": "" if normalize == "NA" else normalize,
        "cache_last_channel_dim1": len(model.encoder.layers),
        "cache_last_channel_dim2": sc.last_channel_cache_size,
        "cache_last_channel_dim3": model.encoder.d_model,
        "cache_last_time_dim1": len(model.encoder.layers),
        "cache_last_time_dim2": model.encoder.d_model,
        "cache_last_time_dim3": model.encoder.conv_context_size[0],
        "subsampling_factor": 8,
        "model_type": "EncDecHybridRNNTCTCBPEModel",
        "url": str(nemo_path),
    })


def rust_eval(model_dir: Path, test_manifest: Path) -> float | None:
    asr = REPO / "target" / "release" / "asr"
    if not asr.exists() or not test_manifest.exists():
        return None
    out = model_dir.parent / f".eval-{model_dir.name}.json"
    subprocess.run([str(asr), "--model", str(model_dir), "eval", str(test_manifest), "--json", str(out)],
                   check=True, capture_output=True)
    wer = json.loads(out.read_text())["wer"]
    out.unlink()
    return wer


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("nemo", type=Path)
    ap.add_argument("--name", required=True, help="model name, e.g. user01-v1")
    ap.add_argument("--models", type=Path, default=REPO / "models")
    ap.add_argument("--zip-dir", type=Path, default=REPO / "research" / "runs" / "packages")
    ap.add_argument("--base-model", type=Path,
                    default=REPO / "models" / "sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-80ms",
                    help="model dir to compare against on the test set")
    a = ap.parse_args(argv)

    from .import_sherpa_nemo import main as write_model_json

    out = a.models / a.name
    export(a.nemo, out)
    write_model_json([str(out)])
    meta = json.loads((out / "model.json").read_text())
    meta["name"] = a.name
    (out / "model.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    (out / "tokens.txt").unlink()

    info_path = a.nemo.parent / "train-info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    card = {"name": a.name, "trained_from": info.get("base"), "speaker": info.get("speaker"),
            "training": info}
    if info.get("dataset"):
        test = Path(info["dataset"]) / "test.jsonl"
        card["test_wer_new"] = rust_eval(out, test)
        if a.base_model.exists():
            card["test_wer_base"] = rust_eval(a.base_model, test)
    (out / "model-card.json").write_text(json.dumps(card, indent=2))

    a.zip_dir.mkdir(parents=True, exist_ok=True)
    pkg = a.zip_dir / f"speakeasy-model-{a.name}.zip"
    with zipfile.ZipFile(pkg, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for f in ("model.onnx", "model.json", "model-card.json"):
            z.write(out / f, f"{a.name}/{f}")
    if card.get("test_wer_new") is not None:
        print(f"held-out WER in the app runtime: base {card.get('test_wer_base', float('nan')):.3f} "
              f"-> {a.name} {card['test_wer_new']:.3f}")
    print(f"model dir: {out}\npackage to send: {pkg} ({pkg.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
