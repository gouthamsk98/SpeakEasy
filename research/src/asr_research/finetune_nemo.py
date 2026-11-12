# SPDX-License-Identifier: GPL-3.0-or-later
"""Fine-tune the streaming FastConformer (the model the app runs) on one speaker's data.

  asr-finetune ../data/datasets/user01 --out runs/user01-v1
  # next round, continue from the previous personal model:
  asr-finetune ../data/datasets/user01 --base runs/user01-v1/model.nemo --out runs/user01-v2

Then: asr-export-nemo runs/user01-v1/model.nemo --name user01-v1
"""
from __future__ import annotations

import os

# NeMo's numba CUDA kernels (RNNT loss, SpecAugment) crash or fail to compile with recent
# numba/CUDA combinations. Use NeMo's pure PyTorch paths instead: slower, but small
# personal datasets make that irrelevant. Must be set before numba is imported.
os.environ.setdefault("NUMBA_DISABLE_CUDA", "1")

import argparse
import json
import time
from pathlib import Path

BASE = "stt_en_fastconformer_hybrid_large_streaming_80ms"


def load(base: str):
    import nemo.collections.asr as nemo_asr

    if base.endswith(".nemo"):
        return nemo_asr.models.ASRModel.restore_from(base)
    return nemo_asr.models.ASRModel.from_pretrained(model_name=base)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("dataset", type=Path, help="folder made by asr-dataset-import")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", default=BASE, help="pretrained name or a previous .nemo")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=4,
                    help="utterances per step; story takes are long, keep this small on 8 GB GPUs")
    ap.add_argument("--freeze-encoder-layers", type=int, default=0,
                    help="freeze the first N conformer layers (less overfitting on tiny data)")
    ap.add_argument("--rnnt-loss", default="pytorch",
                    help="RNNT loss implementation: pytorch (robust) or warprnnt_numba (fast)")
    ap.add_argument("--ctc-weight", type=float, default=0.5,
                    help="weight of the CTC loss (the app decodes with CTC)")
    a = ap.parse_args(argv)

    import lightning.pytorch as pl
    import torch
    from omegaconf import open_dict

    torch.set_float32_matmul_precision("high")
    model = load(a.base)
    model.change_decoding_strategy(decoder_type="ctc")
    if hasattr(model, "joint"):  # hybrid RNNT+CTC: swap the transducer loss implementation
        from nemo.collections.asr.losses.rnnt import RNNTLoss

        model.loss = RNNTLoss(num_classes=model.joint.num_classes_with_blank - 1,
                              loss_name=a.rnnt_loss)
        model.joint.set_loss(model.loss)

    common = {"sample_rate": 16000, "num_workers": 4, "pin_memory": True,
              "max_duration": 45.0, "min_duration": 0.1, "trim_silence": False}
    model.setup_training_data({**common, "manifest_filepath": str(a.dataset / "nemo_train.json"),
                               "batch_size": a.batch, "shuffle": True})
    model.setup_validation_data({**common, "manifest_filepath": str(a.dataset / "nemo_test.json"),
                                 "batch_size": a.batch, "shuffle": False})
    steps_per_epoch = max(1, len(model._train_dl))
    with open_dict(model.cfg):
        model.cfg.optim = {
            "name": "adamw", "lr": a.lr, "betas": [0.9, 0.98], "weight_decay": 1e-3,
            "sched": {"name": "CosineAnnealing", "warmup_steps": max(1, steps_per_epoch),
                      "min_lr": a.lr / 20},
        }
        if "aux_ctc" in model.cfg:
            model.cfg.aux_ctc.ctc_loss_weight = a.ctc_weight
    if hasattr(model, "ctc_loss_weight"):
        model.ctc_loss_weight = a.ctc_weight

    for i, layer in enumerate(model.encoder.layers):
        if i < a.freeze_encoder_layers:
            for p in layer.parameters():
                p.requires_grad = False

    trainer = pl.Trainer(
        devices=1, accelerator="gpu" if torch.cuda.is_available() else "cpu",
        max_epochs=a.epochs, precision="bf16-mixed" if torch.cuda.is_available() else 32,
        logger=False, enable_checkpointing=False, log_every_n_steps=5,
        check_val_every_n_epoch=1, num_sanity_val_steps=0,
    )
    model.set_trainer(trainer)
    model.setup_optimization(model.cfg.optim)

    def val_wer():
        r = trainer.validate(model, verbose=False)
        return float(r[0].get("val_wer", float("nan"))) if r else float("nan")

    before = val_wer()
    t0 = time.time()
    trainer.fit(model)
    after = val_wer()
    a.out.mkdir(parents=True, exist_ok=True)
    model.save_to(str(a.out / "model.nemo"))
    info = {
        "base": a.base, "dataset": str(a.dataset.resolve()), "epochs": a.epochs, "lr": a.lr,
        "batch": a.batch, "freeze_encoder_layers": a.freeze_encoder_layers,
        "ctc_weight": a.ctc_weight, "train_minutes": round((time.time() - t0) / 60, 1),
        "offline_test_wer_before": before, "offline_test_wer_after": after,
        "speaker": json.loads((a.dataset / "consent.json").read_text()).get("speaker"),
    }
    (a.out / "train-info.json").write_text(json.dumps(info, indent=2))
    print(f"\nheld-out WER (NeMo offline CTC): {before:.3f} -> {after:.3f}")
    print(f"saved {a.out / 'model.nemo'}")


if __name__ == "__main__":
    main()
