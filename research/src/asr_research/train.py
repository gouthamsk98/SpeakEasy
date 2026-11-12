# SPDX-License-Identifier: GPL-3.0-or-later
"""Train a streaming CTC model.

  asr-train configs/baseline_ctc.yaml
  # personalise a base model for one speaker (few minutes of their speech):
  asr-train configs/baseline_ctc.yaml --init-from runs/base/best.pt --freeze-layers 1
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import jiwer
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import AsrDataset, collate, spec_augment
from .model import build_model, greedy_ctc
from .tokenizer import CharTokenizer


def compute_cmvn(ds: AsrDataset, max_utts=500):
    s = sq = n = 0
    for i in range(min(len(ds), max_utts)):
        f = ds[i][0].double()
        s, sq, n = s + f.sum(0), sq + (f * f).sum(0), n + f.shape[0]
    mean = s / n
    return mean.float(), (sq / n - mean * mean).clamp(min=1e-6).sqrt().float()


@torch.no_grad()
def evaluate(model, loader, tok, device):
    model.eval()
    hyps, refs = [], []
    for x, flen, _, _, texts in loader:
        logp, olen = model(x.to(device), flen.to(device))
        for i in range(len(texts)):
            ids, _ = greedy_ctc(logp[i, : olen[i]])
            hyps.append(tok.decode(ids))
            refs.append(" ".join(texts[i].lower().split()))
    model.train()
    return jiwer.cer(refs, hyps), jiwer.wer(refs, hyps), list(zip(refs, hyps))[:3]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--init-from", type=Path, help="checkpoint to fine-tune from")
    ap.add_argument("--freeze-layers", type=int, default=0,
                    help="freeze input projection + first N LSTM layers (speaker adaptation)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    base = args.config.parent
    cfg = yaml.safe_load(args.config.read_text())
    tok = CharTokenizer.load(base / cfg["tokens"])
    tc = cfg["train"]
    train_ds = AsrDataset(base / cfg["data"]["train"], tok, cfg["data"]["max_seconds"],
                          tc["speed_perturb"])
    valid_ds = AsrDataset(base / cfg["data"]["valid"], tok, cfg["data"]["max_seconds"])
    dl = DataLoader(train_ds, tc["batch_size"], shuffle=True, collate_fn=collate, num_workers=4,
                    drop_last=len(train_ds) > tc["batch_size"])
    vdl = DataLoader(valid_ds, tc["batch_size"], collate_fn=collate, num_workers=2)

    model = build_model(cfg, len(tok)).to(args.device)
    if args.init_from:
        ck = torch.load(args.init_from, map_location=args.device)
        assert ck["vocab"] == tok.vocab, "vocab mismatch with the checkpoint"
        model.load_state_dict(ck["model"])
        print(f"initialised from {args.init_from}")
    else:
        mean, std = compute_cmvn(train_ds)
        model.feat_mean.copy_(mean)
        model.feat_std.copy_(std)
    for name, p in model.named_parameters():
        if name.startswith("inp.") and args.freeze_layers > 0:
            p.requires_grad = False
        for k in range(args.freeze_layers):
            if name.startswith("lstm.") and name.endswith(f"_l{k}"):
                p.requires_grad = False

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=tc["lr"],
                            weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, tc["lr"], total_steps=tc["epochs"] * len(dl))
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    use_amp = tc["amp"] and args.device.startswith("cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)
    out = base / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    best = float("inf")

    for epoch in range(tc["epochs"]):
        t0, total = time.time(), 0.0
        for x, flen, y, ylen, _ in tqdm(dl, desc=f"epoch {epoch}", leave=False):
            x = spec_augment(x, **tc["specaugment"]).to(args.device)
            with torch.autocast(args.device.split(":")[0], enabled=use_amp):
                logp, olen = model(x, flen.to(args.device))
            loss = ctc(logp.float().transpose(0, 1), y.to(args.device), olen, ylen.to(args.device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        cer, wer, ex = evaluate(model, vdl, tok, args.device)
        print(f"epoch {epoch}: loss {total / max(1, len(dl)):.3f}  CER {cer:.3f}  WER {wer:.3f}"
              f"  ({time.time() - t0:.0f}s)  e.g. {ex[:1]}")
        ck = {"model": model.state_dict(), "config": cfg, "vocab": tok.vocab, "epoch": epoch,
              "cer": cer, "wer": wer}
        torch.save(ck, out / "last.pt")
        if cer < best:
            best = cer
            torch.save(ck, out / "best.pt")
    print(f"best CER {best:.3f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
