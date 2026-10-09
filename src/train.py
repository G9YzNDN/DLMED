"""Train one experiment. The model is always the same; --tasks decides which losses are used.

Examples (the four planned experiments):
    A  python src/train.py --tasks cls  --name A_cls_seed0
    B  python src/train.py --tasks seg  --name B_seg_seed0
    C  python src/train.py --tasks both --name C_mtl_equal_seed0
    D  python src/train.py --tasks both --weighting uncertainty --name D_mtl_uncert_seed0

The best epoch is chosen on the validation set. The test set is never touched here.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import BrainTumorDataset
from engine import predict, selection_score
from losses import MultiTaskLoss
from model import build_model


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--splits", default="splits/splits.csv")
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--name", default=None, help="run folder name (default: built from settings)")
    ap.add_argument("--tasks", choices=["cls", "seg", "both"], default="both")
    ap.add_argument("--weighting", choices=["fixed", "uncertainty"], default="fixed")
    ap.add_argument("--w_seg", type=float, default=1.0)
    ap.add_argument("--w_cls", type=float, default=1.0)
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.tasks == "cls":
        args.w_seg = 0.0
    elif args.tasks == "seg":
        args.w_cls = 0.0
    if args.weighting == "uncertainty" and args.tasks != "both":
        ap.error("--weighting uncertainty needs --tasks both")
    if args.name is None:
        args.name = f"{args.tasks}_{args.weighting}_wseg{args.w_seg}_wcls{args.w_cls}_seed{args.seed}"
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = get_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = Path(args.out_dir) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(f"Run: {run_dir}  device: {device}")

    train_ds = BrainTumorDataset(args.data_dir, args.splits, "train", args.size, train=True)
    val_ds = BrainTumorDataset(args.data_dir, args.splits, "val", args.size)
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers,
                          pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    val_dl = DataLoader(val_ds, args.batch_size, num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0)
    print(f"train {len(train_ds)}  val {len(val_ds)} images")

    model = build_model(args.encoder, pretrained=not args.no_pretrained).to(device)
    criterion = MultiTaskLoss(args.weighting, args.w_seg, args.w_cls).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best_score = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, sums, n = time.time(), {"loss": 0.0, "loss_seg": 0.0, "loss_cls": 0.0}, 0
        for batch in train_dl:
            x = batch["image"].to(device)
            mask, label = batch["mask"].to(device), batch["label"].to(device)
            with torch.autocast(device_type=device, enabled=use_amp):
                seg_logits, cls_logits = model(x)
            loss, parts = criterion(seg_logits.float(), cls_logits.float(), mask, label)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            sums["loss"] += loss.item()
            sums["loss_seg"] += parts["loss_seg"]
            sums["loss_cls"] += parts["loss_cls"]
            n += 1
        scheduler.step()

        _, val = predict(model, val_dl, device)
        score = selection_score(val, args.tasks)
        row = {
            "epoch": epoch, **{f"train_{k}": v / n for k, v in sums.items()},
            "val_macro_f1": val["cls"]["macro_f1"], "val_accuracy": val["cls"]["accuracy"],
            "val_dice": val["seg"]["dice"], "val_iou": val["seg"]["iou"], "val_score": score,
            "lr": scheduler.get_last_lr()[0], "time_s": time.time() - t0,
        }
        if args.weighting == "uncertainty":
            row["log_var_seg"], row["log_var_cls"] = criterion.log_vars.tolist()
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        improved = score > best_score
        if improved:
            best_score = score
            torch.save({"model": model.state_dict(), "config": vars(args), "epoch": epoch},
                       run_dir / "best.pt")
        print(f"ep {epoch:3d}  loss {row['train_loss']:.4f}  val F1 {row['val_macro_f1']:.4f}  "
              f"Dice {row['val_dice']:.4f}  {'*' if improved else ''}  ({row['time_s']:.0f}s)")

    print(f"Best val score {best_score:.4f} -> {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
