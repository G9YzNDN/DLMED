"""Evaluate a trained run on the held-out test set (run only after choosing settings on val).

Usage:
    python src/evaluate.py --run runs/C_mtl_equal_seed0

Writes into the run folder:
    test_metrics.json       metrics of the task(s) this run was trained for
    test_predictions.csv    per-image label, prediction, probabilities, Dice, IoU
    confusion_matrix.png    (if classification was trained)
    seg_best_worst.png      best / worst Dice examples vs ground truth (if segmentation was trained)
    misclassified.png       misclassified examples (if classification was trained)
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import MEAN, STD, BrainTumorDataset
from engine import predict
from model import build_model

CLASS_NAMES = ["meningioma", "glioma", "pituitary"]


def plot_confusion(cm, path):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(3), CLASS_NAMES, rotation=30)
    ax.set_yticks(range(3), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def plot_examples(model, ds, ids, titles, path, device, show_mask=True, ncols=4):
    """Grid of examples (up to `ncols` per row): image with ground truth (green) and prediction (red) contours."""
    if not ids:
        return
    idx = {v: i for i, v in enumerate(ds.df.id)}
    ncols = min(len(ids), ncols)
    nrows = -(-len(ids) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.flat[len(ids):]:
        ax.axis("off")
    for ax, id_, title in zip(axes.flat, ids, titles):
        item = ds[idx[id_]]
        seg_logits, _ = model(item["image"][None].to(device))
        img = item["image"][0].numpy() * STD + MEAN
        ax.imshow(img, cmap="gray")
        if show_mask:
            ax.contour(item["mask"][0].numpy(), levels=[0.5], colors="lime", linewidths=1)
            pred = (torch.sigmoid(seg_logits)[0, 0] > 0.5).float().cpu().numpy()
            if pred.any():
                ax.contour(pred, levels=[0.5], colors="red", linewidths=1)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run folder containing best.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--data_dir", default=None, help="override data_dir from the run config")
    ap.add_argument("--splits", default=None, help="override splits csv from the run config")
    ap.add_argument("--n_examples", type=int, default=4)
    args = ap.parse_args()

    run = Path(args.run)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(run / "best.pt", map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg["encoder"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = BrainTumorDataset(args.data_dir or cfg["data_dir"], args.splits or cfg["splits"],
                           args.split, cfg["size"])
    dl = DataLoader(ds, batch_size=cfg["batch_size"], num_workers=cfg["workers"])
    per_image, metrics = predict(model, dl, device)

    tasks = cfg["tasks"]
    result = {"run": run.name, "tasks": tasks, "split": args.split, "best_epoch": ckpt["epoch"]}
    if tasks in ("cls", "both"):
        result["cls"] = metrics["cls"]
    if tasks in ("seg", "both"):
        result["seg"] = metrics["seg"]
    (run / f"{args.split}_metrics.json").write_text(json.dumps(result, indent=2))

    df = pd.DataFrame({
        "id": per_image["id"], "label": per_image["label"], "pred": per_image["pred"],
        **{f"prob_{c}": per_image["probs"][:, k] for k, c in enumerate(CLASS_NAMES)},
        "dice": per_image["dice"], "iou": per_image["iou"],
    })
    df.to_csv(run / f"{args.split}_predictions.csv", index=False)

    n = args.n_examples
    if "cls" in result:
        c = result["cls"]
        print(f"[cls] acc {c['accuracy']:.4f}  macro P {c['macro_precision']:.4f}  "
              f"R {c['macro_recall']:.4f}  F1 {c['macro_f1']:.4f}")
        plot_confusion(c["confusion_matrix"], run / "confusion_matrix.png")
        wrong = df[df.label != df.pred].head(n)
        plot_examples(model, ds, list(wrong.id),
                      [f"{r.id}: true {CLASS_NAMES[r.label]}\npred {CLASS_NAMES[r.pred]}" for r in wrong.itertuples()],
                      run / "misclassified.png", device, show_mask=(tasks == "both"))
    if "seg" in result:
        s = result["seg"]
        print(f"[seg] Dice {s['dice']:.4f} (+/- {s['dice_std']:.4f})  IoU {s['iou']:.4f}")
        ranked = df.sort_values("dice")
        picks = pd.concat([ranked.tail(n).iloc[::-1], ranked.head(n)])
        plot_examples(model, ds, list(picks.id),
                      [f"{r.id} ({CLASS_NAMES[r.label]})\nDice {r.dice:.3f}" for r in picks.itertuples()],
                      run / "seg_best_worst.png", device, ncols=n)
    print(f"Saved results to {run}")


if __name__ == "__main__":
    main()
