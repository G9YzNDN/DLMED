"""Build the final results package (tables + slide-ready figures) from finished runs.

Run after all experiments are trained and evaluated on the test set:
    python src/analyze.py --runs_dir runs --out_dir results

Expects run folders named A_cls_seed*, B_seg_seed*, C_mtl_equal_seed*, D_mtl_uncert_seed*,
each with history.csv, test_metrics.json and test_predictions.csv.
"""
import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXPS = {  # id -> (label, colour); colours = fixed categorical order of the reference palette
    "A_cls": ("A: Cls-only", "#2a78d6"),
    "B_seg": ("B: Seg-only", "#eb6834"),
    "C_mtl_equal": ("C: MTL equal", "#1baf7a"),
    "D_mtl_uncert": ("D: MTL uncertainty", "#eda100"),
}
CLASSES = ["meningioma", "glioma", "pituitary"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold", "legend.frameon": False,
})


def load_runs(runs_dir):
    runs = {}
    for exp in EXPS:
        for d in sorted(Path(runs_dir).glob(f"{exp}_seed*")):
            if not (d / "test_metrics.json").exists():
                continue
            runs.setdefault(exp, []).append({
                "seed": int(d.name.rsplit("seed", 1)[1]),
                "dir": d,
                "metrics": json.loads((d / "test_metrics.json").read_text()),
                "history": pd.read_csv(d / "history.csv"),
                "pred": pd.read_csv(d / "test_predictions.csv", dtype={"id": str}),
                "config": json.loads((d / "config.json").read_text()),
            })
    return runs


def metric_table(runs):
    """One row per run with every test metric of the task(s) it was trained for."""
    rows = []
    for exp, rs in runs.items():
        for r in rs:
            m = r["metrics"]
            row = {"experiment": exp, "seed": r["seed"], "best_epoch": m["best_epoch"]}
            if "cls" in m:
                c = m["cls"]
                row.update(accuracy=c["accuracy"], macro_precision=c["macro_precision"],
                           macro_recall=c["macro_recall"], macro_f1=c["macro_f1"],
                           **{f"f1_{k}": v for k, v in zip(CLASSES, c["per_class_f1"])})
            if "seg" in m:
                s = m["seg"]
                row.update(dice=s["dice"], iou=s["iou"],
                           **{f"dice_{k}": s.get(f"dice_{k}") for k in CLASSES})
            rows.append(row)
    return pd.DataFrame(rows)


def mean_std_table(df):
    cols = [c for c in df.columns if c not in ("experiment", "seed", "best_epoch")]
    g = df.groupby("experiment", sort=False)[cols]
    mean, std = g.mean(), g.std()
    out = pd.DataFrame(index=mean.index)
    out["n_seeds"] = g.size()
    for c in cols:
        out[c] = [f"{m:.4f} ± {s:.4f}" if pd.notna(m) else "-" for m, s in zip(mean[c], std[c])]
    return mean, std, out


def dot_panel(ax, exps, mean, std, df, metric, title):
    """Dot plot: one hollow dot per seed, filled marker = mean, line = mean ± 1 std.
    A dot plot (not bars) because the y-axis is zoomed in and bars must start at zero."""
    vals = df[df.experiment.isin(exps)][metric]
    span = max(vals.max() - vals.min(), 0.01)
    lo, hi = vals.min() - 0.35 * span, vals.max() + 0.35 * span
    for i, e in enumerate(exps):
        label, colour = EXPS[e]
        m, sd = mean.loc[e, metric], std.loc[e, metric]
        sd = 0 if np.isnan(sd) else sd
        ax.plot([i, i], [m - sd, m + sd], color=colour, lw=3, solid_capstyle="round", zorder=2)
        seeds = df[df.experiment == e][metric]
        ax.scatter(np.full(len(seeds), i + 0.18), seeds, s=40, color=SURFACE, edgecolor=colour,
                   linewidth=1.8, zorder=3)
        ax.scatter([i], [m], s=140, color=colour, edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.annotate(f"{m:.3f} ± {sd:.3f}", (i, m + sd), xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold", color=INK)
    ax.set_xticks(range(len(exps)), [EXPS[e][0].replace(": ", ":\n") for e in exps])
    ax.set_xlim(-0.6, len(exps) - 0.4)
    ax.set_ylim(lo, hi + 0.15 * (hi - lo))
    ax.set_title(title)
    ax.grid(axis="x", visible=False)


def fig_main(df, mean, std, out):
    cls_exps = [e for e in ("A_cls", "C_mtl_equal", "D_mtl_uncert") if e in mean.index]
    seg_exps = [e for e in ("B_seg", "C_mtl_equal", "D_mtl_uncert") if e in mean.index]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    dot_panel(axes[0], cls_exps, mean, std, df, "macro_f1", "Classification — Macro-F1 (test)")
    dot_panel(axes[1], seg_exps, mean, std, df, "dice", "Segmentation — Dice (test)")
    axes[0].set_ylabel("Macro-F1")
    axes[1].set_ylabel("Dice coefficient")
    fig.text(0.5, -0.02, "Big dot: mean of seeds · line: ± 1 std · small dots: individual seeds · "
             "note the zoomed-in y-axis", ha="center", color=INK2, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_dice_by_type(mean, std, out):
    exps = [e for e in ("B_seg", "C_mtl_equal", "D_mtl_uncert") if e in mean.index]
    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.26
    for j, e in enumerate(exps):
        label, colour = EXPS[e]
        vals = [mean.loc[e, f"dice_{k}"] for k in CLASSES]
        errs = [std.loc[e, f"dice_{k}"] for k in CLASSES]
        xs = np.arange(3) + (j - 1) * w
        ax.bar(xs, vals, width=w, color=colour, edgecolor=SURFACE, linewidth=2, label=label)
        ax.errorbar(xs, vals, yerr=errs, fmt="none", color=INK, capsize=4, lw=1.2)
    ax.set_xticks(range(3), [c.capitalize() for c in CLASSES])
    ax.set_ylabel("Dice coefficient")
    ax.set_ylim(0, 1)
    ax.set_title("Segmentation Dice by tumor type (test, mean ± std of 3 seeds)", pad=34)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_curves(runs, out):
    """Per-task training loss and validation metric, seed 0 of every experiment that trains that task.
    (D's total loss includes the learned log-variance terms, so per-task losses are what can be compared.)"""
    cls_e = ("A_cls", "C_mtl_equal", "D_mtl_uncert")
    seg_e = ("B_seg", "C_mtl_equal", "D_mtl_uncert")
    panels = [("train_loss_cls", "Training loss — classification (CE)", cls_e, None),
              ("train_loss_seg", "Training loss — segmentation (BCE + Dice)", seg_e, None),
              ("val_macro_f1", "Validation Macro-F1", cls_e, (0.75, 1.0)),
              ("val_dice", "Validation Dice", seg_e, (0.5, 0.85))]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for ax, (col, title, exps, ylim) in zip(axes.flat, panels):
        for e in exps:
            if e not in runs:
                continue
            h = next((r["history"] for r in runs[e] if r["seed"] == 0), runs[e][0]["history"])
            ax.plot(h.epoch, h[col], color=EXPS[e][1], lw=2, label=EXPS[e][0])
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Epoch")
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=10)
    fig.suptitle("Training curves (seed 0)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_uncertainty(runs, out):
    if "D_mtl_uncert" not in runs:
        return
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for r in runs["D_mtl_uncert"]:
        h = r["history"]
        ax.plot(h.epoch, np.exp(-h.log_var_seg), color="#eb6834", lw=2, alpha=0.9,
                label="Segmentation weight" if r["seed"] == 0 else None)
        ax.plot(h.epoch, np.exp(-h.log_var_cls), color="#2a78d6", lw=2, alpha=0.9,
                label="Classification weight" if r["seed"] == 0 else None)
    ax.axhline(1, color=INK2, lw=1, ls="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Effective loss weight  exp(−s)")
    ax.set_title("D: learned task weights (one line per seed)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_confusion(runs, out):
    exps = [e for e in ("A_cls", "C_mtl_equal", "D_mtl_uncert") if e in runs]
    fig, axes = plt.subplots(1, len(exps), figsize=(5.2 * len(exps), 4.8))
    for ax, e in zip(np.atleast_1d(axes), exps):
        cm = sum(np.array(r["metrics"]["cls"]["confusion_matrix"]) for r in runs[e])
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]:.0%})", ha="center", va="center", fontsize=11,
                        color="white" if norm[i, j] > 0.6 else INK)
        ax.set_xticks(range(3), [c.capitalize() for c in CLASSES], rotation=20)
        ax.set_yticks(range(3), [c.capitalize() for c in CLASSES])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{EXPS[e][0]}\n(test, summed over {len(runs[e])} seeds)", fontsize=12)
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def size_bins(area):
    edges = [0, 1000, 2500, 5000, np.inf]
    labels = ["< 1k", "1k–2.5k", "2.5k–5k", "> 5k"]
    return pd.cut(area, edges, labels=labels)


def fig_dice_vs_size(runs, meta, out):
    exps = [e for e in ("B_seg", "C_mtl_equal", "D_mtl_uncert") if e in runs]
    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.26
    for j, e in enumerate(exps):
        p = pd.concat([r["pred"] for r in runs[e]]).merge(meta[["id", "tumor_area"]], on="id")
        g = p.groupby(size_bins(p.tumor_area), observed=False).dice.mean()
        xs = np.arange(len(g)) + (j - 1) * w
        ax.bar(xs, g.values, width=w, color=EXPS[e][1], edgecolor=SURFACE, linewidth=2, label=EXPS[e][0])
    counts = size_bins(runs[exps[0]][0]["pred"].merge(meta, on="id").tumor_area).value_counts(sort=False)
    ax.set_xticks(range(len(counts)), [f"{k} px\n(n={v})" for k, v in counts.items()])
    ax.set_xlabel("Tumor area in the original image (pixels), n = test images per bin")
    ax.set_ylabel("Mean Dice")
    ax.set_ylim(0, 1)
    ax.set_title("Segmentation Dice by tumor size (test, all seeds pooled)", pad=34)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_architecture(out):
    """Schematic of the shared-encoder multi-task model (no data needed)."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    def box(x, y, w, h, text, colour, sub=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    facecolor=colour, edgecolor="none", alpha=0.18))
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    facecolor="none", edgecolor=colour, linewidth=2))
        ax.text(x + w / 2, y + h / 2 + (0.18 if sub else 0), text, ha="center", va="center",
                fontsize=12.5, fontweight="bold", color=INK)
        if sub:
            ax.text(x + w / 2, y + h / 2 - 0.25, sub, ha="center", va="center", fontsize=10.5, color=INK2)

    def arrow(x0, y0, x1, y1, style="-"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
                                     color=INK2, linewidth=1.8, linestyle=style))

    box(0.2, 2.0, 2.0, 1.2, "MRI slice", INK2, "1 × 256 × 256")
    box(3.0, 1.6, 2.6, 2.0, "Shared encoder", "#4a3aa7", "ResNet-34 (ImageNet)")
    box(6.6, 3.2, 2.6, 1.3, "U-Net decoder", "#eb6834", "upsampling + skips")
    box(6.6, 0.6, 2.6, 1.3, "Classification head", "#2a78d6", "pool → dropout → linear")
    box(10.0, 3.2, 3.0, 1.3, "Tumor mask", "#eb6834", "loss: BCE + Dice")
    box(10.0, 0.6, 3.0, 1.3, "Tumor type (3 classes)", "#2a78d6", "loss: cross-entropy")
    arrow(2.2, 2.6, 3.0, 2.6)
    arrow(5.6, 3.1, 6.6, 3.85)
    arrow(5.6, 2.1, 6.6, 1.25)
    arrow(9.2, 3.85, 10.0, 3.85)
    arrow(9.2, 1.25, 10.0, 1.25)
    ax.add_patch(FancyArrowPatch((4.3, 3.6), (6.6, 4.2), connectionstyle="arc3,rad=-0.35", arrowstyle="-|>",
                                 mutation_scale=14, color=INK2, linewidth=1.2, linestyle="--"))
    ax.text(5.0, 4.55, "skip connections", fontsize=10.5, color=INK2, ha="center")
    ax.text(6.5, 0.05, "Total loss  L = w_seg · L_seg + w_cls · L_cls     "
            "A: w_seg = 0 · B: w_cls = 0 · C: both 1 · D: learned (uncertainty weighting)",
            ha="center", fontsize=11.5, color=INK)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_dataset_examples(meta, data_dir, out):
    import cv2
    fig, axes = plt.subplots(3, 4, figsize=(12, 9.6))
    for r, c in enumerate(CLASSES):
        for j, row in enumerate(meta[meta.class_name == c].sample(4, random_state=1).itertuples()):
            img = cv2.imread(str(Path(data_dir) / "images" / f"{row.id}.png"), cv2.IMREAD_GRAYSCALE)
            msk = cv2.imread(str(Path(data_dir) / "masks" / f"{row.id}.png"), cv2.IMREAD_GRAYSCALE)
            ax = axes[r, j]
            ax.imshow(img, cmap="gray")
            ax.contour(msk > 0, levels=[0.5], colors="#1baf7a", linewidths=1.5)
            ax.set_title(f"{c.capitalize()} · image {row.id}", fontsize=10.5)
            ax.axis("off")
    fig.suptitle("Dataset examples — tumor outline from the expert mask (green)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def error_analysis(runs, meta):
    """Numbers for the error-analysis slide (pooled over seeds)."""
    out = {}
    for e in ("C_mtl_equal", "D_mtl_uncert"):
        if e not in runs:
            continue
        p = pd.concat([r["pred"] for r in runs[e]]).merge(meta[["id", "tumor_area"]], on="id")
        wrong = p.label != p.pred
        out[e] = {
            "n_predictions": int(len(p)),
            "misclassified": int(wrong.sum()),
            "dice_when_class_correct": float(p.loc[~wrong, "dice"].mean()),
            "dice_when_class_wrong": float(p.loc[wrong, "dice"].mean()) if wrong.any() else None,
            "median_area_correct": float(p.loc[~wrong, "tumor_area"].median()),
            "median_area_wrong": float(p.loc[wrong, "tumor_area"].median()) if wrong.any() else None,
            "share_dice_below_0.5": float((p.dice < 0.5).mean()),
            "share_dice_zero": float((p.dice == 0).mean()),
        }
    for e in ("B_seg",):
        if e in runs:
            p = pd.concat([r["pred"] for r in runs[e]])
            out[e] = {"share_dice_below_0.5": float((p.dice < 0.5).mean()),
                      "share_dice_zero": float((p.dice == 0).mean())}
    return out


def errors_by_patient(runs, meta):
    """Classification errors grouped by test patient (all seeds pooled) — shows whether errors are spread out
    or concentrated in a few patients."""
    rows = []
    for e in ("A_cls", "C_mtl_equal", "D_mtl_uncert"):
        if e not in runs:
            continue
        p = pd.concat([r["pred"] for r in runs[e]]).merge(meta[["id", "patient", "class_name"]], on="id")
        wrong = p.label != p.pred
        t = (p.assign(wrong=wrong).groupby(["patient", "class_name"])
             .agg(slices_x_seeds=("wrong", "size"), errors=("wrong", "sum")).reset_index())
        t = t[t.errors > 0].sort_values("errors", ascending=False)
        t["predicted_as"] = [
            ", ".join(f"{CLASSES[k]} ({v})" for k, v in p[(p.patient == pt) & wrong].pred.value_counts().items())
            for pt in t.patient]
        t["share_of_all_errors"] = (t.errors / max(int(wrong.sum()), 1)).round(3)
        t.insert(0, "experiment", e)
        rows.append(t)
    return pd.concat(rows) if rows else pd.DataFrame()


def paired_differences(df):
    """Descriptive differences matched by seed label; original augmentations were not seeded."""
    rows = []
    for mtl in ("C_mtl_equal", "D_mtl_uncert"):
        for base, metric in (("A_cls", "macro_f1"), ("B_seg", "dice")):
            a = df[df.experiment == mtl].set_index("seed")[metric]
            b = df[df.experiment == base].set_index("seed")[metric]
            d = (a - b).dropna()
            if len(d):
                rows.append({"comparison": f"{mtl} - {base}", "metric": metric,
                             "per_seed": [round(v, 4) for v in d.tolist()],
                             "mean_diff": round(d.mean(), 4), "seeds_better": int((d > 0).sum()),
                             "n_seeds": len(d)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--metadata", default="data/processed/metadata.csv")
    ap.add_argument("--example_run", default="C_mtl_equal_seed0", help="run whose example images are copied")
    args = ap.parse_args()

    out = Path(args.out_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.runs_dir)
    if not runs:
        raise SystemExit(f"No finished runs in {args.runs_dir}")
    meta = pd.read_csv(args.metadata, dtype={"id": str})

    df = metric_table(runs)
    mean, std, table = mean_std_table(df)
    df.to_csv(out / "all_runs_test.csv", index=False)
    table.to_csv(out / "summary_test.csv", encoding="utf-8-sig")
    paired = paired_differences(df)
    paired.to_csv(out / "paired_differences.csv", index=False)
    errors = error_analysis(runs, meta)
    by_patient = errors_by_patient(runs, meta)
    by_patient.to_csv(out / "errors_by_patient.csv", index=False)
    (out / "error_analysis.json").write_text(json.dumps(errors, indent=2))

    fig_architecture(fig_dir / "0_architecture.png")
    data_dir = Path(args.metadata).parent
    if (data_dir / "images").exists():
        fig_dataset_examples(meta, data_dir, fig_dir / "0_dataset_examples.png")
    fig_main(df, mean, std, fig_dir / "1_main_comparison.png")
    fig_curves(runs, fig_dir / "2_training_curves.png")
    fig_confusion(runs, fig_dir / "3_confusion_matrices.png")
    fig_dice_by_type(mean, std, fig_dir / "4_dice_by_tumor_type.png")
    fig_dice_vs_size(runs, meta, fig_dir / "5_dice_vs_tumor_size.png")
    fig_uncertainty(runs, fig_dir / "6_uncertainty_weights.png")
    ex = Path(args.runs_dir) / args.example_run
    for name in ("seg_best_worst.png", "misclassified.png"):
        if (ex / name).exists():
            shutil.copy(ex / name, fig_dir / f"7_{args.example_run}_{name}")

    pd.set_option("display.width", 200)
    print(table.T.to_string())
    print("\nPaired differences (multi-task minus single-task, per seed):")
    print(paired.to_string(index=False))
    print("\nError analysis:", json.dumps(errors, indent=2))
    print("\nClassification errors by patient:")
    print(by_patient.to_string(index=False))
    print(f"\nSaved tables and figures to {out}")


if __name__ == "__main__":
    main()
