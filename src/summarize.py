"""Collect <split>_metrics.json from every run and make one comparison table (mean +/- std over seeds).

Usage:
    python src/summarize.py --runs_dir runs --split test

Runs are grouped by name with the trailing "_seed<N>" removed, e.g. C_mtl_equal_seed0/1/2 -> C_mtl_equal.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

COLUMNS = ["accuracy", "macro_precision", "macro_recall", "macro_f1",
           "dice", "iou", "dice_meningioma", "dice_glioma", "dice_pituitary"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None, help="csv path (default: <runs_dir>/summary_<split>.csv)")
    args = ap.parse_args()

    rows = []
    for f in sorted(Path(args.runs_dir).glob(f"*/{args.split}_metrics.json")):
        m = json.loads(f.read_text())
        row = {"experiment": re.sub(r"_seed\d+$", "", m["run"]), "run": m["run"], "tasks": m["tasks"]}
        for task in ("cls", "seg"):
            row.update({k: v for k, v in m.get(task, {}).items() if k in COLUMNS})
        rows.append(row)
    if not rows:
        raise SystemExit(f"No {args.split}_metrics.json found under {args.runs_dir}")

    df = pd.DataFrame(rows)
    cols = [c for c in COLUMNS if c in df]
    g = df.groupby("experiment")[cols]
    mean, std, n = g.mean(), g.std().fillna(0), g.size()
    table = mean.copy().astype(object)
    for c in cols:
        table[c] = [f"{m:.4f} ± {s:.4f}" if pd.notna(m) else "-" for m, s in zip(mean[c], std[c])]
    table.insert(0, "n_seeds", n)

    out = Path(args.out or Path(args.runs_dir) / f"summary_{args.split}.csv")
    table.to_csv(out, encoding="utf-8-sig")  # BOM so Excel shows ± correctly
    df.to_csv(out.with_name(out.stem + "_all_runs.csv"), index=False)
    print(table.to_string())
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
