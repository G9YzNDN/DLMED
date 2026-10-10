#!/usr/bin/env bash
# Pick the learning rate on the VALIDATION set using the multi-task model (experiment C, seed 0).
# The test set is not used here. Result: runs/tune_lr_*/history.csv -> best val_score per lr.
# Usage: bash scripts/tune_lr.sh [extra train.py args]
set -e
OUT_DIR=runs
args=("$@")
for i in "${!args[@]}"; do
  if [ "${args[$i]}" = "--out_dir" ]; then OUT_DIR="${args[$((i+1))]}"; fi
done
for LR in 1e-4 3e-4 1e-3; do
  python src/train.py --tasks both --seed 0 --lr $LR --name tune_lr_$LR "$@"
done
python - "$OUT_DIR" <<'EOF'
import pandas as pd, sys
from pathlib import Path
for f in sorted(Path(sys.argv[1]).glob("tune_lr_*/history.csv")):
    h = pd.read_csv(f); b = h.loc[h.val_score.idxmax()]
    print(f"{Path(f).parent.name:16s} best val_score {b.val_score:.4f} "
          f"(F1 {b.val_macro_f1:.4f}, Dice {b.val_dice:.4f}) at epoch {int(b.epoch)}")
EOF
