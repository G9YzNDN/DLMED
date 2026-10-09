#!/usr/bin/env bash
# Train and test all four experiments (A-D) for several seeds, then build the comparison table.
# Usage: bash scripts/run_experiments.sh [extra train.py args, e.g. --data_dir /content/drive/MyDrive/brain_tumor/processed --out_dir /content/drive/MyDrive/brain_tumor/runs]
set -e
SEEDS="0 1 2"
EXTRA="$@"

# Pick --out_dir from the extra args so evaluate/summarize look in the same place
OUT_DIR=runs
args=("$@")
for i in "${!args[@]}"; do
  if [ "${args[$i]}" = "--out_dir" ]; then OUT_DIR="${args[$((i+1))]}"; fi
done

for SEED in $SEEDS; do
  python src/train.py --tasks cls  --seed $SEED --name A_cls_seed$SEED $EXTRA
  python src/train.py --tasks seg  --seed $SEED --name B_seg_seed$SEED $EXTRA
  python src/train.py --tasks both --seed $SEED --name C_mtl_equal_seed$SEED $EXTRA
  python src/train.py --tasks both --weighting uncertainty --seed $SEED --name D_mtl_uncert_seed$SEED $EXTRA
  for EXP in A_cls B_seg C_mtl_equal D_mtl_uncert; do
    python src/evaluate.py --run "$OUT_DIR/${EXP}_seed$SEED"
  done
done

python src/summarize.py --runs_dir "$OUT_DIR" --split test
