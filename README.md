# Brain Tumor Segmentation + Tumor-Type Classification (Multi-task Learning)

One model with a **shared encoder** learns two tasks together from a T1-weighted contrast-enhanced MRI slice:

1. **Segmentation** of the tumor region (U-Net decoder, metric: Dice / IoU)
2. **Classification** of tumor type: meningioma / glioma / pituitary (metric: Precision / Recall / Macro-F1, confusion matrix)

Main question: *does learning where the tumor is help classify its type (and vice versa)?*

## Dataset

**Figshare Brain Tumor Dataset** (Jun Cheng): 3,064 slices from 233 patients, `.mat` (MATLAB v7.3) files containing
`image`, `tumorMask`, `label` (1 = meningioma, 2 = glioma, 3 = pituitary) and `PID` (patient ID).

- Download: <https://figshare.com/articles/dataset/brain_tumor_dataset/1512427> (4 zip files)
- Do **not** use the Kaggle "Brain Tumor MRI" re-uploads: they have no masks and no patient IDs.
- Citation: Cheng J. et al., "Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition", PLoS ONE 10(10), 2015.

Put all extracted `.mat` files under `data/raw/` (sub-folders are fine). The `data/` folder is git-ignored.
On Colab, keep data and runs in a shared Google Drive folder.

## Setup

**Colab / Kaggle:** use `notebooks/colab_train.ipynb` (installs what it needs).

**Local machine with an NVIDIA GPU (Windows):**

```bash
python -m venv C:/Users/<you>/.venvs/dlmed
C:/Users/<you>/.venvs/dlmed/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
C:/Users/<you>/.venvs/dlmed/Scripts/python -m pip install -r requirements.txt
C:/Users/<you>/.venvs/dlmed/Scripts/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

- RTX 50-series GPUs need the CUDA 12.8 build (`cu128`) as above; older GPUs also work with it.
- If the project folder is inside OneDrive/Dropbox, keep the venv, data and runs **outside** it
  (e.g. `--data_dir C:/brain_tumor/processed --out_dir C:/brain_tumor/runs`) so gigabytes are not synced.
- If DataLoader workers cause errors on Windows, add `--workers 0`.

## Usage

```bash
# 1. Convert .mat -> PNG and make the patient-level split (writes splits/splits.csv)
python src/prepare_data.py --raw_dir data/raw --out_dir data/processed

# 2. Train one experiment (best epoch picked on validation)
python src/train.py --tasks both --name C_mtl_equal_seed0

# 3. Evaluate on the held-out test set
python src/evaluate.py --run runs/C_mtl_equal_seed0

# 4. Run all experiments x 3 seeds and build the comparison table
bash scripts/run_experiments.sh
python src/summarize.py --runs_dir runs --split test
```

Colab: open `notebooks/colab_train.ipynb`.

## Data split (no patient leakage)

Slices from the same patient are near-duplicates, so the split is done **by patient**
(`StratifiedGroupKFold`, grouped by `PID`, stratified by tumor type): about 70 / 15 / 15 % train / val / test.
`prepare_data.py` asserts that no patient appears in two splits.
**Everyone uses the committed `splits/splits.csv`.** The test set is only used after all settings are chosen on validation.

## Experiments

All experiments use the same model (`src/model.py`: U-Net, ResNet-34 ImageNet encoder, classification head on the
deepest encoder features), the same data, augmentation, epochs and seeds. Single-task baselines switch the other loss off.

| ID | Command | Purpose |
|---|---|---|
| A | `--tasks cls` | classification only |
| B | `--tasks seg` | segmentation only |
| C | `--tasks both` | multi-task, equal loss weights |
| D | `--tasks both --weighting uncertainty` | multi-task, learned loss weights (Kendall et al., 2018) |

Fixed weights can also be tried, e.g. `--tasks both --w_seg 1 --w_cls 0.5`.

Default settings: image 256x256, batch 16, AdamW lr 3e-4, weight decay 1e-4, cosine schedule, 30 epochs, mixed precision on GPU.
Segmentation loss = BCE + Dice; classification loss = cross-entropy.
Augmentation (train only, applied to image and mask together): horizontal flip, affine (scale ±10 %, shift ±5 %, rotate ±15°), brightness/contrast.

## Outputs (per run, in `runs/<name>/`)

| File | Content |
|---|---|
| `config.json` | all settings of the run |
| `history.csv` | train losses and validation metrics per epoch |
| `best.pt` | checkpoint of the best validation epoch |
| `test_metrics.json` | test metrics for the trained task(s), incl. per-class F1 and per-tumor-type Dice |
| `test_predictions.csv` | per-image prediction, class probabilities, Dice, IoU (for error analysis) |
| `confusion_matrix.png`, `misclassified.png`, `seg_best_worst.png` | figures for the slides |

## Project structure

```
src/prepare_data.py   .mat -> PNG, metadata, patient-level split
src/dataset.py        PyTorch dataset + augmentation
src/model.py          shared-encoder multi-task model
src/losses.py         task losses and multi-task weighting
src/metrics.py        Dice / IoU / classification metrics
src/engine.py         inference loop shared by training and evaluation
src/train.py          training (one experiment)
src/evaluate.py       test-set evaluation and figures
src/summarize.py      comparison table across runs and seeds
scripts/run_experiments.sh
notebooks/colab_train.ipynb
splits/               shared split files (committed)
```

## Team

| Person | Role | Main files |
|---|---|---|
| 1 | Data & split | `prepare_data.py`, `splits/` |
| 2 | Classification head, cls loss/metrics, experiment A | `losses.py`, `metrics.py` |
| 3 | Segmentation decoder, seg loss/metrics, experiment B | `losses.py`, `metrics.py` |
| 4 | Multi-task model & training, experiments C/D | `model.py`, `train.py` |
| 5 | Evaluation & analysis | `evaluate.py`, `summarize.py`, analysis notebook |
| 6 | Slides, README, submission | `README.md`, slides |

## Workflow

1. `git pull` before you start.
2. Work on your own branch (`git checkout -b <topic>`) and open a Pull Request into `main`.
3. Never commit data, checkpoints or run outputs (see `.gitignore`).
