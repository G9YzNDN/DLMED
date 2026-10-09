# Team Tasks (6 people)

The starter code already runs end to end (data prep → train → evaluate → summary table).
Each person **owns** one part: check it, improve it, run its experiments, write its slides and present it.
Everyone must understand the whole pipeline for the Q&A (3 of the 15 points).

## Order of work

```
Phase 1  Data ready        Person 1 makes splits.csv + processed.zip   (everyone waits on this)
         Sanity check      Person 4 trains 2-3 epochs to prove the pipeline works on Colab
Phase 2  Freeze settings   Person 4 tunes lr / epochs on VALIDATION with experiment C, then settings are locked
Phase 3  Run experiments   A (P2), B (P3), C + D (P4), each x 3 seeds, all with the locked settings
Phase 4  Evaluate          Person 5 builds the comparison table and error analysis
Phase 5  Slides + rehearse Person 6 assembles, everyone presents their part
```

Rule: **nobody looks at test results until Phase 4.** All choices are made on validation.

---

## Person 1 — Data & split

Code: `src/prepare_data.py`, `splits/`

- [ ] Download the 4 zip files from Figshare (link in README) into the shared Drive folder `brain_tumor/raw/`
- [ ] Run `prepare_data.py` and check the output: 3,064 images, 233 patients, 3 classes, no empty masks
- [ ] Look at ~10 images per class with the mask overlaid to confirm masks line up with the tumor
- [ ] Check that no patient is in two splits (the script asserts this; also confirm in `splits/data_summary.csv`)
- [ ] Commit `splits/splits.csv` and `splits/data_summary.csv`; upload `processed.zip` to Drive
- [ ] Later: help Person 5 analyse errors by tumor size (`tumor_area` in `metadata.csv`)

**Slides:** problem & objective, dataset source and citation, class distribution table, example images + masks,
why we split by patient (slices from one patient are near-duplicates → leakage).

## Person 2 — Classification (experiment A)

Code: classification head (`model.py`, `aux_params`), `cls_loss` in `losses.py`, `classification_metrics` in `metrics.py`

- [ ] Understand where the classification head sits (deepest encoder features → pooling → linear)
- [ ] Run experiment A for seeds 0, 1, 2 with the locked settings, then `evaluate.py` on each
- [ ] Check class balance (glioma has about 2x the images of meningioma); try class-weighted CE on **validation only** and report if it helps
- [ ] Optional: Grad-CAM on a few cases to show where the classifier looks (cls-only vs multi-task)

**Slides:** classification method, Precision / Recall / Macro-F1, confusion matrix, which classes get confused.

## Person 3 — Segmentation (experiment B)

Code: U-Net decoder (`model.py`), `seg_loss` / `dice_loss` in `losses.py`, `dice_iou_per_image` in `metrics.py`

- [ ] Understand the U-Net decoder and why the loss is BCE + Dice
- [ ] Run experiment B for seeds 0, 1, 2, then `evaluate.py` on each
- [ ] Check `seg_best_worst.png`: green = ground truth, red = prediction
- [ ] Compare Dice per tumor type (pituitary tumors are often small)
- [ ] Optional: check whether a threshold other than 0.5 is better on validation

**Slides:** segmentation method, Dice / IoU, image comparisons with ground truth, per-tumor-type Dice.

## Person 4 — Multi-task model & training (experiments C, D)

Code: `model.py`, `train.py`, `MultiTaskLoss` in `losses.py`, `scripts/run_experiments.sh`

- [ ] Phase 1: sanity run on Colab (`--epochs 2`) and share the notebook steps that worked with the team
- [ ] Phase 2: tune lr and epochs on validation using experiment C; write the final settings in the README and tell the team
- [ ] Run C (equal weights) and D (uncertainty weighting) for seeds 0, 1, 2, then `evaluate.py`
- [ ] Optional: a few fixed weights (e.g. `--w_cls 0.5`, `--w_cls 2`) chosen on validation
- [ ] Plot training curves from `history.csv`, and for D the learned weights (`log_var_seg`, `log_var_cls`)

**Slides:** architecture diagram (shared encoder + 2 heads), loss formulas, training setup table, why the comparison is fair
(same model, data, augmentation, epochs and seeds; baselines just switch one loss off).

## Person 5 — Evaluation & analysis

Code: `src/evaluate.py`, `src/summarize.py`, `src/engine.py`, analysis notebook

- [ ] Build the final table A/B/C/D with `summarize.py` (mean ± std over 3 seeds)
- [ ] Answer the main question: does multi-task help classification? Does it help segmentation?
      If the difference is smaller than the std, say so honestly.
- [ ] Error analysis from `test_predictions.csv`: misclassified cases, lowest-Dice cases, Dice vs tumor size,
      are misclassified cases also badly segmented?
- [ ] Write limitations: single dataset/center, 2D slices only, small test set (~35 patients), T1-CE only

**Slides:** results table, bar chart with error bars, correct and wrong examples, limitations, conclusion.

## Person 6 — Slides, README & submission

Code: `README.md`, `notebooks/`, slides

- [ ] Make the slide template now (one font, one colour scheme, readable font size, labelled axes)
- [ ] Draw the pipeline/architecture figures together with Person 4
- [ ] Collect each person's slides and keep the style consistent; make sure every figure has a title and axis labels
- [ ] Final check: clone the repo fresh on Colab and run it from the README alone
- [ ] Organise rehearsals, time the talk, and prepare a list of likely questions and answers

**Slides:** title, outline, conclusion; owns the whole deck's look.

---

## Likely questions (everyone should be able to answer)

- Why split by patient and not by image?
- What is a shared encoder, and why could two tasks help each other?
- Why Dice for segmentation and Macro-F1 for classification (not just accuracy)?
- How do you know the comparison between A/B/C/D is fair?
- What does uncertainty weighting do?
- Why didn't multi-task help much / help a lot? What are the limitations?
