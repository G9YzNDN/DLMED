# Team Plan — Review, Slides, Presentation

**Status: the project is finished.** Headline: multi-task learning improved classification (Macro-F1 0.916 → 0.938) and kept segmentation equal (Dice ≈ 0.775). Data, code, all experiments, test results and figures are in this repo.
Our job now: **(1) understand and check** our part, **(2) make our slides**, **(3) present and answer questions.**

You do **not** need to run any code. Everything you need is on GitHub:

| What | Where |
|---|---|
| Results in plain language, key numbers, Q&A answers | [`results/RESULTS.md`](results/RESULTS.md) ← **start here** |
| Slide-ready figures | [`results/figures/`](results/figures/) |
| Tables (CSV) | [`results/`](results/) |
| How the code works / how to run it | [`README.md`](README.md) |

## Timeline

| Step | What | Who |
|---|---|---|
| 1 | Read `results/RESULTS.md` fully (≈ 20 min) | Everyone |
| 2 | Review your part using the checklist below; write questions/problems in the group chat | Everyone |
| 3 | Make your slides in the shared template (Person 6 sends it) | Everyone |
| 4 | Person 6 merges slides and checks style | Person 6 |
| 5 | Rehearse twice with a timer, practise the Q&A list | Everyone |

Each person presents their own slides (≈ 2 minutes each). Everyone must be able to explain the **big picture**:
*one shared encoder, two heads (tumor mask + tumor type), compared with single-task models.*

---

## Person 1 — Problem & Data

**Review**
- [ ] `results/RESULTS.md` sections "Problem" and "Data"
- [ ] `splits/data_summary.csv` — images and patients per split and class
- [ ] `src/prepare_data.py` — how `.mat` files become images + masks, and how the split is made

**Slides (3)**
1. Problem & goal: why segment *and* classify brain tumors; our question "does learning both together help?"
2. Dataset: Figshare (Cheng et al. 2015), 3,064 MRI slices, 3 tumor types, example images with masks (`figures/0_dataset_examples.png`)
3. Data split: split **by patient** (70/15/15), the leakage we found and fixed (suffixed patient IDs → 209 patients), split table

**Be ready to answer:** Why split by patient, not by image? What leakage did we find and how did we fix it?

## Person 2 — Classification

**Review**
- [ ] RESULTS.md "Classification results"
- [ ] `results/figures/3_confusion_matrices.png`
- [ ] `src/losses.py` (`cls_loss`), `src/metrics.py` (`classification_metrics`)

**Slides (2)**
1. How classification works: encoder features → pooling → linear layer → 3 classes; cross-entropy loss; metrics Precision / Recall / Macro-F1
2. Results: experiment A vs C vs D (Macro-F1 table), confusion matrix, which classes get confused and why

**Be ready to answer:** Why Macro-F1 and not only accuracy? Which class is hardest and why? Why might segmentation help classification?

## Person 3 — Segmentation

**Review**
- [ ] RESULTS.md "Segmentation results"
- [ ] `results/figures/4_dice_by_tumor_type.png`, `5_dice_vs_tumor_size.png`, `7_*_seg_best_worst.png`
- [ ] `src/losses.py` (`seg_loss`, `dice_loss`), `src/metrics.py` (`dice_iou_per_image`)

**Slides (2)**
1. How segmentation works: U-Net decoder, BCE + Dice loss, Dice / IoU metrics
2. Results: experiment B vs C vs D (Dice table), best and worst examples vs ground truth, Dice by tumor type and size

**Be ready to answer:** What is Dice? Why BCE + Dice? Why are gliomas the hardest to segment? Why didn't multi-task help segmentation?

## Person 4 — Multi-task Model & Experiments

**Review**
- [ ] RESULTS.md "Method" and "Experiments"
- [ ] `src/model.py`, `src/train.py`, `MultiTaskLoss` in `src/losses.py`
- [ ] `results/figures/2_training_curves.png`, `6_uncertainty_weights.png`, `results/tuning_lr_validation.csv`

**Slides (3)**
1. Architecture diagram (`figures/0_architecture.png`): input → shared ResNet-34 encoder → U-Net decoder (mask) + classification head (type)
2. Experiments A/B/C/D and why the comparison is fair (same model, data, augmentation, epochs, 3 seeds; lr chosen on validation only)
3. Training setup table (image size, batch, optimizer, lr, epochs, augmentation) + training curves; how uncertainty weighting works

**Be ready to answer:** What is a shared encoder? What does uncertainty weighting do? How did you choose the learning rate?

## Person 5 — Evaluation & Analysis

**Review**
- [ ] RESULTS.md "Main result", "Error analysis", "Limitations"
- [ ] `results/figures/1_main_comparison.png`, `results/summary_test.csv`, `results/paired_differences.csv`,
      `results/errors_by_patient.csv`, `results/error_analysis.json`, `results/figures/7_*_misclassified.png`

**Slides (3)**
1. Main result: A/B/C/D comparison (mean ± std over 3 seeds) — does multi-task help each task?
2. Error analysis: one patient (sellar meningioma) causes most classification errors; misclassified slices are well segmented; worst segmentations are gliomas
3. Limitations & future work

**Be ready to answer:** Is the difference between models real or just noise? What are the main limitations?

## Person 6 — Slides, Story & Presentation

**Review**
- [ ] Whole RESULTS.md and README.md
- [ ] Open the repo as an outsider: is it clear how to run it? (README "Setup" and "Usage")

**Slides (3) + whole deck**
1. Title (project name, team names, course)
2. Outline
3. Conclusion: answer to our question in one sentence + 3 key takeaways
- Make the template **before** step 3: one font, one colour set (use the experiment colours from the figures: A blue, B orange, C green, D yellow), big text (≥ 18 pt), every chart with title and axis labels
- Merge everyone's slides, keep style consistent, add slide numbers, time the rehearsal

**Be ready to answer:** Summarise the whole project in 30 seconds.

---

## Slide order (≈ 16 slides, ≈ 12–15 minutes)

| # | Slide | Person | Rubric item |
|---|---|---|---|
| 1 | Title | 6 | – |
| 2 | Outline | 6 | – |
| 3 | Problem & goal | 1 | Problem & data (2) |
| 4 | Dataset | 1 | Problem & data |
| 5 | Patient-level split & leakage fix | 1 | Problem & data |
| 6 | Architecture (shared encoder + 2 heads) | 4 | Difficulty (4): multi-task |
| 7 | Experiments A–D & fair comparison | 4 | Model & experiments (2) |
| 8 | Training setup & curves | 4 | Model & experiments |
| 9 | Classification method | 2 | Model & experiments |
| 10 | Segmentation method | 3 | Model & experiments |
| 11 | Main result A/B/C/D | 5 | Evaluation (2) |
| 12 | Classification results + confusion matrix | 2 | Evaluation |
| 13 | Segmentation results + examples vs ground truth | 3 | Evaluation |
| 14 | Error analysis | 5 | Evaluation |
| 15 | Limitations & future work | 5 | Evaluation |
| 16 | Conclusion | 6 | – |

The other two rubric items are **slide design (2)** (Person 6 keeps it clean and consistent) and
**presentation & Q&A (3)** (everyone presents their part and answers questions).
