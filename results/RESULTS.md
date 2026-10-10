# Results — Brain Tumor Segmentation + Classification with Multi-task Learning

Plain-language summary of the whole project for the team. Every number here comes from the files in this folder.
Figures for the slides are in [`figures/`](figures/).

**Review note (2026-10-10):** all 12 saved checkpoints were re-evaluated on the full test set and reproduced
their saved predictions. The tables and figures below remain the original results. Those runs did not seed
Albumentations, so the seed labels do not imply identical augmentation draws or exact retraining reproducibility.
The corrected code seeds augmentation and data-loader workers for **new** runs; it does not retroactively change
these results. See [README.md](../README.md) and [codextodo.md](../codextodo.md).

## 1. Problem

Brain MRI scans are read by radiologists who must (1) **find the tumor** and (2) **decide what type it is**,
because treatment depends on the type. We train a deep-learning model to do both from one MRI slice:

- **Segmentation**: draw the tumor region pixel by pixel (output: a mask)
- **Classification**: name the tumor type — meningioma, glioma or pituitary tumor

**Research question:** if one model learns both tasks together (multi-task learning), does each task get better
than when it is learned alone?

## 2. Data

| | |
|---|---|
| Source | Figshare *brain tumor dataset* ([Cheng et al., PLoS ONE 2015](https://doi.org/10.1371/journal.pone.0140381)), CC BY 4.0 |
| Acquisition sites | Nanfang Hospital and General Hospital, Tianjin Medical University (two hospitals) |
| Images | 3,064 T1-weighted contrast-enhanced MRI slices (3,049 at 512×512, 15 at 256×256) |
| Labels | expert tumor mask + tumor type for every slice |
| Classes | meningioma 708 · glioma 1,426 · pituitary 930 slices |

Figure: `figures/0_dataset_examples.png`

**Preparation:** `.mat` files → images scaled to 0–255 + binary masks (`src/prepare_data.py`); resized to 256×256 for training.

**Split by patient, not by image.** One patient gives many very similar slices. If slices of the same patient were in
both train and test, the test score would be too optimistic (data leakage). We split by patient
(`StratifiedGroupKFold`: grouped by patient, balanced by tumor type).

**Conservative grouping to reduce possible leakage.** The source reports 233 patients; the files contain
233 distinct patient IDs. Eighteen groups of glioma IDs have letter-suffixed variants
(e.g. `MR040240`, `MR040240B`, `MR040240C`, `MR040240D`). We have not confirmed whether the variants belong
to the same person. Fourteen such groups crossed the initial raw-ID split, so we kept each group together as a
precaution. This produces **209 conservative groups**, not a verified correction to the source's patient count:

| Split | Patient groups | Images | Meningioma | Glioma | Pituitary |
|---|---|---|---|---|---|
| Train | 151 | 2,194 | 505 | 1,019 | 670 |
| Validation | 29 | 434 | 102 | 203 | 129 |
| Test | 29 | 436 | 101 | 204 | 131 |

No recorded ID or conservative group appears in two committed splits (checked automatically).
The `patients` columns in `splits/data_summary.csv` count these groups. Test results were not used to tune
hyperparameters; the later review repeated inference to verify the saved outputs.

## 3. Method

Figure: `figures/0_architecture.png`

- **Shared encoder:** ResNet-34 pre-trained on ImageNet, reads the MRI slice and extracts features
- **Segmentation head:** U-Net decoder (upsampling + skip connections from the encoder) → tumor mask
- **Classification head:** global average pooling → dropout 0.2 → linear layer → 3 tumor types
- **Losses:** segmentation = BCE + Dice; classification = cross-entropy
- **Total loss:** `L = w_seg · L_seg + w_cls · L_cls`

| Setting | Value |
|---|---|
| Input | 1 channel, 256×256, normalised |
| Augmentation (train only, image and mask together) | horizontal flip, scale ±10 %, shift ±5 %, rotate ±15°, brightness/contrast ±15 % |
| Optimizer | AdamW, weight decay 1e-4, cosine learning-rate schedule |
| Learning rate | 3e-4 (chosen on validation, see below) |
| Epochs / batch | 30 / 16, mixed precision |
| Model selection | best epoch on the **validation** set (cls: Macro-F1, seg: Dice, multi-task: their average) |
| Hardware | NVIDIA RTX 5070 Laptop GPU (8 GB), ≈ 13 s per epoch, ≈ 7 min per run |

**Learning-rate choice (validation only, experiment C, seed 0)** — `tuning_lr_validation.csv`:

| lr | Val Macro-F1 | Val Dice | Val score (average) |
|---|---|---|---|
| 1e-4 | 0.977 | 0.742 | 0.860 |
| **3e-4** | **0.986** | **0.769** | **0.877** |
| 1e-3 | 0.979 | 0.754 | 0.867 |

## 4. Experiments

| ID | What is trained | Loss weights | Purpose |
|---|---|---|---|
| **A** | classification only | w_seg = 0, w_cls = 1 | single-task baseline for classification |
| **B** | segmentation only | w_seg = 1, w_cls = 0 | single-task baseline for segmentation |
| **C** | both tasks | w_seg = 1, w_cls = 1 | multi-task, equal weights |
| **D** | both tasks | learned (uncertainty weighting, Kendall et al. 2018) | multi-task, weights tuned automatically |

**Controls in the comparison:** all four use the same network, data split, augmentation policy, learning rate,
epochs and seed labels (0, 1, 2). A and B switch one loss off. We report mean ± standard deviation over three runs.
The original augmentations were not seeded, so this is not a comparison with identical random image transforms.
Learning-rate tuning was performed on C only; this is another limitation of the comparison.

## 5. Main result

> **Answer in this experiment:** learning both tasks together gave **higher tumor-type classification scores**
> (Macro-F1 0.916 → 0.938, higher for all 3 matched seed labels) and **similar mean segmentation scores**
> (Dice 0.774 vs 0.775). Equal loss weights (C) had higher mean scores than learned weights (D).

Figure: `figures/1_main_comparison.png` · Table: `summary_test.csv` · Every run: `all_runs_test.csv`

**Test set, mean ± standard deviation over 3 runs** (436 slices from 29 held-out patient groups):

| Experiment | Accuracy | Macro-Precision | Macro-Recall | **Macro-F1** | **Dice** | IoU |
|---|---|---|---|---|---|---|
| A: Cls-only | 0.932 ± 0.008 | 0.931 ± 0.010 | 0.911 ± 0.011 | 0.916 ± 0.010 | – | – |
| B: Seg-only | – | – | – | – | 0.774 ± 0.006 | **0.681 ± 0.007** |
| **C: MTL equal** | **0.952 ± 0.002** | **0.953 ± 0.005** | **0.932 ± 0.003** | **0.938 ± 0.003** | **0.775 ± 0.008** | 0.679 ± 0.009 |
| D: MTL uncertainty | 0.940 ± 0.009 | 0.942 ± 0.008 | 0.917 ± 0.012 | 0.923 ± 0.012 | 0.769 ± 0.006 | 0.673 ± 0.005 |

**Run-to-run comparison:** `paired_differences.csv` matches runs by their seed labels. These are descriptive
differences; the original runs did not share a controlled augmentation sequence. They do not constitute a
significance or equivalence test:

| Comparison | Seed 0 | Seed 1 | Seed 2 | Mean | Better in |
|---|---|---|---|---|---|
| C − A (Macro-F1) | +0.012 | +0.018 | +0.036 | **+0.022** | **3 / 3 seeds** |
| D − A (Macro-F1) | +0.010 | +0.006 | +0.007 | +0.007 | 3 / 3 seeds |
| C − B (Dice) | −0.013 | +0.008 | +0.007 | +0.001 | 2 / 3 seeds |
| D − B (Dice) | −0.015 | +0.008 | −0.009 | −0.006 | 1 / 3 seeds |

- Classification: the gain of C (+0.022) is about 2× the run-to-run std of A and positive in every seed → a consistent improvement.
- Segmentation: the mean Dice scores are close and differences change sign between runs. This does not prove equivalence or absence of harm.
- Only 3 runs per experiment and 29 held-out groups were evaluated; the uncertainty across different patient samples is not measured by the seed standard deviation.

## 6. Classification results

Figure: `figures/3_confusion_matrices.png` (summed over 3 seeds)

| F1 per class | Meningioma | Glioma | Pituitary |
|---|---|---|---|
| A: Cls-only | 0.851 ± 0.025 | 0.974 ± 0.008 | 0.923 ± 0.003 |
| C: MTL equal | **0.889 ± 0.006** | **0.994 ± 0.001** | **0.931 ± 0.006** |
| D: MTL uncertainty | 0.863 ± 0.022 | 0.990 ± 0.007 | 0.918 ± 0.014 |

- **Meningioma is the hardest class** in every model. Its main confusion is with pituitary tumors (see Error analysis).
- Multi-task removed most of the *other* mistakes: meningioma → glioma dropped from 17 to 5, and glioma errors from 15 to 2 (3 seeds pooled).
- **Why can segmentation help classification?** The mask loss tells the shared encoder *where the tumor is* at every pixel.
  This extra supervision pushes the features to describe the tumor itself instead of unrelated image content.
  (This is our explanation; we did not verify it with attention maps — see Future work.)

## 7. Segmentation results

Figures: `figures/4_dice_by_tumor_type.png`, `figures/5_dice_vs_tumor_size.png`,
`figures/7_C_mtl_equal_seed0_seg_best_worst.png` (top row: best 4, bottom row: worst 4; green = expert, red = model)

| Dice per tumor type | Meningioma | Glioma | Pituitary |
|---|---|---|---|
| B: Seg-only | 0.930 ± 0.004 | 0.662 ± 0.015 | 0.829 ± 0.000 |
| C: MTL equal | 0.937 ± 0.001 | 0.663 ± 0.015 | 0.825 ± 0.007 |
| D: MTL uncertainty | 0.932 ± 0.004 | 0.652 ± 0.012 | 0.825 ± 0.007 |

- **Meningiomas have the highest mean overlap here** (Dice ≈ 0.93). Several examples have clear, rounded outlines.
- **Gliomas have the lowest mean overlap here** (Dice ≈ 0.66). Irregular outlines and low contrast in the examples are possible explanations, not clinically verified causes.
- The **median** Dice (0.865 for C) is much higher than the mean (0.775): most slices are segmented well, and a few
  complete failures pull the mean down. About 4 % of slice/run predictions have Dice = 0 (no overlap),
  and **almost all of them are gliomas** (45 of 51 predictions for C, pooled over seeds).
- Tumor size matters only a little: Dice ≈ 0.70–0.74 for the smallest tumors (< 1,000 px) vs ≈ 0.75–0.82 for larger ones.
- **Possible explanation for the similar Dice scores:** dense mask supervision may leave less benefit from an
  additional image-level label. We did not isolate this mechanism in an experiment.

## 8. Error analysis

Figures: `figures/7_C_mtl_equal_seed0_misclassified.png` · Tables: `errors_by_patient.csv`, `error_analysis.json`

**1. One patient ID accounts for most classification errors.** ID 103673 contributes 54 meningioma → pituitary
error predictions (18 slices × 3 seeds) in each of A, C and D. All three classifiers get these slices wrong in
every seed. That is 61 % of A's errors and
86 % of C's errors. The images suggest a location/appearance resemblance to examples labelled pituitary.
This is a visual hypothesis: we have no clinical annotation confirming an exact sellar location, and no clinician
review establishing why the classifier failed.

| Errors (3 seeds pooled) | All test patients | Without patient 103673 | Accuracy without 103673 |
|---|---|---|---|
| A: Cls-only | 89 | 35 | 97.2 % |
| C: MTL equal | 63 | **9** | **99.3 %** |
| D: MTL uncertainty | 78 | 24 | 98.1 % |

→ Multi-task learning removed **about 3/4 of the remaining errors**, but it could not fix the look-alike case.

The exclusion above is exploratory error analysis, not the primary evaluation. Keep patient 103673 in every
headline score; the same test slices are counted once per seed in this pooled table.

**2. Misclassified slices are segmented well.** For C, misclassified slices have mean Dice **0.92** vs 0.77 for correctly
classified ones. This shows that many classification errors can coexist with good mask overlap. It does not
establish the cause of the errors or prove where the classification head attends.

**3. Segmentation failures are gliomas.** The worst cases (Dice = 0) are gliomas where the model outlines a different
bright structure or misses a faint, irregular tumor (bottom row of the best/worst figure).

**4. Learned weights (D) stayed similar.** `figures/6_uncertainty_weights.png`: both learned weights rose
from 1 to ≈ 1.9 and stayed close. D had lower mean scores than C in these runs. We did not isolate why;
larger loss weights alone do not establish a larger optimizer step or explain the difference.

## 9. Limitations

1. **Small test set:** 29 patient groups. A single patient ID (103673) changes accuracy by about 4 percentage points. A 5-fold
   cross-validation by patient would give more reliable numbers.
2. **Only 3 seeds and no formal statistical test.** The classification gain is consistent across seeds but not proven.
3. **Learning rate was tuned on the multi-task model (C) only** and reused for A, B and D. This could favour C slightly;
   tuning each experiment separately would be fairer.
4. **One dataset from two hospitals, one MRI sequence (T1 contrast).** We did not perform an external or hospital-held-out evaluation.
5. **2D slices, each predicted alone.** A doctor looks at the whole 3D scan; we do not combine slices of the same patient.
6. **Every slice contains a tumor.** Classification must choose one of the three tumor types; healthy-brain rejection was not evaluated.
7. The validation set is small; validation Macro-F1 reaches ≈ 0.99, so choosing the best epoch is somewhat noisy.
8. **Original augmentation randomness was not seeded.** The three seed labels do not fully control the training input sequence. The corrected code affects new runs only.
9. **Patient identity behind suffixed IDs is unverified.** The 209 groups are a conservative splitting rule, not a confirmed revised patient count.

**Future work:** patient-level cross-validation; combine slices into a patient-level decision; 3D models; attention maps
(Grad-CAM) to check where the classifier looks; test on another dataset (e.g. BraTS for gliomas); add location features
to study whether location information helps separate the confusing cases, with clinical review.

## 10. Conclusion

- One model with a shared encoder can **segment brain tumors (Dice 0.775) and classify their type (Macro-F1 0.938, accuracy 95.2 %)** at the same time.
- **Multi-task had higher classification scores in these runs**, while mean segmentation Dice was similar. This is not proof of a universal improvement or equivalent segmentation performance.
- Simple equal loss weights worked better than learned uncertainty weights here.
- Classification errors are concentrated in **one patient ID**; segmentation failures are mainly **glioma predictions**. Their clinical causes remain hypotheses.

## 11. Questions we may be asked (with short answers)

| Question | Answer |
|---|---|
| Why is this multi-task learning? | One network, one shared encoder, two outputs trained together with one combined loss; we report each task's metric separately. |
| Why split by patient? | Slices from one patient look almost the same; if they are in train and test, the test score is too optimistic (leakage). |
| What grouping issue did you find? | Eighteen groups of glioma IDs had suffix variants; 14 groups crossed the initial split. We grouped them conservatively (233 IDs → 209 groups); shared identity is unverified. |
| Why Macro-F1 and not accuracy? | Classes are unbalanced (glioma ≈ 2× meningioma). Macro-F1 averages F1 over classes so each type counts equally. |
| What is Dice? | Overlap between predicted and expert mask: 2 × overlap area ÷ (predicted area + true area). 1 = perfect, 0 = no overlap. |
| Why BCE + Dice loss? | BCE learns each pixel; Dice directly optimises overlap and handles the small tumor area vs large background. |
| How was the comparison controlled? | Same architecture, split, augmentation policy, lr and epochs. Original augmentation draws were not seeded, and lr was tuned on C only; both are limitations. |
| How did you choose hyper-parameters? | lr from {1e-4, 3e-4, 1e-3} on validation only (3e-4 won). Test results were not used to choose settings; inference was later repeated for verification. |
| What is uncertainty weighting? | Each task gets a learnable weight exp(−s) plus a penalty s, so the model can balance tasks itself (Kendall et al. 2018). |
| Why did D not beat C? | Its weights stayed close and its mean scores were lower. We did not run an experiment isolating the cause. |
| Is +0.022 F1 significant? | It is positive for all three matched seed labels, but we have no formal significance test or patient-level confidence interval. |
| Why does the model confuse meningioma with pituitary? | Most errors come from one patient ID. Location and appearance resemblance is a possible explanation, not a clinically confirmed cause. |
| Why is glioma Dice low? | The examples suggest irregular borders and low contrast may contribute; we did not verify the mechanism or measure expert disagreement. |
| Could this be used in a hospital? | Not yet: one dataset, 2D slices, no healthy scans, small test set. It is a research prototype. |
| What would you do next? | Patient-level cross-validation, 3D / multi-slice models, Grad-CAM, external dataset. |

## 12. Re-run the protocol

For Windows setup and the tested package snapshot, see [README.md](../README.md). The following commands
assume an active Python environment in Bash. They use new folders so archived runs stay intact.
The seeding fix means new numbers may differ from the published original runs; keep their reports separate.

```bash
python src/prepare_data.py --raw_dir data/raw/mat --out_dir data/processed   # uses the committed splits/splits.csv
bash scripts/tune_lr.sh --out_dir runs/reproduced/tuning --workers 4           # validation-only lr choice
# Choose the lr from your new validation runs; 3e-4 is the original value below
bash scripts/run_experiments.sh --out_dir runs/reproduced/final --lr 3e-4 --epochs 30 --workers 4
python src/analyze.py --runs_dir runs/reproduced/final --out_dir runs/reproduced/results
```
