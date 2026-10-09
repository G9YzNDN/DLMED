# Results — Brain Tumor Segmentation + Classification with Multi-task Learning

Plain-language summary of the whole project for the team. Every number here comes from the files in this folder.
Figures for the slides are in [`figures/`](figures/).

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
| Source | Figshare *brain tumor dataset* (Cheng et al., PLoS ONE 2015), CC BY 4.0 |
| Images | 3,064 T1-weighted contrast-enhanced MRI slices (3,049 at 512×512, 15 at 256×256) |
| Labels | expert tumor mask + tumor type for every slice |
| Classes | meningioma 708 · glioma 1,426 · pituitary 930 slices |

Figure: `figures/0_dataset_examples.png`

**Preparation:** `.mat` files → images scaled to 0–255 + binary masks (`src/prepare_data.py`); resized to 256×256 for training.

**Split by patient, not by image.** One patient gives many very similar slices. If slices of the same patient were in
both train and test, the test score would be too optimistic (data leakage). We split by patient
(`StratifiedGroupKFold`: grouped by patient, balanced by tumor type).

**Leakage we found and fixed.** The dataset lists 233 patient IDs, but 18 glioma IDs come with letter suffixes
(e.g. `MR040240`, `MR040240B`, `MR040240C`, `MR040240D`) — these look like repeat scans of the same person.
Using the raw IDs, 14 of these people ended up in more than one split. We merged suffixed IDs into one patient:

| Split | Patients | Images | Meningioma | Glioma | Pituitary |
|---|---|---|---|---|---|
| Train | 151 | 2,194 | 505 | 1,019 | 670 |
| Validation | 29 | 434 | 102 | 203 | 129 |
| Test | 29 | 436 | 101 | 204 | 131 |

No patient appears in two splits (checked automatically). The test set was used **only once, at the very end**.

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

**Why the comparison is fair:** all four use the *same* network, data split, augmentation, learning rate, epochs and
seeds (0, 1, 2). A and B are the same network with one loss switched off. Each experiment is run with 3 seeds and we
report mean ± standard deviation, so we can see whether differences are larger than run-to-run noise.

## 5. Main result

> **Answer to our question:** learning both tasks together **improves tumor-type classification**
> (Macro-F1 0.916 → 0.938, better in all 3 seeds) and **keeps segmentation the same** (Dice 0.774 vs 0.775).
> Equal loss weights (C) worked better than learned weights (D).

Figure: `figures/1_main_comparison.png` · Table: `summary_test.csv` · Every run: `all_runs_test.csv`

**Test set, mean ± standard deviation over 3 seeds** (436 slices from 29 unseen patients):

| Experiment | Accuracy | Macro-Precision | Macro-Recall | **Macro-F1** | **Dice** | IoU |
|---|---|---|---|---|---|---|
| A: Cls-only | 0.932 ± 0.008 | 0.931 ± 0.010 | 0.911 ± 0.011 | 0.916 ± 0.010 | – | – |
| B: Seg-only | – | – | – | – | 0.774 ± 0.006 | **0.681 ± 0.007** |
| **C: MTL equal** | **0.952 ± 0.002** | **0.953 ± 0.005** | **0.932 ± 0.003** | **0.938 ± 0.003** | **0.775 ± 0.008** | 0.680 ± 0.009 |
| D: MTL uncertainty | 0.940 ± 0.009 | 0.942 ± 0.009 | 0.917 ± 0.012 | 0.924 ± 0.012 | 0.769 ± 0.006 | 0.673 ± 0.005 |

**Is the difference real or noise?** Same seed = same starting weights and data order, so we compare seed by seed
(`paired_differences.csv`):

| Comparison | Seed 0 | Seed 1 | Seed 2 | Mean | Better in |
|---|---|---|---|---|---|
| C − A (Macro-F1) | +0.012 | +0.018 | +0.036 | **+0.022** | **3 / 3 seeds** |
| D − A (Macro-F1) | +0.010 | +0.006 | +0.007 | +0.007 | 3 / 3 seeds |
| C − B (Dice) | −0.013 | +0.008 | +0.007 | +0.001 | 2 / 3 seeds |
| D − B (Dice) | −0.015 | +0.008 | −0.009 | −0.006 | 1 / 3 seeds |

- Classification: the gain of C (+0.022) is about 2× the run-to-run std of A and positive in every seed → a consistent improvement.
- Segmentation: differences are smaller than the std and change sign between seeds → **no real difference**.
- With only 3 seeds and 29 test patients this is evidence, not statistical proof (see Limitations).

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

- **Meningiomas are segmented almost perfectly** (Dice ≈ 0.93): they are round, bright and have clear borders.
- **Gliomas are hardest** (Dice ≈ 0.66): irregular shape, unclear borders that grow into normal brain, uneven brightness.
- The **median** Dice (0.865 for C) is much higher than the mean (0.775): most slices are segmented well, and a few
  complete failures pull the mean down. About 4 % of slices have Dice = 0 (tumor completely missed or wrong region),
  and **almost all of them are gliomas** (45 of 51 for C).
- Tumor size matters only a little: Dice ≈ 0.70–0.74 for the smallest tumors (< 1,000 px) vs ≈ 0.75–0.82 for larger ones.
- **Why no gain for segmentation?** Segmentation already gets a label for every pixel; one class label per image adds
  very little new information to it.

## 8. Error analysis

Figures: `figures/7_C_mtl_equal_seed0_misclassified.png` · Tables: `errors_by_patient.csv`, `error_analysis.json`

**1. One patient causes most classification errors.** All 54 meningioma → pituitary errors (18 slices × 3 seeds) come from
**one test patient (ID 103673)**, and *every* model gets this patient wrong in *every* seed. That is 61 % of A's errors and
86 % of C's errors. The images show why: this meningioma sits **at the sella turcica, exactly where pituitary tumors grow**,
and looks round and bright like a pituitary tumor. Radiologists also have to work to tell these two apart.

| Errors (3 seeds pooled) | All test patients | Without patient 103673 | Accuracy without 103673 |
|---|---|---|---|
| A: Cls-only | 89 | 35 | 97.2 % |
| C: MTL equal | 63 | **9** | **99.3 %** |
| D: MTL uncertainty | 78 | 24 | 98.1 % |

→ Multi-task learning removed **about 3/4 of the remaining errors**, but it could not fix the look-alike case.

**2. Misclassified slices are segmented well.** For C, misclassified slices have mean Dice **0.92** vs 0.77 for correctly
classified ones. The model finds the tumor; it just names it wrong. So classification errors are *not* caused by
failing to locate the tumor — they come from tumors whose location and appearance look like another type.

**3. Segmentation failures are gliomas.** The worst cases (Dice = 0) are gliomas where the model outlines a different
bright structure or misses a faint, irregular tumor (bottom row of the best/worst figure).

**4. Learned weights (D) did not rebalance the tasks.** `figures/6_uncertainty_weights.png`: both learned weights rose
from 1 to ≈ 1.9 and stayed almost equal. D therefore behaved like C with a larger effective step size, which fits its
slightly lower and less stable results.

## 9. Limitations

1. **Small test set:** 29 patients. A single patient (103673) changes accuracy by about 4 percentage points. A 5-fold
   cross-validation by patient would give more reliable numbers.
2. **Only 3 seeds and no formal statistical test.** The classification gain is consistent across seeds but not proven.
3. **Learning rate was tuned on the multi-task model (C) only** and reused for A, B and D. This could favour C slightly;
   tuning each experiment separately would be fairer.
4. **One dataset, one hospital source, one MRI sequence (T1 contrast).** We do not know how well it works on other scanners.
5. **2D slices, each predicted alone.** A doctor looks at the whole 3D scan; we do not combine slices of the same patient.
6. **Every slice contains a tumor.** The model was never shown healthy brains, so it always predicts a tumor.
7. The validation set is small; validation Macro-F1 reaches ≈ 0.99, so choosing the best epoch is somewhat noisy.

**Future work:** patient-level cross-validation; combine slices into a patient-level decision; 3D models; attention maps
(Grad-CAM) to check where the classifier looks; test on another dataset (e.g. BraTS for gliomas); add location features
to separate sellar meningiomas from pituitary tumors.

## 10. Conclusion

- One model with a shared encoder can **segment brain tumors (Dice 0.775) and classify their type (Macro-F1 0.938, accuracy 95.2 %)** at the same time.
- **Multi-task learning helped classification** consistently and **did not hurt segmentation**.
- Simple equal loss weights worked better than learned uncertainty weights here.
- The remaining errors are mostly **one look-alike patient** (a meningioma at the pituitary location) and **hard gliomas**.

## 11. Questions we may be asked (with short answers)

| Question | Answer |
|---|---|
| Why is this multi-task learning? | One network, one shared encoder, two outputs trained together with one combined loss; we report each task's metric separately. |
| Why split by patient? | Slices from one patient look almost the same; if they are in train and test, the test score is too optimistic (leakage). |
| What leakage did you find? | 18 glioma patient IDs had letter suffixes (repeat scans). 14 were split across sets; we merged them (233 → 209 patients). |
| Why Macro-F1 and not accuracy? | Classes are unbalanced (glioma ≈ 2× meningioma). Macro-F1 averages F1 over classes so each type counts equally. |
| What is Dice? | Overlap between predicted and expert mask: 2 × overlap area ÷ (predicted area + true area). 1 = perfect, 0 = no overlap. |
| Why BCE + Dice loss? | BCE learns each pixel; Dice directly optimises overlap and handles the small tumor area vs large background. |
| How is the comparison fair? | Same network, data split, augmentation, lr, epochs and 3 seeds; A and B only switch one loss off. |
| How did you choose hyper-parameters? | lr from {1e-4, 3e-4, 1e-3} on the validation set only (3e-4 won). Test set used once at the end. |
| What is uncertainty weighting? | Each task gets a learnable weight exp(−s) plus a penalty s, so the model can balance tasks itself (Kendall et al. 2018). |
| Why did D not beat C? | Its learned weights stayed nearly equal (≈ 1.9 each), so it did not rebalance; it only made steps larger and results noisier. |
| Is +0.022 F1 significant? | Positive in all 3 seeds and ≈ 2× the std, so consistent; but 3 seeds / 29 patients is not a formal proof. |
| Why does the model confuse meningioma with pituitary? | One patient's meningioma lies at the sella, where pituitary tumors are; location and look are similar. |
| Why is glioma Dice low? | Gliomas have irregular, unclear borders and uneven contrast; even experts disagree on their edges. |
| Could this be used in a hospital? | Not yet: one dataset, 2D slices, no healthy scans, small test set. It is a research prototype. |
| What would you do next? | Patient-level cross-validation, 3D / multi-slice models, Grad-CAM, external dataset. |

## 12. Reproduce

```bash
python src/prepare_data.py --raw_dir data/raw/mat --out_dir data/processed   # uses the committed splits/splits.csv
bash scripts/tune_lr.sh                                                      # validation-only lr choice
bash scripts/run_experiments.sh --lr 3e-4 --epochs 30                        # A–D × 3 seeds + test evaluation
python src/analyze.py --runs_dir runs --out_dir results                      # this folder
```
