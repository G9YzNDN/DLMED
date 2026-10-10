# Classification method — study notes

## What is predicted?

Input: one normalised, grayscale T1-weighted contrast-enhanced MRI slice, resized to 256 × 256.

Output: one of three tumor types. The original dataset labels are converted to zero-based labels in [prepare_data.py](../src/prepare_data.py):

| Original label | Training label | Class |
|---|---|---|
| 1 | 0 | Meningioma |
| 2 | 1 | Glioma |
| 3 | 2 | Pituitary tumor |

Every sample in this dataset has a tumor label. There is **no healthy / no-tumor class**; this classifier cannot establish whether an arbitrary scan contains a tumor. Predictions are made per slice, not per patient.

## How the head works

The implementation uses `smp.Unet` with `aux_params={"classes": 3, "dropout": 0.2}` in [model.py](../src/model.py). The classification head reads the deepest features of the shared ResNet-34 encoder:

```text
MRI slice → shared encoder → deepest feature map
          → global average pooling → flatten → dropout (0.2)
          → linear layer → three logits
```

- The encoder learns image features and is initialised with ImageNet-pretrained weights.
- Global average pooling averages each feature channel across its spatial positions, producing a feature vector.
- Dropout is active during training to reduce reliance on individual features; it is disabled during evaluation.
- The linear layer produces three unnormalised scores (logits).
- At evaluation, [engine.py](../src/engine.py) applies softmax to obtain probabilities, then chooses the highest-probability class with argmax. Those probabilities have not been evaluated for calibration.

The tumor mask is a **training target for the segmentation branch**, not an extra input to the classifier. At inference the model receives the MRI image. Both heads use shared features; the classification head does not receive the predicted mask as its input.

## Loss and joint learning

[losses.py](../src/losses.py) uses `cross_entropy(logits, labels)` for classification. Conceptually, the per-image loss is `−log(probability assigned to the correct class)`, averaged over the batch. Pass logits directly to this loss; a separate softmax before it is unnecessary.

| Experiment | Training objective | Role |
|---|---|---|
| A | Classification cross-entropy only | Baseline |
| C | Segmentation BCE + Dice, plus classification cross-entropy; both weights 1 | Joint learning |
| D | Same two losses with learned uncertainty weights | Alternative joint weighting |

Both losses update the shared encoder in C and D. Segmentation supervision may encourage features useful for classification, but the repository does not establish that mechanism experimentally. Treat it as a possible explanation.

B trains segmentation only, so its untrained classification output is not included in the classification comparison.

## Metrics

For a particular class, treat that class as positive and the other classes as negative:

| Metric | Formula | Meaning |
|---|---|---|
| Precision | `TP / (TP + FP)` | Among predictions of this class, how many are correct? |
| Recall | `TP / (TP + FN)` | Among true samples of this class, how many are found? |
| F1 | `2 × precision × recall / (precision + recall)` | Balance of precision and recall |
| Macro-F1 | Mean of the three per-class F1 values | Gives equal weight to each class |
| Accuracy | Correct predictions / all predictions | Gives equal weight to each slice |

[metrics.py](../src/metrics.py) computes these with scikit-learn and uses `zero_division=0`. Macro precision and macro recall are averaged separately; **Macro-F1 is not generally their harmonic mean**. Glioma has more slices than the other classes, so accuracy alone can hide weaker performance on a smaller class.

Confusion matrix: row = true class; column = predicted class; diagonal = correct predictions; off-diagonal = errors. Read counts together with class support; unequal row totals affect raw counts.

## Evaluation and comparison

Training uses only train data. [train.py](../src/train.py) selects the checkpoint using validation Macro-F1 for A, and the average of validation Macro-F1 and Dice for C/D. Test evaluation happens separately in [evaluate.py](../src/evaluate.py).

The committed split has 436 test slices from 29 patient groups. Suffix-based grouping is a conservative repository assumption about related IDs; absence of cross-split group overlap does not independently prove patient identities.

Experiments share an architecture, split, settings and seed labels. These controls support comparison, but the learning rate was tuned on C only and the checkpoint-selection criteria differ by task. Matching seed labels does not independently verify identical augmentation or GPU computations. Three seeds estimate training variability on one fixed split; they do not measure generalisation across new patient splits.

The original 12 runs did not seed Albumentations' internal random generator. Their checkpoints and scores are preserved; the corrected training code seeds augmentation, workers and the sampler for new runs only. Differences by recorded seed label are descriptive comparisons, not matched augmentation trials or an exact replay of the corrected protocol.
