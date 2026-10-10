# Codex — project fixes and submission checklist

Updated: **2026-10-10** (Asia/Bangkok).
Team plan: [TASKS.md](TASKS.md) · Results: [results/RESULTS.md](results/RESULTS.md) · Run instructions: [README.md](README.md).

## What is finished

- [x] Original data preparation, four experiments A–D × three seeds, result tables and ten figures.
- [x] Reviewed all 3,064 raw/processed images and masks; no recorded patient ID or conservative group crosses splits.
- [x] Re-ran inference from all 12 original checkpoints on all 436 test images. Predictions match the saved reports.
- [x] Checked saved predictions against metrics, per-run scores, mean/std tables and descriptive per-seed differences.

## Fixes from the review

- [x] **Protect finished runs:** training refuses an existing run folder before writing any files. Use a new name or output folder for trials.
- [x] **Control augmentation randomness:** seed Albumentations, the training sampler and each data-loading worker. New runs save their software versions in `environment.txt`.
- [x] **Record tested versions:** `requirements-lock.txt` captures the review environment (Windows, Python 3.14.7, CUDA 12.8). It is not a historical snapshot saved during original training.
- [x] **Fix Windows instructions:** README uses the same environment's Python for every command and explains how Bash finds it.
- [x] **Keep quoted paths intact:** experiment scripts preserve arguments containing spaces; tuning summaries use the requested output folder.
- [x] **Update Colab trial:** install project requirements, use a separate trial name, and evaluate the short trial on validation.
- [x] **Correct data claims:** two acquisition hospitals; 233 recorded IDs → 209 conservative groups. Shared identity behind suffix variants remains unverified.
- [x] **Correct conclusions:** similar mean Dice is not proof of equivalence; clinical explanations and reasons D underperforms C remain hypotheses.
- [x] **Update team plan and older log:** keep the six review topics, shared slide work and rehearsal; explain the original-run limitation throughout.
- [x] **Keep the latest team additions:** preserve the classification study materials, align their rounding and limitations, and record **GUY** as responsible for **Topic 4 — Multi-task modelling and experiments** in TASKS.md.

## Verification of these fixes

- [x] Three regression tests passed: same-seed augmentation, repeated two-worker loading across two epochs, and refusal to overwrite a saved run.
- [x] Two-epoch multi-task uncertainty-weighted GPU check on real data (2,194 train / 434 validation images, two workers, no pretrained weights), followed by validation evaluation. Outputs: `runs/review_checks/gpu_fix_check_seed0/` (git-ignored). This is a pipeline check, not a new final experiment.
- [x] SHA-256 comparison of 104 original experiment/result/split files passed; archived checkpoints, predictions, histories, tables, figures and split CSVs are unchanged.
- [x] Python syntax, notebook structure, all three README PowerShell blocks, Bash syntax and paths containing spaces passed. No original score was replaced by the short GPU check.
- [x] Regenerated classification reports with their built-in input checks; classification numeric tables stayed unchanged. Verified report hashes, local document links and GUY's topic assignment.
- [x] Commit and push the reviewed fixes to GitHub.

## Important distinction for the presentation

The published tables/figures are from the **original 12 runs**, before augmentation seeding was corrected.
They are verified observations, but the three seed labels do not guarantee identical augmentation draws or an
exact training replay. The fix applies to new runs only; a new full experiment must have its own output folder
and results report. Do not present the original scores as a rerun of the corrected protocol.

Full retraining of all 12 experiments is not part of this review-fix pass. Colab and another teammate's machine
have not been tested. Repeatability depends on library versions, worker count and hardware; equality across
different GPUs is not guaranteed.

## Team still needs to do before submission

- [x] GUY chose Topic 4 — Multi-task modelling and experiments.
- [ ] Remaining members choose their topics and fill in their names in TASKS.md.
- [ ] Read the corrected results and practise explaining the model, split, metrics and limitations.
- [ ] Create the slides together, with sources and readable figures. Label train/validation/test correctly.
- [ ] Include both task outputs and metrics, A–D comparisons, correct/wrong examples and limitations.
- [ ] Check the code/README and slide files required by the instructor are included in the submission.
- [ ] Test any live demo on the presentation laptop using a new run folder.
- [ ] Rehearse twice within the instructor's time limit and practise Q&A.

The technical evidence supports the four technical rubric categories (10 points total). Slides (2 points) and
presentation/Q&A (3 points) still depend on the team's finished work; this checklist does not award a score.
