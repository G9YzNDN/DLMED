# Guy × Claude — progress & todo

Project: Brain Tumor Segmentation + Tumor-Type Classification (Multi-task Learning, 4-point difficulty level).
Team plan: [TASKS.md](TASKS.md) · Results: [results/RESULTS.md](results/RESULTS.md) · How to run: [README.md](README.md)

## Status: project finished (2026-10-09) — team review and slides next

**Headline:** multi-task learning improved classification (Macro-F1 0.916 → 0.938, better in 3/3 seeds) and kept
segmentation equal (Dice 0.774 vs 0.775). Equal loss weights beat learned (uncertainty) weights.

## Done

- [x] **2026-10-09** Reviewed the friend's proposal against the rubric → topic accepted
- [x] **2026-10-09** Created GitHub repo `G9YzNDN/DLMED` (public) and the project code
  (data prep, shared-encoder model, training `--tasks cls/seg/both`, fixed/uncertainty loss weighting, evaluation, Colab notebook)
- [x] **2026-10-09** Smoke-tested the pipeline on fake data; set up the local GPU environment (RTX 5070 Laptop, PyTorch 2.11 + CUDA 12.8)
- [x] **2026-10-09** Downloaded the Figshare dataset (checksums verified) → `DLMED/data/` (git-ignored); 3,064 images, masks checked visually
- [x] **2026-10-09** Found and fixed patient leakage (suffixed glioma IDs = same person) → 209 patients, split 151 / 29 / 29
- [x] **2026-10-09** Learning rate chosen on validation only (1e-4 / **3e-4** / 1e-3) → `results/tuning_lr_validation.csv`
- [x] **2026-10-09** Final experiments A–D × 3 seeds (12 runs, ≈ 7 min each) + test evaluation
- [x] **2026-10-09** Analysis: summary table, per-seed comparison, per-class / per-type / per-size results,
  error analysis by patient (one sellar meningioma patient = most classification errors), learned weights
- [x] **2026-10-09** 10 slide-ready figures in `results/figures/` (architecture, dataset, main result, curves, confusion matrices, Dice by type/size, weights, examples)
- [x] **2026-10-09** Wrote `results/RESULTS.md` (findings, limitations, Q&A answers) and rewrote `TASKS.md` as a review + slides plan

## Todo (me)

- [ ] Invite the 5 teammates as collaborators (repo → Settings → Collaborators) — only needed if they will push
- [ ] Send the team the repo link and say: read `results/RESULTS.md`, then follow `TASKS.md`
- [ ] Confirm who is Person 1–6 (my role: ______)
- [ ] Optional: share `data/raw/zips/` on Google Drive if a teammate wants to re-run on Colab

## Todo (team — see TASKS.md)

- [ ] Everyone reads RESULTS.md and reviews their part
- [ ] Person 6 sends the slide template; everyone makes their slides
- [ ] Merge slides, rehearse twice with a timer, practise the Q&A list

## Notes

- Python env: `C:/Users/guymy/.venvs/dlmed/Scripts/python`
- Data: `DLMED/data/` · trained runs + checkpoints: `DLMED/runs/` (both git-ignored, stay on this laptop)
- Rebuild results after any change: `python src/analyze.py --runs_dir runs --out_dir results`
