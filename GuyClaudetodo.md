# Guy × Claude — progress & todo

Project: Brain Tumor Segmentation + Tumor-Type Classification (Multi-task Learning, aims at the 4-point difficulty level).
Team plan: [TASKS.md](TASKS.md) · How to run: [README.md](README.md)

## Done

- [x] **2026-10-09** Reviewed the friend's proposal against the rubric → topic accepted (MTL = 4 pts if both tasks are trained jointly and reported separately)
- [x] **2026-10-09** Created GitHub repo `G9YzNDN/DLMED` (public) and pushed the project scaffold
  - data prep (.mat → PNG, patient-level split), shared-encoder model, training with `--tasks cls/seg/both`,
    loss weighting (fixed / uncertainty), test evaluation + figures, summary table, Colab notebook
- [x] **2026-10-09** Smoke-tested the whole pipeline (prepare → train A–D → evaluate → summarize) on fake `.mat` data on CPU
- [x] **2026-10-09** Wrote team roles for 6 people in TASKS.md
- [x] **2026-10-09** Decided final experiments run on my laptop (RTX 5070 Laptop, 8 GB); Colab/Kaggle only for trying things out
- [x] **2026-10-09** Local GPU environment set up at `C:\Users\guymy\.venvs\dlmed` (PyTorch 2.11 + CUDA 12.8); GPU check passed
  - Benchmark: ~90 ms/step (batch 16, 256×256), ~1.4 GB GPU memory → about 12 s/epoch of training on the real data,
    so roughly 10 min per 30-epoch run and ~2 h for all 12 final runs (estimate; first real run will confirm)
  - Fixed slow epochs on Windows (DataLoader workers now persistent)
- [x] **2026-10-09** Downloaded the Figshare dataset (4 zips, ~880 MB, checksums verified) into `DLMED/data/raw/` (git-ignored)
- [x] **2026-10-09** Ran `prepare_data.py` on the real data: 3,064 images, 233 patient IDs; masks checked visually (`data/sample_overlays.png`)
- [x] **2026-10-09** Found patient leakage: 18 glioma IDs with letter suffixes (MR040240, MR040240B, ...) = same person,
  14 of them were spread across splits → merged them (209 patients) and re-split; committed `splits/splits.csv`
  - train 151 / val 29 / test 29 patients (2,194 / 434 / 436 images), no patient in two splits

## Todo

### Now
- [ ] Invite the 5 teammates as collaborators (repo → Settings → Collaborators)
- [ ] Send the repo link + TASKS.md to the team; confirm who is Person 1–6 (my role: ______)
- [ ] Create the shared Google Drive folder `brain_tumor/` (`raw/`, `runs/`) and share it

### Data (with Person 1)
- [ ] Tell Person 1 the data + split are done; they own the data slides (class table, split, leakage finding)
- [ ] Upload the 4 zips to Drive `brain_tumor/raw/` and/or `processed.zip` for teammates on Colab/Kaggle

### Training on my laptop
- [ ] First real sanity run: `--tasks both --epochs 2`, check speed per epoch and GPU memory
- [ ] Person 4 locks settings (lr, epochs) on validation
- [ ] Run final experiments A–D × seeds 0, 1, 2 (`scripts/run_experiments.sh`)
- [ ] Copy `runs/` to Drive for Person 5

### Wrap-up
- [ ] Results table + error analysis (Person 5)
- [ ] Slides, README final check, rehearsal (Person 6)

## Notes

- Python env: `C:/Users/guymy/.venvs/dlmed/Scripts/python` (outside OneDrive on purpose)
- Data / runs: `DLMED/data/` and `DLMED/runs/` (git-ignored; inside OneDrive, so they sync to the cloud)
- Do not look at test results before settings are locked on validation.
