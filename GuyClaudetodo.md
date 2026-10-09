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

## Todo

### Now
- [ ] Invite the 5 teammates as collaborators (repo → Settings → Collaborators)
- [ ] Send the repo link + TASKS.md to the team; confirm who is Person 1–6 (my role: ______)
- [ ] Create the shared Google Drive folder `brain_tumor/` (`raw/`, `runs/`) and share it

### Data (with Person 1)
- [ ] Download the 4 Figshare zip files → `C:/brain_tumor/raw/` (and Drive `brain_tumor/raw/`)
- [ ] Run `prepare_data.py` on the real data, check 3,064 images / 233 patients, commit `splits/splits.csv`
- [ ] Upload `processed.zip` to Drive for teammates on Colab/Kaggle

### Training on my laptop
- [ ] First real sanity run: `--tasks both --epochs 2`, check speed per epoch and GPU memory
- [ ] Person 4 locks settings (lr, epochs) on validation
- [ ] Run final experiments A–D × seeds 0, 1, 2 (`scripts/run_experiments.sh` with the `C:/brain_tumor` paths)
- [ ] Copy `runs/` to Drive for Person 5

### Wrap-up
- [ ] Results table + error analysis (Person 5)
- [ ] Slides, README final check, rehearsal (Person 6)

## Notes

- Python env: `C:/Users/guymy/.venvs/dlmed/Scripts/python` (outside OneDrive on purpose)
- Data / runs: `C:/brain_tumor/` (outside OneDrive so GBs are not synced)
- Do not look at test results before settings are locked on validation.
