# Project rules — RSNA Knee Abnormality Detection

## Constraints driving these rules

- Kaggle free GPU quota: ~30 hrs/week, 12-hour session cap.
- Final scored submission runs with **internet disabled**.
- Notebooks-only code competition: the graded run must be self-contained.
- Runtime budget for the scored inference run: **<= 9 hours**.
- Full competition dataset is ~570GB (4407 train studies, multi-series
  DICOM each) — far too large for local disk (this machine has ~48GB free
  as of 2026-08-07). Local downloads are a small dev sample only.

## Standing rules

1. **Always checkpoint training to disk, frequently.**
   Every training script/notebook must save model weights (and optimizer +
   epoch/step state if resuming matters) to `outputs/checkpoints/` on a
   schedule that survives a session getting killed mid-run — e.g. every N
   steps/epochs, not just at the end. A 12-hour session cap means "train to
   completion in one shot" is not a safe assumption; design every training
   run as resumable from the last checkpoint.

2. **Design the final inference notebook to run fully offline.**
   It must not call `pip install`, download from the internet, or otherwise
   assume connectivity. It loads pre-saved weights (uploaded as a Kaggle
   Dataset or committed as Notebook output) and any non-standard package
   already vendored/installed at commit time. Test this assumption explicitly
   — e.g. by disabling internet in the Kaggle notebook settings — before
   trusting a submission.

3. **Keep the pipeline runnable end-to-end at every stage**, even with a
   dummy/constant-probability model. Don't let "no working submission.csv"
   persist while iterating on model quality — a trivial valid submission
   should exist before, and keep working alongside, real model development.

4. **Separate exploration/training compute from inference compute.** Training
   notebooks (using GPU quota) are not the same artifact as the final
   inference notebook. The inference notebook should be small, fast, and
   only do: load weights -> load test data -> predict -> write
   `submission.csv`.

5. **`data/` is never committed.** Only code, configs, small metadata, and
   documentation belong in git. Model weights go to a Kaggle Dataset (or
   Notebook output), not into this git repo, unless small enough and
   explicitly intended to be versioned.

6. **Local machine = code authoring and small-sample testing only.** This
   machine does not have room for the ~570GB full dataset. The only data
   that ever lives in local `data/` is the small sample pulled by
   `scripts/download_sample.py` (root metadata CSVs + a couple of full
   study/series folders, tens of MB) — enough to exercise code paths
   (DICOM reading, CSV schema, submission-format validation), never enough
   to train a real model or draw conclusions about class balance/dataset
   scale. All real data access, all real training, and all real inference
   against the full dataset happen exclusively inside Kaggle kernels
   (`kaggle kernels push`, dataset attached server-side via
   `competition_sources` — no local download involved). See "Kaggle kernel
   workflow" in README.md for the push/pull commands. Before trusting any
   number derived from the local sample (class balance, row counts, image
   properties), prefer re-checking it against the full data from inside a
   kernel, the way `kernels/dev/dev_smoke_test.py` does.
