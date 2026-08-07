# RSNA Knee Abnormality Detection

Kaggle Notebooks-only code competition. Multimodal: knee MRI + paired radiology
reports (multilingual, per RSNA's challenge announcement) -> 12 binary
abnormality labels, scored by macro-averaged AUC-ROC. Final submission runs
on Kaggle with **internet disabled**, must write `submission.csv`, **<= 9h** runtime.

Competition page: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection

## Project layout

```
data/            gitignored - small local dev SAMPLE only, see "Local sample data" below.
                 The full ~570GB dataset never lives here - see CLAUDE.md rule 6.
notebooks/       exploration / prototyping (Kaggle-notebook-compatible scripts)
scripts/         one-off utilities: survey_competition_files.py, download_sample.py
src/
  data/          dataset classes, file indexing, loaders
  preprocessing/ DICOM decoding, slice selection, text cleaning
  models/        image / text / fusion model definitions
  training/      training loops, checkpointing
  inference/     offline inference -> submission.csv
kernels/         one folder per pushed Kaggle kernel (kernel-metadata.json + code)
  dev/           dev smoke-test kernel - runs against the FULL dataset on Kaggle's
                 servers, no local download needed. See "Kaggle kernel workflow".
configs/         yaml configs (paths, hyperparams, target list)
outputs/
  checkpoints/   saved model weights (gitignored, but dir tracked)
  submissions/   generated submission.csv files (gitignored)
  logs/          training logs (gitignored)
  kaggle_runs/   pulled kernel output/logs (gitignored)
  figures/       exploration plots
```

`data/` and large binary outputs are gitignored — see `.gitignore`. Only code,
configs, and small metadata artifacts should ever be committed.

## Environment note

Local Python here is **3.14**, which is too new for stable `torch` wheels as of
this writing. Two options:
1. Do exploration/EDA locally with a lighter env (`pandas`, `pydicom`, `pillow`,
   `matplotlib` install fine on 3.14) and do all model training inside a Kaggle
   Notebook, which ships a working torch/CUDA environment already.
2. Create a local 3.10/3.11 virtualenv (e.g. via `pyenv` or `conda`) if you want
   to train/debug locally before pushing to Kaggle.

Given this is a Notebooks-only competition anyway, option 1 (light local EDA,
heavy lifting on Kaggle) is recommended and is what the rest of this doc assumes.

## Kaggle CLI setup

The CLI is already installed (`kaggle==2.2.3` found on this machine). If you
ever need it fresh:

```bash
pip install kaggle
```

### 1. Get your API token

1. Go to https://www.kaggle.com/settings (Account tab) -> "Create New Token".
2. This downloads `kaggle.json` containing your username + API key.

### 2. Install the credentials

PowerShell (Windows):
```powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.kaggle" | Out-Null
Move-Item -Force "$env:USERPROFILE\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json"
# Restrict permissions (best-effort on Windows; the CLI just checks the file exists)
icacls "$env:USERPROFILE\.kaggle\kaggle.json" /inheritance:r /grant:r "$($env:USERNAME):(R,W)"
```

Git Bash / WSL equivalent:
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Verify auth works:
```bash
kaggle competitions list -s knee
```

**Gotcha:** the `kaggle.json` you install must belong to the *exact* Kaggle
account you accept the rules with. If `kaggle competitions list -s knee`
shows `userHasEntered: False` for this competition even after clicking
"I Understand and Accept" on the website, check `kaggle config view` — if
the username doesn't match the account you used in the browser, get a fresh
token from the correct account's Settings page and replace `kaggle.json`.
Downloads 403 with no useful message when this happens.

### 3. Accept the competition rules

You must accept the rules on the competition **website** before the API will
let you download data (the CLI cannot accept rules for you):
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules
-> click "I Understand and Accept".

### 4. Local sample data (small, for dev only)

The full dataset is **~570GB** (4407 train studies, multi-series DICOM
each) — do **not** `kaggle competitions download -c ... ` the whole thing
locally. Instead, pull a small sample just big enough to exercise code
logic (DICOM reading, CSV schema, submission format):

```bash
cd "path/to/RSNA Knee Abnormality Detection"
python scripts/survey_competition_files.py     # lists files without downloading, writes data/_file_listing.ndjson
python scripts/download_sample.py              # downloads root CSVs + a preset small study/series sample (~95MB)
```

Re-running `download_sample.py` is idempotent (skips files already on disk).
Pass `--prefix "train_series/<StudyInstanceUID>/<SeriesInstanceUID>/"`
(repeatable) to grab specific additional series found via the survey script.

This sample is **not representative** of class balance or dataset scale —
see `configs/config.yaml` `local_sample` section and CLAUDE.md rule 6.
Then run `notebooks/01_data_exploration.py` against it to sanity-check code
against real file structure (schema is already confirmed and reflected in
`configs/config.yaml` as of 2026-08-07 — re-run only if something looks off).

## Kaggle kernel workflow (real data, real training, real inference)

Per CLAUDE.md rule 6: all real data access, training, and inference happen
inside Kaggle kernels, where the full competition dataset is attached
server-side (no local download) and Kaggle's free GPU quota is available.
Each kernel is a folder under `kernels/` containing a `kernel-metadata.json`
(see `kaggle kernels init -p <folder>` to scaffold a new one) and its code
file(s).

```bash
# Push local code to Kaggle and start a run
kaggle kernels push -p kernels/dev

# Poll until it finishes (RUNNING -> COMPLETE or ERROR)
kaggle kernels status dograbrij/rsna-knee-dev-smoke-test

# Full run logs (stdout/stderr) if something fails
kaggle kernels logs dograbrij/rsna-knee-dev-smoke-test

# Pull output files (e.g. submission.csv) + logs back locally
kaggle kernels output dograbrij/rsna-knee-dev-smoke-test -p outputs/kaggle_runs/dev
```

Iterate by editing the code file(s) in `kernels/<name>/` locally, then
`kaggle kernels push -p kernels/<name>` again — this creates a new version
and re-runs it. The kernel id in `kernel-metadata.json` (`dograbrij/<slug>`)
must stay the same across pushes to update the same kernel rather than
create a new one.

`kernels/dev/` is a working example: a CPU-only, internet-on dev kernel that
attaches the competition data (`competition_sources` in its
kernel-metadata.json) and confirms real-scale findings against the small
local sample. Note the mount path it discovered: this competition's data
lands at `/kaggle/input/competitions/<slug>/`, not the flatter
`/kaggle/input/<slug>/` some Kaggle examples assume.

The **final scored submission kernel** is a different artifact once a real
model exists: `enable_internet=false`, loads pre-saved weights, no training
code — see CLAUDE.md rules 1-2 and 4.

## Workflow

1. **Data exploration** (`notebooks/01_data_exploration.py` locally against
   the small sample; `kernels/dev/dev_smoke_test.py` for full-scale checks)
   — confirm file structure, label schema, class balance. No training.
2. **Trivial baseline submission** (`src/inference/make_trivial_submission.py`
   locally; the same logic runs inside `kernels/dev/dev_smoke_test.py` against
   the real full data) — proves the end-to-end submission pipeline is valid
   before any model exists.
3. **Baseline unimodal image model** — see `docs/baseline_plan.md`. Trained
   inside a Kaggle kernel (needs the full dataset + GPU). Fast, simple, gets
   a real (non-constant) AUC on the leaderboard.
4. **Multimodal (image + report text)** — only after (3) is working and checked in.

See `CLAUDE.md` for standing rules on checkpointing, offline inference, and
the local/Kaggle-kernel split, given Kaggle's GPU quota / session /
no-internet / disk constraints.
