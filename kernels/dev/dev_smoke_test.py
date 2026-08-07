"""Dev smoke test - runs INSIDE a Kaggle kernel with the full competition
dataset attached (via kernel-metadata.json `competition_sources`), no local
download needed. Purpose: prove the push/run/pull workflow works, and check
whether findings from the small local sample (docs/baseline_plan.md) hold at
full scale - in particular the ~1.3% structured-label coverage and the
train/test Report-column asymmetry found locally on 2026-08-07.

Self-contained on purpose (no imports from this repo's src/ package) - the
kernel environment doesn't have this project installed. Column names /
target list are hardcoded to match configs/config.yaml; keep them in sync
if that file changes.

Not the final scored submission - this is a CPU-only, internet-on dev
kernel for iteration. The final inference kernel (once a real model exists)
must be a separate kernel with enable_internet=false per CLAUDE.md rule 2.
"""

import os

import pandas as pd

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]

SLUG = "rsna-knee-abnormality-detection"
# Confirmed 2026-08-07: this competition mounts under /kaggle/input/competitions/<slug>/,
# not the flatter /kaggle/input/<slug>/ some Kaggle docs/examples show for datasets.
CANDIDATE_DATA_DIRS = [
    f"/kaggle/input/{SLUG}",
    f"/kaggle/input/competitions/{SLUG}",
]
DATA_DIR = next((d for d in CANDIDATE_DATA_DIRS if os.path.isdir(d)), None)
if DATA_DIR is None:
    found = []
    for root, dirs, files in os.walk("/kaggle/input"):
        if "train.csv" in files:
            found.append(root)
    if len(found) == 1:
        DATA_DIR = found[0]
    else:
        raise FileNotFoundError(
            f"Couldn't locate the competition data under /kaggle/input. Tried "
            f"{CANDIDATE_DATA_DIRS}, walked and found train.csv in: {found}. Check "
            f"kernel-metadata.json competition_sources and the kernel's Data pane on kaggle.com."
        )
print(f"DATA_DIR: {DATA_DIR}")
print("Top-level contents:", sorted(os.listdir(DATA_DIR)))

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

print(f"\ntrain.csv shape: {train.shape}")
print(f"test.csv shape: {test.shape}  columns: {list(test.columns)}")
print(f"sample_submission.csv shape: {sample_sub.shape}")

has_label = train[TARGETS].notna().any(axis=1)
print(f"\nRows with >=1 non-null target label: {has_label.sum()} / {len(train)} ({has_label.mean():.1%})")
print(f"Report column non-null: {train['Report'].notna().sum()} / {len(train)}")

for study_dir in ("train_series", "test_series"):
    p = os.path.join(DATA_DIR, study_dir)
    if os.path.isdir(p):
        n_studies = len(os.listdir(p))
        print(f"{study_dir}: {n_studies} study folders")

# Trivial-but-valid submission: per-target base rate from labeled train
# rows (falls back to 0.5 if a target has zero labeled rows). Mirrors
# src/inference/make_trivial_submission.py - kept inline since the kernel
# can't import this repo's src/ package.
id_cols = [c for c in sample_sub.columns if c not in TARGETS]
out = sample_sub[id_cols].copy()
for t in TARGETS:
    rate = train[t].mean()
    out[t] = float(min(max(rate, 0.02), 0.98)) if pd.notna(rate) else 0.5

out.to_csv("submission.csv", index=False)
print(f"\nWrote submission.csv, shape={out.shape}")
assert list(out.columns) == list(sample_sub.columns)
assert len(out) == len(sample_sub)
assert out[TARGETS].isna().sum().sum() == 0
print("Sanity checks passed.")
