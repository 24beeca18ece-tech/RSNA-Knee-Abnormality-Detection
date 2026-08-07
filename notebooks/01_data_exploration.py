# %% [markdown]
# # RSNA Knee Abnormality Detection — Data Exploration
#
# Written as a `# %%` cell-marker script (opens as a notebook in VS Code /
# Jupytext / Kaggle "Upload notebook"). Safe to run repeatedly — every cell
# is read-only against `data/`.
#
# Goals:
# 1. List the file structure under `data/`.
# 2. Read the labels/metadata file(s).
# 3. Check image format(s) and study count.
# 4. Sample a few paired radiology reports.
# 5. Check class balance across the 12 target labels.
#
# Run this only after downloading + unzipping the competition data
# (see README.md "Kaggle CLI setup"). If `data/` is empty, every cell below
# will say so instead of crashing.

# %%
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    # Reports are multilingual (accented Spanish/German/Dutch etc.) - avoid
    # mojibake when the terminal's default codepage isn't UTF-8 (common on
    # Windows). Purely cosmetic; the underlying data is read correctly either way.
    sys.stdout.reconfigure(encoding="utf-8")

try:
    import yaml
    _cfg_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml" if "__file__" in dir() else Path("../configs/config.yaml")
except ImportError:
    yaml = None
    _cfg_path = None

PROJECT_ROOT = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path("..")
DATA_ROOT = PROJECT_ROOT / "data"


def winlong(p: Path) -> Path:
    """Windows MAX_PATH (260 char) workaround. This dataset's DICOM UID
    folder names are ~40 chars each, 3 levels deep, which blows past the
    limit on plain Windows paths. \\\\?\\ opts into the extended-length path
    API. No-op on non-Windows (e.g. Kaggle kernels, which run Linux) since
    it's never needed there.
    """
    if os.name != "nt":
        return p
    s = str(p.resolve())
    return Path(s if s.startswith("\\\\?\\") else "\\\\?\\" + s)

# Confirmed against the real train.csv/sample_submission.csv on 2026-08-07
# (exact names, including spaces and the apostrophe - matters for any code
# doing exact-match against these columns, e.g. make_trivial_submission.py).
TARGET_COLS_GUESS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]

print(f"Project root: {PROJECT_ROOT}")
print(f"Data root:    {DATA_ROOT}  (exists: {DATA_ROOT.exists()})")

# %% [markdown]
# ## 1. File structure

# %%
def walk_summary(root: Path, max_depth: int = 3, max_items_per_dir: int = 15):
    """Print a depth-limited tree and a global extension histogram."""
    if not root.exists():
        print(f"{root} does not exist yet — download the data first (see README.md).")
        return

    ext_counter = Counter()
    total_files = 0

    def _walk(path: Path, depth: int):
        nonlocal total_files
        try:
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            return
        shown = entries[:max_items_per_dir]
        for entry in shown:
            prefix = "  " * depth
            if entry.is_dir():
                n_children = sum(1 for _ in entry.iterdir())
                print(f"{prefix}{entry.name}/  ({n_children} items)")
                if depth < max_depth:
                    _walk(entry, depth + 1)
            else:
                total_files += 1
                ext_counter[entry.suffix.lower()] += 1
                if depth <= max_depth:
                    print(f"{prefix}{entry.name}")
        if len(entries) > max_items_per_dir:
            print("  " * depth + f"... ({len(entries) - max_items_per_dir} more)")

    _walk(root, 0)
    print("\nExtension counts (entire subtree, not just shown items):")
    for ext, count in ext_counter.most_common():
        print(f"  {ext or '(no ext)'}: {count}")
    print(f"Total files scanned: {total_files}")


walk_summary(DATA_ROOT)

# %% [markdown]
# ## 2. Labels / metadata file
#
# Auto-detects candidate CSV/JSON files at the top of `data/` and previews
# them. Update `configs/config.yaml` -> `paths.train_labels_csv` once you
# know the real filename.

# %%
def find_candidate_tables(root: Path):
    if not root.exists():
        return []
    candidates = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".csv", ".json", ".parquet") and p.stat().st_size < 500_000_000:
            candidates.append(p)
    return sorted(candidates)


candidate_tables = find_candidate_tables(DATA_ROOT)
print(f"Found {len(candidate_tables)} candidate metadata/label files:")
for p in candidate_tables:
    print(f"  {p.relative_to(DATA_ROOT)}  ({p.stat().st_size / 1024:.1f} KB)")

# %%
def load_table(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        if path.suffix.lower() == ".json":
            return pd.read_json(path)
    except Exception as e:  # noqa: BLE001 - exploration script, want to see any failure
        print(f"  Failed to load {path.name}: {e}")
        return None


loaded_tables: dict[str, pd.DataFrame] = {}
for p in candidate_tables:
    df = load_table(p)
    if df is not None:
        loaded_tables[p.name] = df
        print(f"\n=== {p.relative_to(DATA_ROOT)} === shape={df.shape}")
        print(df.head(3).to_string())
        print("dtypes:")
        print(df.dtypes)

# %% [markdown]
# Identify which loaded table looks like the labels file (contains most of
# the 12 target names, case-insensitively / with underscore-vs-space fuzz).

# %%
def normalize(s: str) -> str:
    return s.lower().replace(" ", "").replace("_", "").replace("-", "")


def find_labels_table(tables: dict[str, pd.DataFrame], target_names: list[str]):
    """Prefer the table with the most matched target columns; break ties by
    row count. This matters here because sample_submission.csv matches the
    same target columns as the real labels file but has far fewer rows (one
    per test study, all filled with a placeholder 0.5) — row count is a
    reliable way to tell "the real labels" from "the submission template".
    """
    normalized_targets = {normalize(t) for t in target_names}
    best_name, best_df, best_hits = None, None, -1
    for name, df in tables.items():
        cols_norm = {normalize(c) for c in df.columns}
        hits = len(cols_norm & normalized_targets)
        if hits > best_hits or (hits == best_hits and best_df is not None and len(df) > len(best_df)):
            best_name, best_df, best_hits = name, df, hits
    return best_name, best_df, best_hits


labels_name, labels_df, hits = find_labels_table(loaded_tables, TARGET_COLS_GUESS)
if labels_df is not None and hits > 0:
    print(f"Best guess for labels file: {labels_name} ({hits}/{len(TARGET_COLS_GUESS)} target columns matched)")
else:
    print("Could not confidently identify the labels file. Inspect the tables printed above manually.")

# %% [markdown]
# ## 3. Image format(s) and study count

# %%
IMAGE_EXTENSIONS = {".dcm", ".png", ".jpg", ".jpeg", ".npy", ".npz", ".tiff"}


def scan_images(root: Path):
    if not root.exists():
        return Counter(), set()
    ext_counter = Counter()
    top_level_dirs = set()
    for p in root.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTENSIONS and winlong(p).is_file():
            ext_counter[p.suffix.lower()] += 1
            try:
                rel_parts = p.relative_to(root).parts
                if rel_parts:
                    top_level_dirs.add(rel_parts[0])
            except ValueError:
                pass
    return ext_counter, top_level_dirs


img_ext_counts, img_top_dirs = scan_images(DATA_ROOT)
print("Image file extension counts:")
for ext, count in img_ext_counts.most_common():
    print(f"  {ext}: {count}")
print(f"Top-level dirs containing images (sample): {sorted(img_top_dirs)[:10]}")

# %%
# If DICOM files are present, read a handful of headers to check modality,
# pixel spacing, and how studies/series are organized on disk.
dcm_files = list(DATA_ROOT.rglob("*.dcm"))[:5] if DATA_ROOT.exists() else []
if dcm_files:
    try:
        import pydicom

        for f in dcm_files:
            ds = pydicom.dcmread(winlong(f), stop_before_pixels=True)
            print(f"\n{f.relative_to(DATA_ROOT)}")
            for tag in ("Modality", "StudyInstanceUID", "SeriesInstanceUID", "Rows", "Columns", "PixelSpacing"):
                print(f"  {tag}: {getattr(ds, tag, 'N/A')}")
    except ImportError:
        print("pydicom not installed — `pip install pydicom` to inspect DICOM headers.")
else:
    print("No .dcm files found (either not downloaded yet, or images are in another format — see extension counts above).")

# %%
# Study count: number of distinct study identifiers, inferred from directory
# names one level below a detected image root, or from a study-id column in
# the labels table if one exists.
if img_top_dirs:
    print(f"Distinct top-level study/series directories found: {len(img_top_dirs)}")

if labels_df is not None:
    id_like_cols = [c for c in labels_df.columns if "study" in c.lower() or "id" in c.lower()]
    print(f"ID-like columns in labels table: {id_like_cols}")
    for c in id_like_cols:
        print(f"  {c}: {labels_df[c].nunique()} unique values")

# %% [markdown]
# ## 4. Sample paired radiology reports
#
# Looks for free-text report files (.txt/.json) or a text column inside the
# labels table, and prints a few samples. Per RSNA's 2026 challenge
# description, reports may be multilingual — worth checking language mix here.

# %%
report_text_files = []
if DATA_ROOT.exists():
    for ext in (".txt", ".json"):
        report_text_files.extend(list(DATA_ROOT.rglob(f"*report*{ext}")))
report_text_files = report_text_files[:5]

if report_text_files:
    for f in report_text_files:
        print(f"\n=== {f.relative_to(DATA_ROOT)} ===")
        try:
            content = winlong(f).read_text(encoding="utf-8", errors="replace")
            print(content[:500])
        except Exception as e:  # noqa: BLE001
            print(f"  could not read: {e}")
elif labels_df is not None:
    # pandas may back string columns with a native "str" dtype rather than
    # legacy "object" depending on version - check both.
    text_cols = [
        c for c in labels_df.columns
        if pd.api.types.is_string_dtype(labels_df[c])
        and labels_df[c].dropna().astype(str).str.len().mean() > 40
    ]
    print(f"No standalone report files found; candidate free-text columns in labels table: {text_cols}")
    for c in text_cols[:2]:
        print(f"\n--- sample values from '{c}' ---")
        for v in labels_df[c].dropna().head(3):
            print(f"  {str(v)[:400]}\n")
else:
    print("No report files or labels table found yet.")

# %% [markdown]
# ## 5. Class balance across the 12 targets

# %%
if labels_df is not None:
    matched_cols = [c for c in labels_df.columns if normalize(c) in {normalize(t) for t in TARGET_COLS_GUESS}]
    if matched_cols:
        balance = labels_df[matched_cols].apply(pd.Series.value_counts, normalize=True).T
        print("Positive-rate per target (fraction of 1s where inferable):")
        for c in matched_cols:
            vc = labels_df[c].value_counts(normalize=True, dropna=False)
            print(f"  {c:20s} n={labels_df[c].notna().sum():6d}  {dict(vc.round(3))}")
    else:
        print("Couldn't match target columns automatically. Actual columns in labels table:")
        print(list(labels_df.columns))
else:
    print("No labels table loaded — run this after downloading data.")

# %% [markdown]
# ## Next steps
# - Fill in the TODOs in `configs/config.yaml` (`paths.*`) based on what was
#   found above.
# - Update `TARGET_COLS_GUESS` in this script / `targets:` in config.yaml if
#   the real column names differ.
# - Once paths are confirmed, run `src/inference/make_trivial_submission.py`
#   to produce a first valid `submission.csv`.
