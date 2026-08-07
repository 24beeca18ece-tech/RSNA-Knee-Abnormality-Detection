"""Download a small local dev sample of the competition data: the root
metadata CSVs plus a handful of full study/series folders, picked from a
file listing produced by survey_competition_files.py.

Deliberately NOT `kaggle competitions download -c ...` (the full ~570GB
dataset) - this is for testing code logic locally only. Real training runs
inside a Kaggle kernel against the full attached dataset - see CLAUDE.md.

Usage:
    python scripts/survey_competition_files.py  # writes data/_file_listing.ndjson
    python scripts/download_sample.py           # uses the defaults below
    python scripts/download_sample.py --prefix "train_series/<uid>/" --prefix "test_series/<uid>/<uid>/"
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "rsna-knee-abnormality-detection"
ROOT_FILES = {"train.csv", "train_series.csv", "test.csv", "test_series.csv", "sample_submission.csv"}

# One train study (5 of its 8 series, ~85 MB) + one full test series
# (~13 MB), picked as some of the smallest folders seen in a 30-page listing
# survey on 2026-08-07. Pinned to exact series prefixes (not the whole study
# prefix) so this stays idempotent/stable rather than pulling in more series
# on every re-run. See docs/baseline_plan.md / configs/config.yaml `local_sample`.
_STUDY = "train_series/1.2.826.0.1.3680043.8.498.10029520033957968044048068502185417806"
DEFAULT_PREFIXES = [
    f"{_STUDY}/1.2.826.0.1.3680043.8.498.10138400860303228304047985771421435784/",
    f"{_STUDY}/1.2.826.0.1.3680043.8.498.11788164428905677775308855476550714595/",
    f"{_STUDY}/1.2.826.0.1.3680043.8.498.12061584096631077508632129628209972719/",
    f"{_STUDY}/1.2.826.0.1.3680043.8.498.12283902861056341541201338138068066294/",
    f"{_STUDY}/1.2.826.0.1.3680043.8.498.12595857485273978815469036481976397270/",
    "test_series/1.2.826.0.1.3680043.8.498.10062861783145312629332250977456991776/1.2.826.0.1.3680043.8.498.34851887739080902195133258975769749352/",
]


def winlong(p: Path) -> str:
    """Windows MAX_PATH (260 char) workaround - this dataset's nested DICOM
    UID folder names routinely blow past it. No-op on non-Windows."""
    if os.name != "nt":
        return str(p)
    s = str(p.resolve())
    return s if s.startswith("\\\\?\\") else "\\\\?\\" + s


def exists_nonempty(p: Path) -> bool:
    # plain Path.exists() silently returns False on Windows for paths past
    # MAX_PATH instead of raising - always check via the long-path form.
    lp = Path(winlong(p))
    return lp.exists() and lp.stat().st_size > 0


def main(listing_path: Path, data_dir: Path, prefixes: list[str]) -> None:
    with open(listing_path, encoding="utf-8") as f:
        all_files = [json.loads(line) for line in f]

    to_download = [
        (d["name"], d["size"])
        for d in all_files
        if d["name"] in ROOT_FILES or any(d["name"].startswith(p) for p in prefixes)
    ]
    print(f"Planning {len(to_download)} files, {sum(s for _, s in to_download) / 1e6:.2f} MB total")

    api = KaggleApi()
    api.authenticate()

    downloaded, skipped, failed = 0, 0, []
    for i, (name, size) in enumerate(to_download, 1):
        local_file = data_dir / name
        if exists_nonempty(local_file):
            skipped += 1
            continue
        dest_dir = local_file.parent
        Path(winlong(dest_dir)).mkdir(parents=True, exist_ok=True)
        print(f"[{i}/{len(to_download)}] {name} ({size / 1e6:.2f} MB)")
        for attempt in range(3):
            try:
                api.competition_download_file(COMPETITION, name, path=winlong(dest_dir), force=True, quiet=True)
                downloaded += 1
                break
            except Exception as e:  # noqa: BLE001 - transient network errors happen; retry a few times
                print(f"  retry {attempt + 1}/3 after error: {e}")
                time.sleep(2)
        else:
            failed.append(name)

    print(f"\nDone. Downloaded {downloaded}, already had {skipped}, failed {len(failed)}.")
    if failed:
        print("Failed files (re-run this script to retry - it skips what's already there):")
        for name in failed:
            print(f"  {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listing", default="data/_file_listing.ndjson")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--prefix", action="append", dest="prefixes", default=None,
                         help="Repeatable. Path prefix to include (e.g. 'train_series/<StudyUID>/'). Defaults to a small preset sample.")
    args = parser.parse_args()
    main(Path(args.listing), Path(args.data_dir), args.prefixes or DEFAULT_PREFIXES)
