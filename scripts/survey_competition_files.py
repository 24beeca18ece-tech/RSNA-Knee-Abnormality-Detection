"""List competition files WITHOUT downloading them, to find small ones worth
pulling into a local dev sample. Paginates the Kaggle API file listing and
aggregates by top-level dir / by study-series folder.

Full dataset is ~570GB (per RSNA), too large to enumerate exhaustively in
one run - by default this only pages through the first N entries, enough to
surface several small series folders. Increase --max-pages for a wider (but
slower - each page is a network round trip) sweep.

Usage:
    python scripts/survey_competition_files.py
    python scripts/survey_competition_files.py --max-pages 100 --out data/_file_listing.ndjson
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "rsna-knee-abnormality-detection"


def main(max_pages: int, out_path: Path) -> None:
    api = KaggleApi()
    api.authenticate()

    top_level = defaultdict(lambda: [0, 0])
    series_bytes = defaultdict(lambda: [0, 0])
    root_files = []

    token = None
    page = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        while page < max_pages:
            resp = api.competition_list_files(COMPETITION, page_token=token, page_size=200)
            if not resp.files:
                break
            for file in resp.files:
                name, size = file.name, file.total_bytes
                f.write(json.dumps({"name": name, "size": size}) + "\n")
                parts = name.split("/")
                if len(parts) == 1:
                    root_files.append((name, size))
                else:
                    top_level[parts[0]][0] += 1
                    top_level[parts[0]][1] += size
                    if len(parts) >= 3:
                        key = "/".join(parts[:3])
                        series_bytes[key][0] += 1
                        series_bytes[key][1] += size
            page += 1
            token = resp.next_page_token
            if not token:
                break

    print(f"Paged {page} pages, wrote raw listing to {out_path}")
    print("\nRoot-level files:")
    for name, size in root_files:
        print(f"  {name}: {size} bytes")
    print("\nTop-level dir aggregates (from pages scanned):")
    for name, (count, size) in sorted(top_level.items(), key=lambda x: -x[1][1]):
        print(f"  {name}: {count} files, {size / 1e6:.1f} MB")
    print("\nSmallest study/series folders seen (top 15 by total bytes):")
    for key, (count, size) in sorted(series_bytes.items(), key=lambda x: x[1][1])[:15]:
        print(f"  {key}: {count} files, {size / 1e6:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-pages", type=int, default=30, help="Pages of 200 files each (default 30 = 6000 files)")
    parser.add_argument("--out", default="data/_file_listing.ndjson")
    args = parser.parse_args()
    main(args.max_pages, Path(args.out))
