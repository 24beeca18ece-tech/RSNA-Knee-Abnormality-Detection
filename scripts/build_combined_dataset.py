"""Combine the 58 real structured-labeled rows with everything extracted so
far in data/processed/weak_labels.csv into one training-ready dataset.

Re-run this any time (e.g. after each daily Gemini trickle run) to refresh
data/processed/combined_training_labels.csv with the latest weak-label
coverage. label_source distinguishes provenance per row: "structured" (the
58 gold rows) vs "weak_claude"/"weak_groq"/"weak_gemini" - keep this
distinction downstream (e.g. weight structured rows higher, or hold them
out as a clean validation set) rather than treating all rows as equally
trustworthy. See docs/baseline_plan.md "Decided sequencing" for why.

Usage:
    python scripts/build_combined_dataset.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
ID_COL = "StudyInstanceUID"
COLUMNS = [ID_COL, "label_source", "confidence"] + TARGETS + ["Report"]


def main() -> None:
    df = pd.read_csv("data/train.csv")
    structured = df[df[TARGETS].notna().all(axis=1)].copy()
    structured["label_source"] = "structured"
    structured["confidence"] = "gold"
    structured = structured[COLUMNS]

    weak_path = Path("data/processed/weak_labels.csv")
    if weak_path.exists():
        weak = pd.read_csv(weak_path)
        weak = weak[COLUMNS]
    else:
        weak = pd.DataFrame(columns=COLUMNS)

    combined = pd.concat([structured, weak], ignore_index=True)
    assert combined[ID_COL].is_unique, "duplicate StudyInstanceUID across structured/weak - investigate"

    out_path = Path("data/processed/combined_training_labels.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)

    print(f"Wrote {out_path}  shape={combined.shape}")
    print("\nBy label_source:")
    print(combined["label_source"].value_counts().to_string())
    print("\nBy confidence:")
    print(combined["confidence"].value_counts(dropna=False).to_string())

    total_unlabeled = df[df[TARGETS].isna().all(axis=1)].shape[0]
    print(f"\nCoverage: {len(combined)} / {58 + total_unlabeled} total rows "
          f"({len(weak)} / {total_unlabeled} report-only rows weak-labeled so far)")


if __name__ == "__main__":
    main()
