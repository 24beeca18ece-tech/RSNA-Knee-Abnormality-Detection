"""Produce a trivial-but-valid submission.csv with no trained model.

Purpose: prove the end-to-end submission pipeline (read sample_submission ->
write correctly-shaped predictions) works before any model exists. This is
step 0, ahead of the fast unimodal baseline described in docs/baseline_plan.md.

Usage:
    python src/inference/make_trivial_submission.py
    python src/inference/make_trivial_submission.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_fallback_probs(cfg: dict) -> dict[str, float] | float:
    """Prefer per-target base rates from the training labels if available
    (still trivial/no-model, but a sharper prior than a flat 0.5 everywhere).
    Falls back to a single flat probability otherwise.
    """
    targets = cfg["targets"]
    flat_p = cfg["baseline"]["fallback_probability"]

    labels_path = PROJECT_ROOT / cfg["paths"]["train_labels_csv"]
    if not labels_path.exists():
        print(f"[trivial-submission] No train labels found at {labels_path}; using flat p={flat_p} for all targets.")
        return flat_p

    try:
        labels_df = pd.read_csv(labels_path)
    except Exception as e:  # noqa: BLE001
        print(f"[trivial-submission] Failed to read {labels_path} ({e}); using flat p={flat_p}.")
        return flat_p

    rates = {}
    for t in targets:
        if t in labels_df.columns:
            rate = labels_df[t].mean()
            # keep it away from exact 0/1 so log-loss-style metrics don't blow up
            # if this csv is ever scored by something other than pure AUC
            rates[t] = float(min(max(rate, 0.02), 0.98))
        else:
            rates[t] = flat_p

    if all(v == flat_p for v in rates.values()):
        print(f"[trivial-submission] No target columns matched in {labels_path}; using flat p={flat_p}.")
        return flat_p

    print("[trivial-submission] Using per-target base rates from train labels:")
    for t, r in rates.items():
        print(f"  {t:20s} p={r:.3f}")
    return rates


def main(config_path: str) -> None:
    cfg = load_config(PROJECT_ROOT / config_path if not Path(config_path).is_absolute() else Path(config_path))

    sample_sub_path = PROJECT_ROOT / cfg["paths"]["sample_submission_csv"]
    if not sample_sub_path.exists():
        raise FileNotFoundError(
            f"{sample_sub_path} not found. Download + unzip the competition data first "
            f"(see README.md 'Kaggle CLI setup'), then re-run."
        )

    sample_sub = pd.read_csv(sample_sub_path)
    targets = cfg["targets"]
    id_cols = [c for c in sample_sub.columns if c not in targets]
    matched_targets = [t for t in targets if t in sample_sub.columns]

    if not matched_targets:
        raise ValueError(
            f"None of the configured targets {targets} match sample_submission.csv columns "
            f"{list(sample_sub.columns)}. Update `targets:` in configs/config.yaml after "
            f"running notebooks/01_data_exploration.py."
        )
    if len(matched_targets) < len(targets):
        missing = set(targets) - set(matched_targets)
        print(f"[trivial-submission] Warning: {missing} not found in sample_submission.csv columns.")

    probs = compute_fallback_probs(cfg)

    out = sample_sub[id_cols].copy()
    for t in matched_targets:
        out[t] = probs[t] if isinstance(probs, dict) else probs

    out_dir = PROJECT_ROOT / cfg["paths"]["submissions_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "submission_trivial.csv"
    out.to_csv(out_path, index=False)

    print(f"\n[trivial-submission] Wrote {out_path}  shape={out.shape}")
    print(out.head())

    # Sanity checks that mirror what Kaggle's scorer will implicitly require.
    assert list(out.columns) == list(sample_sub.columns), "column order/names must match sample_submission.csv exactly"
    assert len(out) == len(sample_sub), "row count must match sample_submission.csv exactly"
    assert out[matched_targets].isna().sum().sum() == 0, "no NaNs allowed in predictions"
    print("[trivial-submission] Shape/column/NaN sanity checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
