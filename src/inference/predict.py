"""Step 1 plumbing-test inference: load a checkpoint from
src/training/train_baseline.py, run it on the test set, write a valid
submission.csv. Score from this checkpoint is expected to be meaningless
(see docs/baseline_plan.md) - this only proves the inference path works.

Usage:
    python -m src.inference.predict --data-root data --checkpoint outputs/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import KneeSliceDataset
from src.models.image_baseline import KneeImageBaseline
from src.training.train_baseline import pick_device

ID_COL = "StudyInstanceUID"


def main(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    device = pick_device()

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    targets = ckpt["targets"]
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_macro_auc={ckpt['val_macro_auc']})")

    model = KneeImageBaseline(n_targets=len(targets), backbone=ckpt["backbone"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_csv_path = Path(args.test_csv) if args.test_csv else data_root / "test.csv"
    series_csv_path = Path(args.series_csv) if args.series_csv else data_root / "test_series.csv"
    sample_sub_path = Path(args.sample_submission) if args.sample_submission else data_root / "sample_submission.csv"
    test_df = pd.read_csv(test_csv_path)
    series_meta = pd.read_csv(series_csv_path)
    sample_sub = pd.read_csv(sample_sub_path)

    # KneeSliceDataset expects target columns to exist (for the label tensor,
    # unused at inference) - add dummy zero columns for test rows.
    test_df = test_df.copy()
    for t in targets:
        test_df[t] = 0.0

    series_root = data_root / "test_series"
    test_ds = KneeSliceDataset(test_df, series_meta, series_root, targets, image_size=ckpt["image_size"])
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    all_probs = []
    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            all_probs.append(torch.sigmoid(logits).cpu())
    probs = torch.cat(all_probs).numpy()

    out = test_df[[ID_COL]].copy()
    for j, t in enumerate(targets):
        out[t] = probs[:, j]
    # Match sample_submission.csv's exact column order.
    out = out[list(sample_sub.columns)]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  shape={out.shape}")

    assert list(out.columns) == list(sample_sub.columns), "column mismatch vs sample_submission.csv"
    assert len(out) == len(sample_sub), "row count mismatch vs sample_submission.csv"
    assert out[targets].isna().sum().sum() == 0, "NaNs in predictions"
    print("Shape/column/NaN sanity checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data", help="Root containing test_series/ (image data)")
    parser.add_argument("--test-csv", default=None, help="Override for test.csv (default: <data-root>/test.csv)")
    parser.add_argument("--series-csv", default=None, help="Override for test_series.csv (default: <data-root>/test_series.csv)")
    parser.add_argument("--sample-submission", default=None, help="Override for sample_submission.csv (default: <data-root>/sample_submission.csv)")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--output", default="outputs/submissions/submission_v0.csv")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    main(args)
