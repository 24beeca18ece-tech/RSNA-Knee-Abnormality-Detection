"""Step 1 plumbing-test training: v0 image baseline on ONLY the rows with a
real structured label (58 of 4407 in the full dataset as of 2026-08-07).

This is NOT a real training run - see docs/baseline_plan.md "Decided
sequencing". Goal is proving DICOM loading -> training -> checkpointing ->
AUC logging work end-to-end. A bad/noisy AUC here is expected and fine; do
not tune this. Real leverage comes from Step 2 (report-text weak-labeling).

Usage:
    python -m src.training.train_baseline --data-root data --epochs 5
    python -m src.training.train_baseline --data-root /kaggle/input/competitions/rsna-knee-abnormality-detection --pretrained --epochs 5
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.data.dataset import KneeSliceDataset
from src.models.image_baseline import KneeImageBaseline

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
ID_COL = "StudyInstanceUID"


def macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, int]:
    """Macro-averaged AUC, skipping any target column with only one class
    present in y_true (roc_auc_score is undefined there) - expected and
    common with a val split this small. Returns (auc, n_targets_used)."""
    aucs = []
    for j in range(y_true.shape[1]):
        col = y_true[:, j]
        if len(np.unique(col)) < 2:
            continue
        aucs.append(roc_auc_score(col, y_pred[:, j]))
    return (float(np.mean(aucs)) if aucs else float("nan")), len(aucs)


def pick_device() -> torch.device:
    """torch.cuda.is_available() can be True while the GPU is still unusable
    - e.g. a Kaggle P100 (CUDA capability sm_60) with a PyTorch build that
      only ships kernels for sm_70+ raises `no kernel image is available`
      on the first real op, not at is_available()/device construction time.
    Confirm with a tiny real op and fall back to CPU rather than crashing
    mid-training."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.zeros(1, device="cuda") + 1
        return torch.device("cuda")
    except RuntimeError as e:
        print(f"cuda.is_available()=True but a test op failed ({e}); falling back to cpu.")
        return torch.device("cpu")


def main(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    device = pick_device()
    print(f"device: {device}")

    labels_path = Path(args.labels_csv) if args.labels_csv else data_root / "train.csv"
    series_csv_path = Path(args.series_csv) if args.series_csv else data_root / "train_series.csv"
    labels_df = pd.read_csv(labels_path)
    labeled = labels_df[labels_df[TARGETS].notna().all(axis=1)].reset_index(drop=True)
    print(f"labeled rows (Step 1 training set): {len(labeled)} / {len(labels_df)}")
    series_meta = pd.read_csv(series_csv_path)

    train_df, val_df = train_test_split(labeled, test_size=args.val_frac, random_state=args.seed)
    print(f"train/val split: {len(train_df)} / {len(val_df)} (single split, no k-fold - see baseline_plan.md)")

    series_root = data_root / "train_series"
    train_ds = KneeSliceDataset(train_df, series_meta, series_root, TARGETS, image_size=args.image_size)
    val_ds = KneeSliceDataset(val_df, series_meta, series_root, TARGETS, image_size=args.image_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = KneeImageBaseline(n_targets=len(TARGETS), backbone=args.backbone, pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    ckpt_dir = Path(args.checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.logs_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"train_baseline_{run_id}.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_macro_auc", "n_targets_with_auc", "epoch_seconds"])

    best_auc = float("-inf")
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t_epoch = time.time()
        model.train()
        train_loss_sum, n_train = 0.0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * imgs.size(0)
            n_train += imgs.size(0)
        train_loss = train_loss_sum / max(n_train, 1)

        model.eval()
        val_loss_sum, n_val = 0.0, 0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss_sum += loss.item() * imgs.size(0)
                n_val += imgs.size(0)
                all_probs.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        val_loss = val_loss_sum / max(n_val, 1)
        val_auc, n_auc = macro_auc(np.concatenate(all_labels), np.concatenate(all_probs))

        epoch_seconds = time.time() - t_epoch
        print(f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_macro_auc={val_auc:.4f} (over {n_auc}/{len(TARGETS)} targets)  [{epoch_seconds:.1f}s]")
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, val_auc, n_auc, round(epoch_seconds, 2)])

        # Checkpoint every epoch (CLAUDE.md rule 1) - must survive a mid-run kill.
        ckpt = {
            "epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "val_macro_auc": val_auc, "targets": TARGETS, "backbone": args.backbone, "image_size": args.image_size,
        }
        torch.save(ckpt, ckpt_dir / f"epoch_{epoch}.pt")
        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"  -> new best.pt (val_macro_auc={val_auc:.4f})")

    torch.save(ckpt, ckpt_dir / "last.pt")
    if best_auc == float("-inf"):
        # every epoch's AUC was undefined (all-single-class val folds) - still
        # need a best.pt for the inference step to load.
        torch.save(ckpt, ckpt_dir / "best.pt")
        print("Note: val_macro_auc was undefined every epoch (expected/plausible with 58 rows) - best.pt = last epoch.")

    total_seconds = time.time() - t_start
    print(f"\nTotal training wall-clock: {total_seconds:.1f}s ({total_seconds/60:.2f} min)")
    print(f"Log written to {log_path}")
    print(f"Checkpoints in {ckpt_dir} (best.pt, last.pt, epoch_N.pt)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data", help="Root containing train_series/ (image data)")
    parser.add_argument("--labels-csv", default=None, help="Override for train.csv (default: <data-root>/train.csv)")
    parser.add_argument("--series-csv", default=None, help="Override for train_series.csv (default: <data-root>/train_series.csv)")
    parser.add_argument("--checkpoints-dir", default="outputs/checkpoints")
    parser.add_argument("--logs-dir", default="outputs/logs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--pretrained", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
