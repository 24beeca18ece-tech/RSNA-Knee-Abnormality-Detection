"""Step 3 self-supervised (SimCLR) pretraining of the image encoder on ALL
4407 studies' images - no labels needed, so every study contributes
(labeled or not), unlike Step 1/2 which only ever trained on the rows with
some label. See docs/baseline_plan.md "Step 3".

Auto-resumes from the latest checkpoint on start (unlike train_baseline.py,
which is short enough per-run not to need this) - SSL pretraining over 4407
images will likely span multiple Kaggle sessions given the 12h cap
(CLAUDE.md rule 1), so losing progress on a killed session would be
expensive here in a way it wasn't for Step 1's 58-row runs.

Usage:
    python -m src.training.train_ssl_pretrain --data-root data --epochs 100
    python -m src.training.train_ssl_pretrain --data-root /kaggle/input/competitions/rsna-knee-abnormality-detection --epochs 100
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.ssl_dataset import KneeSSLDataset
from src.models.ssl_model import SimCLRModel, nt_xent_loss
from src.training.train_baseline import pick_device

ID_COL = "StudyInstanceUID"


def find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    candidates = sorted(ckpt_dir.glob("ssl_epoch_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    return candidates[-1] if candidates else None


def main(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    device = pick_device()
    print(f"device: {device}")

    # Every study with usable series metadata, labeled or not - this is the
    # entire point of Step 3 (see module docstring).
    series_csv_path = Path(args.series_csv) if args.series_csv else data_root / "train_series.csv"
    series_meta = pd.read_csv(series_csv_path)
    study_ids = series_meta[ID_COL].unique().tolist()
    if args.limit:
        study_ids = study_ids[: args.limit]
    print(f"pretraining on {len(study_ids)} studies (no labels used)")

    series_root = data_root / "train_series"
    dataset = KneeSSLDataset(study_ids, series_meta, series_root, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.num_workers, drop_last=True)

    model = SimCLRModel(backbone=args.backbone, pretrained=args.pretrained,
                         projection_dim=args.projection_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_dir = Path(args.checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.logs_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    latest = find_latest_checkpoint(ckpt_dir)
    if latest is not None and not args.no_resume:
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {latest} - starting at epoch {start_epoch}")
        if start_epoch > args.epochs:
            print(f"Checkpoint epoch {ckpt['epoch']} already >= --epochs {args.epochs}; nothing to do.")
            return

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"ssl_pretrain_{run_id}.csv"
    write_header = not log_path.exists()
    if write_header:
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "lr", "epoch_seconds"])

    t_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        t_epoch = time.time()
        model.train()
        loss_sum, n_batches = 0.0, 0
        for view1, view2 in loader:
            view1, view2 = view1.to(device), view2.to(device)
            optimizer.zero_grad()
            z1, z2 = model(view1), model(view2)
            loss = nt_xent_loss(z1, z2, temperature=args.temperature)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            n_batches += 1
        scheduler.step()
        train_loss = loss_sum / max(n_batches, 1)
        epoch_seconds = time.time() - t_epoch
        current_lr = scheduler.get_last_lr()[0]

        print(f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  lr={current_lr:.2e}  [{epoch_seconds:.1f}s]")
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, train_loss, current_lr, round(epoch_seconds, 2)])

        ckpt = {
            "epoch": epoch, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "train_loss": train_loss, "backbone": args.backbone, "image_size": args.image_size,
        }
        torch.save(ckpt, ckpt_dir / f"ssl_epoch_{epoch}.pt")
        torch.save(ckpt, ckpt_dir / "ssl_latest.pt")
        # Only the backbone (not optimizer/projection head) is needed for
        # fine-tuning - a small separate file so finetune.py doesn't need
        # to know about SimCLRModel's internals at all.
        torch.save({"backbone_state": model.backbone.state_dict(), "backbone": args.backbone,
                     "image_size": args.image_size, "epoch": epoch}, ckpt_dir / "ssl_backbone_latest.pt")

    total_seconds = time.time() - t_start
    print(f"\nTotal pretraining wall-clock this run: {total_seconds:.1f}s ({total_seconds / 60:.2f} min)")
    print(f"Log: {log_path}")
    print(f"Checkpoints in {ckpt_dir} (ssl_epoch_N.pt, ssl_latest.pt, ssl_backbone_latest.pt)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data", help="Root containing train_series/ (image data)")
    parser.add_argument("--series-csv", default=None, help="Override for train_series.csv (default: <data-root>/train_series.csv)")
    parser.add_argument("--checkpoints-dir", default="outputs/checkpoints")
    parser.add_argument("--logs-dir", default="outputs/logs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--pretrained", action="store_true", default=False,
                         help="start from ImageNet weights rather than random init")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-resume", action="store_true", default=False,
                         help="ignore any existing checkpoint and start fresh")
    parser.add_argument("--limit", type=int, default=None, help="cap studies used, for smoke testing")
    args = parser.parse_args()
    main(args)
