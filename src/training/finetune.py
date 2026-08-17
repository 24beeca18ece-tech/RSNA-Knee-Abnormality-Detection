"""Step 3 fine-tuning: load a Step-3-pretrained (SimCLR) backbone and train
a classifier head on the combined labeled set from Step 2
(data/processed/combined_training_labels.csv - 58 structured + all weak
rows extracted so far). See docs/baseline_plan.md "Step 3".

Validation design: held out against the 58 REAL structured rows specifically
(label_source == "structured"), not a random split of the mixed pool - the
58 gold rows are the only fully-trustworthy labels available, so they're a
much more meaningful val set than a random slice that's itself mostly weak
labels. Training happens on the weak rows (optionally excluding
low-confidence ones with --exclude-low-confidence).

Usage:
    python -m src.training.finetune --data-root data --ssl-checkpoint outputs/checkpoints/ssl_backbone_latest.pt --epochs 20
    python -m src.training.finetune --data-root data --epochs 20   # no SSL checkpoint = ImageNet/random init baseline for comparison
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
from torch.utils.data import DataLoader

from src.data.dataset import KneeSliceDataset
from src.models.image_baseline import KneeImageBaseline
from src.training.train_baseline import TARGETS, macro_auc, pick_device

ID_COL = "StudyInstanceUID"


def load_ssl_backbone(model: KneeImageBaseline, ssl_checkpoint: Path, device: torch.device) -> None:
    """Accepts either checkpoint shape train_ssl_pretrain.py produces:
    - ssl_backbone_latest.pt: {"backbone_state": model.backbone.state_dict(), ...}
      (the intended lightweight fine-tuning artifact - no prefix stripping needed)
    - ssl_epoch_N.pt / ssl_latest.pt: {"model_state": SimCLRModel.state_dict(), ...}
      (the full checkpoint incl. optimizer/scheduler - keys are prefixed
      "backbone."/"projection_head." since it's the whole SimCLRModel, so
      only the "backbone."-prefixed subset is extracted and the prefix
      stripped before loading).
    Raises a clear error if neither expected key is present, rather than a
    bare KeyError, and reports how many backbone parameter tensors were
    actually loaded so a silent partial/empty load isn't mistaken for success.
    """
    ckpt = torch.load(ssl_checkpoint, map_location=device, weights_only=False)
    if "backbone_state" in ckpt:
        backbone_state = ckpt["backbone_state"]
    elif "model_state" in ckpt:
        prefix = "backbone."
        backbone_state = {k[len(prefix):]: v for k, v in ckpt["model_state"].items() if k.startswith(prefix)}
        if not backbone_state:
            raise KeyError(f"{ssl_checkpoint} has 'model_state' but no keys prefixed 'backbone.' - "
                            f"found prefixes: {sorted({k.split('.')[0] for k in ckpt['model_state']})}")
    else:
        raise KeyError(f"{ssl_checkpoint} has neither 'backbone_state' nor 'model_state' - "
                        f"found top-level keys: {sorted(ckpt.keys())}")

    missing, unexpected = model.backbone.load_state_dict(backbone_state, strict=True)
    assert not missing and not unexpected, (missing, unexpected)  # strict=True already raises on mismatch; belt and suspenders
    n_tensors = len(backbone_state)
    print(f"Loaded SSL-pretrained backbone from {ssl_checkpoint} "
          f"(SSL epoch {ckpt.get('epoch', '?')}, {n_tensors} backbone tensors)")


def main(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    device = pick_device()
    print(f"device: {device}")

    combined_path = Path(args.combined_csv)
    df = pd.read_csv(combined_path)
    print(f"combined dataset: {len(df)} rows ({combined_path})")
    print(df["label_source"].value_counts().to_string())

    val_df = df[df["label_source"] == "structured"].reset_index(drop=True)
    train_df = df[df["label_source"] != "structured"].reset_index(drop=True)
    if args.exclude_low_confidence:
        before = len(train_df)
        train_df = train_df[train_df["confidence"] != "low"].reset_index(drop=True)
        print(f"--exclude-low-confidence: dropped {before - len(train_df)} rows")
    print(f"train: {len(train_df)} (weak-labeled)  val: {len(val_df)} (structured gold)")

    series_csv_path = Path(args.series_csv) if args.series_csv else data_root / "train_series.csv"
    series_meta = pd.read_csv(series_csv_path)
    series_root = data_root / "train_series"

    train_ds = KneeSliceDataset(train_df, series_meta, series_root, TARGETS, image_size=args.image_size)
    val_ds = KneeSliceDataset(val_df, series_meta, series_root, TARGETS, image_size=args.image_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = KneeImageBaseline(n_targets=len(TARGETS), backbone=args.backbone,
                               pretrained=args.pretrained and not args.ssl_checkpoint).to(device)
    if args.ssl_checkpoint:
        load_ssl_backbone(model, Path(args.ssl_checkpoint), device)

    if args.freeze_backbone_epochs > 0:
        for p in model.backbone.parameters():
            p.requires_grad = False
        print(f"Backbone frozen for the first {args.freeze_backbone_epochs} epoch(s) (linear probe first).")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    ckpt_dir = Path(args.checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.logs_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"finetune_{run_id}.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_macro_auc", "n_targets_with_auc", "epoch_seconds"])

    best_auc = float("-inf")
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_backbone_epochs + 1 and args.freeze_backbone_epochs > 0:
            for p in model.backbone.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            print(f"Unfroze backbone at epoch {epoch}.")

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

        ckpt = {
            "epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "val_macro_auc": val_auc, "targets": TARGETS, "backbone": args.backbone, "image_size": args.image_size,
            "ssl_checkpoint": args.ssl_checkpoint,
        }
        torch.save(ckpt, ckpt_dir / f"finetune_epoch_{epoch}.pt")
        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            torch.save(ckpt, ckpt_dir / "finetune_best.pt")
            print(f"  -> new finetune_best.pt (val_macro_auc={val_auc:.4f})")

    torch.save(ckpt, ckpt_dir / "finetune_last.pt")
    total_seconds = time.time() - t_start
    print(f"\nTotal fine-tuning wall-clock: {total_seconds:.1f}s ({total_seconds / 60:.2f} min)")
    print(f"Best val_macro_auc: {best_auc:.4f}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data", help="Root containing train_series/ (image data)")
    parser.add_argument("--combined-csv", default="data/processed/combined_training_labels.csv")
    parser.add_argument("--series-csv", default=None, help="Override for train_series.csv (default: <data-root>/train_series.csv)")
    parser.add_argument("--ssl-checkpoint", default=None, help="Path to ssl_backbone_latest.pt from train_ssl_pretrain.py")
    parser.add_argument("--checkpoints-dir", default="outputs/checkpoints")
    parser.add_argument("--logs-dir", default="outputs/logs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--pretrained", action="store_true", default=False,
                         help="ImageNet init if no --ssl-checkpoint given (ignored if --ssl-checkpoint is set)")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0,
                         help="linear-probe the head for this many epochs before unfreezing the backbone")
    parser.add_argument("--exclude-low-confidence", action="store_true", default=False)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    main(args)
