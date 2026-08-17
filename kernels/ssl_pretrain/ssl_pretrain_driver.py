"""Step 3 SSL pretraining driver, run inside a Kaggle kernel with the full
competition dataset attached and this repo's src/ package attached as a
Dataset. This FIRST run is a validation pass (small --limit, few epochs) to
prove the pipeline works end-to-end on Kaggle's real infra (full DICOM
data, real GPU) before committing to the full multi-session pretraining run
- see docs/baseline_plan.md "Step 3".

enable_internet=false in kernel-metadata.json - SSL pretraining starts from
random init (pretrained=False), no ImageNet weights needed, so this can
already run the way the final scored inference notebook will have to.
"""

import argparse
import os
import sys
import time

import torch

print("/kaggle/input contents:", sorted(os.listdir("/kaggle/input")) if os.path.isdir("/kaggle/input") else "MISSING")

SRC_DATASET_CANDIDATES = [
    "/kaggle/input/rsna-knee-src",
    "/kaggle/input/datasets/rsna-knee-src",
    "/kaggle/input/datasets/dograbrij/rsna-knee-src",
]
src_mount = next((d for d in SRC_DATASET_CANDIDATES if os.path.isdir(d) and os.path.isdir(os.path.join(d, "src"))), None)
if src_mount is None:
    found = [root for root, dirs, files in os.walk("/kaggle/input") if os.path.basename(root) == "src" and "__init__.py" in files]
    if len(found) == 1:
        src_mount = os.path.dirname(found[0])
    else:
        raise FileNotFoundError(f"Couldn't find the attached src/ package dataset. found={found}")
print(f"src_mount: {src_mount}")
sys.path.insert(0, src_mount)

from src.training import train_ssl_pretrain  # noqa: E402

SLUG = "rsna-knee-abnormality-detection"
CANDIDATE_DATA_DIRS = [f"/kaggle/input/{SLUG}", f"/kaggle/input/competitions/{SLUG}"]


def find_data_dir() -> str:
    for d in CANDIDATE_DATA_DIRS:
        if os.path.isdir(d):
            return d
    for root, dirs, files in os.walk("/kaggle/input"):
        if "train_series.csv" in files:
            return root
    raise FileNotFoundError(f"Couldn't locate competition data. Tried {CANDIDATE_DATA_DIRS}.")


def main() -> None:
    t0 = time.time()
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")

    data_dir = find_data_dir()
    print(f"DATA_DIR: {data_dir}")

    os.makedirs("/kaggle/working/checkpoints", exist_ok=True)
    os.makedirs("/kaggle/working/logs", exist_ok=True)

    # Validation pass: small subset + few epochs, to prove the full
    # pipeline (data loading -> augmentation -> SimCLR -> checkpointing)
    # works on Kaggle's real infra before the full multi-session run.
    args = argparse.Namespace(
        data_root=data_dir, series_csv=None,
        checkpoints_dir="/kaggle/working/checkpoints", logs_dir="/kaggle/working/logs",
        epochs=3, batch_size=64, lr=3e-4, weight_decay=1e-6, temperature=0.5,
        image_size=224, projection_dim=128, backbone="resnet18", pretrained=False,
        num_workers=2, no_resume=False, limit=200,
    )
    train_ssl_pretrain.main(args)

    print(f"\nTOTAL wall-clock: {time.time() - t0:.1f}s ({(time.time() - t0) / 60:.2f} min)")


if __name__ == "__main__":
    main()
