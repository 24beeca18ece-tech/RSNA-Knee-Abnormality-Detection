"""Step 1 plumbing-test driver, run inside a Kaggle kernel with the full
competition dataset attached (competition_sources) and this repo's src/
package attached as a Dataset (dataset_sources: dograbrij/rsna-knee-src).

Trains the v0 image baseline on ONLY the 58 rows with a real structured
label, then runs inference on the test set. Score is expected to be noise
- see docs/baseline_plan.md "Decided sequencing". Purpose is proving
DICOM loading -> training -> checkpointing -> submission.csv works
end-to-end on Kaggle's infra, not producing a good model.

Imports this repo's actual src/ modules (not a duplicated reimplementation)
so local dev and the Kaggle run share the same code path.
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
        raise FileNotFoundError(
            f"Couldn't find the attached src/ package dataset. Tried {SRC_DATASET_CANDIDATES}, "
            f"walked and found src/ candidates: {found}. /kaggle/input contents: {sorted(os.listdir('/kaggle/input'))}"
        )
print(f"src_mount: {src_mount}, contents: {sorted(os.listdir(src_mount))}")
sys.path.insert(0, src_mount)
from src.training import train_baseline  # noqa: E402
from src.inference import predict  # noqa: E402

SLUG = "rsna-knee-abnormality-detection"
CANDIDATE_DATA_DIRS = [f"/kaggle/input/{SLUG}", f"/kaggle/input/competitions/{SLUG}"]


def find_data_dir() -> str:
    for d in CANDIDATE_DATA_DIRS:
        if os.path.isdir(d):
            return d
    for root, dirs, files in os.walk("/kaggle/input"):
        if "train.csv" in files:
            return root
    raise FileNotFoundError(f"Couldn't locate competition data under /kaggle/input. Tried {CANDIDATE_DATA_DIRS}.")


def main() -> None:
    t0 = time.time()
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")

    data_dir = find_data_dir()
    print(f"DATA_DIR: {data_dir}")

    os.makedirs("/kaggle/working/checkpoints", exist_ok=True)
    os.makedirs("/kaggle/working/logs", exist_ok=True)

    train_args = argparse.Namespace(
        data_root=data_dir, labels_csv=None, series_csv=None,
        checkpoints_dir="/kaggle/working/checkpoints", logs_dir="/kaggle/working/logs",
        epochs=8, batch_size=8, lr=3e-4, image_size=224, val_frac=0.2, seed=42,
        backbone="resnet18", num_workers=2, pretrained=True,
    )
    print("\n=== Training (Step 1: 58 labeled rows only) ===")
    t_train = time.time()
    train_baseline.main(train_args)
    print(f"Training phase wall-clock: {time.time() - t_train:.1f}s")

    predict_args = argparse.Namespace(
        data_root=data_dir, test_csv=None, series_csv=None, sample_submission=None,
        checkpoint="/kaggle/working/checkpoints/best.pt", output="/kaggle/working/submission.csv",
        batch_size=8, num_workers=2,
    )
    print("\n=== Inference ===")
    t_infer = time.time()
    predict.main(predict_args)
    print(f"Inference phase wall-clock: {time.time() - t_infer:.1f}s")

    print(f"\nTOTAL wall-clock: {time.time() - t0:.1f}s ({(time.time() - t0) / 60:.2f} min)")


if __name__ == "__main__":
    main()
