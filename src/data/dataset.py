"""Minimal single-slice-per-study PyTorch Dataset for the v0 image baseline.
See docs/baseline_plan.md "Data handling" / "Step 1" for the design rationale.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing.dicom_utils import ID_COL, load_study_slice_pil

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class KneeSliceDataset(Dataset):
    """One (image, label) pair per study: a single representative DICOM
    slice, resized/normalized, paired with its 12 target labels.

    Missing data (no series metadata, no series folder, no slice files) is
    treated as a hard error, not silently skipped or zero-filled - for a
    tiny 58-row plumbing run every row matters, and this makes gaps in the
    attached dataset visible immediately rather than producing a silently
    smaller effective batch.
    """

    def __init__(self, labels_df, series_meta_df, series_root: Path, targets: list[str], image_size: int = 224):
        self.labels_df = labels_df.reset_index(drop=True)
        self.series_meta_df = series_meta_df
        self.series_root = Path(series_root)
        self.targets = targets
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.labels_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.labels_df.iloc[idx]
        study_uid = row[ID_COL]

        img = load_study_slice_pil(study_uid, self.series_meta_df, self.series_root, self.image_size)
        img_t = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0).repeat(3, 1, 1)
        img_t = (img_t - IMAGENET_MEAN) / IMAGENET_STD

        label = torch.tensor(row[self.targets].to_numpy(dtype=np.float32))
        return img_t, label
