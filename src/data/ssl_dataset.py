"""Unlabeled dataset for Step 3 self-supervised (SimCLR) pretraining.
See docs/baseline_plan.md "Step 3" and src/models/ssl_model.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset

from src.preprocessing.dicom_utils import load_study_slice_pil

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_simclr_augmentations(image_size: int) -> transforms.Compose:
    """Standard SimCLR augmentation recipe, adapted for grayscale-derived
    knee MRI slices: no hue/saturation jitter (meaningless on a replicated
    grayscale image), and a milder random-crop scale range than SimCLR's
    usual (0.2, 1.0) - the knee joint is roughly centered and a very
    aggressive crop risks cropping it out entirely rather than just
    changing viewpoint/scale, unlike SimCLR's usual natural-image setting."""
    blur_kernel = max(3, (int(0.1 * image_size) // 2) * 2 + 1)
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4),
        transforms.RandomApply([transforms.GaussianBlur(blur_kernel, sigma=(0.1, 2.0))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class KneeSSLDataset(Dataset):
    """One pair of independently-augmented views per study, no labels -
    works for ANY study (labeled or not), which is the entire point of
    Step 3: use every one of the 4407 studies' images, not just the 813
    with a weak/structured label so far.
    """

    def __init__(self, study_ids: list[str], series_meta_df, series_root: Path, image_size: int = 224):
        self.study_ids = list(study_ids)
        self.series_meta_df = series_meta_df
        self.series_root = Path(series_root)
        self.image_size = image_size
        self.augment = build_simclr_augmentations(image_size)

    def __len__(self) -> int:
        return len(self.study_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        study_uid = self.study_ids[idx]
        img = load_study_slice_pil(study_uid, self.series_meta_df, self.series_root, self.image_size)
        img_rgb = img.convert("RGB")
        return self.augment(img_rgb), self.augment(img_rgb)
