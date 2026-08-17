"""DICOM series/slice selection and loading for the v0 single-slice baseline.
See docs/baseline_plan.md "Data handling" for the rationale behind each choice.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image

ID_COL = "StudyInstanceUID"
SERIES_ID_COL = "SeriesInstanceUID"


def winlong(p: Path) -> Path:
    """Windows MAX_PATH (260 char) workaround - this dataset's nested DICOM
    UID folder names routinely exceed it locally. No-op on non-Windows
    (Kaggle kernels run Linux, where this is never needed)."""
    if os.name != "nt":
        return p
    s = str(p.resolve())
    return Path(s if s.startswith("\\\\?\\") else "\\\\?\\" + s)


def select_series(study_uid: str, series_meta_df) -> str | None:
    """Pick one series per study: prefer Sagittal + fluid-sensitive (most
    relevant to ACL/meniscus/cartilage/effusion findings per the challenge
    targets), fall back to any Sagittal series, then to whatever's first.
    Returns None if the study has no series metadata at all.
    """
    rows = series_meta_df[series_meta_df[ID_COL] == study_uid]
    if rows.empty:
        return None
    sagittal_fs = rows[(rows["Anatomical_Plane"] == "Sagittal") & (rows["Fluid_Sensitive"] == 1)]
    if not sagittal_fs.empty:
        return sagittal_fs.iloc[0][SERIES_ID_COL]
    sagittal = rows[rows["Anatomical_Plane"] == "Sagittal"]
    if not sagittal.empty:
        return sagittal.iloc[0][SERIES_ID_COL]
    return rows.iloc[0][SERIES_ID_COL]


def pick_middle_slice(series_dir: Path) -> Path | None:
    """Middle slice by filename-sorted index - cheap, no DICOM header reads
    needed (see docs/baseline_plan.md). NOTE: DICOM filenames here are
    SOPInstanceUIDs (effectively random), so filename order is NOT the same
    as anatomical slice order - this picks *a* consistent, deterministic
    slice per series, not necessarily the true middle of the stack. That's
    an accepted v0 simplification (see baseline_plan.md "v0.1 improvement").
    """
    files = sorted(Path(winlong(series_dir)).glob("*.dcm"))
    if not files:
        return None
    return files[len(files) // 2]


def load_slice_array(dcm_path: Path) -> np.ndarray:
    """Read one DICOM slice's pixel data as a float32 2D array, with
    RescaleSlope/RescaleIntercept applied if present (defaults 1.0/0.0)."""
    ds = pydicom.dcmread(winlong(Path(dcm_path)))
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return arr * slope + intercept


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Per-image min-max normalization to [0, 255] uint8. Simple and robust
    starting point for v0 - see baseline_plan.md for revisiting this with
    proper MRI windowing later."""
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255.0).astype(np.uint8)


def load_study_slice_pil(study_uid: str, series_meta_df, series_root: Path, image_size: int) -> Image.Image:
    """One representative grayscale slice for a study, resized to a square
    (image_size, image_size) PIL Image - the shared image-loading step
    behind both the labeled dataset (src/data/dataset.py) and the
    self-supervised pretraining dataset (src/data/ssl_dataset.py), which
    otherwise differ only in what they attach as the target. Raises
    FileNotFoundError if the study has no usable series/slice, same as
    KneeSliceDataset - a gap in the attached data should be visible, not
    silently skipped."""
    series_uid = select_series(study_uid, series_meta_df)
    if series_uid is None:
        raise FileNotFoundError(f"No series metadata for study {study_uid}")
    series_dir = Path(series_root) / study_uid / series_uid
    slice_path = pick_middle_slice(series_dir)
    if slice_path is None:
        raise FileNotFoundError(f"No .dcm files under {series_dir}")
    arr = load_slice_array(slice_path)
    img_u8 = normalize_to_uint8(arr)
    return Image.fromarray(img_u8, mode="L").resize((image_size, image_size), Image.BILINEAR)
