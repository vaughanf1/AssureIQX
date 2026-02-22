"""PyTorch dataset class for BTXRD bone tumor radiographs.

Loads images from split manifest CSVs (produced by Phase 3 splitting) and
returns (tensor, label_index) tuples suitable for DataLoader batching.

Images are lazily loaded in __getitem__ to avoid holding all images in memory.
Albumentations transforms are applied per-sample, enabling different
augmentation pipelines for train/val/test modes.

Implemented in Phase 3.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# 3-class label mapping for BTXRD bone tumor classification
CLASS_TO_IDX: dict[str, int] = {"Normal": 0, "Benign": 1, "Malignant": 2}
IDX_TO_CLASS: dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}


class BTXRDDataset(Dataset):
    """PyTorch Dataset for BTXRD bone tumor radiograph classification.

    Reads a split manifest CSV with columns (image_id, split, label) and
    loads images from a directory on demand. Each sample returns a
    (tensor, label_index) tuple.

    Args:
        manifest_csv: Path to split manifest CSV (image_id, split, label).
        images_dir: Directory containing the radiograph image files.
        transform: Optional albumentations Compose pipeline to apply.

    Raises:
        ValueError: If manifest is missing required columns or contains
            unknown labels.
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        images_dir: str | Path,
        transform: object | None = None,
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        self.images_dir = Path(images_dir)
        self.transform = transform

        self.df = pd.read_csv(self.manifest_csv)

        # Validate required columns
        required_columns = {"image_id", "label"}
        missing = required_columns - set(self.df.columns)
        if missing:
            raise ValueError(
                f"Manifest CSV is missing required columns: {missing}. "
                f"Found columns: {list(self.df.columns)}"
            )

        # Validate all labels are known
        unknown_labels = set(self.df["label"].unique()) - set(CLASS_TO_IDX.keys())
        if unknown_labels:
            raise ValueError(
                f"Unknown labels found in manifest: {unknown_labels}. "
                f"Expected one of: {list(CLASS_TO_IDX.keys())}"
            )

        logger.info(
            "Loaded %s: %d samples, class distribution: %s",
            self.manifest_csv.name,
            len(self.df),
            self.df["label"].value_counts().to_dict(),
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Load and transform a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image_tensor, label_index) where image_tensor has
            shape (3, H, W) float32 and label_index is 0/1/2.
        """
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        label_str = row["label"]
        label_idx = CLASS_TO_IDX[label_str]

        # Load image and convert to RGB (handles grayscale radiographs)
        image_path = self.images_dir / image_id
        image = Image.open(image_path).convert("RGB")

        # Convert to numpy HWC uint8 array for albumentations
        image = np.array(image)

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label_idx

    @property
    def class_counts(self) -> dict[str, int]:
        """Return per-class sample counts from the manifest.

        Returns:
            Dict mapping class name to count, e.g.
            {"Normal": 500, "Benign": 300, "Malignant": 200}.
        """
        return self.df["label"].value_counts().to_dict()

    @property
    def labels(self) -> list[int]:
        """Return list of integer label indices for all samples.

        Useful for computing class weights or stratified sampling in
        Phase 4 training loop.

        Returns:
            List of label indices (0, 1, or 2) in manifest order.
        """
        return [CLASS_TO_IDX[label] for label in self.df["label"].tolist()]


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader with standard settings for BTXRD training.

    Args:
        dataset: PyTorch Dataset instance.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle samples each epoch.
        num_workers: Number of parallel data loading workers.
        pin_memory: Whether to pin memory for faster GPU transfer.

    Returns:
        Configured DataLoader instance.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def load_annotation_mask_raw(ann_path: str | Path) -> np.ndarray | None:
    """Load a LabelMe JSON annotation and rasterize to a binary mask.

    Returns the mask at the original image dimensions (no resizing).
    Supports polygon and rectangle shape types.

    Args:
        ann_path: Path to LabelMe JSON annotation file.

    Returns:
        Binary uint8 mask of shape (H, W) with values 0 or 1,
        or None if the file cannot be loaded.
    """
    ann_path = Path(ann_path)
    if not ann_path.exists():
        return None

    with open(ann_path) as f:
        data = json.load(f)

    h = data["imageHeight"]
    w = data["imageWidth"]
    mask = np.zeros((h, w), dtype=np.uint8)

    for shape in data["shapes"]:
        pts = np.array(shape["points"], dtype=np.int32)
        shape_type = shape["shape_type"]

        if shape_type == "polygon":
            cv2.fillPoly(mask, [pts], 1)
        elif shape_type == "rectangle":
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 1, -1)

    return mask


class BTXRDAnnotatedDataset(Dataset):
    """Annotation-aware dataset that adds ROI-guided samples for tumor images.

    For tumor images (Benign/Malignant) with LabelMe annotations, this dataset
    creates an additional ROI-guided sample where non-tumor regions are dimmed
    by 50%. This effectively doubles the tumor class samples (helping class
    imbalance) and guides model attention toward annotated regions.

    Normal images are passed through unchanged (they have no annotations).

    Args:
        manifest_csv: Path to split manifest CSV (image_id, split, label).
        images_dir: Directory containing the radiograph image files.
        annotations_dir: Directory containing LabelMe JSON annotations.
        transform: Optional albumentations Compose pipeline to apply.
        dim_factor: Factor to dim non-ROI regions (0.5 = 50% dimming).
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        images_dir: str | Path,
        annotations_dir: str | Path,
        transform: object | None = None,
        dim_factor: float = 0.5,
    ) -> None:
        self.base = BTXRDDataset(manifest_csv, images_dir, transform=None)
        self.annotations_dir = Path(annotations_dir)
        self.transform = transform
        self.dim_factor = dim_factor

        # Build index: original samples + ROI-guided duplicates for annotated tumors
        self._index: list[tuple[int, bool]] = []  # (base_idx, is_roi_guided)
        self._annotation_cache: dict[int, np.ndarray | None] = {}

        for i in range(len(self.base)):
            self._index.append((i, False))  # original sample

            label_str = self.base.df.iloc[i]["label"]
            if label_str in ("Benign", "Malignant"):
                image_id = self.base.df.iloc[i]["image_id"]
                ann_stem = Path(image_id).stem
                ann_path = self.annotations_dir / f"{ann_stem}.json"
                mask = load_annotation_mask_raw(ann_path)
                if mask is not None:
                    self._annotation_cache[i] = mask
                    self._index.append((i, True))  # ROI-guided duplicate

        n_roi = sum(1 for _, is_roi in self._index if is_roi)
        logger.info(
            "BTXRDAnnotatedDataset: %d base + %d ROI-guided = %d total samples",
            len(self.base),
            n_roi,
            len(self._index),
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        base_idx, is_roi_guided = self._index[idx]
        row = self.base.df.iloc[base_idx]
        label_idx = CLASS_TO_IDX[row["label"]]

        image_path = self.base.images_dir / row["image_id"]
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        if is_roi_guided:
            mask = self._annotation_cache[base_idx]
            # Resize mask to match image dimensions
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(
                    mask, (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            # Dim non-ROI regions
            dimmed = (image * self.dim_factor).astype(np.uint8)
            mask_3ch = np.stack([mask] * 3, axis=-1)
            image = np.where(mask_3ch, image, dimmed)

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label_idx

    @property
    def labels(self) -> list[int]:
        """Return list of integer label indices for all samples (including ROI-guided)."""
        result = []
        for base_idx, _ in self._index:
            label_str = self.base.df.iloc[base_idx]["label"]
            result.append(CLASS_TO_IDX[label_str])
        return result
