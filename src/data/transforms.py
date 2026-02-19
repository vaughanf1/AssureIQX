"""Albumentations augmentation pipelines for train, validation, and test modes.

Pipeline design rationale:
- CLAHE is applied BEFORE Normalize because CLAHE requires uint8 input.
  Normalize converts to float32, so CLAHE must come first in the pipeline.
- Training augmentations include spatial transforms (flip, rotate) for
  regularization, plus CLAHE for contrast enhancement on radiographs.
- Val/test pipelines are deterministic (resize + normalize only) to ensure
  reproducible evaluation metrics.

Implemented in Phase 3.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet normalization constants (used for transfer learning with
# pretrained backbones like EfficientNet-B0)
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int = 224) -> A.Compose:
    """Training augmentation pipeline.

    Order: CLAHE -> HorizontalFlip -> Rotate -> Resize -> Normalize -> ToTensorV2

    CLAHE must precede Normalize because it operates on uint8 pixel values.
    Spatial transforms (flip, rotate) are applied before resize for efficiency.

    Args:
        image_size: Target height and width for output images.

    Returns:
        Albumentations Compose pipeline producing (C, H, W) float32 tensors.
    """
    return A.Compose(
        [
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=0, fill=0, p=0.5),
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """Validation preprocessing pipeline (deterministic, no augmentation).

    Order: Resize -> Normalize -> ToTensorV2

    Args:
        image_size: Target height and width for output images.

    Returns:
        Albumentations Compose pipeline producing (C, H, W) float32 tensors.
    """
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


# Test transforms are identical to validation transforms (deterministic)
get_test_transforms = get_val_transforms
