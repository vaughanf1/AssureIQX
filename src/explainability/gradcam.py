"""Grad-CAM heatmap generation and overlay utilities.

Provides reusable functions for:
- Denormalizing ImageNet-normalized tensors for visualization
- Generating Grad-CAM heatmaps via pytorch-grad-cam
- Creating heatmap overlays on RGB images
- Loading LabelMe JSON annotations as binary masks
- Computing IoU between Grad-CAM heatmaps and annotation masks
- Selecting TP/FP/FN examples from test predictions
- Building matplotlib image grids for curated galleries

All functions are stateless and operate on numpy arrays or torch tensors.
Used by scripts/gradcam.py (Phase 6) and scripts/infer.py (Phase 6).

Implemented in Phase 6.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pytorch_grad_cam import GradCAM  # noqa: E402
from pytorch_grad_cam.utils.image import show_cam_on_image  # noqa: E402
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget  # noqa: E402

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

logger = logging.getLogger(__name__)

# Precompute numpy arrays for denormalization
_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32)
_STD = np.array(IMAGENET_STD, dtype=np.float32)


def denormalize_tensor(tensor) -> np.ndarray:
    """Convert a normalized CHW tensor to HWC float32 [0,1] numpy array.

    Reverses ImageNet normalization: img = tensor * std + mean, then clips
    to [0, 1] to prevent overflow artifacts in visualization.

    Args:
        tensor: Torch tensor of shape (C, H, W), ImageNet-normalized.

    Returns:
        Numpy array of shape (H, W, C), float32 in [0, 1].
    """
    img = tensor.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
    img = img * _STD + _MEAN
    return np.clip(img, 0, 1).astype(np.float32)


def generate_gradcam(model, input_tensor, target_class_idx: int) -> np.ndarray:
    """Generate a Grad-CAM heatmap for a single input tensor.

    Uses pytorch-grad-cam with context manager pattern for clean hook
    cleanup. Targets the layer returned by ``model.gradcam_target_layer``.

    IMPORTANT: Do NOT wrap calls to this function in ``torch.no_grad()`` --
    Grad-CAM requires gradient computation. The model MUST be in eval mode.

    Args:
        model: BTXRDClassifier instance in eval mode.
        input_tensor: Tensor of shape (1, C, H, W) on the correct device.
        target_class_idx: Class index to target for Grad-CAM.

    Returns:
        Grayscale heatmap of shape (H, W), float32 in [0, 1].
    """
    target_layers = [model.gradcam_target_layer]
    targets = [ClassifierOutputTarget(target_class_idx)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        heatmap = grayscale_cam[0]  # (H, W)

    if heatmap.max() == 0.0:
        logger.warning(
            "Grad-CAM heatmap is all zeros for class %d. "
            "This may indicate a degenerate case (very high confidence "
            "or wrong target layer).",
            target_class_idx,
        )

    return heatmap.astype(np.float32)


def create_overlay(rgb_img: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Overlay a Grad-CAM heatmap onto an RGB image.

    Uses ``show_cam_on_image`` from pytorch-grad-cam for proper colormap
    application and alpha blending.

    Args:
        rgb_img: HWC float32 image in [0, 1].
        heatmap: HW float32 heatmap in [0, 1].

    Returns:
        HWC uint8 overlay image in [0, 255].
    """
    return show_cam_on_image(rgb_img, heatmap, use_rgb=True)


def load_annotation_mask(ann_path: str | Path, target_size: int = 224) -> np.ndarray:
    """Load a LabelMe JSON annotation and rasterize to a binary mask.

    Supports both polygon and rectangle shape types from LabelMe exports.
    The mask is created at original image resolution, then resized to
    ``target_size x target_size`` with nearest-neighbor interpolation to
    preserve binary values.

    Args:
        ann_path: Path to LabelMe JSON annotation file.
        target_size: Target mask size (square). Default 224 to match
            model input size.

    Returns:
        Binary uint8 mask of shape (target_size, target_size) with
        values 0 or 1.
    """
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
        else:
            logger.warning(
                "Unknown shape type '%s' in %s, skipping.", shape_type, ann_path
            )

    mask_resized = cv2.resize(
        mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST
    )
    return mask_resized


def compute_cam_iou(
    cam_heatmap: np.ndarray,
    annotation_mask: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute IoU between a thresholded Grad-CAM heatmap and annotation mask.

    Binarizes the heatmap at the given threshold, then computes the
    Intersection over Union (Jaccard index) with the annotation mask.

    Args:
        cam_heatmap: (H, W) float32 heatmap in [0, 1].
        annotation_mask: (H, W) binary uint8 mask (0 or 1).
        threshold: Binarization threshold for the heatmap. Default 0.5.

    Returns:
        IoU score as float in [0, 1]. Returns 0.0 if union is zero.
    """
    cam_binary = (cam_heatmap >= threshold).astype(np.uint8)
    intersection = np.logical_and(cam_binary, annotation_mask).sum()
    union = np.logical_or(cam_binary, annotation_mask).sum()
    return float(intersection / union) if union > 0 else 0.0


def select_examples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_idx: int,
    k: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select top-k TP, FP, FN indices for a given class.

    Selection strategy:
    - TP: sorted by highest confidence for class (most confident correct)
    - FP: sorted by highest confidence for class (most confident mistakes)
    - FN: sorted by lowest confidence for true class (model was most wrong)

    Args:
        y_true: Ground truth labels, shape (N,).
        y_pred: Predicted labels, shape (N,).
        y_prob: Predicted probabilities, shape (N, C).
        class_idx: Target class index.
        k: Number of examples to select per category.

    Returns:
        Tuple of (tp_indices, fp_indices, fn_indices) as numpy arrays.
        Each may contain fewer than k elements if fewer examples exist.
    """
    tp_mask = (y_true == class_idx) & (y_pred == class_idx)
    fp_mask = (y_true != class_idx) & (y_pred == class_idx)
    fn_mask = (y_true == class_idx) & (y_pred != class_idx)

    tp_indices = np.where(tp_mask)[0]
    fp_indices = np.where(fp_mask)[0]
    fn_indices = np.where(fn_mask)[0]

    # TP: highest confidence correct predictions
    tp_sorted = tp_indices[np.argsort(-y_prob[tp_indices, class_idx])][
        : min(k, len(tp_indices))
    ]
    # FP: highest confidence false positives (most confident mistakes)
    fp_sorted = fp_indices[np.argsort(-y_prob[fp_indices, class_idx])][
        : min(k, len(fp_indices))
    ]
    # FN: lowest confidence for true class (model was most wrong)
    fn_sorted = fn_indices[np.argsort(y_prob[fn_indices, class_idx])][
        : min(k, len(fn_indices))
    ]

    return tp_sorted, fp_sorted, fn_sorted


def build_gallery_grid(
    images: list[np.ndarray],
    titles: list[str],
    rows: int,
    cols: int,
    output_path: str | Path,
    suptitle: str = "",
) -> None:
    """Create a matplotlib figure grid of images with titles.

    Unused subplots are hidden. The figure is saved at DPI 150 and
    closed immediately to prevent memory leaks.

    Args:
        images: List of (H, W, 3) images (uint8 or float32).
        titles: List of title strings for each subplot.
        rows: Number of grid rows.
        cols: Number of grid columns.
        output_path: Path to save the PNG figure.
        suptitle: Optional figure super-title.
    """
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    # Normalize axes to 2D array for uniform indexing
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    for idx, (img, title) in enumerate(zip(images, titles)):
        r, c = divmod(idx, cols)
        axes[r][c].imshow(img)
        axes[r][c].set_title(title, fontsize=9)
        axes[r][c].axis("off")

    # Hide unused subplots
    for idx in range(len(images), rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold")

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved gallery grid: %s", output_path)
