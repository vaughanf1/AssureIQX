#!/usr/bin/env python3
"""Generate Grad-CAM heatmaps for selected examples.

Loads the best checkpoint, selects representative images per class,
and generates Grad-CAM overlay visualisations saved to
``paths.results_dir/gradcam/``.

Outputs:
  - gallery_Normal.png     -- TP/FP/FN gallery grid for Normal class
  - gallery_Benign.png     -- TP/FP/FN gallery grid for Benign class
  - gallery_Malignant.png  -- TP/FP/FN gallery grid for Malignant class
  - annotation_comparison.png -- 4-panel comparison for tumor images
  - annotation_report.json -- IoU scores and summary statistics

Usage:
    python scripts/gradcam.py --config configs/default.yaml

Implemented in: Phase 6
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import BTXRDDataset, IDX_TO_CLASS  # noqa: E402
from src.data.transforms import get_test_transforms  # noqa: E402
from src.evaluation.metrics import run_inference  # noqa: E402
from src.explainability.gradcam import (  # noqa: E402
    build_gallery_grid,
    compute_cam_iou,
    create_overlay,
    denormalize_tensor,
    generate_gradcam,
    load_annotation_mask,
    select_examples,
)
from src.models.factory import create_model, get_device, load_checkpoint  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.reproducibility import set_seed  # noqa: E402

logger = logging.getLogger(__name__)


def _json_serializable(obj: object) -> object:
    """JSON serialization helper for numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")


def _generate_class_gallery(
    class_idx: int,
    class_name: str,
    tp_indices: np.ndarray,
    fp_indices: np.ndarray,
    fn_indices: np.ndarray,
    dataset: BTXRDDataset,
    model: torch.nn.Module,
    device: torch.device,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
) -> None:
    """Generate a per-class gallery grid with TP, FP, FN rows.

    Each row shows Grad-CAM overlay images with titles indicating
    true class, predicted class, and confidence.

    Args:
        class_idx: Target class index.
        class_name: Target class name.
        tp_indices: Dataset indices for true positives.
        fp_indices: Dataset indices for false positives.
        fn_indices: Dataset indices for false negatives.
        dataset: Test dataset instance.
        model: Trained model in eval mode.
        device: Torch device.
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities.
        output_path: Path to save the gallery PNG.
    """
    categories = [
        ("TP", tp_indices),
        ("FP", fp_indices),
        ("FN", fn_indices),
    ]

    # Determine grid dimensions
    max_cols = max(len(tp_indices), len(fp_indices), len(fn_indices), 1)
    rows = 3  # TP, FP, FN

    images = []
    titles = []

    for cat_label, indices in categories:
        for idx in indices:
            tensor, _ = dataset[idx]
            input_tensor = tensor.unsqueeze(0).to(device)

            # Target the PREDICTED class for Grad-CAM
            pred_cls = int(y_pred[idx])
            heatmap = generate_gradcam(model, input_tensor, pred_cls)

            rgb_img = denormalize_tensor(tensor)
            overlay = create_overlay(rgb_img, heatmap)

            images.append(overlay)
            true_name = IDX_TO_CLASS[int(y_true[idx])]
            pred_name = IDX_TO_CLASS[pred_cls]
            prob = y_prob[idx, pred_cls]
            titles.append(f"{cat_label}: {true_name}->{pred_name} ({prob:.2f})")

        # Pad with empty slots if fewer examples than max_cols
        for _ in range(max_cols - len(indices)):
            images.append(np.zeros((224, 224, 3), dtype=np.uint8))
            titles.append("")

    build_gallery_grid(
        images,
        titles,
        rows,
        max_cols,
        output_path,
        suptitle=f"{class_name} -- TP / FP / FN Gallery",
    )


def _generate_annotation_comparison(
    dataset: BTXRDDataset,
    model: torch.nn.Module,
    device: torch.device,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    annotations_dir: Path,
    output_dir: Path,
    image_size: int = 224,
    max_images: int = 5,
) -> dict:
    """Generate annotation comparison panels for tumor images.

    Selects correctly classified tumor images (Benign TP + Malignant TP)
    that have LabelMe annotations, generates Grad-CAM heatmaps, and
    creates 4-panel comparison figures.

    Args:
        dataset: Test dataset instance.
        model: Trained model in eval mode.
        device: Torch device.
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities.
        annotations_dir: Directory containing LabelMe JSON annotations.
        output_dir: Directory to save comparison figures and report.
        image_size: Model input size for mask resizing.
        max_images: Maximum number of comparison panels to generate.

    Returns:
        Annotation report dict with per-image IoU scores and summary.
    """
    # Find correctly classified tumor images (Benign=1, Malignant=2)
    tumor_tp_indices = []
    for class_idx in [1, 2]:  # Benign, Malignant
        tp_mask = (y_true == class_idx) & (y_pred == class_idx)
        tp_indices = np.where(tp_mask)[0]
        # Sort by confidence (highest first) for most illustrative examples
        sorted_tp = tp_indices[np.argsort(-y_prob[tp_indices, class_idx])]
        tumor_tp_indices.extend(sorted_tp.tolist())

    # Filter to those with annotations
    comparison_indices = []
    for idx in tumor_tp_indices:
        image_id = dataset.df.iloc[idx]["image_id"]
        # Strip image extension and append .json
        ann_stem = Path(image_id).stem
        ann_path = annotations_dir / f"{ann_stem}.json"
        if ann_path.exists():
            comparison_indices.append((idx, ann_path))
        if len(comparison_indices) >= max_images:
            break

    if not comparison_indices:
        logger.warning("No tumor images with annotations found for comparison.")
        return {"per_image": [], "mean_iou": 0.0, "summary": "No images available."}

    logger.info(
        "Generating annotation comparison for %d tumor images.",
        len(comparison_indices),
    )

    per_image_results = []
    n_panels = len(comparison_indices)
    fig, axes = plt.subplots(n_panels, 4, figsize=(20, 5 * n_panels))

    # Handle single panel case (axes is 1D)
    if n_panels == 1:
        axes = np.array([axes])

    for panel_idx, (dataset_idx, ann_path) in enumerate(comparison_indices):
        tensor, _ = dataset[dataset_idx]
        input_tensor = tensor.unsqueeze(0).to(device)

        pred_cls = int(y_pred[dataset_idx])
        true_cls = int(y_true[dataset_idx])
        class_name = IDX_TO_CLASS[true_cls]
        prob = y_prob[dataset_idx, pred_cls]

        # Generate Grad-CAM
        heatmap = generate_gradcam(model, input_tensor, pred_cls)
        rgb_img = denormalize_tensor(tensor)
        overlay = create_overlay(rgb_img, heatmap)

        # Load annotation mask
        ann_mask = load_annotation_mask(ann_path, target_size=image_size)

        # Compute IoU
        iou = compute_cam_iou(heatmap, ann_mask, threshold=0.5)

        image_id = dataset.df.iloc[dataset_idx]["image_id"]
        per_image_results.append(
            {
                "image_id": image_id,
                "class": class_name,
                "confidence": float(prob),
                "iou": float(iou),
            }
        )

        # Plot 4-panel comparison
        axes[panel_idx][0].imshow(rgb_img)
        axes[panel_idx][0].set_title(
            f"Original ({class_name}, p={prob:.2f})", fontsize=10
        )
        axes[panel_idx][0].axis("off")

        axes[panel_idx][1].imshow(ann_mask, cmap="Reds", interpolation="nearest")
        axes[panel_idx][1].set_title("Tumor Annotation", fontsize=10)
        axes[panel_idx][1].axis("off")

        axes[panel_idx][2].imshow(heatmap, cmap="jet", interpolation="nearest")
        axes[panel_idx][2].set_title("Grad-CAM Heatmap", fontsize=10)
        axes[panel_idx][2].axis("off")

        axes[panel_idx][3].imshow(overlay)
        axes[panel_idx][3].set_title(f"Overlay (IoU={iou:.2f})", fontsize=10)
        axes[panel_idx][3].axis("off")

    fig.suptitle(
        "Grad-CAM vs Tumor Annotation Comparison", fontsize=14, fontweight="bold"
    )
    fig.tight_layout()
    comparison_path = output_dir / "annotation_comparison.png"
    fig.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved annotation comparison: %s", comparison_path)

    # Compute summary statistics
    iou_scores = [r["iou"] for r in per_image_results]
    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0

    # Qualitative summary
    focal_count = sum(1 for s in iou_scores if s > 0.1)
    summary = (
        f"Analyzed {len(per_image_results)} correctly classified tumor images. "
        f"Mean IoU between Grad-CAM attention and expert annotations: {mean_iou:.3f}. "
        f"{focal_count}/{len(per_image_results)} images show focal attention "
        f"overlapping annotated tumor regions (IoU > 0.1)."
    )

    report = {
        "per_image": per_image_results,
        "mean_iou": mean_iou,
        "num_images": len(per_image_results),
        "iou_threshold": 0.5,
        "summary": summary,
    }

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM heatmaps for selected examples."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Config overrides in key.subkey=value format",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    cfg = load_config(args.config, overrides=args.override)
    set_seed(cfg.get("seed", 42))

    # Device
    device = get_device(cfg.get("device", "auto"))
    logger.info("Using device: %s", device)

    # Paths
    images_dir = PROJECT_ROOT / cfg["data"]["raw_dir"] / "images"
    annotations_dir = PROJECT_ROOT / cfg["data"]["raw_dir"] / "annotations"
    results_dir = PROJECT_ROOT / cfg["paths"]["results_dir"]
    output_dir = results_dir / "gradcam"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────
    checkpoint_path = PROJECT_ROOT / cfg["inference"]["default_checkpoint"]
    logger.info("Loading checkpoint: %s", checkpoint_path)

    ckpt = load_checkpoint(str(checkpoint_path), device="cpu")
    ckpt_config = ckpt["config"]
    class_names = ckpt["class_names"]

    model = create_model(ckpt_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    logger.info(
        "Model loaded: %s (epoch %d, val_loss=%.6f)",
        ckpt_config["model"]["backbone"],
        ckpt["epoch"],
        ckpt["val_loss"],
    )

    # ── Load test dataset ──────────────────────────────────────
    test_csv = PROJECT_ROOT / cfg["data"]["splits_dir"] / "stratified_test.csv"
    image_size = cfg["data"]["image_size"]
    test_dataset = BTXRDDataset(
        test_csv,
        images_dir,
        get_test_transforms(image_size),
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=False,
    )
    logger.info("Test set: %d samples", len(test_dataset))

    # ── Run inference ──────────────────────────────────────────
    logger.info("Running inference on test set...")
    y_true, y_pred, y_prob = run_inference(model, test_loader, device)

    accuracy = float(np.mean(y_true == y_pred))
    logger.info("Inference complete: %d predictions, accuracy=%.3f", len(y_true), accuracy)

    # Log per-class distribution
    for i, name in enumerate(class_names):
        count = int((y_true == i).sum())
        correct = int(((y_true == i) & (y_pred == i)).sum())
        logger.info("  %s: %d samples, %d correct (%.1f%%)", name, count, correct, 100 * correct / max(count, 1))

    # ── Generate per-class galleries (EXPL-01 + EXPL-02) ──────
    k = cfg["gradcam"]["examples_per_class"]
    logger.info("Generating per-class galleries with k=%d examples...", k)

    for class_idx, class_name in enumerate(class_names):
        tp_indices, fp_indices, fn_indices = select_examples(
            y_true, y_pred, y_prob, class_idx, k=k
        )
        logger.info(
            "  %s: TP=%d, FP=%d, FN=%d (selected from available)",
            class_name,
            len(tp_indices),
            len(fp_indices),
            len(fn_indices),
        )

        if len(tp_indices) == 0 and len(fp_indices) == 0 and len(fn_indices) == 0:
            logger.warning("  No examples found for %s, skipping gallery.", class_name)
            continue

        gallery_path = output_dir / f"gallery_{class_name}.png"
        _generate_class_gallery(
            class_idx=class_idx,
            class_name=class_name,
            tp_indices=tp_indices,
            fp_indices=fp_indices,
            fn_indices=fn_indices,
            dataset=test_dataset,
            model=model,
            device=device,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            output_path=gallery_path,
        )

    # ── Annotation comparison (EXPL-03) ───────────────────────
    logger.info("Generating annotation comparison for tumor images...")
    report = _generate_annotation_comparison(
        dataset=test_dataset,
        model=model,
        device=device,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        annotations_dir=annotations_dir,
        output_dir=output_dir,
        image_size=image_size,
        max_images=5,
    )

    # Save annotation report
    report_path = output_dir / "annotation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=_json_serializable)
    logger.info("Saved annotation report: %s", report_path)

    # ── Summary ────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("GRAD-CAM GENERATION COMPLETE")
    logger.info("=" * 70)
    logger.info("Output directory: %s", output_dir)
    logger.info("Galleries: %s", ", ".join(f"gallery_{n}.png" for n in class_names))
    logger.info("Annotation comparison: annotation_comparison.png")
    logger.info("Annotation report: annotation_report.json")
    if report["per_image"]:
        logger.info("Mean IoU: %.3f (across %d tumor images)", report["mean_iou"], report["num_images"])
        logger.info("Summary: %s", report["summary"])
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
