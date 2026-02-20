#!/usr/bin/env python3
"""Evaluate trained model on both split strategies.

Loads the best checkpoint for each split strategy, runs inference on
the corresponding test set, computes EVAL-01 through EVAL-05 metrics,
and generates publication-quality plots.

Outputs per split:
  - roc_curves.png           (EVAL-01)
  - pr_curves.png            (EVAL-02)
  - confusion_matrix.png     (EVAL-04, absolute)
  - confusion_matrix_normalized.png (EVAL-04, row-normalized)
  - metrics_summary.json     (EVAL-01/02/03 summary)
  - classification_report.json (EVAL-05)

Usage:
    python scripts/eval.py --config configs/default.yaml

Implemented in: Phase 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import BTXRDDataset, create_dataloader
from src.data.transforms import get_test_transforms
from src.evaluation.metrics import compute_all_metrics, run_inference
from src.evaluation.visualization import (
    plot_confusion_matrices,
    plot_pr_curves,
    plot_roc_curves,
)
from src.models.factory import create_model, get_device, load_checkpoint
from src.utils.config import load_config
from src.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)

# Map config split names to directory names, checkpoint filenames, and test CSVs
SPLIT_MAP = {
    "stratified": {
        "dir_name": "stratified",
        "checkpoint": "best_stratified.pt",
        "test_csv": "stratified_test.csv",
    },
    "center": {
        "dir_name": "center_holdout",
        "checkpoint": "best_center.pt",
        "test_csv": "center_test.csv",
    },
}


def _json_serializable(obj: object) -> object:
    """JSON serialization helper for numpy types.

    Args:
        obj: Object to serialize.

    Returns:
        JSON-serializable Python type.

    Raises:
        TypeError: If type is not handled.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained model on both split strategies."
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

    # Image directory (shared across splits)
    images_dir = PROJECT_ROOT / cfg["data"]["raw_dir"] / "images"

    # Iterate over each split strategy defined in config
    split_strategies = cfg["evaluation"]["split_strategies"]
    logger.info("Evaluating %d split strategies: %s", len(split_strategies), split_strategies)
    logger.info("=" * 70)

    for split_name in split_strategies:
        if split_name not in SPLIT_MAP:
            logger.warning("Unknown split strategy '%s', skipping.", split_name)
            continue

        split_info = SPLIT_MAP[split_name]
        logger.info("--- Evaluating split: %s ---", split_name)

        # ── Paths ─────────────────────────────────────────────
        checkpoint_path = (
            PROJECT_ROOT / cfg["paths"]["checkpoints_dir"] / split_info["checkpoint"]
        )
        test_csv_path = (
            PROJECT_ROOT / cfg["data"]["splits_dir"] / split_info["test_csv"]
        )
        results_dir = (
            PROJECT_ROOT / cfg["paths"]["results_dir"] / split_info["dir_name"]
        )
        results_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Checkpoint: %s", checkpoint_path)
        logger.info("Test CSV:   %s", test_csv_path)
        logger.info("Results:    %s", results_dir)

        # ── Load checkpoint ───────────────────────────────────
        ckpt = load_checkpoint(str(checkpoint_path), device="cpu")
        ckpt_config = ckpt["config"]
        class_names = ckpt["class_names"]
        logger.info(
            "Loaded checkpoint from epoch %d (val_loss=%.6f)",
            ckpt["epoch"],
            ckpt["val_loss"],
        )

        # ── Create model ─────────────────────────────────────
        model = create_model(ckpt_config)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        model.eval()
        logger.info("Model loaded: %s", ckpt_config["model"]["backbone"])

        # ── Create test dataset and dataloader ────────────────
        test_dataset = BTXRDDataset(
            test_csv_path,
            images_dir,
            get_test_transforms(cfg["data"]["image_size"]),
        )
        test_loader = create_dataloader(
            test_dataset,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=cfg["data"]["num_workers"],
        )
        logger.info("Test set: %d samples", len(test_dataset))

        # ── Run inference ─────────────────────────────────────
        y_true, y_pred, y_prob = run_inference(model, test_loader, device)
        logger.info("Inference complete: %d predictions", len(y_true))

        # ── Compute metrics ───────────────────────────────────
        metrics = compute_all_metrics(y_true, y_pred, y_prob, class_names)

        # ── Generate plots ────────────────────────────────────
        plot_roc_curves(metrics, class_names, results_dir / "roc_curves.png")
        logger.info("Saved ROC curves")

        plot_pr_curves(metrics, class_names, results_dir / "pr_curves.png")
        logger.info("Saved PR curves")

        plot_confusion_matrices(metrics["confusion_matrix"], class_names, results_dir)
        logger.info("Saved confusion matrices")

        # ── Save metrics summary JSON ─────────────────────────
        metrics_summary = {
            "split_strategy": split_name,
            "test_set_size": int(len(y_true)),
            "macro_auc": metrics["macro_auc"],
            "per_class_auc": metrics["roc_auc"],
            "per_class_average_precision": metrics["average_precision"],
            "per_class_sensitivity": metrics["sensitivity"],
            "per_class_specificity": metrics["specificity"],
            "malignant_sensitivity": metrics["malignant_sensitivity"],
            "accuracy": metrics["accuracy"],
        }

        summary_path = results_dir / "metrics_summary.json"
        with open(summary_path, "w") as f:
            json.dump(metrics_summary, f, indent=2, default=_json_serializable)
        logger.info("Saved metrics summary: %s", summary_path)

        # ── Save classification report JSON ───────────────────
        report_path = results_dir / "classification_report.json"
        with open(report_path, "w") as f:
            json.dump(
                metrics["classification_report"],
                f,
                indent=2,
                default=_json_serializable,
            )
        logger.info("Saved classification report: %s", report_path)

        # ── Log headline metrics ──────────────────────────────
        logger.info(
            "HEADLINE [%s] | Malignant Sensitivity: %.3f | Macro AUC: %.3f | Accuracy: %.3f",
            split_name,
            metrics["malignant_sensitivity"],
            metrics["macro_auc"],
            metrics["accuracy"],
        )
        logger.info("=" * 70)

    logger.info("Evaluation complete for all split strategies.")


if __name__ == "__main__":
    main()
