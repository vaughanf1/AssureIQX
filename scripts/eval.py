#!/usr/bin/env python3
"""Evaluate trained model on both split strategies.

Loads the best checkpoint for each split strategy, runs inference on
the corresponding test set, computes EVAL-01 through EVAL-08 metrics,
generates publication-quality plots, bootstrap CIs, and a comparison table.

Outputs per split:
  - roc_curves.png           (EVAL-01)
  - pr_curves.png            (EVAL-02)
  - confusion_matrix.png     (EVAL-04, absolute)
  - confusion_matrix_normalized.png (EVAL-04, row-normalized)
  - metrics_summary.json     (EVAL-01/02/03 summary)
  - classification_report.json (EVAL-05)
  - bootstrap_ci.json        (EVAL-07)

Outputs at results root:
  - comparison_table.json    (EVAL-06/08)
  - comparison_table.csv     (EVAL-06/08)

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
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import BTXRDDataset, create_dataloader
from src.data.transforms import get_test_transforms
from src.evaluation.bootstrap import bootstrap_confidence_intervals
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

# BTXRD paper baseline (Yao et al., Scientific Data 2025)
# Source: PMC11739492 -- YOLOv8s-cls results
BTXRD_BASELINE = {
    "model": "YOLOv8s-cls",
    "source": "Yao et al., Scientific Data 2025 (PMC11739492)",
    "split": "Random 80/20 (no patient grouping)",
    "image_size": 600,
    "epochs": 300,
    "per_class_precision": {"Normal": 0.913, "Benign": 0.881, "Malignant": 0.734},
    "per_class_recall": {"Normal": 0.898, "Benign": 0.875, "Malignant": 0.839},
    "caveats": [
        "Random 80/20 split without patient-level grouping (potential data leakage)",
        "Validation set used for reporting (no separate held-out test set)",
        "Different image size (600px vs 224px)",
        "Different architecture (YOLOv8s-cls vs EfficientNet-B0)",
        "Different training duration (300 epochs vs early stopping at ~5 epochs)",
        "Paper reports mAP@0.5 (detection metric) which is not directly comparable to AUC",
        "No macro AUC or specificity reported in paper -- only precision/recall per class",
    ],
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


def _build_comparison_table(
    collected_results: dict,
    results_root: Path,
) -> None:
    """Build and save the comparison table (JSON + CSV).

    Aggregates per-split metrics, BTXRD baseline, and generalization gap
    into a single comparison table for EVAL-06 and EVAL-08.

    Args:
        collected_results: Dict keyed by split name with metrics and bootstrap CIs.
        results_root: Root results directory (e.g. results/).
    """
    # Extract key metrics for gap calculation
    stratified = collected_results.get("stratified", {})
    center_holdout = collected_results.get("center_holdout", {})

    s_metrics = stratified.get("metrics_summary", {})
    c_metrics = center_holdout.get("metrics_summary", {})

    s_report = stratified.get("classification_report", {})
    c_report = center_holdout.get("classification_report", {})

    s_ci = stratified.get("bootstrap_ci", {})
    c_ci = center_holdout.get("bootstrap_ci", {})

    # Generalization gap: center_holdout minus stratified (negative = center worse)
    generalization_gap = {}
    if s_metrics and c_metrics:
        generalization_gap = {
            "description": "center_holdout minus stratified (negative means center-holdout is worse)",
            "macro_auc_gap": c_metrics.get("macro_auc", 0) - s_metrics.get("macro_auc", 0),
            "malignant_sensitivity_gap": (
                c_metrics.get("malignant_sensitivity", 0) - s_metrics.get("malignant_sensitivity", 0)
            ),
            "accuracy_gap": c_metrics.get("accuracy", 0) - s_metrics.get("accuracy", 0),
        }

    # Build full comparison JSON
    comparison = {
        "stratified": {
            "metrics_summary": s_metrics,
            "classification_report": s_report,
            "bootstrap_ci": {
                "macro_auc": s_ci.get("macro_auc", {}),
                "malignant_sensitivity": s_ci.get("sensitivity_Malignant", {}),
            },
        },
        "center_holdout": {
            "metrics_summary": c_metrics,
            "classification_report": c_report,
            "bootstrap_ci": {
                "macro_auc": c_ci.get("macro_auc", {}),
                "malignant_sensitivity": c_ci.get("sensitivity_Malignant", {}),
            },
        },
        "btxrd_baseline": BTXRD_BASELINE,
        "generalization_gap": generalization_gap,
    }

    # Save JSON
    json_path = results_root / "comparison_table.json"
    with open(json_path, "w") as f:
        json.dump(comparison, f, indent=2, default=_json_serializable)
    logger.info("Saved comparison table JSON: %s", json_path)

    # Build CSV comparison table
    # Helper to get a metric safely
    def _get(d: dict, *keys, default: str = "-") -> str:
        """Navigate nested dict and format as string."""
        val = d
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
            if val is None:
                return default
        if isinstance(val, float):
            return f"{val:.3f}"
        return str(val)

    class_names = ["Normal", "Benign", "Malignant"]
    rows = []

    # Macro AUC
    rows.append({
        "Metric": "Macro AUC",
        "Stratified": _get(s_metrics, "macro_auc"),
        "Center Holdout": _get(c_metrics, "macro_auc"),
        "BTXRD Baseline": "-",
        "Gap": _get(generalization_gap, "macro_auc_gap"),
    })

    # Accuracy
    rows.append({
        "Metric": "Accuracy",
        "Stratified": _get(s_metrics, "accuracy"),
        "Center Holdout": _get(c_metrics, "accuracy"),
        "BTXRD Baseline": "-",
        "Gap": _get(generalization_gap, "accuracy_gap"),
    })

    # Per-class sensitivity (recall)
    for cls in class_names:
        btxrd_val = BTXRD_BASELINE["per_class_recall"].get(cls)
        btxrd_str = f"{btxrd_val:.3f}" if btxrd_val is not None else "-"
        rows.append({
            "Metric": f"{cls} Sensitivity",
            "Stratified": _get(s_metrics, "per_class_sensitivity", cls),
            "Center Holdout": _get(c_metrics, "per_class_sensitivity", cls),
            "BTXRD Baseline": btxrd_str,
            "Gap": _get(generalization_gap, "malignant_sensitivity_gap")
            if cls == "Malignant"
            else "-",
        })

    # Per-class specificity
    for cls in class_names:
        rows.append({
            "Metric": f"{cls} Specificity",
            "Stratified": _get(s_metrics, "per_class_specificity", cls),
            "Center Holdout": _get(c_metrics, "per_class_specificity", cls),
            "BTXRD Baseline": "-",
            "Gap": "-",
        })

    # Per-class F1 (from classification report)
    for cls in class_names:
        rows.append({
            "Metric": f"{cls} F1",
            "Stratified": _get(s_report, cls, "f1-score"),
            "Center Holdout": _get(c_report, cls, "f1-score"),
            "BTXRD Baseline": "-",
            "Gap": "-",
        })

    # Macro F1
    rows.append({
        "Metric": "Macro F1",
        "Stratified": _get(s_report, "macro avg", "f1-score"),
        "Center Holdout": _get(c_report, "macro avg", "f1-score"),
        "BTXRD Baseline": "-",
        "Gap": "-",
    })

    df = pd.DataFrame(rows)
    csv_path = results_root / "comparison_table.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved comparison table CSV: %s", csv_path)

    # Log comparison summary
    logger.info("=" * 70)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 70)
    logger.info(
        "%-25s %12s %15s %15s %10s",
        "Metric", "Stratified", "Center Holdout", "BTXRD Baseline", "Gap",
    )
    logger.info("-" * 77)
    for row in rows:
        logger.info(
            "%-25s %12s %15s %15s %10s",
            row["Metric"],
            row["Stratified"],
            row["Center Holdout"],
            row["BTXRD Baseline"],
            row["Gap"],
        )
    logger.info("=" * 70)
    logger.info(
        "NOTE: BTXRD baseline uses random 80/20 split without patient grouping. "
        "See comparison_table.json for %d caveats.",
        len(BTXRD_BASELINE["caveats"]),
    )


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
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Override checkpoint directory (default: from config paths.checkpoints_dir)",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    cfg = load_config(args.config, overrides=args.override)

    # Apply checkpoint directory override if provided
    if args.checkpoint_dir is not None:
        cfg["paths"]["checkpoints_dir"] = args.checkpoint_dir
    set_seed(cfg.get("seed", 42))

    # Device
    device = get_device(cfg.get("device", "auto"))
    logger.info("Using device: %s", device)

    # Image directory (shared across splits)
    images_dir = PROJECT_ROOT / cfg["data"]["raw_dir"] / "images"

    # Results root for comparison table
    results_root = PROJECT_ROOT / cfg["paths"]["results_dir"]
    results_root.mkdir(parents=True, exist_ok=True)

    # Iterate over each split strategy defined in config
    split_strategies = cfg["evaluation"]["split_strategies"]
    logger.info("Evaluating %d split strategies: %s", len(split_strategies), split_strategies)
    logger.info("=" * 70)

    # Collector for comparison table (keyed by dir_name: "stratified" or "center_holdout")
    collected_results: dict[str, dict] = {}

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

        # ── Bootstrap confidence intervals (EVAL-07) ──────────
        n_iters = cfg["evaluation"]["bootstrap_iterations"]  # 1000
        conf_level = cfg["evaluation"]["confidence_level"]  # 0.95
        seed = cfg.get("seed", 42)

        logger.info(
            "Computing bootstrap CIs (%d iterations, %.0f%% confidence)...",
            n_iters,
            conf_level * 100,
        )
        ci_results = bootstrap_confidence_intervals(
            y_true, y_pred, y_prob, class_names,
            n_iterations=n_iters,
            confidence_level=conf_level,
            seed=seed,
        )

        ci_path = results_dir / "bootstrap_ci.json"
        with open(ci_path, "w") as f:
            json.dump(ci_results, f, indent=2, default=_json_serializable)
        logger.info("Saved bootstrap CIs: %s", ci_path)

        # Log bootstrap headline
        auc_ci = ci_results["macro_auc"]
        mal_ci = ci_results["sensitivity_Malignant"]
        logger.info(
            "Bootstrap CIs (%d iterations): "
            "Macro AUC = %.3f [%.3f, %.3f] (n_valid=%d), "
            "Malignant Sensitivity = %.3f [%.3f, %.3f] (n_valid=%d)",
            n_iters,
            auc_ci["mean"], auc_ci["ci_lower"], auc_ci["ci_upper"], auc_ci["n_valid"],
            mal_ci["mean"], mal_ci["ci_lower"], mal_ci["ci_upper"], mal_ci["n_valid"],
        )

        # ── Log headline metrics ──────────────────────────────
        logger.info(
            "HEADLINE [%s] | Malignant Sensitivity: %.3f | Macro AUC: %.3f | Accuracy: %.3f",
            split_name,
            metrics["malignant_sensitivity"],
            metrics["macro_auc"],
            metrics["accuracy"],
        )
        logger.info("=" * 70)

        # ── Collect for comparison table ──────────────────────
        collected_results[split_info["dir_name"]] = {
            "metrics_summary": metrics_summary,
            "classification_report": metrics["classification_report"],
            "bootstrap_ci": ci_results,
        }

    # ── Build comparison table (EVAL-06 / EVAL-08) ────────────
    if len(collected_results) >= 2:
        _build_comparison_table(collected_results, results_root)
    else:
        logger.warning(
            "Comparison table requires at least 2 splits; only %d evaluated.",
            len(collected_results),
        )

    logger.info("Evaluation complete for all split strategies.")


if __name__ == "__main__":
    main()
