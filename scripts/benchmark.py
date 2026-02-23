#!/usr/bin/env python3
"""Orchestrate model improvement experiment grid.

Runs a systematic comparison of architectures and training configurations
on the BTXRD dataset. Each experiment is trained on both stratified and
center-holdout splits, evaluated, and results are collected into a
comparison table.

Experiment grid:
  E1: EfficientNet-B0 baseline (224px, no annotations)
  E2: EfficientNet-B0 + annotations (380px)
  E3: EfficientNet-B3 + annotations (380px)
  E4: ResNet-50-CBAM + annotations (380px)

Outputs:
  results/experiments/{experiment_id}/  -- per-experiment eval results
  results/experiments/comparison.csv    -- cross-experiment comparison table
  results/experiments/comparison.json   -- machine-readable comparison
  checkpoints/backup_b0_baseline/       -- backup of pre-existing checkpoints

Usage:
    python scripts/benchmark.py --config configs/default.yaml
    python scripts/benchmark.py --config configs/default.yaml --experiments E1_b0_baseline,E3_b3_annotated
    python scripts/benchmark.py --config configs/default.yaml --skip-eval

Implemented in: Phase 9
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ── Experiment Grid ──────────────────────────────────────────────────
EXPERIMENT_GRID = [
    {
        "id": "E1_b0_baseline",
        "backbone": "efficientnet_b0",
        "image_size": 224,
        "batch_size": 32,
        "dropout": 0.2,
        "epochs": 80,
        "annotations": False,
        "attn_layer": None,
    },
    {
        "id": "E2_b0_annotated",
        "backbone": "efficientnet_b0",
        "image_size": 380,
        "batch_size": 16,
        "dropout": 0.2,
        "epochs": 80,
        "annotations": True,
        "attn_layer": None,
    },
    {
        "id": "E3_b3_annotated",
        "backbone": "efficientnet_b3",
        "image_size": 380,
        "batch_size": 8,
        "dropout": 0.3,
        "epochs": 80,
        "annotations": True,
        "attn_layer": None,
    },
    {
        "id": "E4_resnet50_cbam",
        "backbone": "resnet50",
        "image_size": 380,
        "batch_size": 8,
        "dropout": 0.3,
        "epochs": 80,
        "annotations": True,
        "attn_layer": "cbam",
    },
    {
        "id": "E5_paper_replication",
        "backbone": "efficientnet_b0",
        "image_size": 600,
        "batch_size": 8,
        "dropout": 0.2,
        "epochs": 300,
        "annotations": False,
        "attn_layer": None,
        "split_strategies": ["random"],
        "early_stopping_patience": 999,
    },
]

SPLIT_STRATEGIES = ["stratified", "center"]
SPLIT_PREFIX_MAP = {"stratified": "stratified", "center": "center", "random": "random"}
SPLIT_DIR_MAP = {"stratified": "stratified", "center": "center_holdout", "random": "random"}


def backup_checkpoints(checkpoints_dir: Path) -> bool:
    """Back up existing checkpoints to backup_b0_baseline/ subdirectory.

    Args:
        checkpoints_dir: Path to the checkpoints directory.

    Returns:
        True if backup was created, False if backup already exists or
        no checkpoints to back up.
    """
    backup_dir = checkpoints_dir / "backup_b0_baseline"
    if backup_dir.exists():
        logger.info("Backup directory already exists: %s -- skipping backup", backup_dir)
        return False

    pt_files = list(checkpoints_dir.glob("*.pt"))
    if not pt_files:
        logger.info("No checkpoints found to back up in %s", checkpoints_dir)
        return False

    backup_dir.mkdir(parents=True, exist_ok=True)
    for pt_file in pt_files:
        dest = backup_dir / pt_file.name
        shutil.copy2(pt_file, dest)
        logger.info("Backed up: %s -> %s", pt_file.name, dest)

    logger.info("Backed up %d checkpoint(s) to %s", len(pt_files), backup_dir)
    return True


def build_override_args(experiment: dict, split_strategy: str) -> list[str]:
    """Build --override CLI arguments for train.py from experiment config.

    Args:
        experiment: Experiment dict from EXPERIMENT_GRID.
        split_strategy: One of "stratified" or "center".

    Returns:
        List of strings like ["--override", "model.backbone=resnet50", ...].
    """
    overrides = [
        f"model.backbone={experiment['backbone']}",
        f"data.image_size={experiment['image_size']}",
        f"training.batch_size={experiment['batch_size']}",
        f"model.dropout={experiment['dropout']}",
        f"training.split_strategy={split_strategy}",
    ]

    if experiment.get("epochs"):
        overrides.append(f"training.epochs={experiment['epochs']}")
        patience = experiment.get("early_stopping_patience", 20)
        overrides.append(f"training.early_stopping_patience={patience}")

    if not experiment["annotations"]:
        overrides.append("data.annotations_dir=null")

    if experiment["attn_layer"]:
        overrides.append(f"model.attn_layer={experiment['attn_layer']}")

    # Pass all overrides as values to a single --override flag.
    # argparse nargs="*" treats each --override as a new list, so only
    # the last one would survive if we used multiple --override flags.
    if overrides:
        return ["--override"] + overrides
    return []


def run_experiment(
    experiment: dict,
    config_path: str,
    skip_eval: bool,
    checkpoints_dir: Path,
    results_base: Path,
) -> dict:
    """Run a single experiment across both split strategies.

    Args:
        experiment: Experiment dict from EXPERIMENT_GRID.
        config_path: Path to the YAML config file.
        skip_eval: If True, skip evaluation after training.
        checkpoints_dir: Path to the checkpoints directory.
        results_base: Base path for experiment results (results/experiments/).

    Returns:
        Dict with experiment status for each split.
    """
    exp_id = experiment["id"]
    exp_results_dir = results_base / exp_id
    exp_results_dir.mkdir(parents=True, exist_ok=True)

    statuses = {}
    exp_splits = experiment.get("split_strategies", SPLIT_STRATEGIES)

    for split in exp_splits:
        split_prefix = SPLIT_PREFIX_MAP[split]
        split_dir_name = SPLIT_DIR_MAP[split]
        run_key = f"{exp_id}/{split}"

        logger.info("=" * 70)
        logger.info("EXPERIMENT: %s | SPLIT: %s", exp_id, split)
        logger.info("=" * 70)

        start_time = time.time()

        # ── Train ─────────────────────────────────────────
        train_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train.py"),
            "--config", config_path,
        ] + build_override_args(experiment, split)

        logger.info("Training command: %s", " ".join(train_cmd))

        try:
            subprocess.run(
                train_cmd,
                check=True,
                cwd=str(PROJECT_ROOT),
            )
            logger.info("Training completed for %s", run_key)
        except subprocess.CalledProcessError as e:
            logger.error("Training FAILED for %s (exit code %d)", run_key, e.returncode)
            statuses[split] = {
                "status": "train_failed",
                "duration_s": time.time() - start_time,
                "error": str(e),
            }
            continue

        # ── Evaluate ──────────────────────────────────────
        if not skip_eval:
            # Create experiment checkpoint directory and copy checkpoint there
            exp_ckpt_dir = exp_results_dir / "checkpoints"
            exp_ckpt_dir.mkdir(parents=True, exist_ok=True)

            src_ckpt = checkpoints_dir / f"best_{split_prefix}.pt"
            if src_ckpt.exists():
                dst_ckpt = exp_ckpt_dir / f"best_{split_prefix}.pt"
                shutil.copy2(src_ckpt, dst_ckpt)
                logger.info("Copied checkpoint: %s -> %s", src_ckpt, dst_ckpt)
            else:
                logger.warning("Checkpoint not found: %s -- skipping eval", src_ckpt)
                statuses[split] = {
                    "status": "no_checkpoint",
                    "duration_s": time.time() - start_time,
                }
                continue

            # Run eval.py with checkpoint-dir override and experiment results dir
            # All overrides passed as values to a single --override flag
            # (argparse nargs="*" treats each --override as a new list)
            eval_overrides = [
                f"paths.results_dir=results/experiments/{exp_id}",
                f"evaluation.split_strategies=[{split}]",
            ]
            model_overrides = build_override_args(experiment, split)
            # build_override_args returns ["--override", "k1=v1", "k2=v2", ...]
            # Extract just the override values (skip the "--override" prefix)
            if model_overrides:
                eval_overrides.extend(model_overrides[1:])  # skip first "--override"

            eval_cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "eval.py"),
                "--config", config_path,
                "--checkpoint-dir", str(exp_ckpt_dir),
                "--override",
            ] + eval_overrides

            logger.info("Eval command: %s", " ".join(eval_cmd))

            try:
                subprocess.run(
                    eval_cmd,
                    check=True,
                    cwd=str(PROJECT_ROOT),
                )
                logger.info("Evaluation completed for %s", run_key)
            except subprocess.CalledProcessError as e:
                logger.error("Evaluation FAILED for %s (exit code %d)", run_key, e.returncode)
                statuses[split] = {
                    "status": "eval_failed",
                    "duration_s": time.time() - start_time,
                    "error": str(e),
                }
                continue

        # ── Rename checkpoint with experiment ID ──────────
        src_ckpt = checkpoints_dir / f"best_{split_prefix}.pt"
        dst_ckpt = checkpoints_dir / f"best_{split_prefix}_{exp_id}.pt"
        if src_ckpt.exists():
            shutil.move(str(src_ckpt), str(dst_ckpt))
            logger.info("Renamed checkpoint: %s -> %s", src_ckpt.name, dst_ckpt.name)

        duration = time.time() - start_time
        statuses[split] = {
            "status": "success",
            "duration_s": duration,
        }
        logger.info(
            "Completed %s in %.1f seconds (%.1f min)",
            run_key, duration, duration / 60,
        )

    return statuses


def generate_comparison(results_base: Path, experiments: list[dict]) -> None:
    """Generate cross-experiment comparison table from eval results.

    Reads metrics_summary.json from each experiment's results directory
    and builds comparison CSV + JSON files.

    Args:
        results_base: Base path for experiment results (results/experiments/).
        experiments: List of experiment dicts that were run.
    """
    rows = []

    for exp in experiments:
        exp_id = exp["id"]
        row = {
            "experiment_id": exp_id,
            "backbone": exp["backbone"],
            "image_size": exp["image_size"],
            "annotations": exp["annotations"],
        }

        for split, dir_name in [("stratified", "stratified"), ("center", "center_holdout"), ("random", "random")]:
            metrics_path = results_base / exp_id / dir_name / "metrics_summary.json"
            if metrics_path.exists():
                try:
                    with open(metrics_path) as f:
                        metrics = json.load(f)
                    row[f"{split}_accuracy"] = metrics.get("accuracy")
                    row[f"{split}_macro_auc"] = metrics.get("macro_auc")
                    mal_sens = metrics.get("per_class_sensitivity", {}).get(
                        "Malignant", metrics.get("malignant_sensitivity")
                    )
                    row[f"{split}_malignant_sensitivity"] = mal_sens
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(
                        "Failed to read metrics for %s/%s: %s", exp_id, split, e
                    )
                    row[f"{split}_accuracy"] = None
                    row[f"{split}_macro_auc"] = None
                    row[f"{split}_malignant_sensitivity"] = None
            else:
                logger.warning("Metrics not found: %s", metrics_path)
                row[f"{split}_accuracy"] = None
                row[f"{split}_macro_auc"] = None
                row[f"{split}_malignant_sensitivity"] = None

        rows.append(row)

    if not rows:
        logger.warning("No experiment results found -- skipping comparison generation")
        return

    # ── Save CSV ──────────────────────────────────────────
    csv_path = results_base / "comparison.csv"
    fieldnames = [
        "experiment_id",
        "backbone",
        "image_size",
        "annotations",
        "stratified_accuracy",
        "stratified_macro_auc",
        "stratified_malignant_sensitivity",
        "center_accuracy",
        "center_macro_auc",
        "center_malignant_sensitivity",
        "random_accuracy",
        "random_macro_auc",
        "random_malignant_sensitivity",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved comparison CSV: %s", csv_path)

    # ── Save JSON ─────────────────────────────────────────
    json_path = results_base / "comparison.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    logger.info("Saved comparison JSON: %s", json_path)

    # ── Determine best experiment ─────────────────────────
    # Primary: highest stratified accuracy
    # Tiebreaker 1: highest malignant sensitivity (stratified)
    # Tiebreaker 2: highest center-holdout accuracy
    valid_rows = [r for r in rows if r.get("stratified_accuracy") is not None]
    if valid_rows:
        best = max(
            valid_rows,
            key=lambda r: (
                r.get("stratified_accuracy") or 0,
                r.get("stratified_malignant_sensitivity") or 0,
                r.get("center_accuracy") or 0,
            ),
        )
        logger.info("=" * 70)
        logger.info("BEST EXPERIMENT: %s", best["experiment_id"])
        logger.info(
            "  Stratified Accuracy: %.3f | Macro AUC: %s | Malignant Sensitivity: %s",
            best.get("stratified_accuracy", 0),
            f"{best.get('stratified_macro_auc', 0):.3f}"
            if best.get("stratified_macro_auc") is not None
            else "-",
            f"{best.get('stratified_malignant_sensitivity', 0):.3f}"
            if best.get("stratified_malignant_sensitivity") is not None
            else "-",
        )
        logger.info("=" * 70)

    # ── Print comparison table ────────────────────────────
    logger.info("")
    logger.info("COMPARISON TABLE")
    logger.info("-" * 120)
    header = (
        f"{'Experiment':<25} {'Backbone':<20} {'ImgSize':>7} {'Ann':>4} "
        f"{'Strat Acc':>10} {'Strat AUC':>10} {'Strat MalSens':>14} "
        f"{'Center Acc':>11} {'Center AUC':>11} {'Center MalSens':>15} "
        f"{'Rand Acc':>10} {'Rand AUC':>10} {'Rand MalSens':>13}"
    )
    logger.info(header)
    logger.info("-" * 155)

    for row in rows:
        def _fmt(val: float | None) -> str:
            return f"{val:.3f}" if val is not None else "-"

        is_best = (
            valid_rows
            and row["experiment_id"] == best["experiment_id"]
        )
        marker = " ***" if is_best else ""

        line = (
            f"{row['experiment_id']:<25} {row['backbone']:<20} "
            f"{row['image_size']:>7} {'Yes' if row['annotations'] else 'No':>4} "
            f"{_fmt(row.get('stratified_accuracy')):>10} "
            f"{_fmt(row.get('stratified_macro_auc')):>10} "
            f"{_fmt(row.get('stratified_malignant_sensitivity')):>14} "
            f"{_fmt(row.get('center_accuracy')):>11} "
            f"{_fmt(row.get('center_macro_auc')):>11} "
            f"{_fmt(row.get('center_malignant_sensitivity')):>15} "
            f"{_fmt(row.get('random_accuracy')):>10} "
            f"{_fmt(row.get('random_macro_auc')):>10} "
            f"{_fmt(row.get('random_malignant_sensitivity')):>13}"
            f"{marker}"
        )
        logger.info(line)

    logger.info("-" * 155)
    if valid_rows:
        logger.info("*** = Best experiment (by stratified accuracy)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run model improvement benchmark (experiment grid).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Experiments:
  E1_b0_baseline        EfficientNet-B0 at 224px, no annotations
  E2_b0_annotated       EfficientNet-B0 at 380px, with annotations
  E3_b3_annotated       EfficientNet-B3 at 380px, with annotations
  E4_resnet50_cbam      ResNet-50-CBAM at 380px, with annotations
  E5_paper_replication  EfficientNet-B0 at 600px, 300 epochs, random 80/20 split

E1-E4 train on stratified + center-holdout splits.
E5 trains only on the random split (paper replication protocol).
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default="all",
        help=(
            "Comma-separated experiment IDs to run, or 'all' "
            "(default: all). Example: E1_b0_baseline,E3_b3_annotated"
        ),
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation after training (train only)",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Select experiments ────────────────────────────────
    if args.experiments == "all":
        selected = EXPERIMENT_GRID
    else:
        requested_ids = {e.strip() for e in args.experiments.split(",")}
        selected = [e for e in EXPERIMENT_GRID if e["id"] in requested_ids]
        unknown = requested_ids - {e["id"] for e in selected}
        if unknown:
            logger.error(
                "Unknown experiment IDs: %s. Valid: %s",
                unknown,
                [e["id"] for e in EXPERIMENT_GRID],
            )
            sys.exit(1)

    logger.info("Selected %d experiment(s): %s", len(selected), [e["id"] for e in selected])

    # ── Paths ─────────────────────────────────────────────
    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    results_base = PROJECT_ROOT / "results" / "experiments"
    results_base.mkdir(parents=True, exist_ok=True)

    # ── Back up existing checkpoints ──────────────────────
    backup_checkpoints(checkpoints_dir)

    # ── Run experiments ───────────────────────────────────
    all_statuses: dict[str, dict] = {}
    total_start = time.time()

    for experiment in selected:
        exp_id = experiment["id"]
        logger.info("")
        logger.info("#" * 70)
        logger.info("# STARTING EXPERIMENT: %s", exp_id)
        logger.info("# Backbone: %s | Image Size: %d | Annotations: %s | Attn: %s",
                     experiment["backbone"], experiment["image_size"],
                     experiment["annotations"], experiment["attn_layer"] or "none")
        logger.info("#" * 70)

        statuses = run_experiment(
            experiment=experiment,
            config_path=args.config,
            skip_eval=args.skip_eval,
            checkpoints_dir=checkpoints_dir,
            results_base=results_base,
        )
        all_statuses[exp_id] = statuses

    total_duration = time.time() - total_start

    # ── Generate comparison table ─────────────────────────
    if not args.skip_eval:
        logger.info("")
        logger.info("Generating comparison table...")
        generate_comparison(results_base, selected)

    # ── Print summary ─────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 70)
    logger.info("%-30s %-15s %-15s %s", "Experiment/Split", "Status", "Duration", "Notes")
    logger.info("-" * 70)

    for exp_id, splits in all_statuses.items():
        for split, info in splits.items():
            status = info.get("status", "unknown")
            duration = info.get("duration_s", 0)
            error = info.get("error", "")
            notes = error[:40] if error else ""
            logger.info(
                "%-30s %-15s %8.1f min   %s",
                f"{exp_id}/{split}",
                status,
                duration / 60,
                notes,
            )

    logger.info("-" * 70)
    logger.info(
        "Total benchmark time: %.1f min (%.1f hours)",
        total_duration / 60,
        total_duration / 3600,
    )
    logger.info("=" * 70)

    # ── Exit code ─────────────────────────────────────────
    failures = sum(
        1
        for splits in all_statuses.values()
        for info in splits.values()
        if info.get("status") not in ("success",)
    )
    if failures:
        logger.warning("%d experiment run(s) failed", failures)
        sys.exit(1)


if __name__ == "__main__":
    main()
