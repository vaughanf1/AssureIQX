#!/usr/bin/env python3
"""Train EfficientNet-B0 classifier on BTXRD dataset.

Loads split manifests, builds data loaders with augmentations,
fine-tunes an EfficientNet-B0 backbone, and saves the best
checkpoint to ``paths.checkpoints_dir``.

Key design decisions:
- WEIGHTED CrossEntropyLoss for training (handles class imbalance)
- UNWEIGHTED CrossEntropyLoss for validation (clean early stopping signal)
- Per-class recall logged every epoch to detect Malignant class collapse
- Best checkpoint (lowest unweighted val_loss) and final checkpoint saved

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --override training.epochs=10
    python scripts/train.py --config configs/default.yaml --override training.split_strategy=center

Implemented in: Phase 4
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend -- must be before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import recall_score
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import BTXRDDataset, create_dataloader
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.factory import (
    EarlyStopping,
    compute_class_weights,
    create_model,
    get_device,
    save_checkpoint,
)
from src.utils.config import load_config
from src.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)

CLASS_NAMES = ["Normal", "Benign", "Malignant"]


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train the model for one epoch.

    Args:
        model: Model to train.
        loader: Training data loader.
        criterion: Loss function (weighted CrossEntropyLoss).
        optimizer: Optimizer instance.
        device: Device to run on.

    Returns:
        Tuple of (epoch_loss, epoch_accuracy) where epoch_loss is the
        average loss per sample and epoch_accuracy is fraction correct.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += batch_size

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 3,
) -> tuple[float, float, np.ndarray]:
    """Evaluate the model on validation data.

    Args:
        model: Model to evaluate.
        loader: Validation data loader.
        criterion: Loss function (unweighted CrossEntropyLoss).
        device: Device to run on.
        num_classes: Number of classes for per-class recall.

    Returns:
        Tuple of (epoch_loss, epoch_accuracy, per_class_recall) where
        per_class_recall is a numpy array of shape (num_classes,).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Val  ", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += batch_size

            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0

    per_class_recall = recall_score(
        all_labels,
        all_preds,
        average=None,
        labels=list(range(num_classes)),
        zero_division=0,
    )

    return epoch_loss, epoch_acc, per_class_recall


def save_training_log(log_rows: list[dict], filepath: str | Path) -> None:
    """Write training metrics to a CSV file.

    Args:
        log_rows: List of dicts, each with keys: epoch, train_loss,
            val_loss, val_acc, val_recall_normal, val_recall_benign,
            val_recall_malignant, lr.
        filepath: Output CSV path.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_acc",
        "val_recall_normal",
        "val_recall_benign",
        "val_recall_malignant",
        "lr",
    ]

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)

    logger.info("Training log saved to %s", filepath)


def plot_loss_curves(log_rows: list[dict], filepath: str | Path) -> None:
    """Plot training and validation loss curves and per-class recall curves.

    Generates two PNG files:
    - Loss curve (train_loss vs val_loss) at the given filepath
    - Recall curve (per-class recall over epochs) at recall_curve.png
      in the same directory

    Args:
        log_rows: List of dicts with training metrics.
        filepath: Output path for loss curve PNG.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    epochs = [row["epoch"] for row in log_rows]
    train_losses = [row["train_loss"] for row in log_rows]
    val_losses = [row["val_loss"] for row in log_rows]

    # Loss curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    ax.plot(epochs, val_losses, label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    logger.info("Loss curve saved to %s", filepath)

    # Per-class recall curve
    recall_normal = [row["val_recall_normal"] for row in log_rows]
    recall_benign = [row["val_recall_benign"] for row in log_rows]
    recall_malignant = [row["val_recall_malignant"] for row in log_rows]

    recall_filepath = filepath.parent / "recall_curve.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, recall_normal, label="Normal", linewidth=2)
    ax.plot(epochs, recall_benign, label="Benign", linewidth=2)
    ax.plot(epochs, recall_malignant, label="Malignant", linewidth=2, color="red")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Recall")
    ax.set_title("Per-Class Recall Over Training")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(recall_filepath, dpi=150)
    plt.close(fig)
    logger.info("Recall curve saved to %s", recall_filepath)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train EfficientNet-B0 classifier on BTXRD dataset."
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
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cfg = load_config(args.config, overrides=args.override)
    set_seed(cfg.get("seed", 42))

    # ── Device ────────────────────────────────────────────
    device = get_device(cfg.get("device", "auto"))
    logger.info("Using device: %s", device)

    # ── Split strategy ────────────────────────────────────
    split_strategy = cfg["training"]["split_strategy"]
    split_prefix = "stratified" if split_strategy == "stratified" else "center"
    logger.info("Split strategy: %s (prefix: %s)", split_strategy, split_prefix)

    # ── File paths ────────────────────────────────────────
    train_csv = PROJECT_ROOT / cfg["data"]["splits_dir"] / f"{split_prefix}_train.csv"
    val_csv = PROJECT_ROOT / cfg["data"]["splits_dir"] / f"{split_prefix}_val.csv"
    images_dir = PROJECT_ROOT / cfg["data"]["raw_dir"] / "images"
    results_dir = PROJECT_ROOT / cfg["paths"]["results_dir"] / (
        "stratified" if split_strategy == "stratified" else "center_holdout"
    )
    checkpoints_dir = PROJECT_ROOT / cfg["paths"]["checkpoints_dir"]

    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Train CSV: %s", train_csv)
    logger.info("Val CSV:   %s", val_csv)
    logger.info("Images:    %s", images_dir)
    logger.info("Results:   %s", results_dir)
    logger.info("Checkpoints: %s", checkpoints_dir)

    # ── Datasets ──────────────────────────────────────────
    image_size = cfg["data"]["image_size"]
    train_dataset = BTXRDDataset(
        train_csv, images_dir, get_train_transforms(image_size)
    )
    val_dataset = BTXRDDataset(
        val_csv, images_dir, get_val_transforms(image_size)
    )
    logger.info(
        "Train: %d samples, Val: %d samples",
        len(train_dataset),
        len(val_dataset),
    )

    # ── Data loaders ──────────────────────────────────────
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]
    train_loader = create_dataloader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = create_dataloader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # ── Model ─────────────────────────────────────────────
    model = create_model(cfg).to(device)
    logger.info("Model created: %s", cfg["model"]["backbone"])

    # ── Class weights (from training set only) ────────────
    class_weights = compute_class_weights(
        train_dataset.labels, cfg["model"]["num_classes"]
    ).to(device)
    logger.info("Class weights: %s", class_weights.tolist())

    # ── Loss functions ────────────────────────────────────
    # CRITICAL: Weighted loss for training, unweighted for validation/early stopping
    train_criterion = nn.CrossEntropyLoss(weight=class_weights)
    val_criterion = nn.CrossEntropyLoss()

    # ── Optimizer ─────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # ── Scheduler ─────────────────────────────────────────
    epochs = cfg["training"]["epochs"]
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # ── Early stopping ────────────────────────────────────
    early_stopping = EarlyStopping(
        patience=cfg["training"]["early_stopping_patience"]
    )

    # ── Training loop ─────────────────────────────────────
    log_rows: list[dict] = []
    best_val_loss = float("inf")
    best_epoch = 0
    malignant_zero_count = 0  # Track consecutive epochs with zero Malignant recall
    start_time = time.time()

    logger.info("Starting training for up to %d epochs...", epochs)
    logger.info("=" * 70)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, train_criterion, optimizer, device
        )

        # Validate (with UNWEIGHTED loss for clean early stopping signal)
        val_loss, val_acc, per_class_recall = validate(
            model, val_loader, val_criterion, device, cfg["model"]["num_classes"]
        )

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Build log row
        log_row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "val_recall_normal": round(float(per_class_recall[0]), 6),
            "val_recall_benign": round(float(per_class_recall[1]), 6),
            "val_recall_malignant": round(float(per_class_recall[2]), 6),
            "lr": round(current_lr, 6),
        }
        log_rows.append(log_row)

        # Print epoch summary
        epoch_time = time.time() - epoch_start
        logger.info(
            "Epoch %d/%d (%.1fs) | "
            "Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.4f | "
            "Recall [N: %.3f, B: %.3f, M: %.3f] | LR: %.6f",
            epoch,
            epochs,
            epoch_time,
            train_loss,
            val_loss,
            val_acc,
            per_class_recall[0],
            per_class_recall[1],
            per_class_recall[2],
            current_lr,
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_path = checkpoints_dir / f"best_{split_prefix}.pt"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                val_loss=val_loss,
                config=cfg,
                class_weights=class_weights,
                filepath=str(best_path),
            )
            logger.info("  -> Saved best checkpoint (val_loss=%.6f)", val_loss)

        # Malignant class collapse detection
        if per_class_recall[2] == 0.0:
            malignant_zero_count += 1
            if malignant_zero_count >= 2:
                logger.warning(
                    "WARNING: Malignant recall is 0.0 for %d consecutive epochs! "
                    "Possible class collapse.",
                    malignant_zero_count,
                )
        else:
            malignant_zero_count = 0

        # Early stopping check
        if early_stopping(val_loss):
            logger.info(
                "Early stopping triggered at epoch %d (patience=%d)",
                epoch,
                early_stopping.patience,
            )
            break

    # ── Post-training ─────────────────────────────────────
    total_time = time.time() - start_time
    logger.info("=" * 70)

    # Save final checkpoint
    final_path = checkpoints_dir / f"final_{split_prefix}.pt"
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        val_loss=val_loss,
        config=cfg,
        class_weights=class_weights,
        filepath=str(final_path),
    )
    logger.info("Saved final checkpoint: %s", final_path)

    # Save training log CSV
    log_csv_path = results_dir / "training_log.csv"
    save_training_log(log_rows, log_csv_path)

    # Plot loss curves and recall curves
    loss_plot_path = results_dir / "loss_curve.png"
    plot_loss_curves(log_rows, loss_plot_path)

    # Print training summary
    final_recall = per_class_recall
    logger.info("Training Summary")
    logger.info("-" * 40)
    logger.info("Split strategy:    %s", split_strategy)
    logger.info("Total epochs:      %d", epoch)
    logger.info("Best epoch:        %d", best_epoch)
    logger.info("Best val_loss:     %.6f", best_val_loss)
    logger.info("Final val_loss:    %.6f", val_loss)
    logger.info("Final val_acc:     %.4f", val_acc)
    logger.info(
        "Final recall:      Normal=%.3f, Benign=%.3f, Malignant=%.3f",
        final_recall[0],
        final_recall[1],
        final_recall[2],
    )
    logger.info("Best checkpoint:   %s", checkpoints_dir / f"best_{split_prefix}.pt")
    logger.info("Final checkpoint:  %s", final_path)
    logger.info("Training log:      %s", log_csv_path)
    logger.info("Loss curve:        %s", loss_plot_path)
    logger.info("Recall curve:      %s", results_dir / "recall_curve.png")
    logger.info("Total time:        %.1f seconds (%.1f min)", total_time, total_time / 60)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
