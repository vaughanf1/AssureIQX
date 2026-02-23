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

from src.data.dataset import BTXRDAnnotatedDataset, BTXRDDataset, create_dataloader
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
    scaler: torch.amp.GradScaler | None = None,
    amp_dtype: torch.dtype | None = None,
) -> tuple[float, float]:
    """Train the model for one epoch.

    Args:
        model: Model to train.
        loader: Training data loader.
        criterion: Loss function (weighted CrossEntropyLoss).
        optimizer: Optimizer instance.
        device: Device to run on.
        scaler: GradScaler for CUDA AMP (None for MPS/CPU).
        amp_dtype: dtype for autocast (torch.float16 or None to disable AMP).

    Returns:
        Tuple of (epoch_loss, epoch_accuracy) where epoch_loss is the
        average loss per sample and epoch_accuracy is fraction correct.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = amp_dtype is not None
    device_type = "cuda" if device.type == "cuda" else device.type

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if use_amp:
            with torch.autocast(device_type=device_type, dtype=amp_dtype):
                outputs = model(images)
                loss = criterion(outputs, labels)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    amp_dtype: torch.dtype | None = None,
) -> tuple[float, float, np.ndarray]:
    """Evaluate the model on validation data.

    Args:
        model: Model to evaluate.
        loader: Validation data loader.
        criterion: Loss function (unweighted CrossEntropyLoss).
        device: Device to run on.
        num_classes: Number of classes for per-class recall.
        amp_dtype: dtype for autocast (torch.float16 or None to disable AMP).

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
    use_amp = amp_dtype is not None
    device_type = "cuda" if device.type == "cuda" else device.type

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Val  ", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            if use_amp:
                with torch.autocast(device_type=device_type, dtype=amp_dtype):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
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
    SPLIT_PREFIX = {"stratified": "stratified", "center": "center", "random": "random"}
    split_prefix = SPLIT_PREFIX.get(split_strategy, split_strategy)
    logger.info("Split strategy: %s (prefix: %s)", split_strategy, split_prefix)

    # ── File paths ────────────────────────────────────────
    train_csv = PROJECT_ROOT / cfg["data"]["splits_dir"] / f"{split_prefix}_train.csv"
    val_csv = PROJECT_ROOT / cfg["data"]["splits_dir"] / f"{split_prefix}_val.csv"

    # Use pre-resized cached images if available (10x faster I/O)
    image_size = cfg["data"]["image_size"]
    cache_dir = PROJECT_ROOT / "data_cache" / f"images_{image_size}"
    raw_images_dir = PROJECT_ROOT / cfg["data"]["raw_dir"] / "images"
    if cache_dir.is_dir() and len(list(cache_dir.glob("*"))) > 0:
        images_dir = cache_dir
        logger.info("Using cached pre-resized images from %s", cache_dir)
    else:
        images_dir = raw_images_dir
        logger.info("No cached images at %s — loading from raw (slower)", cache_dir)
    RESULTS_DIR_NAME = {"stratified": "stratified", "center": "center_holdout", "random": "random"}
    results_dir = PROJECT_ROOT / cfg["paths"]["results_dir"] / RESULTS_DIR_NAME.get(split_strategy, split_strategy)
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
    annotations_dir_rel = cfg["data"].get("annotations_dir")

    if annotations_dir_rel:
        annotations_dir = PROJECT_ROOT / annotations_dir_rel
        if annotations_dir.is_dir():
            logger.info("Using annotation-guided dataset from %s", annotations_dir)
            train_dataset = BTXRDAnnotatedDataset(
                train_csv, images_dir, annotations_dir,
                get_train_transforms(image_size),
            )
        else:
            logger.warning(
                "annotations_dir configured (%s) but not found, "
                "falling back to standard dataset.",
                annotations_dir,
            )
            train_dataset = BTXRDDataset(
                train_csv, images_dir, get_train_transforms(image_size)
            )
    else:
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

    # ── Class weights (from ORIGINAL distribution, not augmented) ────
    # When using annotation-guided training, compute weights from the base
    # dataset to avoid skewing toward the augmented tumor classes.
    if hasattr(train_dataset, "base"):
        base_labels = train_dataset.base.labels
        logger.info("Computing class weights from base distribution (%d samples)", len(base_labels))
    else:
        base_labels = train_dataset.labels
    class_weights = compute_class_weights(
        base_labels, cfg["model"]["num_classes"]
    ).to(device)
    logger.info("Class weights: %s", class_weights.tolist())

    # ── Loss functions ────────────────────────────────────
    # CRITICAL: Weighted loss for training, unweighted for validation/early stopping
    label_smoothing = cfg["training"].get("label_smoothing", 0.0)
    train_criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=label_smoothing,
    )
    val_criterion = nn.CrossEntropyLoss()
    if label_smoothing > 0:
        logger.info("Label smoothing: %.2f", label_smoothing)

    # ── Optimizer ─────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # ── Scheduler (warmup + cosine annealing) ───────────
    epochs = cfg["training"]["epochs"]
    warmup_epochs = cfg["training"].get("warmup_epochs", 0)

    # Clamp warmup to leave at least 1 epoch for cosine annealing
    if warmup_epochs >= epochs:
        warmup_epochs = max(epochs - 1, 0)
        logger.warning(
            "warmup_epochs clamped to %d (must be < epochs=%d)",
            warmup_epochs, epochs,
        )

    if warmup_epochs > 0:
        warmup_scheduler = lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_epochs,
        )
        cosine_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6,
        )
        scheduler = lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
        logger.info("Scheduler: %d-epoch warmup + cosine annealing", warmup_epochs)
    else:
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6,
        )

    # ── Early stopping ────────────────────────────────────
    early_stopping = EarlyStopping(
        patience=cfg["training"]["early_stopping_patience"]
    )

    # ── Mixed Precision (AMP) Setup ─────────────────────
    amp_dtype = None
    scaler = None
    if device.type == "cuda":
        amp_dtype = torch.float16
        scaler = torch.amp.GradScaler("cuda")
        logger.info("AMP enabled: float16 with GradScaler (CUDA)")
    else:
        # MPS float16 is numerically unstable with class weights + label smoothing
        # Speed gain is minimal with cached images; float32 is safer
        logger.info("AMP disabled (%s)", device.type)

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
            model, train_loader, train_criterion, optimizer, device,
            scaler=scaler, amp_dtype=amp_dtype,
        )

        # Validate (with UNWEIGHTED loss for clean early stopping signal)
        val_loss, val_acc, per_class_recall = validate(
            model, val_loader, val_criterion, device, cfg["model"]["num_classes"],
            amp_dtype=amp_dtype,
        )

        # Clear MPS cache between epochs to prevent memory fragmentation
        if device.type == "mps":
            torch.mps.empty_cache()

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
