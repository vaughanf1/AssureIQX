"""Model factory for creating and loading classifiers from config.

Provides:
- create_model(config) -- builds a BTXRDClassifier from YAML config
- save_checkpoint / load_checkpoint -- checkpoint serialization with full state
- compute_class_weights -- inverse-frequency weights for imbalanced CrossEntropyLoss
- get_device -- auto-detect CUDA > MPS > CPU
- EarlyStopping -- patience-based training termination

All functions operate on CPU tensors by default; callers move to device.

Implemented in Phase 4.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.models.classifier import BTXRDClassifier


def get_device(device_str: str = "auto") -> torch.device:
    """Resolve device string to torch.device.

    Priority when device_str is "auto": CUDA > MPS > CPU.

    Args:
        device_str: One of "auto", "cpu", "cuda", "mps".

    Returns:
        Resolved torch.device.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def create_model(config: dict) -> BTXRDClassifier:
    """Create a BTXRDClassifier from a config dict.

    Reads config["model"] section for backbone, num_classes, pretrained,
    and dropout parameters. If ``model.attn_layer`` is set (e.g., "cbam"),
    it is passed through as ``block_args=dict(attn_layer=...)`` to timm.

    Args:
        config: Full config dict (e.g., from configs/default.yaml).

    Returns:
        Configured BTXRDClassifier instance.
    """
    model_cfg = config["model"]

    # Build extra kwargs for timm.create_model (e.g., CBAM attention)
    kwargs: dict = {}
    attn_layer = model_cfg.get("attn_layer")
    if attn_layer:
        kwargs["block_args"] = dict(attn_layer=attn_layer)

    return BTXRDClassifier(
        backbone=model_cfg["backbone"],
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg["pretrained"],
        drop_rate=model_cfg.get("dropout", 0.2),
        **kwargs,
    )


def compute_class_weights(
    labels: list[int], num_classes: int = 3
) -> torch.Tensor:
    """Compute inverse-frequency class weights for CrossEntropyLoss.

    Formula: weight_c = n_samples / (n_classes * count_c)
    This is identical to sklearn's compute_class_weight('balanced').

    Args:
        labels: List of integer class labels from the training set.
        num_classes: Number of classes.

    Returns:
        Float32 tensor of shape (num_classes,) with per-class weights.
        Returned on CPU -- caller must move to device before use.
    """
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    weights = total / (num_classes * counts.astype(np.float64))
    return torch.tensor(weights, dtype=torch.float32)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object | None,
    epoch: int,
    val_loss: float,
    config: dict,
    class_weights: torch.Tensor,
    filepath: str,
) -> None:
    """Save a training checkpoint with full state for resumption.

    Checkpoint contains: model weights, optimizer state, scheduler state,
    training config, class names, normalization stats, and class weights.

    Args:
        model: Trained model (state_dict will be extracted).
        optimizer: Optimizer with adaptive state.
        scheduler: LR scheduler (or None if not used).
        epoch: Current epoch number.
        val_loss: Validation loss at this checkpoint.
        config: Full training config dict.
        class_weights: Class weight tensor used for weighted loss.
        filepath: Output path for the .pt checkpoint file.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "val_loss": val_loss,
        "config": config,
        "class_names": config["model"]["class_names"],
        "normalization": {
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
        },
        "class_weights": class_weights.tolist(),
    }
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath: str, device: str = "cpu") -> dict:
    """Load a training checkpoint.

    Args:
        filepath: Path to .pt checkpoint file.
        device: Device to map tensors to (default: "cpu" for portability).

    Returns:
        Checkpoint dict with model_state_dict, optimizer_state_dict,
        scheduler_state_dict, config, class_names, normalization, etc.
    """
    return torch.load(filepath, map_location=device, weights_only=False)


class EarlyStopping:
    """Monitor validation loss and stop training when no improvement.

    Tracks best validation loss and counts consecutive epochs without
    improvement. Returns True (stop) when patience is exhausted.

    Args:
        patience: Number of epochs to wait after last improvement.
        min_delta: Minimum decrease in val_loss to qualify as improvement.
    """

    def __init__(self, patience: int = 7, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter: int = 0
        self.best_loss: float = float("inf")
        self.should_stop: bool = False

    def __call__(self, val_loss: float) -> bool:
        """Check if training should stop.

        Args:
            val_loss: Current epoch's validation loss.

        Returns:
            True if patience exhausted (training should stop).
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop
