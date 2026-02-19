# Phase 4: Model Training - Research

**Researched:** 2026-02-19
**Domain:** PyTorch model training, timm EfficientNet-B0, class-imbalanced loss, early stopping, checkpoint management
**Confidence:** HIGH

## Summary

Phase 4 implements the complete training pipeline for an EfficientNet-B0 classifier on the BTXRD bone tumor dataset. The core deliverables are: (1) a model architecture module in `src/models/classifier.py` wrapping timm's `efficientnet_b0` with a 3-class output head, (2) a training script `scripts/train.py` with inverse-frequency weighted cross-entropy loss, early stopping, and checkpoint saving, and (3) training logs and loss curve plots proving convergence and Malignant class viability.

All required libraries are already installed and verified: timm 1.0.15 (EfficientNet-B0 creation), torch 2.6.0 (training loop, optimizer, scheduler, CrossEntropyLoss), matplotlib 3.10.0 (loss curves), tqdm 4.67.1 (progress bars), and scikit-learn 1.7.2 (per-class recall computation). No new dependencies are needed. The development machine has MPS (Apple Silicon) acceleration available; CUDA is not present. The training pipeline must support `auto` device detection (CUDA > MPS > CPU).

Key verified facts: timm 1.0.15 creates EfficientNet-B0 via `timm.create_model('efficientnet_b0', pretrained=True, num_classes=3, drop_rate=0.2)` producing a model with 4.01M parameters, a `Linear(1280, 3)` classifier head, and ImageNet normalization stats matching our existing pipeline. The Grad-CAM target layer for Phase 6 is confirmed as `model.bn2` (or `model.conv_head`) -- both work with pytorch-grad-cam 1.5.5. The training set has 2,621 samples (stratified) and 2,499 samples (center) with 82/79 batches per epoch at batch_size=32. Malignant class has only 240/200 training samples, making weighted loss essential.

**Primary recommendation:** Use full fine-tuning (all parameters trainable) with AdamW optimizer, cosine annealing LR scheduler, and inverse-frequency weighted CrossEntropyLoss. Train both split strategies sequentially. Monitor per-class recall every epoch to confirm Malignant class is not collapsing.

## Standard Stack

### Core (all already in requirements.txt)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| timm | 1.0.15 | `timm.create_model('efficientnet_b0', pretrained=True, num_classes=3)` | De-facto model zoo for PyTorch; handles weight download, classifier head replacement, drop_rate config |
| torch | 2.6.0 | Training loop, `nn.CrossEntropyLoss`, `optim.AdamW`, `lr_scheduler.CosineAnnealingLR` | Pinned in Phase 1 |
| scikit-learn | 1.7.2 | `recall_score(average=None)` for per-class Malignant recall monitoring | Already used for splitting; per-class metrics are standard |
| matplotlib | 3.10.0 | Loss curve plots (train_loss and val_loss vs epoch) | Already in requirements.txt |
| tqdm | 4.67.1 | Training progress bars with loss display | Already in requirements.txt |
| pandas | 2.2.3 | Training log CSV read/write | Already in requirements.txt |
| numpy | 2.2.3 | Class weight computation | Already in requirements.txt |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| csv (stdlib) | N/A | Alternative to pandas for simple CSV append | If line-by-line CSV writing is preferred |
| json (stdlib) | N/A | Saving training metadata alongside checkpoints | Config serialization in checkpoint |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| AdamW | SGD+momentum | AdamW converges faster for transfer learning on small datasets; SGD may generalize better but requires more tuning (momentum, LR warmup) |
| CosineAnnealingLR | ReduceLROnPlateau | Cosine is deterministic (same schedule regardless of loss trajectory); ReduceLROnPlateau is reactive but harder to reproduce |
| Inverse-frequency weights | Focal Loss | Focal loss is more complex, adds a gamma hyperparameter; inverse-frequency weights are simpler and well-proven for moderate imbalance (5.5:1 ratio) |
| Full fine-tuning | Head-only then full | Two-phase training adds complexity; with 2600 training images and a small model (4M params), full fine-tuning with low LR is standard practice |

**Installation:** No new packages needed. All are in `requirements.txt`.

## Architecture Patterns

### Recommended Project Structure

```
src/models/
  classifier.py      # BTXRDClassifier nn.Module wrapping timm model
  factory.py          # create_model() and load_checkpoint() factory functions
  __init__.py         # Exports

scripts/
  train.py            # CLI entry point: config -> dataloaders -> train loop -> checkpoints

configs/
  default.yaml        # Already has model/training/paths sections

checkpoints/          # Output directory
  best_stratified.pt
  final_stratified.pt
  best_center.pt
  final_center.pt

results/
  stratified/
    training_log.csv
    loss_curve.png
  center_holdout/
    training_log.csv
    loss_curve.png
```

### Pattern 1: Model Architecture with timm

**What:** Wrap timm's EfficientNet-B0 in a thin `nn.Module` that exposes the model cleanly and provides convenience methods for checkpoint saving/loading.

**When to use:** Always. The wrapper provides a stable interface for the rest of the pipeline (eval, inference, Grad-CAM).

**Example:**
```python
# Source: Verified against timm 1.0.15 installed in .venv
import timm
import torch
import torch.nn as nn

class BTXRDClassifier(nn.Module):
    """EfficientNet-B0 classifier for 3-class bone tumor classification."""

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        num_classes: int = 3,
        pretrained: bool = True,
        drop_rate: float = 0.2,
    ):
        super().__init__()
        self.model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )
        self.num_classes = num_classes
        self.num_features = self.model.num_features  # 1280 for efficientnet_b0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @property
    def gradcam_target_layer(self):
        """Return the layer to use for Grad-CAM (last conv+bn before pooling)."""
        return self.model.bn2
```

**Verified facts (HIGH confidence):**
- `timm.create_model('efficientnet_b0', pretrained=True, num_classes=3, drop_rate=0.2)` produces a model with 4,011,391 parameters
- Classifier head is `Linear(in_features=1280, out_features=3, bias=True)`
- `model.num_features` returns 1280
- `model.forward_features(x)` returns shape `(batch, 1280, 7, 7)` for 224x224 input
- `model.bn2` is `BatchNormAct2d(1280)` -- confirmed Grad-CAM target layer works with pytorch-grad-cam 1.5.5
- `model.conv_head` is `Conv2d` (1280 channels) -- also works as Grad-CAM target
- Default ImageNet normalization stats: mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) -- matches our transforms.py

### Pattern 2: Inverse-Frequency Weighted CrossEntropyLoss

**What:** Compute class weights from training set label counts using `n_samples / (n_classes * class_count)` formula (identical to sklearn's `compute_class_weight('balanced')`). Pass as `weight` tensor to `nn.CrossEntropyLoss`.

**When to use:** Required for TRAIN-04. Class distribution is 1315:1066:240 (stratified) or 1355:944:200 (center) -- Malignant is ~5.5x underrepresented.

**Example:**
```python
# Source: Verified against PyTorch 2.6.0 CrossEntropyLoss + sklearn compute_class_weight
import numpy as np
import torch

def compute_class_weights(labels: list[int], num_classes: int = 3) -> torch.Tensor:
    """Compute inverse-frequency class weights for CrossEntropyLoss.

    Formula: weight_c = n_samples / (n_classes * count_c)
    This is identical to sklearn's compute_class_weight('balanced').

    Args:
        labels: List of integer class labels from training set.
        num_classes: Number of classes.

    Returns:
        Float32 tensor of shape (num_classes,) with per-class weights.
    """
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    weights = total / (num_classes * counts.astype(np.float64))
    return torch.tensor(weights, dtype=torch.float32)

# Verified weights for stratified split:
# Normal=0.664, Benign=0.820, Malignant=3.640
# Verified weights for center split:
# Normal=0.615, Benign=0.882, Malignant=4.165
```

**Verified facts (HIGH confidence):**
- `nn.CrossEntropyLoss(weight=tensor)` accepts a 1D float tensor of length `num_classes`
- Weight tensor must be on the same device as model/targets
- The `weight` parameter rescales the loss per-class: loss_c = -weight_c * log(softmax(logit_c))
- Unweighted validation loss should be used for early stopping (prevents bias toward Malignant in stopping criterion)
- `label_smoothing` parameter is available (default 0.0) -- not required but could help generalization

### Pattern 3: Training Loop with Early Stopping

**What:** Standard PyTorch training loop with epoch-level validation, early stopping on validation loss, and per-class metric monitoring.

**When to use:** Required for TRAIN-05, TRAIN-06.

**Example:**
```python
# Source: Standard PyTorch training pattern, verified on MPS device
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device, num_classes=3):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(all_labels)
    per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    return epoch_loss, per_class_recall


class EarlyStopping:
    """Early stopping on validation loss with configurable patience."""

    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop
```

**Verified facts (HIGH confidence):**
- MPS (Apple Silicon) supports all required operations: forward pass, backward pass, optimizer step, loss computation -- tested on this machine
- `torch.use_deterministic_algorithms(True, warn_only=True)` works on MPS without errors
- `model.train()` enables dropout; `model.eval()` disables it -- verified
- Validation should use UNWEIGHTED loss for early stopping to avoid bias (weighted loss in training only)

### Pattern 4: Checkpoint Format

**What:** Save comprehensive checkpoints containing everything needed to resume training or run inference.

**When to use:** Required for TRAIN-07.

**Example:**
```python
# Source: Verified with PyTorch 2.6.0 torch.save/load
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "val_loss": val_loss,
    "best_val_loss": best_val_loss,
    "config": config,  # Full YAML config dict
    "class_names": ["Normal", "Benign", "Malignant"],
    "normalization": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "class_weights": class_weights.tolist(),
}

# Save
torch.save(checkpoint, "checkpoints/best_stratified.pt")

# Load (device-portable)
loaded = torch.load("checkpoints/best_stratified.pt", map_location="cpu", weights_only=False)
```

**Verified facts (HIGH confidence):**
- Checkpoint size: ~15.6 MB for EfficientNet-B0
- `map_location='cpu'` correctly moves all tensors to CPU when loading from MPS/CUDA
- `weights_only=False` is required for loading checkpoints with non-tensor data (config dicts, class names); `weights_only=True` also works for PyTorch 2.6.0 with simple nested dict/list/tensor data
- Loading and running forward pass produces valid 3-class softmax outputs (verified)

### Pattern 5: Training Log CSV and Loss Curves

**What:** Log epoch-level metrics to CSV and plot loss curves as PNG.

**When to use:** Required for TRAIN-08.

**Example:**
```python
# CSV format: epoch,train_loss,val_loss,val_acc,val_recall_normal,val_recall_benign,val_recall_malignant,lr
# Source: Standard practice, matches results/{split_name}/training_log.csv path from requirements

import csv
import matplotlib.pyplot as plt

def save_training_log(log_rows: list[dict], filepath: str):
    """Save training log as CSV."""
    fieldnames = [
        "epoch", "train_loss", "val_loss", "val_acc",
        "val_recall_normal", "val_recall_benign", "val_recall_malignant", "lr"
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)


def plot_loss_curves(log_rows: list[dict], filepath: str):
    """Plot training and validation loss curves."""
    epochs = [r["epoch"] for r in log_rows]
    train_loss = [r["train_loss"] for r in log_rows]
    val_loss = [r["val_loss"] for r in log_rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_loss, label="Train Loss", marker="o", markersize=3)
    ax.plot(epochs, val_loss, label="Val Loss", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
```

### Anti-Patterns to Avoid

- **Using weighted loss for validation:** Validation loss must be UNWEIGHTED (`nn.CrossEntropyLoss()` without `weight` parameter) to provide an unbiased signal for early stopping. Weighted loss is for training only.
- **Forgetting `model.eval()` during validation:** Dropout and batch norm behave differently in train vs eval mode. Always call `model.eval()` before validation loop and `model.train()` before training loop.
- **Computing class weights from entire dataset instead of training set:** Weights must be computed from the TRAINING split counts only, not from the full dataset. The training set class distribution differs slightly from the overall distribution, especially for the center-holdout split.
- **Not moving class weight tensor to device:** `nn.CrossEntropyLoss(weight=weights)` where `weights` is on CPU but model/targets are on MPS will cause a device mismatch error.
- **Saving model instead of model.state_dict():** Always save `model.state_dict()` (not the model object) for portability. Saving the model object pickles the class definition, creating fragile dependencies.
- **Not using `torch.no_grad()` during validation:** Forgetting this wastes memory on gradient computation during evaluation.
- **Hardcoding device string:** Use the `device: auto` config pattern to auto-detect CUDA > MPS > CPU.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| EfficientNet-B0 architecture | Manual conv block stacking | `timm.create_model('efficientnet_b0', pretrained=True, num_classes=3)` | timm handles: weight download, classifier head replacement, squeeze-excitation blocks, drop_rate, ImageNet normalization stats |
| Class weight computation | Manual counting and division | `n_samples / (n_classes * counts)` formula or `sklearn.utils.class_weight.compute_class_weight('balanced')` | Formula is well-established; sklearn implementation verified to produce identical results |
| Learning rate scheduling | Manual LR decay logic | `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)` | Handles cosine curve math, eta_min floor, state_dict for checkpoint saving |
| Progress bars | Print statements | `tqdm(loader, desc=f"Epoch {epoch}")` with `.set_postfix(loss=loss)` | Handles terminal width, ETA, iteration speed, postfix formatting |
| Per-class recall | Manual TP/FP/FN counting per class | `sklearn.metrics.recall_score(y_true, y_pred, average=None, zero_division=0)` | Handles edge cases (zero predictions for a class), multi-class correctly |

**Key insight:** The training loop itself is the one piece that must be hand-written (PyTorch does not provide a built-in training loop). But every component WITHIN the loop (model, loss, optimizer, scheduler, metrics) should use well-tested library implementations.

## Common Pitfalls

### Pitfall 1: Malignant Class Collapse

**What goes wrong:** With only 240 Malignant training samples (stratified) or 200 (center), the model can learn to never predict Malignant -- achieving ~93.5% accuracy by predicting only Normal/Benign. The weighted loss is supposed to prevent this, but if weights are miscalculated or the learning rate is too high, collapse can still occur.
**Why it happens:** Severe class imbalance (5.5:1 ratio) combined with cross-entropy's tendency to push logits toward majority classes.
**How to avoid:** (1) Use inverse-frequency weights as specified (Malignant weight ~3.6x). (2) Monitor per-class recall every epoch -- if Malignant recall drops to 0 for 2+ consecutive epochs, the model is collapsing. (3) Log per-class recall in the training CSV so collapse is detectable post-hoc.
**Warning signs:** Malignant recall = 0, overall accuracy plateau at ~93%, confusion matrix showing zero Malignant predictions.

### Pitfall 2: MPS Device Compatibility Issues

**What goes wrong:** Some PyTorch operations are not fully supported on MPS (Apple Silicon), leading to runtime errors or silent numerical differences compared to CUDA/CPU.
**Why it happens:** MPS backend is newer than CUDA; not all operations have deterministic MPS implementations.
**How to avoid:** (1) `torch.use_deterministic_algorithms(True, warn_only=True)` is already configured in `set_seed()`. (2) The core operations (conv2d, linear, cross_entropy, adamw) are all verified working on this machine's MPS. (3) If MPS causes issues, fall back to CPU via `--override device=cpu`.
**Warning signs:** NaN losses, dramatically different results between MPS and CPU runs, RuntimeError mentioning MPS.

### Pitfall 3: Weighted vs Unweighted Validation Loss

**What goes wrong:** Using the SAME weighted loss for both training and validation makes early stopping biased -- it over-values Malignant validation samples, causing early stopping to trigger based on minority class performance rather than overall model quality.
**Why it happens:** Natural assumption that training and validation should use the same loss function.
**How to avoid:** Use weighted `CrossEntropyLoss` for TRAINING only. Use unweighted `CrossEntropyLoss` (no `weight` parameter) for VALIDATION loss that drives early stopping. Report both in the training log.
**Warning signs:** Early stopping triggers very early; validation loss appears unstable despite good per-class metrics.

### Pitfall 4: Forgetting to Move Weight Tensor to Device

**What goes wrong:** `RuntimeError: Expected all tensors to be on the same device` when the class weight tensor is on CPU but the model outputs and targets are on MPS/CUDA.
**Why it happens:** Class weights are computed as a numpy array, converted to a CPU tensor, but not moved to the training device.
**How to avoid:** `weights = compute_class_weights(labels).to(device)` before passing to `CrossEntropyLoss`.
**Warning signs:** Immediate crash at first training step with device mismatch error.

### Pitfall 5: Not Saving Optimizer and Scheduler State in Checkpoint

**What goes wrong:** If training needs to be resumed (e.g., power failure, manual inspection), the optimizer's adaptive state (Adam momentum) and scheduler's position are lost. Resuming from a checkpoint without optimizer state effectively restarts training with a fresh optimizer, potentially causing a spike in loss.
**Why it happens:** Saving only `model_state_dict` seems sufficient for inference, but resumable training requires full state.
**How to avoid:** Always save `optimizer.state_dict()` and `scheduler.state_dict()` in every checkpoint.
**Warning signs:** After resuming, loss suddenly spikes before slowly recovering.

### Pitfall 6: num_workers > 0 on macOS with MPS

**What goes wrong:** On some macOS configurations, `num_workers > 0` in DataLoader can cause multiprocessing issues (deadlocks, memory leaks, or ObjC runtime warnings).
**Why it happens:** macOS fork() behavior with multithreaded libraries can be problematic. Python multiprocessing on macOS uses `spawn` by default in newer versions, which is safer but slower.
**How to avoid:** Start with `num_workers=4` (configured in default.yaml). If DataLoader hangs or crashes, reduce to `num_workers=0` (single-process data loading). The dataset is small enough that `num_workers=0` would still be fast.
**Warning signs:** Training hangs at first batch, or ObjC runtime warnings appear in console.

### Pitfall 7: Checkpoint Naming Collision Between Split Strategies

**What goes wrong:** Training both split strategies overwrites the same checkpoint file because the filename does not include the split strategy name.
**Why it happens:** Default checkpoint path is hardcoded without parameterization.
**How to avoid:** Use `checkpoints/best_{split_strategy}.pt` and `checkpoints/final_{split_strategy}.pt` naming convention. The config already has `training.split_strategy` field.
**Warning signs:** After training center split, stratified checkpoint is missing.

## Code Examples

### Complete Model Factory

```python
# Source: Verified against timm 1.0.15
import timm
import torch
import torch.nn as nn
from pathlib import Path

def create_model(config: dict) -> nn.Module:
    """Create an EfficientNet classifier from config.

    Args:
        config: Full config dict with model.backbone, model.num_classes,
                model.pretrained, model.dropout.

    Returns:
        Configured EfficientNet model.
    """
    model_cfg = config["model"]
    model = timm.create_model(
        model_cfg["backbone"],         # "efficientnet_b0"
        pretrained=model_cfg["pretrained"],  # True
        num_classes=model_cfg["num_classes"],  # 3
        drop_rate=model_cfg.get("dropout", 0.2),  # 0.2
    )
    return model


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    val_loss: float,
    config: dict,
    filepath: str,
):
    """Save a training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "val_loss": val_loss,
        "config": config,
        "class_names": config["model"]["class_names"],
        "normalization": {
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
        },
    }
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath: str, device: str = "cpu") -> dict:
    """Load a training checkpoint.

    Args:
        filepath: Path to .pt checkpoint file.
        device: Device to map tensors to (default: cpu for portability).

    Returns:
        Checkpoint dict with model_state_dict, config, etc.
    """
    return torch.load(filepath, map_location=device, weights_only=False)
```

### Device Auto-Detection

```python
# Source: Verified on this machine (MPS available, CUDA not)
import torch

def get_device(device_str: str = "auto") -> torch.device:
    """Resolve device string to torch.device.

    Priority: CUDA > MPS > CPU when device_str is "auto".
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)
```

### Class Weight Computation from Training Set

```python
# Source: Verified identical to sklearn compute_class_weight('balanced')
import numpy as np
import torch

def compute_class_weights(labels: list[int], num_classes: int = 3) -> torch.Tensor:
    """Compute inverse-frequency class weights.

    Formula: weight_c = n_samples / (n_classes * count_c)

    Returns:
        Float32 tensor of shape (num_classes,).
    """
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    weights = total / (num_classes * counts.astype(np.float64))
    return torch.tensor(weights, dtype=torch.float32)

# Verified values:
# Stratified: [0.664, 0.820, 3.640]
# Center:     [0.615, 0.882, 4.165]
```

### Early Stopping Implementation

```python
# Source: Standard pattern verified in PyTorch ecosystem
class EarlyStopping:
    """Monitor validation loss and stop training on plateau.

    Args:
        patience: Epochs to wait after last improvement.
        min_delta: Minimum decrease to qualify as improvement.
    """

    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop
```

### Training Script Main Flow

```python
# High-level pseudocode for scripts/train.py
def main():
    # 1. Parse args, load config, set seed
    cfg = load_config(args.config, overrides=args.override)
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])

    # 2. Build data loaders (from Phase 3 components)
    split_strategy = cfg["training"]["split_strategy"]  # "stratified" or "center"
    train_dataset = BTXRDDataset(manifest, images_dir, get_train_transforms(image_size))
    val_dataset = BTXRDDataset(manifest, images_dir, get_val_transforms(image_size))
    train_loader = create_dataloader(train_dataset, batch_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size, shuffle=False)

    # 3. Build model, optimizer, scheduler, loss
    model = create_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    class_weights = compute_class_weights(train_dataset.labels).to(device)
    train_criterion = nn.CrossEntropyLoss(weight=class_weights)
    val_criterion = nn.CrossEntropyLoss()  # UNWEIGHTED for early stopping

    # 4. Training loop
    early_stopping = EarlyStopping(patience=cfg["training"]["early_stopping_patience"])
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, train_criterion, optimizer, device)
        val_loss, per_class_recall = validate(model, val_loader, val_criterion, device)
        scheduler.step()

        # Log metrics
        log_row = {epoch, train_loss, val_loss, per_class_recall, lr}
        log_rows.append(log_row)

        # Save best checkpoint
        if val_loss < best_val_loss:
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, cfg, best_path)

        # Early stopping
        if early_stopping(val_loss):
            break

    # 5. Save final checkpoint, training log CSV, loss curve plot
    save_checkpoint(model, optimizer, scheduler, epoch, val_loss, cfg, final_path)
    save_training_log(log_rows, log_csv_path)
    plot_loss_curves(log_rows, plot_path)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual model construction | `timm.create_model()` with pretrained weights | timm 0.3+ (~2021) | One-line model creation with any backbone, automatic weight download, classifier head replacement |
| SGD + StepLR | AdamW + CosineAnnealingLR | ~2019 (Loshchilov & Hutter) | AdamW decouples weight decay from gradient update; cosine schedule provides smooth LR decay |
| Manual early stopping | Library implementations (Lightning, etc.) | ~2020+ | For vanilla PyTorch, the manual `EarlyStopping` class pattern is standard; no need for a framework dependency |
| Class oversampling (WeightedRandomSampler) | Weighted CrossEntropyLoss | Both are current | For moderate imbalance (5:1), weighted loss is simpler and equally effective; oversampling changes effective epoch length |
| `torch.save(model, path)` | `torch.save(model.state_dict(), path)` | PyTorch best practice since ~2019 | state_dict is more portable, avoids class definition pickle issues |
| `torch.load(path)` no args | `torch.load(path, map_location=device, weights_only=False)` | PyTorch 2.6+ | weights_only default changed; map_location ensures device portability |

**Deprecated/outdated:**
- `torch.load()` without `weights_only` parameter: PyTorch 2.6+ issues FutureWarning; explicitly pass `weights_only=False` for checkpoints with non-tensor data
- `torch.optim.lr_scheduler.StepLR`: Still works but cosine annealing is preferred for transfer learning

## Open Questions

1. **Full Fine-Tuning vs. Frozen Backbone**
   - What we know: Full fine-tuning (all 4.01M params trainable) is the standard for small datasets with transfer learning. Head-only training leaves only 3,843 params trainable (0.1%), which is likely insufficient.
   - What's unclear: Whether the training set (2,621 images) is large enough for full fine-tuning without severe overfitting.
   - Recommendation: Use full fine-tuning with the configured learning rate (0.001) and weight decay (0.0001). Early stopping (patience=7) and dropout (0.2) provide regularization. If validation loss diverges rapidly, consider lowering LR to 0.0001 as a CLI override. This is a PoC -- start simple.

2. **Training Both Splits: Sequential vs. Makefile Targets**
   - What we know: The roadmap plans show 04-03 as "Train on both split strategies." The Makefile `train` target runs a single command.
   - What's unclear: Should `make train` train both strategies automatically, or should the user run it twice with different `--override training.split_strategy=...`?
   - Recommendation: The training script should accept `training.split_strategy` from config. `make train` trains the default strategy (stratified). Add `make train-all` target that trains both sequentially. This keeps the simple case simple.

3. **num_workers on macOS/MPS**
   - What we know: The config specifies `data.num_workers: 4`. On macOS, multiprocessing in DataLoader can sometimes hang.
   - What's unclear: Whether num_workers=4 will be stable on this specific machine with MPS.
   - Recommendation: Keep `num_workers: 4` as default. Document that `--override data.num_workers=0` is the fallback if DataLoader hangs. The dataset is small enough that `num_workers=0` would add only a few seconds per epoch.

4. **Validation Loss: Weighted or Unweighted for Logging?**
   - What we know: Early stopping should use unweighted val loss. But the training log CSV should show useful metrics.
   - What's unclear: Should the CSV log weighted val loss, unweighted val loss, or both?
   - Recommendation: Log UNWEIGHTED val loss (used for early stopping) as `val_loss` in the CSV. This is the canonical metric for convergence tracking. Per-class recall columns capture the class imbalance story separately.

## Sources

### Primary (HIGH confidence)

- **timm 1.0.15 (installed, verified):** `timm.create_model('efficientnet_b0', pretrained=True, num_classes=3, drop_rate=0.2)` -- verified model architecture, num_features=1280, classifier head, bn2 layer for Grad-CAM, forward pass shape, parameter count (4,011,391)
- **PyTorch 2.6.0 (installed, verified):** `nn.CrossEntropyLoss(weight=tensor)`, `optim.AdamW`, `lr_scheduler.CosineAnnealingLR`, `torch.save/load` with `map_location` and `weights_only` -- all verified on MPS device
- **pytorch-grad-cam 1.5.5 (installed, verified):** `GradCAM(model=model, target_layers=[model.bn2])` produces valid (1, 224, 224) CAM output -- verified for Phase 6 compatibility
- **scikit-learn 1.7.2 (installed, verified):** `recall_score(average=None, zero_division=0)` and `compute_class_weight('balanced')` -- both produce correct results matching manual computation
- **matplotlib 3.10.0 (installed, verified):** Loss curve plotting to PNG -- verified working

### Secondary (MEDIUM confidence)

- [PyTorch CrossEntropyLoss documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) -- weight parameter, label_smoothing parameter
- [timm EfficientNet documentation](https://huggingface.co/docs/timm/models/efficientnet) -- model variants and configuration
- [PyTorch CosineAnnealingLR documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html) -- T_max, eta_min parameters
- [PyTorch Forums: weighted cross entropy for class imbalance](https://discuss.pytorch.org/t/solving-class-imbalance-by-implementing-weighted-cross-entropy/109691) -- community validation of inverse-frequency weighting approach

### Tertiary (LOW confidence)

- WebSearch results for medical image classification best practices -- confirmed by direct library verification above

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- All libraries installed, verified with actual code execution on this machine; no dependency gaps
- Architecture: HIGH -- Model creation, forward pass, checkpoint save/load, loss computation, optimizer step, scheduler step all verified on MPS device with correct shapes and values
- Pitfalls: HIGH -- All pitfalls derived from verified behavior (device mismatch tested, weighted vs unweighted loss tested, class weight computation validated against sklearn, Grad-CAM target layer confirmed)

**Research date:** 2026-02-19
**Valid until:** 2026-03-19 (stable domain; timm 1.0.15, PyTorch 2.6.0, and all deps are pinned)
