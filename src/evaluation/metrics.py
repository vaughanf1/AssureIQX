"""Classification metrics: accuracy, sensitivity, specificity, AUC, F1.

Provides:
- run_inference(model, dataloader, device) -- collect predictions from test set
- compute_all_metrics(y_true, y_pred, y_prob, class_names) -- EVAL-01 through EVAL-05

Metrics coverage:
  EVAL-01: Per-class ROC curves + AUC (one-vs-rest)
  EVAL-02: Per-class Precision-Recall curves + Average Precision
  EVAL-03: Per-class sensitivity (recall) and specificity
  EVAL-04: Confusion matrix (absolute counts)
  EVAL-05: Full classification report (precision, recall, F1, support)

Implemented in Phase 5.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    multilabel_confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm


def run_inference(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run model inference on a full test set.

    Sets model to eval mode and iterates over the dataloader under
    torch.no_grad(), applying softmax to raw logits exactly once.

    Args:
        model: Trained classifier (outputs raw logits).
        dataloader: Test set DataLoader.
        device: Device to run inference on.

    Returns:
        Tuple of (y_true, y_pred, y_prob):
          - y_true: shape (N,) int64 -- ground truth labels
          - y_pred: shape (N,) int64 -- predicted class (argmax of y_prob)
          - y_prob: shape (N, C) float -- softmax probabilities
    """
    model.eval()

    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Inference", leave=False):
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)

            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels).astype(np.int64)
    y_prob = np.concatenate(all_probs).astype(np.float64)
    y_pred = np.argmax(y_prob, axis=1).astype(np.int64)

    return y_true, y_pred, y_prob


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
) -> dict:
    """Compute all evaluation metrics (EVAL-01 through EVAL-05).

    Args:
        y_true: Ground truth labels, shape (N,).
        y_pred: Predicted labels, shape (N,).
        y_prob: Predicted probabilities, shape (N, C).
        class_names: Ordered list of class names (e.g. ["Normal", "Benign", "Malignant"]).

    Returns:
        Dict with keys:
          roc_fpr, roc_tpr, roc_auc -- per-class ROC data (EVAL-01)
          macro_auc -- macro-averaged AUC
          pr_precision, pr_recall, average_precision -- per-class PR data (EVAL-02)
          sensitivity, specificity -- per-class (EVAL-03)
          confusion_matrix -- ndarray (EVAL-04)
          classification_report -- dict (EVAL-05)
          malignant_sensitivity -- headline metric (float)
          accuracy -- overall accuracy (float)
    """
    num_classes = len(class_names)
    classes = list(range(num_classes))

    # Binarize labels for one-vs-rest metrics
    y_true_bin = label_binarize(y_true, classes=classes)

    # ── EVAL-01: ROC Curves + AUC ────────────────────────────────────
    roc_fpr: dict[str, np.ndarray] = {}
    roc_tpr: dict[str, np.ndarray] = {}
    roc_auc_dict: dict[str, float] = {}

    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_fpr[name] = fpr
        roc_tpr[name] = tpr
        roc_auc_dict[name] = float(auc(fpr, tpr))

    macro_auc = float(
        roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    )

    # ── EVAL-02: Precision-Recall Curves + AP ─────────────────────────
    pr_precision: dict[str, np.ndarray] = {}
    pr_recall: dict[str, np.ndarray] = {}
    avg_precision: dict[str, float] = {}

    for i, name in enumerate(class_names):
        prec, rec, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        pr_precision[name] = prec
        pr_recall[name] = rec
        avg_precision[name] = float(
            average_precision_score(y_true_bin[:, i], y_prob[:, i])
        )

    # ── EVAL-03: Sensitivity + Specificity ────────────────────────────
    mcm = multilabel_confusion_matrix(y_true, y_pred, labels=classes)
    sensitivity: dict[str, float] = {}
    specificity: dict[str, float] = {}

    for i, name in enumerate(class_names):
        tn, fp, fn, tp = mcm[i].ravel()
        sensitivity[name] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity[name] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # ── EVAL-04: Confusion Matrix ─────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # ── EVAL-05: Classification Report ────────────────────────────────
    cls_report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # ── Headline metrics ──────────────────────────────────────────────
    malignant_idx = class_names.index("Malignant")
    malignant_sensitivity = sensitivity["Malignant"]
    accuracy = float(np.mean(y_true == y_pred))

    return {
        # EVAL-01
        "roc_fpr": roc_fpr,
        "roc_tpr": roc_tpr,
        "roc_auc": roc_auc_dict,
        "macro_auc": macro_auc,
        # EVAL-02
        "pr_precision": pr_precision,
        "pr_recall": pr_recall,
        "average_precision": avg_precision,
        # EVAL-03
        "sensitivity": sensitivity,
        "specificity": specificity,
        # EVAL-04
        "confusion_matrix": cm,
        # EVAL-05
        "classification_report": cls_report,
        # Headlines
        "malignant_sensitivity": malignant_sensitivity,
        "accuracy": accuracy,
    }
