"""ROC curves, PR curves, confusion matrices, and training loss plots.

Provides publication-quality matplotlib/seaborn visualizations for
model evaluation results. All functions save to disk and never call
plt.show() (Agg backend is set at module level).

Implemented in Phase 5.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend -- must precede pyplot import

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Color scheme: Normal=Blue, Benign=Orange, Malignant=Red
CLASS_COLORS: list[str] = ["#2196F3", "#FF9800", "#F44336"]


def plot_roc_curves(
    metrics: dict,
    class_names: list[str],
    output_path: str | Path,
) -> None:
    """Plot one-vs-rest ROC curves with macro-average.

    Generates a single figure with per-class ROC curves and a macro-average
    curve computed by interpolating per-class TPR to a common FPR grid.

    Args:
        metrics: Dict from compute_all_metrics containing roc_fpr, roc_tpr, roc_auc.
        class_names: Ordered class names matching metrics keys.
        output_path: File path to save the PNG.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Per-class ROC curves
    mean_fpr = np.linspace(0, 1, 1000)
    tprs = []

    for i, name in enumerate(class_names):
        fpr = metrics["roc_fpr"][name]
        tpr = metrics["roc_tpr"][name]
        auc_val = metrics["roc_auc"][name]

        ax.plot(
            fpr,
            tpr,
            color=CLASS_COLORS[i],
            linewidth=2,
            label=f"{name} (AUC = {auc_val:.3f})",
        )

        # Interpolate for macro-average
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    # Macro-average ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    macro_auc = metrics["macro_auc"]

    ax.plot(
        mean_fpr,
        mean_tpr,
        color="navy",
        linewidth=2,
        linestyle="--",
        label=f"Macro-average (AUC = {macro_auc:.3f})",
    )

    # Diagonal reference
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", alpha=0.3)

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curves(
    metrics: dict,
    class_names: list[str],
    output_path: str | Path,
) -> None:
    """Plot per-class Precision-Recall curves.

    Args:
        metrics: Dict from compute_all_metrics containing pr_precision,
                 pr_recall, average_precision.
        class_names: Ordered class names matching metrics keys.
        output_path: File path to save the PNG.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    for i, name in enumerate(class_names):
        precision = metrics["pr_precision"][name]
        recall = metrics["pr_recall"][name]
        ap_val = metrics["average_precision"][name]

        ax.plot(
            recall,
            precision,
            color=CLASS_COLORS[i],
            linewidth=2,
            label=f"{name} (AP = {ap_val:.3f})",
        )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves (One-vs-Rest)", fontsize=14)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(
    cm: np.ndarray,
    class_names: list[str],
    output_dir: str | Path,
) -> None:
    """Plot absolute and row-normalized confusion matrix heatmaps.

    Saves two files:
      - confusion_matrix.png (absolute counts)
      - confusion_matrix_normalized.png (row-normalized proportions)

    Args:
        cm: Confusion matrix ndarray of shape (C, C).
        class_names: Ordered class names for axis labels.
        output_dir: Directory to save the PNG files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Absolute counts ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)

    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Row-normalized ────────────────────────────────────────────
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_norm = cm.astype(np.float64) / np.where(row_sums == 0, 1, row_sums)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        vmin=0,
        vmax=1,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix (Normalized)", fontsize=14)

    fig.tight_layout()
    fig.savefig(
        output_dir / "confusion_matrix_normalized.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)
