"""Bootstrap confidence intervals for AUC and per-class sensitivity.

Provides:
- bootstrap_confidence_intervals(y_true, y_pred, y_prob, class_names, ...)
  Computes 95% (configurable) confidence intervals via bootstrap resampling
  for macro AUC and per-class sensitivity. Uses percentile method with
  class-presence guard to handle small classes (e.g. 51 Malignant samples).

Implemented in Phase 5.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import recall_score, roc_auc_score


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compute bootstrap confidence intervals for macro AUC and per-class sensitivity.

    Uses bootstrap resampling with the percentile method. Includes a
    class-presence guard that skips degenerate samples where any class
    is entirely absent -- critical for the stratified split where
    Malignant has only 51 test samples.

    Args:
        y_true: Ground truth labels, shape (N,).
        y_pred: Predicted labels, shape (N,).
        y_prob: Predicted probabilities, shape (N, C).
        class_names: Ordered list of class names.
        n_iterations: Number of bootstrap iterations (default: 1000).
        confidence_level: CI level, e.g. 0.95 for 95% CI (default: 0.95).
        seed: Random seed for reproducibility (default: 42).

    Returns:
        Dict with entries for "macro_auc" and "sensitivity_{class_name}" for
        each class. Each entry contains:
          - mean: Mean of bootstrap distribution
          - ci_lower: Lower bound of confidence interval
          - ci_upper: Upper bound of confidence interval
          - n_valid: Number of valid bootstrap iterations (may be < n_iterations
            if some samples had missing classes)
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    num_classes = len(class_names)

    # Collectors
    auc_scores: list[float] = []
    sensitivity_scores: dict[int, list[float]] = {i: [] for i in range(num_classes)}

    for _ in range(n_iterations):
        # Sample with replacement
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        bt = y_true[idx]
        bp = y_pred[idx]
        bprob = y_prob[idx]

        # Class-presence guard: skip if any class is entirely absent
        if len(np.unique(bt)) < num_classes:
            continue

        # Macro AUC
        try:
            auc_val = roc_auc_score(
                bt, bprob, multi_class="ovr", average="macro"
            )
            auc_scores.append(float(auc_val))
        except ValueError:
            # Skip on failure (e.g. single-class sample slipped through)
            continue

        # Per-class sensitivity (recall)
        recalls = recall_score(
            bt,
            bp,
            average=None,
            labels=list(range(num_classes)),
            zero_division=0,
        )
        for i in range(num_classes):
            sensitivity_scores[i].append(float(recalls[i]))

    # Compute CIs using percentile method
    alpha = (1 - confidence_level) / 2
    lower_pct = alpha * 100
    upper_pct = (1 - alpha) * 100

    def _compute_ci(values: list[float]) -> dict:
        arr = np.array(values)
        if len(arr) == 0:
            return {
                "mean": float("nan"),
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
                "n_valid": 0,
            }
        return {
            "mean": float(np.mean(arr)),
            "ci_lower": float(np.percentile(arr, lower_pct)),
            "ci_upper": float(np.percentile(arr, upper_pct)),
            "n_valid": len(arr),
        }

    results: dict[str, dict] = {}
    results["macro_auc"] = _compute_ci(auc_scores)

    for i, name in enumerate(class_names):
        results[f"sensitivity_{name}"] = _compute_ci(sensitivity_scores[i])

    return results
