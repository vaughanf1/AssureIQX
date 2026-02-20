---
phase: 05-evaluation
plan: 01
subsystem: evaluation
tags: [metrics, roc, pr-curve, confusion-matrix, sensitivity, specificity, classification-report]

dependency_graph:
  requires: [04-01, 04-02, 04-03]
  provides: [per-split-metrics, evaluation-plots, metrics-json]
  affects: [05-02, 07-01]

tech_stack:
  added: []
  patterns: [one-vs-rest-metrics, publication-quality-plots, json-metric-export]

key_files:
  created:
    - src/evaluation/metrics.py
    - src/evaluation/visualization.py
    - results/stratified/metrics_summary.json
    - results/stratified/classification_report.json
    - results/stratified/roc_curves.png
    - results/stratified/pr_curves.png
    - results/stratified/confusion_matrix.png
    - results/stratified/confusion_matrix_normalized.png
    - results/center_holdout/metrics_summary.json
    - results/center_holdout/classification_report.json
    - results/center_holdout/roc_curves.png
    - results/center_holdout/pr_curves.png
    - results/center_holdout/confusion_matrix.png
    - results/center_holdout/confusion_matrix_normalized.png
  modified:
    - src/evaluation/__init__.py
    - scripts/eval.py

decisions:
  - id: eval-softmax-once
    description: "Softmax applied exactly once in run_inference; model outputs raw logits"
  - id: eval-ovr-metrics
    description: "One-vs-Rest strategy for ROC and PR curves via label_binarize"
  - id: eval-malignant-headline
    description: "Malignant sensitivity logged as headline metric over overall accuracy"

metrics:
  duration: 10 min
  completed: 2026-02-20
---

# Phase 5 Plan 1: Core Evaluation Pipeline Summary

**One-liner:** Full EVAL-01 through EVAL-05 metric suite with OvR ROC/PR curves, confusion matrices, and per-split JSON exports driven by eval.py orchestration script.

## What Was Done

### Task 1: Implement metrics.py and visualization.py
**Commit:** 1bc4f67

Replaced docstring-only stubs with full implementations:

- **metrics.py** -- `run_inference()` collects predictions under `torch.no_grad()` with softmax applied exactly once; `compute_all_metrics()` computes all five evaluation criteria (ROC+AUC, PR+AP, sensitivity/specificity, confusion matrix, classification report) and returns a structured dict.
- **visualization.py** -- `plot_roc_curves()` renders per-class OvR curves plus macro-average interpolated over 1000 FPR points; `plot_pr_curves()` renders per-class PR curves with AP in legend; `plot_confusion_matrices()` saves both absolute-count and row-normalized heatmaps using seaborn.
- **__init__.py** -- Updated to export all five public functions.

### Task 2: Implement eval.py orchestration script
**Commit:** 2d56643

Replaced `NotImplementedError` placeholder with full evaluation orchestration:

- Iterates over split strategies from config (`stratified`, `center`)
- Loads best checkpoint, reconstructs model from checkpoint config
- Creates test dataset/dataloader with deterministic transforms
- Runs inference, computes all metrics, generates all plots
- Saves `metrics_summary.json` and `classification_report.json` per split
- Logs headline metrics (Malignant sensitivity, macro AUC, accuracy)

## Key Metrics Produced

| Metric | Stratified | Center-Holdout |
|--------|-----------|----------------|
| Macro AUC | 0.846 | 0.627 |
| Malignant Sensitivity | 0.608 | 0.364 |
| Accuracy | 0.679 | 0.472 |
| Malignant AUC | 0.906 | -- |
| Test Set Size | 564 | 808 |

The center-holdout generalization gap is substantial, as expected with cross-center evaluation on a small dataset.

## Deviations from Plan

None -- plan executed exactly as written.

## Verification Results

1. `python scripts/eval.py --config configs/default.yaml` -- ran without errors for both splits
2. `results/stratified/` -- contains all 6 required files (roc_curves.png, pr_curves.png, confusion_matrix.png, confusion_matrix_normalized.png, metrics_summary.json, classification_report.json)
3. `results/center_holdout/` -- contains same 6 files
4. metrics_summary.json contains all required keys (macro_auc, per_class_auc, per_class_sensitivity, per_class_specificity, malignant_sensitivity, accuracy, per_class_average_precision)
5. All PNG files > 10KB (range: 38-114KB)

## Decisions Made

1. **Softmax applied once in run_inference** -- Model outputs raw logits; `F.softmax(logits, dim=1)` applied in `run_inference()` only, preventing double-softmax.
2. **One-vs-Rest metric strategy** -- ROC and PR curves computed via `label_binarize` with OvR approach, consistent with sklearn's multi-class conventions.
3. **Malignant sensitivity as headline metric** -- Logged prominently since it's the clinically relevant metric (detecting cancer matters more than overall accuracy).

## Next Phase Readiness

Plan 05-02 can proceed immediately. It consumes:
- `metrics_summary.json` from both splits (for comparison table)
- `y_true`, `y_pred`, `y_prob` arrays (via re-running inference, or from saved predictions)
- Bootstrap CI computation will wrap `compute_all_metrics()` in a resampling loop
