---
phase: 05-evaluation
verified: 2026-02-20T16:29:02Z
status: passed
score: 7/7 must-haves verified
re_verification: false
---

# Phase 5: Evaluation Verification Report

**Phase Goal:** Both trained models are comprehensively evaluated with clinically relevant metrics, producing a side-by-side comparison that reveals the generalization gap between stratified and center-holdout performance
**Verified:** 2026-02-20T16:29:02Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Running `python scripts/eval.py` produces ROC curve PNGs, PR curve PNGs, confusion matrix heatmaps (absolute + normalized), and classification report JSON in each split's results directory | VERIFIED | All 4 PNG files present in both `results/stratified/` and `results/center_holdout/`; sizes range 38-115 KB confirming non-empty renders. `classification_report.json` present in both with per-class precision/recall/F1. |
| 2 | Per-class sensitivity and specificity are computed and written to metrics_summary.json with Malignant sensitivity as the headline metric | VERIFIED | `metrics_summary.json` in both splits contains `per_class_sensitivity`, `per_class_specificity`, and `malignant_sensitivity` keys with real values. eval.py logs `HEADLINE [%s] | Malignant Sensitivity: %.3f` explicitly. |
| 3 | Bootstrap 95% confidence intervals are computed for macro AUC and per-class sensitivity and saved as JSON in each split's results directory | VERIFIED | `bootstrap_ci.json` exists in both splits. Each contains `macro_auc`, `sensitivity_Normal`, `sensitivity_Benign`, `sensitivity_Malignant` with `mean`, `ci_lower`, `ci_upper`, `n_valid=1000` fields. All 1000 iterations valid. |
| 4 | A comparison table shows side-by-side metrics for both split strategies, making the center-holdout generalization gap explicit | VERIFIED | `results/comparison_table.json` and `results/comparison_table.csv` both exist with populated content. Gap: macro AUC -0.219, Malignant Sensitivity -0.243, Accuracy -0.208. `generalization_gap` section present with direction description. |
| 5 | A comparison against the BTXRD paper's YOLOv8s-cls baseline is included with caveats about split methodology | VERIFIED | `btxrd_baseline` key in `comparison_table.json` contains 7 explicit caveats covering split leakage, image size, architecture, epochs, metric type mismatch, and lack of AUC reporting in paper. |
| 6 | All plots are publication-quality with labeled axes, titles, and legends | VERIFIED | `visualization.py` uses `figsize=(8,8)`, `dpi=150`, named axis labels, legend with AUC/AP values, grid lines, diagonal reference for ROC. Confusion matrices use seaborn heatmap with `annot=True`, labeled axes. No `plt.show()` calls; Agg backend set at module level. |
| 7 | All key wiring from eval.py through metrics, visualization, and bootstrap modules is complete and produces real output | VERIFIED | eval.py calls `run_inference` -> `compute_all_metrics` -> `plot_roc_curves` / `plot_pr_curves` / `plot_confusion_matrices` -> `bootstrap_confidence_intervals` -> `_build_comparison_table` in correct sequence. All results written to disk and confirmed present. |

**Score:** 7/7 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/evaluation/metrics.py` | `compute_all_metrics`, `run_inference` | VERIFIED | 184 lines; exports both functions; full EVAL-01 through EVAL-05 implementation; no stubs; `label_binarize`, `roc_curve`, `precision_recall_curve`, `multilabel_confusion_matrix`, `confusion_matrix`, `classification_report` all called with real results used |
| `src/evaluation/visualization.py` | `plot_roc_curves`, `plot_pr_curves`, `plot_confusion_matrices` | VERIFIED | 211 lines; Agg backend set before pyplot import; all three functions save to disk; macro-average ROC interpolated over 1000-point FPR grid; both absolute and normalized confusion matrices; `plt.close(fig)` after each save |
| `src/evaluation/bootstrap.py` | `bootstrap_confidence_intervals` | VERIFIED | 119 lines; isolated RNG with seed; class-presence guard; percentile CI method; returns mean/ci_lower/ci_upper/n_valid for macro_auc and per-class sensitivity |
| `src/evaluation/__init__.py` | All 5 public functions exported | VERIFIED | Exports `bootstrap_confidence_intervals`, `compute_all_metrics`, `run_inference`, `plot_confusion_matrices`, `plot_pr_curves`, `plot_roc_curves` |
| `scripts/eval.py` | CLI orchestration for both splits | VERIFIED | 501 lines; full pipeline: config load -> device -> per-split loop (checkpoint load, model reconstruct, dataset, inference, metrics, plots, JSON saves, bootstrap CIs) -> comparison table |
| `results/stratified/roc_curves.png` | Publication-quality ROC PNG | VERIFIED | 101,265 bytes |
| `results/stratified/pr_curves.png` | Publication-quality PR PNG | VERIFIED | 83,914 bytes |
| `results/stratified/confusion_matrix.png` | Absolute count heatmap | VERIFIED | 40,524 bytes |
| `results/stratified/confusion_matrix_normalized.png` | Row-normalized heatmap | VERIFIED | 44,420 bytes |
| `results/stratified/metrics_summary.json` | EVAL-01/02/03 metrics summary | VERIFIED | Contains macro_auc (0.846), per_class_auc, per_class_sensitivity, per_class_specificity, malignant_sensitivity (0.608), accuracy |
| `results/stratified/classification_report.json` | Per-class precision/recall/F1 | VERIFIED | Contains Normal, Benign, Malignant with precision/recall/f1-score/support plus macro avg |
| `results/stratified/bootstrap_ci.json` | 95% CIs for stratified split | VERIFIED | n_valid=1000 for all metrics; macro_auc CI [0.814, 0.873]; Malignant sensitivity CI [0.474, 0.743] |
| `results/center_holdout/roc_curves.png` | Publication-quality ROC PNG | VERIFIED | 114,946 bytes |
| `results/center_holdout/pr_curves.png` | Publication-quality PR PNG | VERIFIED | 96,840 bytes |
| `results/center_holdout/confusion_matrix.png` | Absolute count heatmap | VERIFIED | 38,250 bytes |
| `results/center_holdout/confusion_matrix_normalized.png` | Row-normalized heatmap | VERIFIED | 44,213 bytes |
| `results/center_holdout/metrics_summary.json` | EVAL-01/02/03 metrics summary | VERIFIED | Contains macro_auc (0.627), per_class_sensitivity, per_class_specificity, malignant_sensitivity (0.364), accuracy |
| `results/center_holdout/classification_report.json` | Per-class precision/recall/F1 | VERIFIED | Contains all three classes with correct metric keys |
| `results/center_holdout/bootstrap_ci.json` | 95% CIs for center-holdout split | VERIFIED | n_valid=1000; macro_auc CI [0.594, 0.658]; Malignant sensitivity CI [0.270, 0.448] |
| `results/comparison_table.json` | Side-by-side + baseline + gap | VERIFIED | Contains `stratified`, `center_holdout`, `btxrd_baseline` (7 caveats), `generalization_gap` sections with all gap values |
| `results/comparison_table.csv` | Human-readable comparison | VERIFIED | 12 metric rows; columns: Metric, Stratified, Center Holdout, BTXRD Baseline, Gap |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `scripts/eval.py` | `src/evaluation/metrics.py` | `run_inference` + `compute_all_metrics` calls | WIRED | Line 392: `y_true, y_pred, y_prob = run_inference(...)`. Line 396: `metrics = compute_all_metrics(...)`. Return values used immediately for plots and JSON. |
| `scripts/eval.py` | `src/evaluation/visualization.py` | `plot_roc_curves`, `plot_pr_curves`, `plot_confusion_matrices` calls | WIRED | Lines 399, 402, 405: all three plot functions called with real `metrics` dict and real output paths. |
| `scripts/eval.py` | `src/evaluation/bootstrap.py` | `bootstrap_confidence_intervals` called with prediction arrays | WIRED | Line 447: called with `y_true, y_pred, y_prob, class_names` from actual inference. Result saved to `bootstrap_ci.json` and collected for comparison table. |
| `scripts/eval.py` | `results/comparison_table.json` | `_build_comparison_table` called after both splits complete | WIRED | Line 490: `_build_comparison_table(collected_results, results_root)` called when `len(collected_results) >= 2`. Writes both JSON and CSV. |
| `scripts/eval.py` | `src/models/factory.py` | `load_checkpoint` + `create_model` + `get_device` | WIRED | Lines 318, 361, 371: device, checkpoint, and model creation all called with real config values. Model state dict loaded and set to eval mode. |
| `scripts/eval.py` | `src/data/dataset.py` | `BTXRDDataset` + `create_dataloader` | WIRED | Lines 378-388: test dataset created with real CSV path, images dir, and transforms. Dataloader created from it. |
| `metrics.py` | sklearn metrics | `roc_curve`, `precision_recall_curve`, `multilabel_confusion_matrix`, `confusion_matrix`, `classification_report` | WIRED | All sklearn functions imported and called with real arrays. Return values captured in named variables and assembled into return dict. |

### Requirements Coverage

| Requirement | Status | Notes |
|-------------|--------|-------|
| EVAL-01: Per-class ROC curves + AUC (one-vs-rest) | SATISFIED | `roc_curve` + `roc_auc_score` in `metrics.py`; plotted by `visualization.py`; PNGs confirmed present and non-trivial size |
| EVAL-02: Per-class PR curves + AP | SATISFIED | `precision_recall_curve` + `average_precision_score` in `metrics.py`; plotted by `visualization.py`; PNGs confirmed |
| EVAL-03: Per-class sensitivity and specificity | SATISFIED | `multilabel_confusion_matrix` used to extract TN/FP/FN/TP; sensitivity and specificity computed and saved in `metrics_summary.json` |
| EVAL-04: Confusion matrix heatmaps (absolute + normalized) | SATISFIED | Two separate PNG files per split confirmed present; absolute uses `fmt='d'`, normalized uses row-normalization and `fmt='.2f'` |
| EVAL-05: Classification report (precision, recall, F1 per class) | SATISFIED | `classification_report(..., output_dict=True)` called; results saved to `classification_report.json` with per-class and macro avg entries |
| EVAL-06: Evaluation on both split strategies with comparison table | SATISFIED | `comparison_table.json` and `comparison_table.csv` present with both splits side-by-side; gap column explicit |
| EVAL-07: Bootstrap 95% CIs (1000 iterations) | SATISFIED | `bootstrap_confidence_intervals` with `n_iterations=1000`, `confidence_level=0.95`; n_valid=1000 for all metrics in both splits |
| EVAL-08: Comparison against BTXRD paper YOLOv8s-cls baseline with caveats | SATISFIED | `BTXRD_BASELINE` dict hardcoded in `eval.py` with 7 caveats; included in `comparison_table.json` under `btxrd_baseline` key |

### Anti-Patterns Found

None. Zero instances of TODO, FIXME, placeholder, NotImplementedError, or empty-return stubs found in any of the four key files.

### Human Verification Required

#### 1. Visual quality of plots

**Test:** Open `results/stratified/roc_curves.png`, `results/stratified/pr_curves.png`, `results/stratified/confusion_matrix.png`, and `results/stratified/confusion_matrix_normalized.png` in an image viewer.
**Expected:** Axes labeled, legend with AUC/AP values, curves visually distinct per class, color scheme Normal=Blue/Benign=Orange/Malignant=Red. Confusion matrix shows per-class counts with color gradient.
**Why human:** Image rendering correctness cannot be verified by file-size or code-path checks alone.

#### 2. Full end-to-end script re-run

**Test:** Run `python scripts/eval.py --config configs/default.yaml` from the project root. Observe console output.
**Expected:** Both splits complete without errors. Console shows `HEADLINE` log lines with Malignant Sensitivity, Macro AUC, and Accuracy. Bootstrap CI log line appears. Comparison summary table printed at end.
**Why human:** Confirms that trained checkpoints (`best_stratified.pt`, `best_center.pt`) and split CSVs are still accessible and that inference runs on current hardware.

---

## Gaps Summary

No gaps identified. All 7 observable truths are verified. All 21 required artifacts exist with substantive implementations and correct wiring. EVAL-01 through EVAL-08 are all covered.

The generalization gap between stratified (macro AUC 0.846, Malignant sensitivity 0.608) and center-holdout (macro AUC 0.627, Malignant sensitivity 0.364) is explicitly computed and present in all comparison outputs as intended by the phase goal.

---

_Verified: 2026-02-20T16:29:02Z_
_Verifier: Claude (gsd-verifier)_
