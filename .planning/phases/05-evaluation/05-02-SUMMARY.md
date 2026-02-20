---
phase: 05-evaluation
plan: "02"
subsystem: evaluation
tags: [bootstrap, confidence-intervals, comparison-table, baseline, generalization-gap]
requires: ["05-01"]
provides: ["bootstrap CIs for macro AUC and per-class sensitivity", "comparison table with BTXRD baseline and generalization gap"]
affects: ["07-01", "07-02"]
tech-stack:
  added: []
  patterns: ["bootstrap resampling with percentile method", "class-presence guard for small classes"]
key-files:
  created:
    - src/evaluation/bootstrap.py
    - results/comparison_table.json
    - results/comparison_table.csv
  modified:
    - src/evaluation/__init__.py
    - scripts/eval.py
key-decisions:
  - id: "05-02-01"
    decision: "Percentile method for bootstrap CIs with class-presence guard"
    rationale: "Skips degenerate bootstrap samples where any class is absent; critical for stratified split with only 51 Malignant samples"
  - id: "05-02-02"
    decision: "7 caveats for BTXRD baseline comparison"
    rationale: "Paper uses random 80/20 split without patient grouping, different image size (600 vs 224), different architecture, and reports mAP not AUC -- direct comparison is misleading without caveats"
  - id: "05-02-03"
    decision: "Generalization gap computed as center_holdout minus stratified (negative = worse)"
    rationale: "Makes performance degradation from unseen centers explicit and directional"
duration: "14 min"
completed: "2026-02-20"
---

# Phase 5 Plan 02: Bootstrap CIs and Comparison Table Summary

Bootstrap 95% CIs (1000 iterations, percentile method) for macro AUC and per-class sensitivity, plus dual-split comparison table with BTXRD YOLOv8s-cls baseline and explicit generalization gap.

## Performance

- **Duration:** 14 min (includes 2 model inference passes across 1,372 test images)
- **Start:** 2026-02-20T16:09:08Z
- **End:** 2026-02-20T16:22:54Z
- **Tasks:** 2/2
- **Files created:** 3 (bootstrap.py, comparison_table.json, comparison_table.csv)
- **Files modified:** 2 (eval.py, __init__.py)

## Accomplishments

### Bootstrap Confidence Intervals (EVAL-07)

Implemented `bootstrap_confidence_intervals()` in `src/evaluation/bootstrap.py` with:
- Isolated RNG (`np.random.RandomState(seed)`) for reproducibility
- Class-presence guard skipping degenerate samples where any class is absent
- 1000 bootstrap iterations, 95% CI via percentile method
- All 1000 iterations valid for both splits (no samples discarded)

**Stratified split results:**
- Macro AUC: 0.846 [0.814, 0.873]
- Malignant Sensitivity: 0.608 [0.474, 0.743]
- Normal Sensitivity: 0.606 [0.548, 0.665]
- Benign Sensitivity: 0.784 [0.727, 0.840]

**Center-holdout split results:**
- Macro AUC: 0.627 [0.594, 0.658]
- Malignant Sensitivity: 0.364 [0.270, 0.448]
- Normal Sensitivity: 0.262 [0.210, 0.318]
- Benign Sensitivity: 0.643 [0.596, 0.691]

### Comparison Table (EVAL-06 / EVAL-08)

Built `comparison_table.json` and `comparison_table.csv` with:
- Side-by-side metrics for stratified vs. center-holdout splits
- BTXRD paper baseline (YOLOv8s-cls) with 7 comparison caveats
- Explicit generalization gap: AUC -0.219, Malignant Sensitivity -0.243, Accuracy -0.208

## Task Commits

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Implement bootstrap.py CI module | eddbd05 | src/evaluation/bootstrap.py, src/evaluation/__init__.py |
| 2 | Add bootstrap CIs and comparison table to eval.py | 892f146 | scripts/eval.py, results/comparison_table.json, results/comparison_table.csv |

## Files Created/Modified

### Created
- `src/evaluation/bootstrap.py` -- Bootstrap CI computation with class-presence guard
- `results/comparison_table.json` -- Full comparison data (stratified + center_holdout + baseline + gap)
- `results/comparison_table.csv` -- 12-row metric comparison table (Metric / Stratified / Center Holdout / BTXRD Baseline / Gap)

### Modified
- `src/evaluation/__init__.py` -- Added bootstrap_confidence_intervals to package exports
- `scripts/eval.py` -- Added bootstrap CI computation, comparison table generation, BTXRD baseline data, and formatted console output

### Generated (gitignored, reproducible via `make evaluate`)
- `results/stratified/bootstrap_ci.json` -- 95% CIs for stratified split
- `results/center_holdout/bootstrap_ci.json` -- 95% CIs for center-holdout split

## Decisions Made

1. **Percentile method for bootstrap CIs** -- Simple, interpretable, and appropriate for 1000 iterations. The class-presence guard ensures no degenerate samples corrupt the CI estimates, especially critical for the 51-sample Malignant class in the stratified test set.

2. **7 caveats for BTXRD baseline comparison** -- Added explicit caveat about mAP@0.5 (detection metric) not being comparable to AUC, in addition to the 6 standard methodology differences. This prevents readers from drawing false conclusions from the paper comparison.

3. **Generalization gap as center_holdout minus stratified** -- Negative values mean center-holdout performance is worse, making the directionality immediately intuitive. Gap included for key metrics: macro AUC (-0.219), Malignant sensitivity (-0.243), accuracy (-0.208).

## Deviations from Plan

None -- plan executed exactly as written.

## Issues Encountered

- **Per-split result files gitignored:** `results/stratified/bootstrap_ci.json` and `results/center_holdout/bootstrap_ci.json` are gitignored by the `.gitignore` pattern `results/*/bootstrap_*.json`. This is correct behavior -- these files are regenerable from checkpoints via `make evaluate`. The root-level comparison tables (`results/comparison_table.json` and `.csv`) are committed as they represent the cross-split aggregation artifact.

## Key Results for Phase 7 Reporting

The wide Malignant sensitivity CI [0.474, 0.743] for the stratified split (51 test samples) and the substantial generalization gap (-0.219 macro AUC) are key findings that must be prominently reported in the PoC report:

1. **Small Malignant test set:** 51 samples yield a 27-percentage-point CI width for sensitivity
2. **Center generalization:** Center-holdout performance is substantially worse across all metrics
3. **BTXRD comparison:** Paper baseline appears stronger but uses random split without patient grouping -- not directly comparable

## Next Phase Readiness

Phase 5 is now COMPLETE. All EVAL-01 through EVAL-08 requirements are satisfied:
- EVAL-01: Per-class ROC curves + AUC (05-01)
- EVAL-02: Per-class PR curves + AP (05-01)
- EVAL-03: Per-class sensitivity + specificity (05-01)
- EVAL-04: Confusion matrices (05-01)
- EVAL-05: Classification reports (05-01)
- EVAL-06: Comparison table with both splits (05-02)
- EVAL-07: Bootstrap 95% CIs (05-02)
- EVAL-08: BTXRD baseline comparison with caveats (05-02)

Phase 6 (Explainability and Inference) can proceed -- it needs the trained checkpoints (Phase 4) and evaluation results (Phase 5) which are all in place.
