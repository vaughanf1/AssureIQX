---
phase: "07-documentation-and-reports"
plan: "01"
subsystem: "documentation"
tags: ["model-card", "poc-report", "clinical-framing", "mitchell-2019"]
dependency-graph:
  requires: ["01", "02", "03", "04", "05", "06"]
  provides: ["model-card", "poc-report", "clinical-decision-framing"]
  affects: ["07-02"]
tech-stack:
  added: []
  patterns: ["mitchell-2019-model-card", "clinical-decision-framing"]
key-files:
  created:
    - "docs/model_card.md"
    - "docs/poc_report.md"
  modified: []
decisions:
  - id: "07-01-model-card-format"
    description: "Mitchell et al. 2019 format with 9 sections plus references"
  - id: "07-01-clinical-framing"
    description: "Sensitivity at specificity as clinical decision statement"
  - id: "07-01-limitation-count"
    description: "9 documented limitations covering data, model, and evaluation"
metrics:
  duration: "5 min"
  completed: "2026-02-21"
---

# Phase 7 Plan 1: Model Card and PoC Report Summary

**One-liner:** Mitchell et al. 2019 model card and comprehensive PoC report with clinical decision framing, dual-split comparison, and 9 documented limitations

## What Was Done

### Task 1: Model Card (docs/model_card.md)

Created a 259-line model card following the Mitchell et al. (2019) "Model Cards for Model Reporting" format with all 9 required sections:

1. **Model Details** -- EfficientNet-B0, 4M params, full fine-tuning, Adam lr=0.001
2. **Intended Use** -- Research PoC only, NOT FOR CLINICAL USE
3. **Factors** -- Center bias (78% Center 1), anatomical site, tumor subtype
4. **Metrics** -- Macro AUC as primary, Malignant sensitivity as clinical headline
5. **Training Data** -- BTXRD 3,746 images, class distribution, CC BY-NC-ND 4.0
6. **Evaluation Data** -- 564 stratified test, 808 center-holdout test
7. **Quantitative Analyses** -- Both split performance tables with bootstrap CIs, generalization gap table
8. **Caveats and Recommendations** -- All 9 limitations with expanded context
9. **Ethical Considerations** -- Non-clinical use, dataset bias, misuse potential

**Commit:** abe92dc

### Task 2: PoC Report (docs/poc_report.md)

Created a 495-line comprehensive proof-of-concept report with 11 sections:

1. **Title/Metadata** -- Version v1.0, dated 2026-02-21
2. **Disclaimer** -- Prominent NOT FOR CLINICAL USE box
3. **Executive Summary** -- Key findings with clinical decision framing statement
4. **Methods** (6 subsections) -- Dataset, preprocessing, split strategy, architecture, training, evaluation
5. **Results** (4 subsections) -- Stratified, center-holdout, comparison table, BTXRD baseline
6. **Explainability** -- Grad-CAM method, qualitative findings, annotation comparison (mean IoU 0.070)
7. **Limitations** -- All 9 limitations expanded with clinical context
8. **Clinical Relevance** -- Feasibility framing, clinical decision statement, requirements for clinical use
9. **Next Steps** -- 7 recommended actions
10. **References** -- Yao et al., Mitchell et al., Selvaraju et al., Tan & Le
11. **Appendix** -- Full per-class metrics tables, bootstrap CIs, figure references

**Clinical decision framing statement** appears in both executive summary and clinical relevance: "At the default operating point, the model achieves 60.8% sensitivity (95% CI: 47.4%--74.3%) for malignant tumors at 95.7% specificity on the stratified test set."

All 7 BTXRD baseline comparison caveats included verbatim.

**Commit:** 2e2c28b

## Key Metrics Referenced

| Metric | Stratified | Center-Holdout | Gap |
|--------|-----------|----------------|-----|
| Macro AUC | 0.846 (0.814-0.873) | 0.627 (0.594-0.658) | -0.219 |
| Malignant Sensitivity | 60.8% (47.4%-74.3%) | 36.4% (27.0%-44.8%) | -24.3 pp |
| Accuracy | 67.9% | 47.2% | -20.8 pp |

## Decisions Made

1. **Mitchell et al. 2019 format:** Model card follows the standard 9-section format for consistency with ML community best practices.
2. **Clinical decision framing:** Used sensitivity-at-specificity format to communicate performance in clinically meaningful terms.
3. **9 limitations:** Comprehensive limitation set covering data quality (leakage, label noise, pathology), sample size (Malignant count, CIs), model (single architecture, Grad-CAM IoU), and evaluation (center gap, Center 3 sparsity).
4. **7 BTXRD caveats:** All caveats from comparison_table.json included to prevent misleading comparison with paper results.

## Deviations from Plan

None -- plan executed exactly as written.

## Verification Results

- Model card: 259 lines (>150 required), 14 section header matches (>9 required)
- PoC report: 495 lines (>300 required), 9 section header matches (>8 required)
- Key numbers present in both: 0.846, 0.627, 60.8%, 36.4%, -0.219
- NOT FOR CLINICAL USE: 3 occurrences in model card, 2 in PoC report
- Clinical decision framing statement: 3 occurrences in PoC report (>= 2 required)
- All 9 limitations documented in both files
- All 7 BTXRD baseline caveats in PoC report
