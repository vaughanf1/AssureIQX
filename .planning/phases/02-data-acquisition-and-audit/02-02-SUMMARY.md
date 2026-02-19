---
phase: 02-data-acquisition-and-audit
plan: 02
subsystem: data
tags: [audit, matplotlib, seaborn, imagehash, phash, duplicate-detection, leakage-risk, dataset-spec, markdown-report]

# Dependency graph
requires:
  - phase: 02-01
    provides: data_raw/ with 3746 images, 1867 annotations, dataset.csv (37 columns)
provides:
  - "Audit script (scripts/audit.py) generating 7-section markdown report with 5 embedded figures"
  - "Auto-generated docs/data_audit_report.md confirming 1,879/1,525/342 class distribution"
  - "docs/dataset_spec.md documenting all 37 columns, label derivation, provenance, CC BY-NC-ND 4.0 license"
  - "5 PNG figures in docs/figures/ (class_distribution, dimension_histogram, center_breakdown, annotation_coverage, duplicate_detection)"
  - "Duplicate detection results: 21 exact duplicate groups, 20 near-duplicate pairs"
  - "Leakage risk assessment with proxy grouping analysis (295 and 941 unique groups)"
affects: [03-01, 03-02, 04-01, 05-01, 06-01, 07-01, 08-01]

# Tech tracking
tech-stack:
  added: []
  patterns: [headless-matplotlib, phash-duplicate-detection, markdown-report-generation, seaborn-whitegrid]

key-files:
  created: [docs/data_audit_report.md, docs/dataset_spec.md, docs/figures/class_distribution.png, docs/figures/dimension_histogram.png, docs/figures/center_breakdown.png, docs/figures/annotation_coverage.png, docs/figures/duplicate_detection.png]
  modified: [scripts/audit.py]

key-decisions:
  - "No missing values in dataset -- all 37 columns fully populated (142,348 cells)"
  - "21 exact duplicate image pairs found via phash (same hash, distance=0)"
  - "20 near-duplicate pairs (phash distance 1-5) -- may be multi-angle shots of same lesion"
  - "wrist-joint column has zero instances (empty column in schema)"
  - "Mixed image_id extensions: 3,719 .jpeg + 27 .jpg"
  - "295 unique proxy groups via center+age+gender (12.7 images/group avg)"
  - "941 unique proxy groups via center+age+gender+site (4.0 images/group avg)"
  - "5 figures generated (no missing_values figure since dataset is complete)"
  - "CC BY-NC-ND 4.0 license documented with figshare CC BY discrepancy note"

patterns-established:
  - "matplotlib.use('Agg') before pyplot import for headless rendering"
  - "save_figure() pattern: 150 DPI, bbox_inches='tight', facecolor='white', plt.close()"
  - "Section functions return (figure_path_or_None, markdown_text) tuples"
  - "write_audit_report() assembles sections with TOC and figure embeds"

# Metrics
duration: 7min
completed: 2026-02-19
---

# Phase 2 Plan 2: Data Audit and Dataset Specification Summary

**Audit script profiling 3,746 BTXRD images with 7-section markdown report (class distribution 1,879/1,525/342, phash duplicate detection, leakage risk assessment) plus 37-column dataset spec with CC BY-NC-ND 4.0 license**

## Performance

- **Duration:** 7 min
- **Started:** 2026-02-19T20:23:34Z
- **Completed:** 2026-02-19T20:30:52Z
- **Tasks:** 2
- **Files created:** 8 (scripts/audit.py modified, 7 new files)

## Accomplishments

- Complete audit pipeline: `python scripts/audit.py --config configs/default.yaml` generates 7-section report with 5 embedded PNG figures
- Class distribution confirmed: 1,879 Normal (50.2%), 1,525 Benign (40.7%), 342 Malignant (9.1%), 5.5x imbalance ratio
- Per-center breakdown matches paper: Center 1 (2,938, 78.4%), Center 2 (549, 14.7%), Center 3 (259, 6.9%)
- Zero missing values across all 142,348 cells (37 columns x 3,746 rows)
- 100% annotation coverage: all 1,867 tumor images have matching JSON annotations
- Duplicate detection: 21 exact duplicate groups and 20 near-duplicate pairs via perceptual hashing
- Leakage risk documented with proxy grouping analysis: 295 groups (center+age+gender), 941 groups (center+age+gender+site)
- Dataset spec documents all 37 columns with actual CSV header names, types, value ranges, and counts
- Label derivation logic with invariants and Python implementation
- Data provenance per center with source descriptions and class breakdowns

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement audit script with all sections** - `5e2c866` (feat)
2. **Task 2: Create dataset specification document** - `952fcdf` (docs)

## Files Created/Modified

- `scripts/audit.py` - Complete audit pipeline replacing NotImplementedError (7 section functions + report assembly)
- `docs/data_audit_report.md` - Auto-generated 7-section audit report with embedded figures
- `docs/dataset_spec.md` - 37-column dataset specification with label derivation, provenance, and license
- `docs/figures/class_distribution.png` - Bar chart: Normal 1,879 / Benign 1,525 / Malignant 342
- `docs/figures/dimension_histogram.png` - Width (153-3594px) and height (311-4881px) distributions
- `docs/figures/center_breakdown.png` - Stacked bar chart of class distribution per center
- `docs/figures/annotation_coverage.png` - 1,867/1,867 tumor images have annotations (100%)
- `docs/figures/duplicate_detection.png` - Pairwise hash distance histogram for close pairs

## Decisions Made

- **No missing_values figure:** The dataset has zero missing values, so the missing_values.png figure is skipped (returns None). The report text documents this finding.
- **wrist-joint column is empty:** All 3,746 rows have wrist-joint=0. Documented in dataset_spec.md as a known issue.
- **Mixed image_id extensions:** 3,719 files use `.jpeg` and 27 use `.jpg`. Documented for downstream code awareness.
- **Proxy grouping granularity:** Two levels analyzed -- center+age+gender (295 groups, 12.7 avg) and center+age+gender+site (941 groups, 4.0 avg). Neither is sufficient for true patient-level grouping.
- **CC BY-NC-ND 4.0 with discrepancy note:** Paper says NC-ND, figshare says CC BY. Documented both with paper as authoritative.

## Deviations from Plan

None -- plan executed exactly as written.

## Key Audit Findings for Downstream Phases

1. **Class imbalance (5.5x):** Phase 4 training must use class-weighted loss or oversampling
2. **21 exact duplicate pairs:** Phase 3 splitting should be aware; duplicates in train+test inflate metrics
3. **No patient_id:** Center-holdout split (Phase 3) is the primary mitigation for leakage risk
4. **Variable image sizes (153-3594 x 311-4881):** Phase 4 preprocessing must resize/normalize
5. **Center 3 bias:** Only 27 Normal images; center-holdout with Center 3 as test will have limited Normal representation

## Next Phase Readiness

- Phase 2 is complete: data downloaded (02-01) and audited (02-02)
- Ready for Phase 3 (Data Splitting): class distribution, center breakdown, and leakage risk analysis provide the foundation for informed split strategy design
- The duplicate detection results should inform Phase 3 split logic (ensure duplicates land on same side)
- No blockers for Phase 3

---
*Phase: 02-data-acquisition-and-audit*
*Completed: 2026-02-19*
