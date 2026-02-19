---
phase: 03-splits-and-dataset-loader
plan: 01
subsystem: data
tags: [sklearn, imagehash, stratified-split, center-holdout, duplicate-detection, phash]

# Dependency graph
requires:
  - phase: 02-data-acquisition-and-audit
    provides: "dataset.csv metadata, raw images, 21 duplicate pairs info"
provides:
  - "Pure split utility functions (derive_label, compute_duplicate_groups, stratified_split, center_holdout_split, save_split_csv)"
  - "6 CSV split manifests in data/splits/ (stratified + center-holdout)"
  - "Leakage risk documentation in script output"
affects:
  - 03-splits-and-dataset-loader (plan 02 - dataset loader will use split CSVs)
  - 04-training-loop (reads split manifests for data loading)
  - 05-evaluation (evaluates on both stratified and center test sets)

# Tech tracking
tech-stack:
  added: [imagehash, PyWavelets]
  patterns: [duplicate-aware group splitting, representative-based stratification]

key-files:
  created:
    - src/data/split_utils.py
    - data/splits/stratified_train.csv
    - data/splits/stratified_val.csv
    - data/splits/stratified_test.csv
    - data/splits/center_train.csv
    - data/splits/center_val.csv
    - data/splits/center_test.csv
  modified:
    - scripts/split.py

key-decisions:
  - "Duplicate groups use representative-based splitting: first member participates in stratified split, all members assigned to same partition"
  - "Center-holdout validation set is 15% of Center 1 data (not 15% of total), giving ~439 val images"
  - "phash hash_size=8 (64-bit) used for duplicate detection with distance=0 threshold"

patterns-established:
  - "Split CSV format: image_id,split,label (3 columns, no index)"
  - "Dual strategy pattern: every split function returns (train, val, test) DataFrames"
  - "Inline validation checks printed during script execution for auditability"

# Metrics
duration: 15min
completed: 2026-02-19
---

# Phase 3 Plan 01: Split Strategy Summary

**Dual split strategy (stratified 70/15/15 + center-holdout) with phash duplicate grouping producing 6 reproducible CSV manifests**

## Performance

- **Duration:** 15 min
- **Started:** 2026-02-19T21:29:47Z
- **Completed:** 2026-02-19T21:45:32Z
- **Tasks:** 2
- **Files modified:** 8 (1 module, 1 script, 6 CSV manifests)

## Accomplishments
- Implemented 5 pure split utility functions in src/data/split_utils.py
- Generated 6 CSV split manifests (stratified: 2621/561/564, center: 2499/439/808)
- 21 exact duplicate pairs detected via phash and forced to same split side in both strategies
- Class proportions preserved within <1% of original in stratified splits (50.2% Normal, 40.7% Benign, 9.1% Malignant)
- Leakage risk documented in console output (no patient_id, multi-angle images not groupable)
- `make split` produces identical results on repeated runs (seed=42)

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement split utility functions** - `5a733fb` (feat)
2. **Task 2: Implement split script and generate manifests** - `54cd07e` (feat)

## Files Created/Modified
- `src/data/split_utils.py` - Pure functions: derive_label, compute_duplicate_groups, stratified_split, center_holdout_split, save_split_csv
- `scripts/split.py` - CLI entry point with validation, leakage documentation, summary output
- `data/splits/stratified_train.csv` - 2621 images, stratified training split
- `data/splits/stratified_val.csv` - 561 images, stratified validation split
- `data/splits/stratified_test.csv` - 564 images, stratified test split
- `data/splits/center_train.csv` - 2499 images (Center 1), center-holdout training split
- `data/splits/center_val.csv` - 439 images (Center 1), center-holdout validation split
- `data/splits/center_test.csv` - 808 images (Centers 2+3), center-holdout test split

## Decisions Made
- **Representative-based splitting for duplicate groups:** First member of each duplicate pair participates in the stratified split decision; all members are then assigned to the winning partition. This ensures exact duplicates never cross a split boundary.
- **Center-holdout val_ratio applied to Center 1 only:** 15% of Center 1 (2938 images) = 439 val images, rather than 15% of total. This is simpler and gives adequate validation data from the same distribution as training.
- **phash distance=0 threshold:** Only exact hash matches count as duplicates (21 pairs found). Near-duplicates with distance>0 are treated as separate images to avoid over-grouping.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed imagehash in venv**
- **Found during:** Task 2 (running split script)
- **Issue:** imagehash was listed in requirements.txt but not installed in the project's .venv
- **Fix:** Ran `pip install imagehash==4.3.2 Pillow==11.1.0` in the .venv
- **Files modified:** None (package installation only)
- **Verification:** Import succeeds, hashing runs correctly
- **Committed in:** N/A (runtime dependency, not a code change)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Package was already declared in requirements.txt, just needed installation. No scope creep.

## Issues Encountered
- System Python (3.9.6) lacks torch; had to use project's .venv (Python 3.12.12 with all deps). The Makefile's `python` command requires .venv activation or PATH adjustment.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- 6 CSV split manifests ready for Phase 3 Plan 02 (dataset loader / PyTorch Dataset)
- Split manifests define image_id, split, label -- dataset loader needs to map these to image file paths
- Center 3 has only 27 Normal images in test set -- model evaluation should note this imbalance
- Leakage risk from same-lesion multi-angle images remains (documented, not mitigatable without patient_id)

---
*Phase: 03-splits-and-dataset-loader*
*Completed: 2026-02-19*
