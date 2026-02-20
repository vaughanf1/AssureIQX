---
phase: 04-model-training
plan: 02
subsystem: training
tags: [pytorch, efficientnet, cross-entropy, early-stopping, cosine-annealing, tqdm, sklearn]

# Dependency graph
requires:
  - phase: 04-01
    provides: "BTXRDClassifier, create_model, compute_class_weights, get_device, save_checkpoint, EarlyStopping"
  - phase: 03-02
    provides: "BTXRDDataset, create_dataloader, get_train_transforms, get_val_transforms"
  - phase: 01-02
    provides: "load_config, set_seed, configs/default.yaml"
provides:
  - "Complete training script (scripts/train.py) with full training loop"
  - "Weighted CrossEntropyLoss training with unweighted validation for early stopping"
  - "Best and final checkpoint saving to checkpoints/ directory"
  - "CSV training log with per-epoch metrics and per-class recall"
  - "Loss curve and recall curve PNG generation"
  - "Malignant class collapse detection warning"
  - "make train-all target for both split strategies"
affects: [04-03, 05-evaluation, 06-gradcam, 07-inference]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Weighted loss for training, unweighted for validation (early stopping signal)"
    - "Per-class recall monitoring every epoch to detect class collapse"
    - "Dual checkpoint strategy: best (lowest val_loss) and final"
    - "Split-prefixed checkpoint naming: best_stratified.pt, best_center.pt"

key-files:
  created: []
  modified:
    - scripts/train.py
    - Makefile

key-decisions:
  - "Weighted CrossEntropyLoss for training only; unweighted for val/early stopping -- prevents class weight from distorting stopping signal"
  - "Split-prefixed checkpoint names (best_stratified.pt, best_center.pt) -- avoids overwriting when running both splits"
  - "Results directory naming: stratified/ and center_holdout/ -- matches Phase 3 convention"

patterns-established:
  - "Training scripts follow pattern: argparse + config + seed + setup + loop + post-processing"
  - "Checkpoint includes full state (model, optimizer, scheduler, config, class_names, normalization, class_weights)"
  - "Per-class recall via sklearn.metrics.recall_score with zero_division=0"

# Metrics
duration: 3min
completed: 2026-02-20
---

# Phase 4 Plan 02: Training Script Summary

**Complete training pipeline with weighted CrossEntropyLoss, CosineAnnealingLR, early stopping on unweighted val_loss, per-class recall monitoring, and dual checkpoint saving**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-19T23:26:26Z
- **Completed:** 2026-02-20T00:15:00Z
- **Tasks:** 2/2
- **Files modified:** 2

## Accomplishments

- Full training loop with weighted loss (training) and unweighted loss (validation/early stopping)
- Per-class recall logged every epoch with Malignant collapse warning after 2 consecutive zero-recall epochs
- CSV training log + loss curve + recall curve PNG generation
- `make train-all` target for sequential stratified and center-holdout training

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement the complete training script** - `e579ed0` (feat)
2. **Task 2: Add train-all Makefile target** - `c5d88f4` (feat)

## Files Created/Modified

- `scripts/train.py` - Complete training script (513 lines): train_one_epoch, validate, save_training_log, plot_loss_curves, main with full training loop
- `Makefile` - Added train-all target for both split strategies, updated .PHONY

## Decisions Made

- **Weighted vs unweighted loss separation**: Weighted CrossEntropyLoss for training compensates for class imbalance; unweighted CrossEntropyLoss for validation provides a clean early stopping signal not distorted by weights
- **Split-prefixed checkpoint naming**: best_stratified.pt / best_center.pt / final_stratified.pt / final_center.pt -- avoids overwriting when running train-all
- **Results directory structure**: stratified/ and center_holdout/ subdirectories under results/ -- consistent with Phase 3 naming

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Training script is complete and ready to produce model checkpoints
- Plan 04-03 (training dry-run validation) can verify the full pipeline end-to-end
- Phase 5 (Evaluation) will use checkpoints produced by this script
- Phase 6 (Grad-CAM) will load best checkpoints for explainability

---
*Phase: 04-model-training*
*Completed: 2026-02-20*
