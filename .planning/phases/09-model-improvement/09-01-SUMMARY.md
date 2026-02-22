---
phase: 09-model-improvement
plan: 01
subsystem: model
tags: [resnet50, cbam, efficientnet, timm, benchmark, multi-architecture]

# Dependency graph
requires:
  - phase: 04-model-training
    provides: BTXRDClassifier, factory.py, train.py
  - phase: 05-evaluation
    provides: eval.py evaluation pipeline
provides:
  - Multi-architecture BTXRDClassifier (EfficientNet + ResNet-CBAM)
  - Model factory with attn_layer/block_args passthrough
  - Benchmark orchestration script for experiment grid
  - eval.py --checkpoint-dir override for per-experiment evaluation
affects: [09-02-execute-experiments]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "**kwargs passthrough from factory to classifier to timm.create_model"
    - "Architecture-aware gradcam_target_layer (bn2 for EfficientNet, layer4[-1] for ResNet)"
    - "Experiment grid orchestration via subprocess calls to train.py + eval.py"

key-files:
  created:
    - scripts/benchmark.py
  modified:
    - src/models/classifier.py
    - src/models/factory.py
    - scripts/eval.py
    - Makefile

key-decisions:
  - "**kwargs on BTXRDClassifier.__init__ for flexible timm passthrough (block_args, etc.)"
  - "gradcam_target_layer: bn2 for EfficientNet, layer4[-1] for ResNet, then fallback walk"
  - "factory.py reads model.attn_layer from config and converts to block_args dict"
  - "Checkpoint backup to checkpoints/backup_b0_baseline/ before any benchmark runs"
  - "Experiment checkpoints renamed to best_{split}_{exp_id}.pt to avoid overwrites"

patterns-established:
  - "Multi-architecture model support: classifier accepts **kwargs, factory builds kwargs from config"
  - "Benchmark grid pattern: hardcoded EXPERIMENT_GRID with subprocess calls to train.py + eval.py"

# Metrics
duration: 4min
completed: 2026-02-22
---

# Phase 9 Plan 01: Multi-Architecture Support and Benchmark Orchestration Summary

**ResNet-50-CBAM and EfficientNet-B3 support added to classifier/factory with 4-experiment benchmark script ready to run**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-22T00:38:51Z
- **Completed:** 2026-02-22T00:42:50Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- BTXRDClassifier now supports ResNet-50 with CBAM attention (26M params) alongside EfficientNet family via **kwargs passthrough to timm
- Architecture-aware gradcam_target_layer property returns model.layer4[-1] for ResNet and model.bn2 for EfficientNet
- Factory reads model.attn_layer config key and passes as block_args=dict(attn_layer=...) to classifier
- Benchmark script orchestrates 4 experiments x 2 splits with checkpoint backup, comparison table generation, and per-experiment results isolation
- eval.py accepts --checkpoint-dir override for evaluating experiment-specific checkpoints

## Task Commits

Each task was committed atomically:

1. **Task 1: Update classifier.py and factory.py for multi-architecture support** - `67c508e` (feat)
2. **Task 2: Create benchmark.py orchestration script and update Makefile** - `b54c9d9` (feat)

## Files Created/Modified
- `src/models/classifier.py` - Multi-architecture BTXRDClassifier with **kwargs and ResNet gradcam support
- `src/models/factory.py` - Model factory with attn_layer/block_args config passthrough
- `scripts/benchmark.py` - Experiment grid orchestration (4 experiments x 2 splits, comparison table)
- `scripts/eval.py` - Added --checkpoint-dir CLI argument for experiment-specific evaluation
- `Makefile` - Added benchmark target

## Decisions Made
- Used **kwargs on BTXRDClassifier.__init__ rather than explicit block_args parameter, keeping the interface flexible for future timm arguments
- gradcam_target_layer checks bn2 first (EfficientNet), then layer4 (ResNet), then fallback walk -- order matters since EfficientNet is the more common architecture
- Benchmark script uses subprocess.run to call train.py and eval.py rather than importing their functions, maintaining process isolation and respecting CLI override mechanism
- Checkpoints renamed after each experiment run to best_{split}_{exp_id}.pt to prevent overwrites across experiments

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- classifier.py and factory.py ready for multi-architecture training
- benchmark.py ready to execute: `python scripts/benchmark.py --config configs/default.yaml`
- Existing B0 checkpoints will be backed up automatically on first benchmark run
- Plan 09-02 can now run the experiment grid and analyze results

---
*Phase: 09-model-improvement*
*Completed: 2026-02-22*
