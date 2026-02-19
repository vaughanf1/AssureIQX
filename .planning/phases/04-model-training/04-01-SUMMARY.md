---
phase: 04-model-training
plan: 01
subsystem: model
tags: [pytorch, timm, efficientnet, classifier, checkpoint, early-stopping]

# Dependency graph
requires:
  - phase: 03-splits-and-loader
    provides: "IMAGENET_MEAN/STD constants from src.data.transforms"
provides:
  - "BTXRDClassifier nn.Module wrapping timm EfficientNet-B0 with 3-class logit output"
  - "create_model factory building classifier from YAML config"
  - "save_checkpoint / load_checkpoint with full training state"
  - "compute_class_weights for inverse-frequency weighted loss"
  - "get_device auto-detection (CUDA > MPS > CPU)"
  - "EarlyStopping class with configurable patience"
affects: [04-02 training script, 05 evaluation, 06 gradcam and inference, 08 demo]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "timm.create_model wrapper in thin nn.Module for stable interface"
    - "Checkpoint dict format: model_state_dict, optimizer_state_dict, scheduler_state_dict, config, class_names, normalization, class_weights"
    - "Raw logits output (no softmax) -- CrossEntropyLoss handles softmax internally"
    - "CPU-returned class weights -- caller moves to device"

key-files:
  created: []
  modified:
    - "src/models/classifier.py"
    - "src/models/factory.py"
    - "src/models/__init__.py"

key-decisions:
  - "gradcam_target_layer returns model.bn2 (BatchNormAct2d 1280) -- verified for Phase 6"
  - "save_checkpoint includes class_weights as list for checkpoint self-documentation"
  - "compute_class_weights returns CPU tensor -- caller responsible for .to(device)"

patterns-established:
  - "Model wrapper pattern: thin nn.Module delegating to timm backbone"
  - "Factory pattern: create_model(config) reads config['model'] section"
  - "Checkpoint format: dict with 9 keys covering full training state"

# Metrics
duration: 2min
completed: 2026-02-19
---

# Phase 4 Plan 01: Model Architecture and Factory Summary

**BTXRDClassifier wrapping timm EfficientNet-B0 with factory functions for model creation, checkpoint save/load, inverse-frequency class weights, device auto-detection, and early stopping**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-19T23:21:16Z
- **Completed:** 2026-02-19T23:23:17Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments
- BTXRDClassifier produces (batch, 3) logits from (batch, 3, 224, 224) input with num_features=1280 and gradcam_target_layer accessible
- Factory functions fully operational: create_model, compute_class_weights (verified: Normal=0.664, Benign=0.820, Malignant=3.640), get_device (auto selects MPS on this machine), save/load_checkpoint round-trip, EarlyStopping patience enforcement
- All symbols importable from `src.models` -- clean public API for downstream training, eval, inference scripts

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement BTXRDClassifier in classifier.py** - `f15ccdc` (feat)
2. **Task 2: Implement model factory, checkpoints, class weights, device detection, early stopping** - `8e9358b` (feat)
3. **Task 3: Update models __init__.py with public exports** - `cfbfa27` (feat)

## Files Created/Modified
- `src/models/classifier.py` - BTXRDClassifier nn.Module wrapping timm EfficientNet-B0 with forward() and gradcam_target_layer
- `src/models/factory.py` - create_model, save/load_checkpoint, compute_class_weights, get_device, EarlyStopping
- `src/models/__init__.py` - Public exports for all model package symbols

## Decisions Made
- `gradcam_target_layer` returns `model.bn2` (BatchNormAct2d with 1280 channels) -- confirmed working with pytorch-grad-cam 1.5.5 in research
- `save_checkpoint` includes `class_weights` (as `.tolist()`) in the checkpoint dict for full self-documentation
- `compute_class_weights` returns CPU tensor; caller is responsible for `.to(device)` to avoid accidental device assumptions
- No layers frozen -- full fine-tuning as recommended in research for small dataset with transfer learning

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All model architecture and factory utilities are ready for Plan 04-02 (training script)
- `create_model(config)` reads directly from `configs/default.yaml` model section
- Checkpoint format includes everything needed for eval (Phase 5), Grad-CAM (Phase 6), and inference (Phase 6)
- No blockers identified

---
*Phase: 04-model-training*
*Completed: 2026-02-19*
