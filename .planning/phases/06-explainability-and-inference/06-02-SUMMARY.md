---
phase: 06-explainability-and-inference
plan: 02
subsystem: inference
tags: [gradcam, cli, inference, PIL, argparse, softmax]

# Dependency graph
requires:
  - phase: 06-01
    provides: generate_gradcam, denormalize_tensor, create_overlay utilities
  - phase: 04-02
    provides: trained checkpoints with class_names metadata
  - phase: 03-02
    provides: get_test_transforms deterministic preprocessing pipeline
provides:
  - Single-image inference CLI (--image) with Grad-CAM overlay generation
  - Batch inference CLI (--input-dir) with JSON results output
  - Complete scripts/infer.py for clinician and developer use
affects: [07-report, 08-packaging]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "CLI argument validation (exactly one of mutually exclusive args)"
    - "Separate torch.no_grad() for prediction vs gradient-enabled Grad-CAM"
    - "PIL.Image.fromarray for saving overlays (not cv2.imwrite)"

key-files:
  created: []
  modified:
    - scripts/infer.py

key-decisions:
  - "Grad-CAM targets predicted class (not ground truth) since ground truth unavailable at inference time"
  - "Class names loaded from checkpoint metadata (not hardcoded) for portability"
  - "PIL.Image.fromarray used for saving overlays (plan requirement, avoids BGR/RGB confusion)"

patterns-established:
  - "Inference script pattern: load checkpoint -> create model from ckpt config -> load_state_dict -> eval mode"
  - "Batch results saved as JSON array of per-image dicts with scores sub-object"

# Metrics
duration: 3min
completed: 2026-02-20
---

# Phase 6 Plan 02: Inference CLI Summary

**Single-image and batch inference CLI with Grad-CAM overlays, softmax confidences, and JSON batch results using scripts/infer.py**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-20T18:57:18Z
- **Completed:** 2026-02-20T19:00:25Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Complete inference CLI supporting --image (single) and --input-dir (batch) modes
- Grad-CAM overlay PNG generation for each image, saved to configurable output directory
- Console output with class name, per-class softmax confidences, and overlay path
- Batch mode produces batch_results.json with structured per-image predictions
- Robust argument validation and error handling (missing args, nonexistent files, missing checkpoint)

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement scripts/infer.py single-image and batch inference** - `161e86e` (feat)

## Files Created/Modified
- `scripts/infer.py` - Complete CLI script for single-image and batch inference with Grad-CAM overlays (282 lines, replacing 11-line stub)

## Decisions Made
- Grad-CAM targets predicted class (consistent with gradcam.py gallery script from 06-01)
- Class names loaded from checkpoint `ckpt["class_names"]` rather than hardcoded, ensuring portability across different checkpoint versions
- PIL.Image.fromarray used for saving overlay PNGs (avoids BGR/RGB issues with cv2.imwrite)
- Default output directory is `results/inference/` (consistent with project results structure)
- Supported image extensions: .jpg, .jpeg, .png, .bmp, .tiff (covers all common radiograph formats)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Phase 6 (Explainability and Inference) is now complete with both plans delivered
- Grad-CAM gallery (06-01) and inference CLI (06-02) ready for Phase 7 reporting
- Inference script can be referenced in Phase 7 documentation as the primary user-facing tool
- Low Grad-CAM/annotation IoU (0.070 from 06-01) should be documented as limitation in Phase 7

---
*Phase: 06-explainability-and-inference*
*Completed: 2026-02-20*
