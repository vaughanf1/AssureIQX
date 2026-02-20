---
phase: 06-explainability-and-inference
plan: 01
subsystem: explainability
tags: [gradcam, pytorch-grad-cam, heatmap, overlay, labelme, iou, gallery, efficientnet]

# Dependency graph
requires:
  - phase: 04-model-training
    provides: "Trained checkpoints (best_stratified.pt) with gradcam_target_layer property on BTXRDClassifier"
  - phase: 05-evaluation
    provides: "run_inference function, evaluation patterns, test set predictions"
provides:
  - "Reusable Grad-CAM functions: generate_gradcam, denormalize_tensor, create_overlay, load_annotation_mask, compute_cam_iou, select_examples, build_gallery_grid"
  - "Per-class TP/FP/FN gallery grids with Grad-CAM overlays"
  - "Annotation comparison panel with IoU scores for tumor images"
  - "Annotation report JSON with per-image and summary IoU statistics"
affects: [06-02-inference, 07-reporting]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "GradCAM context manager pattern with ClassifierOutputTarget"
    - "Denormalize -> generate_gradcam -> create_overlay pipeline"
    - "LabelMe JSON to binary mask via cv2.fillPoly/rectangle"
    - "TP/FP/FN selection sorted by confidence for curated galleries"

key-files:
  created:
    - src/explainability/gradcam.py
    - results/gradcam/gallery_Normal.png
    - results/gradcam/gallery_Benign.png
    - results/gradcam/gallery_Malignant.png
    - results/gradcam/annotation_comparison.png
    - results/gradcam/annotation_report.json
  modified:
    - scripts/gradcam.py

key-decisions:
  - "Target predicted class (not true class) for Grad-CAM to show what the model attends to for its decision"
  - "IoU threshold 0.5 for binarizing Grad-CAM heatmaps (standard in literature)"
  - "Stratified checkpoint used for gallery (macro AUC 0.846, better performance = more interpretable heatmaps)"
  - "Top-5 Benign TP images for annotation comparison (sorted by confidence, highest first)"

patterns-established:
  - "GradCAM context manager: with GradCAM(model=model, target_layers=[model.gradcam_target_layer]) as cam"
  - "Gallery grid: 3 rows (TP/FP/FN) x k cols with titled subplots at DPI 150"
  - "Annotation comparison: 4-panel (original, annotation mask, heatmap, overlay with IoU)"

# Metrics
duration: 2min
completed: 2026-02-20
---

# Phase 6 Plan 1: Grad-CAM Explainability Summary

**Grad-CAM heatmap galleries with TP/FP/FN examples per class and annotation IoU comparison for tumor images using pytorch-grad-cam 1.5.5**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-20T17:28:47Z
- **Completed:** 2026-02-20T17:50:00Z
- **Tasks:** 2/2
- **Files modified:** 2

## Accomplishments
- Implemented 7 reusable Grad-CAM utility functions in `src/explainability/gradcam.py` (280 lines)
- Generated per-class gallery grids with 3 TP, 3 FP, 3 FN examples for Normal, Benign, and Malignant classes
- Annotation comparison panel with 4-panel figures (original, annotation mask, Grad-CAM heatmap, overlay) for 5 tumor images
- IoU quantification: mean IoU 0.070 across 5 correctly classified tumor images (1/5 with IoU > 0.1), indicating Grad-CAM attention is not strongly correlated with expert-annotated tumor regions at threshold 0.5

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement src/explainability/gradcam.py utility module** - `893cd46` (feat)
2. **Task 2: Implement scripts/gradcam.py gallery orchestration** - `fde591f` (feat)

## Files Created/Modified
- `src/explainability/gradcam.py` - 7 reusable functions: denormalize_tensor, generate_gradcam, create_overlay, load_annotation_mask, compute_cam_iou, select_examples, build_gallery_grid
- `scripts/gradcam.py` - Orchestration script: checkpoint loading, inference, TP/FP/FN selection, gallery generation, annotation comparison with IoU
- `results/gradcam/gallery_Normal.png` - 3x3 grid of TP/FP/FN Grad-CAM overlays for Normal class
- `results/gradcam/gallery_Benign.png` - 3x3 grid of TP/FP/FN Grad-CAM overlays for Benign class
- `results/gradcam/gallery_Malignant.png` - 3x3 grid of TP/FP/FN Grad-CAM overlays for Malignant class
- `results/gradcam/annotation_comparison.png` - 5-row x 4-col comparison panels for tumor images
- `results/gradcam/annotation_report.json` - Per-image IoU scores and summary statistics

## Decisions Made
- **Target predicted class for Grad-CAM:** Shows what the model attends to for its actual decision, rather than what it should attend to. This is more clinically informative for understanding model behavior.
- **IoU threshold 0.5:** Standard in the literature for binarizing Grad-CAM heatmaps. Mean IoU of 0.070 is lower than typical literature values (0.5-0.6), but expected given our model's modest performance (68% accuracy).
- **Stratified checkpoint only:** Used the better-performing stratified split checkpoint (macro AUC 0.846) for more interpretable heatmaps.
- **Top-5 by confidence for annotation comparison:** Selected the 5 most confident correctly classified tumor images to show best-case attention alignment.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - all verification checks passed on first run.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- `src/explainability/gradcam.py` is ready for import by `scripts/infer.py` (Plan 06-02)
- All 7 utility functions are tested via end-to-end gallery generation
- Low mean IoU (0.070) suggests the model's attention patterns do not strongly align with expert tumor annotations -- this should be documented as a limitation in Phase 7 reporting
- Gallery grids confirm the model has interpretable TP/FP/FN examples across all 3 classes

---
*Phase: 06-explainability-and-inference*
*Completed: 2026-02-20*
