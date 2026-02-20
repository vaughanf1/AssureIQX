---
phase: 06-explainability-and-inference
verified: 2026-02-20T19:04:29Z
status: passed
score: 5/5 must-haves verified
re_verification: false
---

# Phase 6: Explainability and Inference Verification Report

**Phase Goal:** Clinicians can inspect Grad-CAM heatmaps showing where the model attends for each prediction, and a CLI tool produces predictions with confidence scores and visual overlays for any input image

**Verified:** 2026-02-20T19:04:29Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `python scripts/gradcam.py` generates Grad-CAM heatmap overlays for TP/FP/FN per class | VERIFIED | `scripts/gradcam.py` (464 lines) fully implemented; `make gradcam` target calls it; all 3 gallery PNGs exist in `results/gradcam/` |
| 2 | 3-5 curated examples per category (TP/FP/FN) per class saved as image grids in `results/gradcam/` | VERIFIED | `gallery_Normal.png` (1.7MB), `gallery_Benign.png` (1.8MB), `gallery_Malignant.png` (1.9MB) all present with substantial file sizes |
| 3 | Annotation comparison panel shows Grad-CAM attention vs LabelMe tumor annotation masks with IoU scores | VERIFIED | `annotation_comparison.png` (1.2MB) exists; `annotation_report.json` contains 5-image comparison with per-image IoU and qualitative summary (mean IoU: 0.070) |
| 4 | `python scripts/infer.py --image path/to/image.jpg --checkpoint path/to/best.pt` outputs class prediction, softmax confidences, and Grad-CAM overlay PNG | VERIFIED | `scripts/infer.py` (382 lines) fully implemented; `results/inference/IMG000001_gradcam.png` (72KB) confirms it was run successfully |
| 5 | `python scripts/infer.py --input-dir path/to/images/` processes a directory of images and outputs predictions for each | VERIFIED | Batch mode implemented (lines 313-378); globs supported extensions, saves per-image overlays and `batch_results.json`; argument validation requires exactly one of `--image` or `--input-dir` |

**Score:** 5/5 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/explainability/gradcam.py` | 7 reusable Grad-CAM functions (min 150 lines) | VERIFIED | 280 lines; all 7 functions present: `denormalize_tensor`, `generate_gradcam`, `create_overlay`, `load_annotation_mask`, `compute_cam_iou`, `select_examples`, `build_gallery_grid` |
| `scripts/gradcam.py` | Gallery orchestration script | VERIFIED | 464 lines; full orchestration from checkpoint loading through annotation comparison; no stubs |
| `scripts/infer.py` | Single-image and batch inference CLI (min 100 lines) | VERIFIED | 382 lines; complete implementation with argparse, error handling, and Grad-CAM integration |
| `results/gradcam/gallery_Normal.png` | TP/FP/FN gallery grid for Normal class | VERIFIED | Exists, 1.7MB (substantive image content) |
| `results/gradcam/gallery_Benign.png` | TP/FP/FN gallery grid for Benign class | VERIFIED | Exists, 1.8MB |
| `results/gradcam/gallery_Malignant.png` | TP/FP/FN gallery grid for Malignant class | VERIFIED | Exists, 1.9MB |
| `results/gradcam/annotation_comparison.png` | 4-panel comparison for tumor images | VERIFIED | Exists, 1.2MB |
| `results/gradcam/annotation_report.json` | IoU scores and summary statistics | VERIFIED | Exists with `per_image` (5 entries), `mean_iou`, `num_images`, `iou_threshold`, `summary` fields |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `scripts/gradcam.py` | `src/explainability/gradcam.py` | `from src.explainability.gradcam import build_gallery_grid, compute_cam_iou, create_overlay, denormalize_tensor, generate_gradcam, load_annotation_mask, select_examples` | WIRED | Line 44-52; all 7 functions imported and called in orchestration |
| `scripts/gradcam.py` | `src/models/factory.py` | `from src.models.factory import create_model, get_device, load_checkpoint` | WIRED | Line 53; used in `main()` for model loading |
| `scripts/gradcam.py` | `src/evaluation/metrics.py` | `run_inference` for test set predictions | WIRED | Line 43; `run_inference(model, test_loader, device)` called at line 381 |
| `scripts/infer.py` | `src/explainability/gradcam.py` | `from src.explainability.gradcam import create_overlay, denormalize_tensor, generate_gradcam` | WIRED | Lines 40-44; all 3 used in `_run_single_inference()` |
| `scripts/infer.py` | `src/models/factory.py` | `from src.models.factory import create_model, get_device, load_checkpoint` | WIRED | Line 45; used in `_load_model()` helper |
| `scripts/infer.py` | `src/data/transforms.py` | `from src.data.transforms import get_test_transforms` | WIRED | Line 39; `get_test_transforms(image_size)` called at line 283 |
| `src/explainability/gradcam.py` | `pytorch_grad_cam` | `GradCAM`, `ClassifierOutputTarget`, `show_cam_on_image` | WIRED | Lines 31-33; used in `generate_gradcam()` and `create_overlay()` |
| `src/models/classifier.py` | `gradcam_target_layer` property | `model.bn2` (final BatchNormAct2d before global avg pool) | WIRED | Lines 65-73; referenced by `target_layers = [model.gradcam_target_layer]` in `generate_gradcam()` |

---

### Requirements Coverage

| Requirement | Description | Status | Evidence |
|-------------|-------------|--------|----------|
| EXPL-01 | Grad-CAM heatmap generation using pytorch-grad-cam on EfficientNet-B0 last conv layer | SATISFIED | `generate_gradcam()` uses `GradCAM` with `model.gradcam_target_layer` (bn2); gallery PNGs exist |
| EXPL-02 | Curated heatmap gallery: 3-5 examples each for TP, FP, FN per class, saved as image grid | SATISFIED | `select_examples()` selects k=3 per category; `build_gallery_grid()` saves 3-row grids; all 3 class gallery PNGs exist |
| EXPL-03 | Heatmap overlay comparison against LabelMe tumor annotations (qualitative: does model attend to tumor region?) | SATISFIED | `annotation_comparison.png` and `annotation_report.json` produced; mean IoU 0.070 with qualitative summary; Note: 5 panels are all Benign TPs (Malignant TPs not reached due to iteration order + max_images=5 cap — see Anti-Patterns section) |
| INFR-01 | Single-image inference script: image path + checkpoint -> class prediction + softmax confidences + Grad-CAM overlay | SATISFIED | `scripts/infer.py --image ... --checkpoint ...` fully implemented; `IMG000001_gradcam.png` confirms it ran |
| INFR-02 | Batch inference mode for evaluating a directory of images | SATISFIED | `--input-dir` mode globs images, saves overlays and `batch_results.json`, prints summary table |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `scripts/gradcam.py` | 185-200 | Annotation comparison iterates Benign TPs before Malignant TPs; with all 231 Benign test images having annotations, the first 5 hits are always Benign, so Malignant never appears in the comparison | Info | The ROADMAP SC3 requires "qualitative comparison against LabelMe tumor annotations" which is satisfied — but the comparison silently excludes Malignant. The plan explicitly intended "Benign TP + Malignant TP" representation. Clinical interpretability would be richer with both tumor types. |
| `Makefile` | 31 | `make infer` calls `python scripts/infer.py --config configs/default.yaml` without `--image` or `--input-dir`; would exit with argument error | Info | Does not affect CLI success criteria (SC4/SC5 reference `python scripts/infer.py` directly). The Makefile target is a convenience stub that requires the user to override. |

No blockers found. No TODO/FIXME/placeholder/NotImplementedError patterns present in any key file.

---

### Human Verification Required

#### 1. Heatmap Visual Quality

**Test:** Open `results/gradcam/gallery_Normal.png`, `gallery_Benign.png`, and `gallery_Malignant.png` and visually inspect the heatmap overlays.
**Expected:** TP rows should show heatmaps highlighting anatomically plausible regions (e.g., lung fields, nodule areas) rather than image borders, artifacts, or uniform coverage. FP and FN rows should suggest interpretable failure modes.
**Why human:** Visual quality of Grad-CAM heatmaps cannot be verified programmatically. This is ROADMAP success criterion 2 and requires clinical or domain judgment.

#### 2. Annotation Comparison Visual Review

**Test:** Open `results/gradcam/annotation_comparison.png` and inspect the 4-panel rows (Original | Tumor Annotation | Grad-CAM Heatmap | Overlay with IoU).
**Expected:** The Grad-CAM heatmap column and the tumor annotation column should be visually comparable (even if IoU is low at 0.070). The overlay column should show the colormap blended onto the original image with the IoU score visible in the title.
**Why human:** The ROADMAP SC3 asks whether "the model attends to the annotated tumor region" — this is a qualitative clinical judgment.

---

### Notable Observations

**Annotation comparison covers only Benign TPs.** All 5 panels in `annotation_comparison.png` are Benign class (verified by reading `annotation_report.json`). Both Benign (231) and Malignant (51) test images have LabelMe annotations. The script processes Benign TPs first (class_idx=1 before class_idx=2) and reaches max_images=5 before any Malignant images are processed. The ROADMAP success criterion 3 is technically satisfied (qualitative comparison IS produced), but the comparison does not represent the Malignant class. This is worth documenting as a limitation in Phase 7 reporting.

**Mean IoU of 0.070 is low but expected.** The model achieves ~68% accuracy on the stratified test set. The SUMMARY correctly identifies this as a finding (not a failure): "Grad-CAM attention is not strongly correlated with expert-annotated tumor regions at threshold 0.5." This is a genuine clinical insight about model behavior.

**`make gradcam` target is fully functional.** Produces all 5 required outputs in `results/gradcam/`. The `make infer` target requires manual `--image` or `--input-dir` arguments and cannot be run standalone, but the success criteria reference `python scripts/infer.py` directly.

---

## Files Verified

- `/Users/vaughanfawcett/AssureXRay/src/explainability/gradcam.py` — 280 lines, 7 functions, no stubs
- `/Users/vaughanfawcett/AssureXRay/scripts/gradcam.py` — 464 lines, full orchestration
- `/Users/vaughanfawcett/AssureXRay/scripts/infer.py` — 382 lines, full CLI
- `/Users/vaughanfawcett/AssureXRay/results/gradcam/gallery_Normal.png` — 1.7MB
- `/Users/vaughanfawcett/AssureXRay/results/gradcam/gallery_Benign.png` — 1.8MB
- `/Users/vaughanfawcett/AssureXRay/results/gradcam/gallery_Malignant.png` — 1.9MB
- `/Users/vaughanfawcett/AssureXRay/results/gradcam/annotation_comparison.png` — 1.2MB
- `/Users/vaughanfawcett/AssureXRay/results/gradcam/annotation_report.json` — valid JSON with 5 per-image entries
- `/Users/vaughanfawcett/AssureXRay/results/inference/IMG000001_gradcam.png` — 72KB (confirms infer.py ran)

---

*Verified: 2026-02-20T19:04:29Z*
*Verifier: Claude (gsd-verifier)*
