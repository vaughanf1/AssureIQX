# Phase 6: Explainability and Inference - Research

**Researched:** 2026-02-20
**Domain:** Grad-CAM explainability for classification CNNs, LabelMe annotation comparison, CLI inference tooling
**Confidence:** HIGH

## Summary

Phase 6 implements Grad-CAM heatmap generation, a curated gallery of TP/FP/FN examples, qualitative comparison of model attention against tumor annotations, and single-image/batch inference scripts. The entire pipeline has been verified end-to-end: pytorch-grad-cam 1.5.5 (already installed) generates valid heatmaps using the `model.gradcam_target_layer` property (which returns `model.model.bn2`, the 1280-channel BatchNormAct2d before global average pooling), and the overlay visualization works on both CPU and MPS devices. LabelMe annotations exist for all 1,867 tumor images (Benign + Malignant) with both rectangle and polygon shape types, readily convertible to binary masks via `cv2.fillPoly` / `cv2.rectangle` for IoU-based comparison against thresholded Grad-CAM heatmaps.

The existing codebase provides strong foundations: `scripts/gradcam.py` and `scripts/infer.py` are stubs with argparse skeletons; `src/explainability/gradcam.py` is a placeholder ready for implementation; the evaluation script (`scripts/eval.py`) demonstrates the exact checkpoint-loading, dataset creation, and inference patterns to reuse; and the config already has `gradcam` and `inference` sections. No new dependencies are required.

The key implementation challenge is selecting TP/FP/FN examples systematically from test set predictions (requires re-running inference on the test set or loading cached predictions), creating properly formatted image grids, and handling the LabelMe annotation-to-mask pipeline for the qualitative comparison. The annotation comparison (EXPL-03) is explicitly qualitative per the requirements, but computing IoU between thresholded Grad-CAM masks and annotation masks provides a quantitative supplement.

**Primary recommendation:** Structure as two modules: (1) `src/explainability/gradcam.py` provides all reusable functions (heatmap generation, overlay creation, annotation mask loading, image grid assembly), (2) `scripts/gradcam.py` orchestrates the curated gallery and annotation comparison, (3) `scripts/infer.py` provides CLI inference with Grad-CAM overlay. Reuse the checkpoint-loading and inference patterns from `scripts/eval.py`.

## Standard Stack

### Core (all already in requirements.txt)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pytorch-grad-cam (grad-cam) | 1.5.5 | `GradCAM` class for heatmap generation, `show_cam_on_image` for overlay visualization, `ClassifierOutputTarget` for per-class targeting | De-facto standard for PyTorch Grad-CAM; supports batch processing, context manager pattern, EfficientNet architecture |
| torch | 2.6.0 | Model loading, inference, softmax computation | Core framework |
| opencv-python (cv2) | 4.13.0 | LabelMe polygon/rectangle to binary mask via `fillPoly`/`rectangle`, image resizing | Already installed as grad-cam dependency; standard for mask operations |
| matplotlib | 3.10.0 | Image grid assembly (`plt.subplots`), Grad-CAM gallery figures, annotation overlay plots | Publication-quality plotting; consistent with Phase 5 |
| numpy | 2.2.3 | Array operations, denormalization, IoU computation | Foundation for all numerical work |
| Pillow | 11.1.0 | Image loading for inference script (handles JPEG, PNG, BMP, etc.) | Already used in dataset class |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| json (stdlib) | N/A | Loading LabelMe annotation JSON files, saving inference results | Annotation parsing |
| pathlib (stdlib) | N/A | Path construction, image glob patterns | Consistent with existing codebase |
| argparse (stdlib) | N/A | CLI argument parsing for `--image`, `--checkpoint`, `--input-dir` | Already in stub scripts |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `pytorch-grad-cam` GradCAM | Manual gradient hook implementation | pytorch-grad-cam handles edge cases (batch norm, ReLU hooks, memory cleanup via context manager); ~5 lines vs ~50+ lines of error-prone hook management |
| `cv2.fillPoly` for annotation masks | `shapely` or `skimage.draw.polygon` | cv2 is already installed (grad-cam dependency), handles both polygons and rectangles natively, and is faster for rasterization |
| `matplotlib` image grids | `torchvision.utils.make_grid` | torchvision make_grid works on tensors, but matplotlib subplots allow per-image titles (class, confidence, TP/FP/FN label) and mixed-size annotations which are needed for the curated gallery |
| IoU for annotation comparison | Pointing Game (% of max activation inside annotation) | IoU is more informative and standard in the literature; Pointing Game gives only a binary yes/no. Both are easy to compute; IoU is preferred |

**Installation:** No new packages needed. All are in `requirements.txt`.

## Architecture Patterns

### Recommended Project Structure

```
src/explainability/
  __init__.py            # Exports (already exists)
  gradcam.py             # Core functions: generate_gradcam(), create_overlay(),
                         #   load_annotation_mask(), compute_cam_iou(),
                         #   build_image_grid(), denormalize_tensor()

scripts/
  gradcam.py             # CLI: load model, run inference, select TP/FP/FN,
                         #   generate curated gallery + annotation comparison
  infer.py               # CLI: single-image and batch inference with Grad-CAM overlay

results/gradcam/
  gallery_Normal.png     # Grid: TP/FP/FN examples for Normal class
  gallery_Benign.png     # Grid: TP/FP/FN examples for Benign class
  gallery_Malignant.png  # Grid: TP/FP/FN examples for Malignant class
  annotation_comparison.png  # Side-by-side: original, annotation mask, Grad-CAM, overlay
  annotation_report.json     # IoU scores and qualitative summary
```

### Pattern 1: GradCAM Generation with Context Manager

**What:** Use pytorch-grad-cam's context manager pattern for clean hook cleanup.
**When to use:** Any time you generate Grad-CAM heatmaps.
**Verified:** Tested end-to-end with BTXRDClassifier on MPS device.

```python
# Source: Verified against pytorch-grad-cam 1.5.5 installed in .venv
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# target_layers must be a list (even for single layer)
target_layers = [model.gradcam_target_layer]  # returns model.model.bn2

with GradCAM(model=model, target_layers=target_layers) as cam:
    # targets=None uses the model's highest-scoring class
    # targets=[ClassifierOutputTarget(class_idx)] targets a specific class
    targets = [ClassifierOutputTarget(class_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    # grayscale_cam shape: (batch_size, H, W), values in [0, 1]
    heatmap = grayscale_cam[0]

# Overlay onto denormalized RGB image (float32, [0, 1], HWC)
overlay = show_cam_on_image(rgb_img, heatmap, use_rgb=True)
# overlay shape: (H, W, 3), dtype=uint8, values [0, 255]
```

### Pattern 2: Denormalization for Overlay Visualization

**What:** Reverse ImageNet normalization to get displayable RGB images.
**When to use:** Before calling `show_cam_on_image` or saving original images alongside heatmaps.

```python
import numpy as np
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD

MEAN = np.array(IMAGENET_MEAN)  # (0.485, 0.456, 0.406)
STD = np.array(IMAGENET_STD)    # (0.229, 0.224, 0.225)

def denormalize_tensor(tensor):
    """Convert normalized CHW tensor to HWC float32 [0,1] for visualization."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
    img = img * STD + MEAN
    return np.clip(img, 0, 1).astype(np.float32)
```

### Pattern 3: TP/FP/FN Selection from Test Predictions

**What:** Systematically select examples for the curated gallery.
**When to use:** For EXPL-02 gallery creation.

```python
# After running inference: y_true, y_pred, y_prob arrays (from run_inference)
# For each class c:
#   TP: y_true == c AND y_pred == c
#   FP: y_true != c AND y_pred == c
#   FN: y_true == c AND y_pred != c
# Select top-k by confidence (for TP: highest prob[c], for FP: highest prob[c],
# for FN: lowest prob[c] -- these are the most informative examples)

def select_examples(y_true, y_pred, y_prob, class_idx, k=3):
    tp_mask = (y_true == class_idx) & (y_pred == class_idx)
    fp_mask = (y_true != class_idx) & (y_pred == class_idx)
    fn_mask = (y_true == class_idx) & (y_pred != class_idx)

    # Sort by confidence for most informative examples
    tp_indices = np.where(tp_mask)[0]
    fp_indices = np.where(fp_mask)[0]
    fn_indices = np.where(fn_mask)[0]

    # TP: highest confidence correct predictions
    tp_sorted = tp_indices[np.argsort(-y_prob[tp_indices, class_idx])][:k]
    # FP: highest confidence false positives (most confident mistakes)
    fp_sorted = fp_indices[np.argsort(-y_prob[fp_indices, class_idx])][:k]
    # FN: lowest confidence misses (model was most wrong)
    fn_sorted = fn_indices[np.argsort(y_prob[fn_indices, class_idx])][:k]

    return tp_sorted, fp_sorted, fn_sorted
```

### Pattern 4: LabelMe Annotation to Binary Mask

**What:** Convert LabelMe JSON annotations to binary masks for IoU comparison.
**When to use:** For EXPL-03 annotation comparison.

```python
import json
import cv2
import numpy as np

def load_annotation_mask(ann_path, target_size=224):
    """Load LabelMe annotation and return resized binary mask."""
    data = json.load(open(ann_path))
    h, w = data['imageHeight'], data['imageWidth']
    mask = np.zeros((h, w), dtype=np.uint8)

    for shape in data['shapes']:
        pts = np.array(shape['points'], dtype=np.int32)
        if shape['shape_type'] == 'polygon':
            cv2.fillPoly(mask, [pts], 1)
        elif shape['shape_type'] == 'rectangle':
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 1, -1)

    # Resize to match model input size
    mask_resized = cv2.resize(mask, (target_size, target_size),
                               interpolation=cv2.INTER_NEAREST)
    return mask_resized
```

### Pattern 5: IoU Between Grad-CAM and Annotation Mask

**What:** Quantitative metric for how well Grad-CAM overlaps with tumor annotations.
**When to use:** For EXPL-03 annotation comparison.

```python
def compute_cam_annotation_iou(cam_heatmap, annotation_mask, threshold=0.5):
    """Compute IoU between thresholded Grad-CAM and annotation mask.

    Args:
        cam_heatmap: (H, W) float32 in [0, 1] from GradCAM
        annotation_mask: (H, W) binary mask from LabelMe
        threshold: Grad-CAM binarization threshold

    Returns:
        IoU score (float), also known as Jaccard index
    """
    cam_binary = (cam_heatmap >= threshold).astype(np.uint8)
    intersection = np.logical_and(cam_binary, annotation_mask).sum()
    union = np.logical_or(cam_binary, annotation_mask).sum()
    return float(intersection / union) if union > 0 else 0.0
```

### Pattern 6: Single-Image Inference Pipeline

**What:** Load raw image, preprocess, run inference, generate Grad-CAM, save overlay.
**When to use:** For `scripts/infer.py` single-image mode.

```python
# Inference pipeline for arbitrary input images:
# 1. Load image with PIL, convert to RGB
# 2. Apply test transforms (Resize 224 + Normalize)
# 3. Run model forward pass (raw logits)
# 4. Apply softmax for confidence scores
# 5. Generate Grad-CAM for predicted class
# 6. Denormalize for overlay
# 7. Save overlay PNG alongside text output
```

### Anti-Patterns to Avoid

- **Double softmax:** Model outputs raw logits. Apply softmax exactly once (in inference, not in model). The `run_inference` function in `metrics.py` already handles this correctly -- reuse this pattern.
- **Forgetting model.eval():** Must call `model.eval()` before inference AND before Grad-CAM. Grad-CAM uses gradients (not `torch.no_grad()`), but batch norm must be in eval mode.
- **Not using context manager for GradCAM:** The `with GradCAM(...) as cam:` pattern ensures hook cleanup. Without it, hooks accumulate and leak memory on repeated calls.
- **Hardcoding target layer name:** Use `model.gradcam_target_layer` property, not string `"bn2"`. This ensures the correct layer even if the model architecture changes.
- **Saving images with wrong color space:** `show_cam_on_image` with `use_rgb=True` returns RGB. If saving with `cv2.imwrite`, convert to BGR first. Using `PIL.Image.fromarray(overlay).save()` handles RGB correctly.
- **Running GradCAM inside torch.no_grad():** GradCAM requires gradient computation. Do NOT wrap GradCAM calls in `torch.no_grad()`. The GradCAM class manages its own gradient state.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Grad-CAM computation | Manual gradient hooks + weight averaging | `pytorch_grad_cam.GradCAM` | Hook management, memory cleanup, batch support, edge cases with batch norm layers are all handled |
| Heatmap overlay on image | Manual colormap + alpha blending | `show_cam_on_image()` from pytorch-grad-cam | Handles colormap (cv2.COLORMAP_JET), alpha blending, uint8 conversion, proper scaling |
| Polygon rasterization | Manual point-in-polygon or shapely | `cv2.fillPoly` + `cv2.rectangle` | Already available via opencv-python (grad-cam dependency), handles complex polygons efficiently |
| Image grid composition | Manual numpy concatenation | `matplotlib.pyplot.subplots` with `ax.imshow` | Handles variable titles, spacing, DPI, figure saving; consistent with Phase 5 plotting style |

**Key insight:** The entire Grad-CAM pipeline from input tensor to overlay PNG is about 10 lines using pytorch-grad-cam utilities. The implementation complexity is in the orchestration (selecting examples, loading annotations, assembling grids), not in the Grad-CAM computation itself.

## Common Pitfalls

### Pitfall 1: GradCAM Returns All-Zero Heatmap

**What goes wrong:** The heatmap is entirely black/zero, producing no visual information.
**Why it happens:** (1) Wrong target layer selected (too early in network, or a layer without spatial dimensions). (2) Gradients are zero because the model is very confident (near-1.0 softmax) -- the gradient of a near-saturated softmax approaches zero. (3) `requires_grad=False` on target layer parameters.
**How to avoid:** (1) Use `model.gradcam_target_layer` which returns `model.bn2` (verified working). (2) Target the predicted class, not a class with probability near 0. (3) Ensure model parameters have `requires_grad=True` (default for loaded checkpoints).
**Warning signs:** Check `grayscale_cam.max() > 0` after generation. Log a warning if max is 0 or very small.

### Pitfall 2: Image Color Space Mismatch

**What goes wrong:** Overlays appear with inverted or wrong colors.
**Why it happens:** Mixing RGB and BGR color spaces. PIL loads as RGB; OpenCV defaults to BGR; `show_cam_on_image` has a `use_rgb` parameter.
**How to avoid:** Always use `use_rgb=True` with `show_cam_on_image`. Save with `PIL.Image.fromarray(overlay)` (expects RGB) instead of `cv2.imwrite` (expects BGR).
**Warning signs:** Heatmap overlays where red appears as blue, or images appear with blue-tinted skin tones.

### Pitfall 3: Denormalization Overflow

**What goes wrong:** Denormalized pixel values exceed [0, 1] range, causing artifacts in overlay.
**Why it happens:** Some extreme pixel values after denormalization go slightly above 1.0 or below 0.0 due to augmentation or model artifacts.
**How to avoid:** Always `np.clip(denormalized, 0, 1)` after denormalization. The `show_cam_on_image` function expects float32 in [0, 1].
**Warning signs:** White or black patches in the overlay that weren't in the original image.

### Pitfall 4: Annotation Mask Size Mismatch

**What goes wrong:** IoU computation returns 0 because mask and heatmap are different sizes.
**Why it happens:** LabelMe annotations are at original image resolution (e.g., 3213x2397), but Grad-CAM heatmaps are at model input size (224x224).
**How to avoid:** Resize annotation mask to 224x224 using `cv2.resize` with `INTER_NEAREST` interpolation (preserves binary mask values, no blurring).
**Warning signs:** Annotation mask shape doesn't match `(224, 224)`.

### Pitfall 5: Not Enough FP/FN Examples for Some Classes

**What goes wrong:** Trying to select 3-5 FP or FN examples when fewer exist in the test set.
**Why it happens:** With 51 Malignant test samples and ~60% sensitivity, there are about 20 FN. But for well-classified classes, there may be fewer FP/FN. Normal class has 282 test samples with ~61% recall, so ~110 FN exist. Each class should have enough examples, but the code must handle the edge case of fewer examples than requested.
**How to avoid:** Use `min(k, len(available_indices))` when selecting examples. If fewer than 3 exist, use all available. Document the counts in the gallery.
**Warning signs:** Index out of bounds errors or empty subplot rows in the gallery grid.

### Pitfall 6: Inference Script Assumes Test Transforms

**What goes wrong:** Inference on arbitrary images produces wrong predictions because augmentation is applied.
**Why it happens:** Using train transforms (with CLAHE, flip, rotation) instead of test transforms (resize + normalize only) for inference.
**How to avoid:** Always use `get_test_transforms(image_size)` for inference, never `get_train_transforms`.
**Warning signs:** Different predictions for the same image on repeated runs (stochastic augmentations).

## Code Examples

### Full Grad-CAM Generation for a Single Image

```python
# Source: Verified end-to-end on MPS with best_stratified.pt checkpoint
import torch
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from src.models.factory import create_model, load_checkpoint, get_device
from src.data.transforms import get_test_transforms, IMAGENET_MEAN, IMAGENET_STD

# Load model from checkpoint
device = get_device("auto")
ckpt = load_checkpoint("checkpoints/best_stratified.pt", device="cpu")
model = create_model(ckpt["config"])
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(device)
model.eval()

# Load and preprocess image
image = Image.open("path/to/image.jpeg").convert("RGB")
image_np = np.array(image)
transform = get_test_transforms(224)
input_tensor = transform(image=image_np)["image"].unsqueeze(0).to(device)

# Run inference
with torch.no_grad():
    logits = model(input_tensor)
    probs = torch.nn.functional.softmax(logits, dim=1)
    pred_idx = probs.argmax(dim=1).item()
    class_names = ckpt["class_names"]
    pred_class = class_names[pred_idx]
    confidence = probs[0, pred_idx].item()

# Generate Grad-CAM (NO torch.no_grad here -- GradCAM needs gradients)
with GradCAM(model=model, target_layers=[model.gradcam_target_layer]) as cam:
    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

# Create overlay
mean = np.array(IMAGENET_MEAN)
std = np.array(IMAGENET_STD)
rgb_img = input_tensor[0].cpu().numpy().transpose(1, 2, 0) * std + mean
rgb_img = np.clip(rgb_img, 0, 1).astype(np.float32)
overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# Save
Image.fromarray(overlay).save("output_gradcam.png")
print(f"Prediction: {pred_class} (confidence: {confidence:.3f})")
```

### Batch Inference with Grad-CAM

```python
# Process batch of images efficiently
with GradCAM(model=model, target_layers=[model.gradcam_target_layer]) as cam:
    for batch_tensors, batch_labels in dataloader:
        batch_tensors = batch_tensors.to(device)

        # Get predictions first
        with torch.no_grad():
            logits = model(batch_tensors)
            probs = torch.nn.functional.softmax(logits, dim=1)
            pred_classes = probs.argmax(dim=1)

        # Generate Grad-CAM for each image targeting its predicted class
        targets = [ClassifierOutputTarget(c.item()) for c in pred_classes]
        grayscale_cams = cam(input_tensor=batch_tensors, targets=targets)
        # grayscale_cams shape: (batch_size, 224, 224)
```

### Image Grid Assembly (Gallery)

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def build_gallery_grid(images, titles, rows, cols, output_path, suptitle=""):
    """Create a grid of images with titles.

    Args:
        images: List of (H, W, 3) uint8 or float32 images
        titles: List of strings for each subplot
        rows, cols: Grid dimensions
        output_path: Path to save PNG
        suptitle: Figure title
    """
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1:
        axes = [axes]
    if cols == 1:
        axes = [[ax] for ax in axes]

    for idx, (img, title) in enumerate(zip(images, titles)):
        r, c = divmod(idx, cols)
        axes[r][c].imshow(img)
        axes[r][c].set_title(title, fontsize=9)
        axes[r][c].axis("off")

    # Hide unused subplots
    for idx in range(len(images), rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

### Annotation Comparison Panel

```python
# For EXPL-03: Side-by-side visualization
# Panel layout per image: [Original | Annotation Mask | Grad-CAM Heatmap | Overlay]
# For tumor images (Benign/Malignant) where annotations exist

def create_comparison_panel(rgb_img, annotation_mask, cam_heatmap, iou, output_path):
    """4-panel comparison: original, annotation, gradcam, overlay."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(rgb_img)
    axes[0].set_title("Original")

    axes[1].imshow(annotation_mask, cmap="Reds", alpha=0.7)
    axes[1].set_title("Tumor Annotation")

    axes[2].imshow(cam_heatmap, cmap="jet")
    axes[2].set_title("Grad-CAM Heatmap")

    overlay = show_cam_on_image(rgb_img, cam_heatmap, use_rgb=True)
    axes[3].imshow(overlay)
    axes[3].set_title(f"Overlay (IoU={iou:.2f})")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual gradient hooks for Grad-CAM | pytorch-grad-cam library | Established by 2021 (v1.3+) | Eliminates hook management bugs; adds batch support, context manager, multiple CAM variants |
| Grad-CAM only | Multiple CAM variants (Grad-CAM++, Score-CAM, Eigen-CAM, HiResCAM) | 2020-2024 | Grad-CAM remains the standard for classification; Grad-CAM++ and HiResCAM offer alternatives for specific failure cases |
| Qualitative-only explainability | IoU-based quantitative evaluation against annotations | 2022-2025 | Mean IoU scores of 0.5-0.6 for Grad-CAM are typical in the literature; our EXPL-03 should compute and report IoU |
| Per-image Grad-CAM | Curated TP/FP/FN galleries | 2020+ | Systematic selection reveals model failure modes; clinically more valuable than random examples |

**Deprecated/outdated:**
- **CAM (original Class Activation Mapping)**: Requires global average pooling directly before classifier. Grad-CAM generalizes to any conv layer. Use Grad-CAM, not CAM.
- **Manual hook-based implementations**: pytorch-grad-cam handles all hook lifecycle. Don't write custom hooks.

## Key Verified Facts

These facts were verified by running actual code against the installed packages and project codebase (not just documentation):

1. **Target layer:** `model.gradcam_target_layer` returns `model.model.bn2` which is `BatchNormAct2d(1280)`. Verified working with pytorch-grad-cam 1.5.5. **[HIGH confidence -- tested]**

2. **GradCAM output shape:** `(batch_size, 224, 224)` float32 in `[0, 1]`. **[HIGH confidence -- tested]**

3. **MPS compatibility:** GradCAM works on MPS (Apple Silicon) device with PyTorch 2.6.0. No fallback to CPU needed. **[HIGH confidence -- tested on device]**

4. **Context manager:** `with GradCAM(model=model, target_layers=[layer]) as cam:` pattern works and properly cleans up hooks. **[HIGH confidence -- tested]**

5. **Batch processing:** GradCAM processes batches natively. Pass batch tensor and list of targets (one per image). **[HIGH confidence -- tested with batch_size=2]**

6. **show_cam_on_image:** Expects float32 HWC image in `[0, 1]` + float32 HW heatmap in `[0, 1]`. Returns uint8 HWC `[0, 255]`. `use_rgb=True` for RGB output. **[HIGH confidence -- tested]**

7. **LabelMe annotations:** 1,867 annotation files, all non-empty. Shape types: rectangle (2,324) and polygon (2,318). Labels: osteochondroma, osteosarcoma, etc. Normal images have NO annotations (0/282 in test set). Benign (231/231) and Malignant (51/51) test images all have annotations. **[HIGH confidence -- counted from files]**

8. **Annotation-to-mask:** `cv2.fillPoly` for polygons, `cv2.rectangle` for rectangles. Resize to 224x224 with `INTER_NEAREST`. **[HIGH confidence -- tested]**

9. **Checkpoint structure:** Contains `model_state_dict`, `config`, `class_names`, `normalization` (mean/std), `class_weights`. **[HIGH confidence -- inspected best_stratified.pt]**

10. **Existing stubs:** `scripts/gradcam.py` has argparse with `--config` and `--override`. Missing: nothing else needed. `scripts/infer.py` has `--config`, `--override`, `--image`, `--input-dir`. Missing: `--checkpoint` argument (required by success criteria 4). **[HIGH confidence -- read files]**

## Data Context for Gallery Selection

From the stratified test set evaluation (564 samples):

| Class | Test Samples | Sensitivity | Expected TP | Expected FP | Expected FN |
|-------|-------------|-------------|-------------|-------------|-------------|
| Normal | 282 | 60.6% | ~171 | ~41* | ~111 |
| Benign | 231 | 78.4% | ~181 | ~125* | ~50 |
| Malignant | 51 | 60.8% | ~31 | ~16* | ~20 |

*FP counts are approximate from the confusion matrix.

All classes have sufficient TP, FP, and FN examples for 3-5 per category. The Malignant class has the fewest samples but still enough (~20 FN, ~31 TP, ~16 FP).

## Open Questions

1. **Grad-CAM IoU threshold selection**
   - What we know: Standard threshold is 0.5 for binarizing Grad-CAM heatmaps. Literature reports mean IoU of 0.5-0.6 for Grad-CAM against annotations.
   - What's unclear: Optimal threshold may vary for radiographs. Multiple thresholds (0.3, 0.5, 0.7) could be reported.
   - Recommendation: Use 0.5 as default, optionally report at multiple thresholds. This is a qualitative analysis per the requirements, so exact threshold is not critical.

2. **Which checkpoint for Grad-CAM gallery**
   - What we know: Config has `inference.default_checkpoint: checkpoints/best_stratified.pt`. Stratified split has better performance (macro AUC 0.846 vs 0.627).
   - What's unclear: Should gallery show both splits or just stratified?
   - Recommendation: Use stratified checkpoint for the main gallery (better performance, more interpretable heatmaps). The inference script should accept any checkpoint via `--checkpoint` flag.

3. **Normal class annotation comparison**
   - What we know: Normal images have no LabelMe annotations (no tumors to annotate). Grad-CAM on correctly classified Normal images should ideally show diffuse attention (no focal hot spots suggesting tumor-like features).
   - What's unclear: How to document "absence of focal attention" qualitatively.
   - Recommendation: Include Normal TP examples in the gallery but skip IoU comparison (no annotation mask exists). Document qualitatively whether Normal TPs show diffuse vs. focal attention patterns.

## Sources

### Primary (HIGH confidence)
- pytorch-grad-cam 1.5.5 -- installed package, tested end-to-end with `GradCAM`, `ClassifierOutputTarget`, `show_cam_on_image`
- timm 1.0.15 EfficientNet-B0 -- verified `model.bn2` (BatchNormAct2d 1280) as Grad-CAM target layer
- OpenCV 4.13.0 -- verified `cv2.fillPoly` and `cv2.rectangle` for annotation mask creation
- Project codebase: `src/models/classifier.py`, `src/models/factory.py`, `src/evaluation/metrics.py`, `scripts/eval.py`, `configs/default.yaml`

### Secondary (MEDIUM confidence)
- [pytorch-grad-cam GitHub README](https://github.com/jacobgil/pytorch-grad-cam) -- API documentation, target layer recommendations
- [pytorch-grad-cam Issue #95](https://github.com/jacobgil/pytorch-grad-cam/issues/95) -- EfficientNet `_conv_head` recommendation (for original EfficientNet, not timm variant; timm uses `bn2`)
- [Advancing AI Interpretability: Grad-CAM IoU analysis](https://www.mdpi.com/2504-4990/7/1/12) -- Mean IoU scores of 0.57 baseline for Grad-CAM against annotations

### Tertiary (LOW confidence)
- [pytorch-grad-cam Discussion #459](https://github.com/jacobgil/pytorch-grad-cam/discussions/459) -- EfficientNet/MobileNet target layer discussion (inconclusive in the thread itself)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- All libraries installed, tested, versions pinned
- Architecture: HIGH -- Full end-to-end pipeline verified on actual model and data
- Pitfalls: HIGH -- Based on verified testing and known library behavior
- Annotation comparison: MEDIUM -- IoU approach is standard but threshold sensitivity not quantified

**Research date:** 2026-02-20
**Valid until:** 2026-03-20 (stable libraries, no breaking changes expected)
