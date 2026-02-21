# Phase 8: Streamlit Demo - Research

**Researched:** 2026-02-21
**Domain:** Streamlit web app for medical image classification with Grad-CAM visualization
**Confidence:** HIGH

## Summary

Phase 8 implements a Streamlit web application that allows non-technical users to upload a bone radiograph image, view a 3-class prediction (Normal / Benign / Malignant) with confidence bar chart, and see a Grad-CAM heatmap overlay -- all with a prominent "NOT FOR CLINICAL USE" disclaimer. The phase depends entirely on Phase 6 infrastructure: the `src/explainability/gradcam.py` module (already fully implemented with `generate_gradcam()`, `create_overlay()`, `denormalize_tensor()`), the `src/models/factory.py` model loading utilities (`create_model()`, `load_checkpoint()`, `get_device()`), and the `src/data/transforms.py` preprocessing pipeline (`get_test_transforms()`). The existing CLI inference script `scripts/infer.py` provides the exact inference pattern to adapt for the Streamlit UI.

Streamlit 1.54.0 (current as of February 2026) is the standard choice for this type of rapid ML demo. It provides `st.file_uploader` for image upload, `st.image` for displaying numpy/PIL images (including Grad-CAM overlays), `st.bar_chart` for confidence visualization (with native `horizontal=True` support), and `st.cache_resource` for caching the PyTorch model across reruns so it loads only once. The existing `app/app.py` stub already has the basic Streamlit scaffolding with `set_page_config` and imports.

The key implementation challenge is structuring the app so that (1) the model loads once via `@st.cache_resource` rather than on every interaction, (2) the Grad-CAM generation runs outside `torch.no_grad()` as established in Phase 6, and (3) the disclaimer is visible on every page state (before and after upload). The app is a single-page application with no routing needed, making this a straightforward Streamlit use case. Plotly is NOT needed -- Streamlit's built-in `st.bar_chart` with `horizontal=True` handles the confidence bars natively.

**Primary recommendation:** Rewrite `app/app.py` as a single-file Streamlit app that imports directly from the existing `src/` modules, using `@st.cache_resource` for model loading and the same inference+Grad-CAM pipeline as `scripts/infer.py`. Add `streamlit>=1.42.0` to requirements.txt (pin to minimum supporting the `horizontal` bar chart parameter). No additional dependencies required.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| streamlit | >=1.42.0 (current: 1.54.0) | Web application framework for ML demos | De-facto standard for Python ML demos; no frontend code needed; `st.file_uploader`, `st.image`, `st.bar_chart`, `st.cache_resource` cover all requirements |
| torch | 2.6.0 (already installed) | Model loading and inference | Existing dependency; used by `src/models/factory.py` |
| pytorch-grad-cam | 1.5.5 (already installed) | Grad-CAM heatmap generation | Existing dependency; used via `src/explainability/gradcam.py` |
| Pillow | 11.1.0 (already installed) | Image loading from uploaded bytes | Existing dependency; `Image.open(uploaded_file).convert("RGB")` |
| numpy | 2.2.3 (already installed) | Array operations for image preprocessing | Existing dependency |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| albumentations | 2.0.8 (already installed) | Test transforms (resize + normalize) | Applied to uploaded image via `get_test_transforms(224)` |
| pandas | 2.2.3 (already installed) | DataFrame for `st.bar_chart` input | Confidence scores formatted as DataFrame for chart |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `st.bar_chart` | `st.plotly_chart` with Plotly Express | Plotly adds a dependency and complexity; `st.bar_chart(horizontal=True)` is sufficient for 3-class confidence bars and requires zero extra dependencies |
| Streamlit | Gradio | Gradio has richer ML-specific widgets but is heavier; Streamlit is already stubbed in `app/app.py` and specified in the Makefile `demo` target |
| Single-file app | Multi-page app | Single-page is correct here -- there is only one workflow (upload -> predict -> visualize) |

**Installation:**
```bash
pip install streamlit>=1.42.0
```

Note: `streamlit>=1.42.0` is the minimum version that supports `st.bar_chart(horizontal=True)`. This parameter was introduced in Streamlit 1.42.0 (December 2025). Current version is 1.54.0.

## Architecture Patterns

### Recommended Project Structure

```
app/
  app.py              # Single-file Streamlit application (rewrite existing stub)

.streamlit/
  config.toml         # Optional: theme/server settings (disable telemetry, set port)

src/                  # All imports come from existing modules:
  models/factory.py   #   create_model(), load_checkpoint(), get_device()
  explainability/
    gradcam.py        #   generate_gradcam(), create_overlay(), denormalize_tensor()
  data/transforms.py  #   get_test_transforms()
```

### Pattern 1: Model Caching with @st.cache_resource

**What:** Load the PyTorch model once and cache it across all reruns and user sessions.
**When to use:** Always -- model loading takes several seconds; must not reload on every widget interaction.
**Source:** Streamlit official docs (st.cache_resource)

```python
import streamlit as st
from src.models.factory import create_model, load_checkpoint, get_device

@st.cache_resource
def load_model(checkpoint_path: str):
    """Load model once, cache across all sessions and reruns."""
    device = get_device("auto")
    ckpt = load_checkpoint(checkpoint_path, device="cpu")
    model = create_model(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    class_names = ckpt["class_names"]
    return model, class_names, device
```

### Pattern 2: Image Upload and Preprocessing Pipeline

**What:** Accept user-uploaded image via `st.file_uploader`, convert to PIL, then apply the same test transforms used in CLI inference.
**When to use:** After user uploads an image.

```python
import numpy as np
from PIL import Image
from src.data.transforms import get_test_transforms

uploaded_file = st.file_uploader(
    "Upload a bone radiograph",
    type=["jpg", "jpeg", "png", "bmp", "tiff"],
)

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(pil_image)

    transform = get_test_transforms(224)
    transformed = transform(image=image_np)
    input_tensor = transformed["image"].unsqueeze(0).to(device)
```

### Pattern 3: Inference + Grad-CAM in One Pass

**What:** Run forward pass for prediction, then generate Grad-CAM (outside torch.no_grad) for the predicted class.
**When to use:** After image preprocessing.

```python
import torch
import torch.nn.functional as F
from src.explainability.gradcam import generate_gradcam, create_overlay, denormalize_tensor

# Inference (with no_grad for prediction only)
with torch.no_grad():
    logits = model(input_tensor)
    probs = F.softmax(logits, dim=1)
    pred_idx = probs.argmax(dim=1).item()
    confidence = probs[0].cpu().numpy()

# Grad-CAM (needs gradients -- NO torch.no_grad here)
heatmap = generate_gradcam(model, input_tensor, pred_idx)
rgb_img = denormalize_tensor(input_tensor.squeeze(0))
overlay = create_overlay(rgb_img, heatmap)
```

### Pattern 4: Confidence Bar Chart with st.bar_chart

**What:** Display horizontal bar chart showing per-class confidence scores.
**When to use:** After inference to show all class probabilities.

```python
import pandas as pd

scores_df = pd.DataFrame({
    "Class": class_names,
    "Confidence": [float(confidence[i]) for i in range(len(class_names))],
})
st.bar_chart(scores_df, x="Class", y="Confidence", horizontal=True)
```

### Pattern 5: Persistent Disclaimer Banner

**What:** Show "NOT FOR CLINICAL USE" disclaimer that is visible regardless of app state.
**When to use:** Always -- must be visible before upload, after upload, and alongside results.

```python
# Place at the TOP of the app, right after set_page_config
st.warning(
    "**NOT FOR CLINICAL USE -- Research Prototype Only.** "
    "This tool is for demonstration purposes only and has not been "
    "validated for clinical decision-making.",
    icon="\\u26A0\\uFE0F",
)
```

### Pattern 6: Spinner During Inference

**What:** Show loading indicator while model runs inference and Grad-CAM.
**When to use:** Wrap inference + Grad-CAM generation.

```python
with st.spinner("Analyzing image..."):
    # inference + gradcam code here
    pass
st.success("Analysis complete!")
```

### Anti-Patterns to Avoid

- **Loading model inside the upload callback:** Model must be cached with `@st.cache_resource`, not loaded each time. Without caching, the ~4M parameter model reloads on every widget interaction (slider, button, upload).
- **Using `st.cache_data` for the model:** `st.cache_data` copies the return value (serializes/deserializes). PyTorch models are not easily serializable this way. Use `st.cache_resource` which returns a singleton reference.
- **Wrapping Grad-CAM in `torch.no_grad()`:** Grad-CAM requires gradient computation. This is already enforced by the `generate_gradcam()` function from Phase 6 but must not be accidentally wrapped by the calling code.
- **Running inference on every rerun without checking upload state:** Always guard inference code with `if uploaded_file is not None:` to avoid errors when no file is uploaded.
- **Forgetting `model.eval()`:** Must set model to eval mode before inference. This is handled inside the cached `load_model()` function, so it persists across reruns.
- **Using `cv2.imwrite` for overlay display:** Streamlit's `st.image` accepts numpy arrays directly with `channels="RGB"`. No need to save to disk or convert color spaces. The `create_overlay()` function returns RGB uint8 arrays.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Image upload widget | Custom file input HTML/JS | `st.file_uploader(type=["jpg","jpeg","png"])` | Handles drag-and-drop, file validation, size limits, BytesIO interface |
| Confidence bar chart | Custom matplotlib figure embedded in Streamlit | `st.bar_chart(df, horizontal=True)` | Native Streamlit chart; interactive; no matplotlib overhead; zero extra code |
| Model caching across reruns | Manual `st.session_state` dict with load checks | `@st.cache_resource` decorator | Handles cache invalidation, thread safety, cache clearing; model persists across all sessions |
| Loading spinner | Custom progress bar or placeholder text | `st.spinner("Analyzing...")` | Built-in animated spinner with context manager pattern |
| Disclaimer/warning banner | Custom HTML/CSS div | `st.warning("...", icon="...")` | Styled correctly, visible, accessible; matches Streamlit theme |
| Layout columns for side-by-side images | Custom HTML grid | `st.columns([1, 1])` | Responsive, theme-aware, handles various screen sizes |
| Grad-CAM generation | Re-implementing Grad-CAM logic | Import from `src/explainability/gradcam` | Phase 6 already has verified, tested functions |
| Model loading pipeline | Custom checkpoint parsing | Import from `src/models/factory` | Phase 4/6 already has `load_checkpoint()` + `create_model()` |

**Key insight:** The entire Streamlit app should contain ZERO custom ML logic. All inference, Grad-CAM, preprocessing, and model loading functions already exist in `src/`. The app is purely a UI wrapper that calls existing functions.

## Common Pitfalls

### Pitfall 1: Model Reloads on Every Interaction

**What goes wrong:** The app becomes unusably slow (2-5 second delay on every click/upload) because the model reloads from disk on each Streamlit rerun.
**Why it happens:** Streamlit reruns the entire script on every user interaction. Without caching, `load_checkpoint()` + `create_model()` + `.load_state_dict()` execute every time.
**How to avoid:** Use `@st.cache_resource` decorator on the model loading function. The decorator ensures the model loads once and the same object is returned on subsequent calls.
**Warning signs:** Console shows repeated "Loading model..." messages; each interaction has multi-second delay.

### Pitfall 2: Grad-CAM Fails Silently Inside torch.no_grad()

**What goes wrong:** Grad-CAM returns an all-zero heatmap, producing a blank overlay.
**Why it happens:** If the inference and Grad-CAM code are both wrapped in `torch.no_grad()`, Grad-CAM cannot compute gradients. The existing `generate_gradcam()` function will log a warning but still return zeros.
**How to avoid:** Keep inference (forward pass + softmax) inside `torch.no_grad()`, but call `generate_gradcam()` OUTSIDE that context. The Phase 6 code already structures this correctly -- follow the same pattern.
**Warning signs:** Overlay image looks identical to the original (no colored heatmap visible). Check for the warning log "Grad-CAM heatmap is all zeros."

### Pitfall 3: Import Path Errors When Running from app/ Directory

**What goes wrong:** `ModuleNotFoundError: No module named 'src'` when launching `streamlit run app/app.py`.
**Why it happens:** Streamlit runs the script with the script's directory as the working directory, or the project root may not be on `sys.path`. The `src/` package is at the project root.
**How to avoid:** Add the project root to `sys.path` at the top of `app/app.py`, using the same pattern as `scripts/infer.py`:
```python
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
```
**Warning signs:** Import errors mentioning `src.models`, `src.explainability`, or `src.data`.

### Pitfall 4: UploadedFile Not Compatible with PIL.Image.open

**What goes wrong:** This is actually NOT a real issue -- `st.file_uploader` returns a `BytesIO`-like object that PIL can read directly. But developers sometimes try to pass file paths instead.
**Why it happens:** Confusion between file paths and file-like objects.
**How to avoid:** Simply pass the uploaded file directly: `Image.open(uploaded_file).convert("RGB")`. The `UploadedFile` class extends `BytesIO`.
**Warning signs:** Attempting to call `.name` or use the object as a path string.

### Pitfall 5: Displaying Numpy Arrays with Wrong Value Range

**What goes wrong:** `st.image` throws `RuntimeError` about pixel values being out of range.
**Why it happens:** The denormalized image is float32 in [0, 1], but might have values slightly outside this range. The overlay from `create_overlay()` is uint8 [0, 255] which displays correctly.
**How to avoid:** For the original image display, convert to uint8 first: `(rgb_img * 255).astype(np.uint8)`. For the overlay, it is already uint8 from `create_overlay()`. Alternatively, pass `clamp=True` to `st.image`.
**Warning signs:** `RuntimeError: All float pixel values must be between 0 and 1` or garbled image display.

### Pitfall 6: MPS/CUDA Device Mismatch in Cached Model

**What goes wrong:** The cached model is on one device but new input tensors are created on a different device.
**Why it happens:** `@st.cache_resource` caches the model on whatever device was detected at first load. If the device detection changes between sessions (unlikely but possible on shared machines), there could be a mismatch.
**How to avoid:** Return `device` from the cached model function and use it consistently for input tensor placement. The recommended `load_model()` function above returns `(model, class_names, device)` for this reason.
**Warning signs:** RuntimeError about tensors on different devices.

### Pitfall 7: Disclaimer Not Visible in All App States

**What goes wrong:** The disclaimer appears only after results are shown, not on the initial upload page.
**Why it happens:** Placing the disclaimer inside an `if uploaded_file:` block.
**How to avoid:** Place the `st.warning()` call at the top of the script, unconditionally, right after `st.set_page_config()` and the page title. Streamlit renders elements in order, so top-level elements appear in all states.
**Warning signs:** Visual inspection shows no disclaimer when the app first loads.

## Code Examples

### Complete Streamlit App Structure (Verified Pattern)

```python
#!/usr/bin/env python3
"""AssureXRay Streamlit Demo Application."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path for src/ imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from src.data.transforms import get_test_transforms
from src.explainability.gradcam import create_overlay, denormalize_tensor, generate_gradcam
from src.models.factory import create_model, get_device, load_checkpoint

# ── Constants ──────────────────────────────────────────────
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_stratified.pt"
IMAGE_SIZE = 224


@st.cache_resource
def load_model(checkpoint_path: str):
    """Load and cache the classification model."""
    device = get_device("auto")
    ckpt = load_checkpoint(checkpoint_path, device="cpu")
    model = create_model(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, ckpt["class_names"], device


def run_inference(model, input_tensor, class_names, device):
    """Run prediction and Grad-CAM on a preprocessed tensor."""
    input_tensor = input_tensor.to(device)

    # Forward pass (no_grad for prediction)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)

    pred_idx = probs.argmax(dim=1).item()
    confidence = probs[0].cpu().numpy()

    # Grad-CAM (needs gradients -- outside no_grad)
    heatmap = generate_gradcam(model, input_tensor, pred_idx)
    rgb_img = denormalize_tensor(input_tensor.squeeze(0))
    overlay = create_overlay(rgb_img, heatmap)

    return {
        "pred_class": class_names[pred_idx],
        "pred_idx": pred_idx,
        "confidence": confidence,
        "class_names": class_names,
        "overlay": overlay,
        "original_rgb": (rgb_img * 255).astype(np.uint8),
    }


def main():
    # ── Page config (must be first Streamlit call) ──────────
    st.set_page_config(
        page_title="AssureXRay",
        page_icon=":material/radiology:",
        layout="wide",
    )

    # ── Title and disclaimer (always visible) ───────────────
    st.title("AssureXRay -- Bone Tumor Classification")
    st.warning(
        "**NOT FOR CLINICAL USE -- Research Prototype Only.** "
        "This tool is for demonstration purposes only and has not been "
        "validated for clinical decision-making.",
    )

    # ── Load model (cached) ─────────────────────────────────
    model, class_names, device = load_model(str(DEFAULT_CHECKPOINT))

    # ── File upload ─────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Upload a bone radiograph",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
    )

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(pil_image)

        # Preprocess
        transform = get_test_transforms(IMAGE_SIZE)
        transformed = transform(image=image_np)
        input_tensor = transformed["image"].unsqueeze(0)

        # Inference + Grad-CAM
        with st.spinner("Analyzing image..."):
            result = run_inference(model, input_tensor, class_names, device)

        # ── Results layout ──────────────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            st.image(pil_image, use_container_width=True)

        with col2:
            st.subheader("Grad-CAM Overlay")
            st.image(result["overlay"], use_container_width=True)

        # Prediction result
        st.subheader(f"Prediction: {result['pred_class']}")

        # Confidence bar chart
        scores_df = pd.DataFrame({
            "Class": result["class_names"],
            "Confidence": [float(result["confidence"][i])
                           for i in range(len(result["class_names"]))],
        })
        st.bar_chart(scores_df, x="Class", y="Confidence", horizontal=True)


if __name__ == "__main__":
    main()
```

### Makefile Target (Already Exists)

```makefile
# Already in Makefile:
demo: ## Launch Streamlit demo app
	streamlit run app/app.py
```

The existing `make demo` target already points to `app/app.py` and uses `streamlit run`, which matches the success criteria.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `st.cache` (deprecated) | `st.cache_resource` for models, `st.cache_data` for data | Streamlit 1.18 (Jan 2023) | Old `st.cache` had confusing `allow_output_mutation` parameter; new decorators are clear about caching semantics |
| `st.set_page_config(layout="wide")` + manual column width CSS | `st.columns([ratio, ratio])` with responsive design | Streamlit 1.0+ (stable since 2021) | No CSS hacking needed for multi-column layouts |
| `matplotlib` figures via `st.pyplot()` for bar charts | `st.bar_chart(horizontal=True)` native charts | Streamlit 1.42.0 (Dec 2025) | Native charts are interactive, theme-aware, and require zero matplotlib/plotly overhead |
| `use_column_width=True` in `st.image` | `width="stretch"` or `use_container_width=True` | Streamlit 1.40+ (2025) | Old parameter is deprecated; new `width` parameter is more flexible |

**Deprecated/outdated:**
- **`st.cache`**: Replaced by `st.cache_data` and `st.cache_resource`. Do NOT use `st.cache`.
- **`use_column_width` in `st.image`**: Deprecated; use `width="stretch"` or pass integer pixel width.
- **`st.beta_columns`**: Renamed to `st.columns` long ago.

## Open Questions

1. **Streamlit version pinning strategy**
   - What we know: The app needs `st.bar_chart(horizontal=True)` which requires >=1.42.0. Current PyPI version is 1.54.0.
   - What's unclear: Whether to pin exactly (e.g., `streamlit==1.54.0`) or use a minimum (e.g., `streamlit>=1.42.0`).
   - Recommendation: Use `streamlit>=1.42.0` in requirements.txt. This allows users to install whatever recent version they have while ensuring the `horizontal` parameter is available. Streamlit has good backward compatibility within minor versions.

2. **Whether to create `.streamlit/config.toml`**
   - What we know: Streamlit can be configured via `.streamlit/config.toml` for theme, server port, telemetry, etc. The project does not currently have this file.
   - What's unclear: Whether a custom theme is needed or if defaults are sufficient.
   - Recommendation: Create a minimal `.streamlit/config.toml` that disables telemetry/usage stats gathering (`[browser] gatherUsageStats = false`) and optionally sets the server port. Theme customization is not required for the demo.

3. **Checkpoint path flexibility**
   - What we know: The default checkpoint is `checkpoints/best_stratified.pt`. The demo should work with this default.
   - What's unclear: Whether users should be able to select a different checkpoint via the UI.
   - Recommendation: Hardcode `best_stratified.pt` as the default. Adding a checkpoint selector is out of scope for the basic demo requirement (DEMO-01). The CLI `scripts/infer.py` already supports `--checkpoint` for advanced users.

4. **use_container_width deprecation status**
   - What we know: Official docs show `use_container_width` as deprecated in favor of `width="stretch"` for `st.image`. However, `use_container_width` still works.
   - What's unclear: Exact version where `width` parameter was added to `st.image`.
   - Recommendation: Use `width="stretch"` for future-proofing if on a recent enough Streamlit version, otherwise `use_container_width=True` as fallback. Test during implementation.

## Sources

### Primary (HIGH confidence)
- [Streamlit official docs: st.file_uploader](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader) -- API parameters, return types, image upload example
- [Streamlit official docs: st.image](https://docs.streamlit.io/develop/api-reference/media/st.image) -- numpy array display, channels parameter, width options
- [Streamlit official docs: st.bar_chart](https://docs.streamlit.io/develop/api-reference/charts/st.bar_chart) -- horizontal parameter, DataFrame input
- [Streamlit official docs: st.cache_resource](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource) -- PyTorch model caching pattern, scope options
- [Streamlit official docs: st.set_page_config](https://docs.streamlit.io/develop/api-reference/configuration/st.set_page_config) -- layout, page_title, page_icon parameters
- [Streamlit official docs: Layout elements](https://docs.streamlit.io/develop/api-reference/layout) -- st.columns, st.sidebar, st.container, st.tabs
- [PyPI: streamlit 1.54.0](https://pypi.org/project/streamlit/) -- Current version, Python >=3.10 requirement
- Project codebase: `src/explainability/gradcam.py` -- verified generate_gradcam(), create_overlay(), denormalize_tensor() APIs
- Project codebase: `src/models/factory.py` -- verified create_model(), load_checkpoint(), get_device() APIs
- Project codebase: `scripts/infer.py` -- full inference pattern including preprocessing and Grad-CAM

### Secondary (MEDIUM confidence)
- [Streamlit discuss: PyTorch model demo](https://discuss.streamlit.io/t/pytorch-model-demo-load-the-model-inference-only-once/45118) -- @st.cache_resource pattern for PyTorch confirmed
- [Streamlit official docs: Caching overview](https://docs.streamlit.io/develop/concepts/architecture/caching) -- cache_resource vs cache_data distinction
- [Streamlit 2025 release notes](https://docs.streamlit.io/develop/quick-reference/release-notes/2025) -- horizontal bar chart introduced in 1.42.0
- [Streamlit blog: Common app problems](https://blog.streamlit.io/common-app-problems-resource-limits/) -- Memory and performance considerations

### Tertiary (LOW confidence)
- [GitHub: Image-Classification-Web-App-using-PyTorch-and-Streamlit](https://github.com/denistanjingyu/Image-Classification-Web-App-using-PyTorch-and-Streamlit) -- Community example of PyTorch + Streamlit pattern (useful reference but not authoritative)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- Streamlit API verified via official docs; all supporting libraries already installed and tested in prior phases
- Architecture: HIGH -- Inference pipeline directly reuses Phase 6 verified code; Streamlit patterns from official docs
- Pitfalls: HIGH -- Based on official Streamlit docs (caching, rerun model), and verified Phase 6 Grad-CAM patterns
- Code examples: HIGH -- Pattern matches `scripts/infer.py` (working code) adapted for Streamlit API (verified from official docs)

**Research date:** 2026-02-21
**Valid until:** 2026-03-21 (Streamlit releases monthly but API is backward-compatible; Phase 6 code is stable)
