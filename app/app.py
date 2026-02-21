#!/usr/bin/env python3
"""AssureXRay Streamlit Demo Application.

Provides an interactive web interface for uploading bone radiograph images
and viewing 3-class predictions (Normal / Benign / Malignant) with
Grad-CAM heatmap overlays and confidence bar charts.

Usage:
    streamlit run app/app.py
    # or
    make demo

Implemented in: Phase 8
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path for src/ imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from src.data.transforms import get_test_transforms  # noqa: E402
from src.explainability.gradcam import (  # noqa: E402
    create_overlay,
    denormalize_tensor,
    generate_gradcam,
)
from src.models.factory import create_model, get_device, load_checkpoint  # noqa: E402

# -- Constants ----------------------------------------------------------------
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_stratified.pt"
IMAGE_SIZE = 224


@st.cache_resource
def load_model(checkpoint_path: str):
    """Load and cache the classification model.

    Uses @st.cache_resource so the model loads once and persists across
    all Streamlit reruns and user sessions.

    Args:
        checkpoint_path: Path to .pt checkpoint file.

    Returns:
        Tuple of (model, class_names, device).
    """
    device = get_device("auto")
    ckpt = load_checkpoint(checkpoint_path, device="cpu")
    model = create_model(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, ckpt["class_names"], device


def run_inference(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    class_names: list[str],
    device: torch.device,
) -> dict:
    """Run prediction and Grad-CAM on a preprocessed input tensor.

    The forward pass runs inside torch.no_grad() for efficiency.
    Grad-CAM runs OUTSIDE no_grad because it requires gradient computation.

    Args:
        model: Trained model in eval mode.
        input_tensor: Tensor of shape (1, C, H, W).
        class_names: List of class name strings.
        device: Torch device for tensor placement.

    Returns:
        Dict with pred_class, pred_idx, confidence, class_names,
        overlay (uint8 numpy HWC), and original_rgb (uint8 numpy HWC).
    """
    input_tensor = input_tensor.to(device)

    # Forward pass (no_grad for prediction only)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)

    pred_idx = probs.argmax(dim=1).item()
    confidence = probs[0].cpu().numpy()

    # Grad-CAM (needs gradients -- OUTSIDE no_grad)
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


def main() -> None:
    """Main Streamlit application entry point."""
    # Page config must be the first Streamlit call
    st.set_page_config(
        page_title="AssureXRay",
        page_icon=":material/radiology:",
        layout="wide",
    )

    # -- Title and disclaimer (always visible) --------------------------------
    st.title("AssureXRay -- Bone Tumor Classification")
    st.warning(
        "**NOT FOR CLINICAL USE -- Research Prototype Only.** "
        "This tool is for demonstration purposes only and has not been "
        "validated for clinical decision-making."
    )

    # -- Load model (cached) --------------------------------------------------
    model, class_names, device = load_model(str(DEFAULT_CHECKPOINT))

    # -- File upload ----------------------------------------------------------
    uploaded_file = st.file_uploader(
        "Upload a bone radiograph",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
    )

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(pil_image)

        # Preprocess with the same deterministic pipeline as evaluation
        transform = get_test_transforms(IMAGE_SIZE)
        transformed = transform(image=image_np)
        input_tensor = transformed["image"].unsqueeze(0)

        # Run inference + Grad-CAM
        with st.spinner("Analyzing image..."):
            result = run_inference(model, input_tensor, class_names, device)

        # -- Results layout ---------------------------------------------------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            st.image(pil_image, use_container_width=True)

        with col2:
            st.subheader("Grad-CAM Overlay")
            st.image(result["overlay"], use_container_width=True)

        # Prediction heading
        st.subheader(f"Prediction: {result['pred_class']}")

        # Confidence bar chart
        scores_df = pd.DataFrame(
            {
                "Class": result["class_names"],
                "Confidence": [
                    float(result["confidence"][i])
                    for i in range(len(result["class_names"]))
                ],
            }
        )
        st.bar_chart(scores_df, x="Class", y="Confidence", horizontal=True)


if __name__ == "__main__":
    main()
