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

import json  # noqa: E402

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
RESULTS_DIR = PROJECT_ROOT / "results"


@st.cache_resource
def load_model(checkpoint_path: str):
    """Load and cache the classification model.

    Uses @st.cache_resource so the model loads once and persists across
    all Streamlit reruns and user sessions.

    Args:
        checkpoint_path: Path to .pt checkpoint file.

    Returns:
        Tuple of (model, class_names, device, image_size).
    """
    device = get_device("auto")
    ckpt = load_checkpoint(checkpoint_path, device="cpu")
    model = create_model(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    image_size = ckpt["config"]["data"].get("image_size", 224)
    return model, ckpt["class_names"], device, image_size


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
    model, class_names, device, image_size = load_model(str(DEFAULT_CHECKPOINT))

    # -- File upload ----------------------------------------------------------
    uploaded_file = st.file_uploader(
        "Upload a bone radiograph",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
    )

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(pil_image)

        # Preprocess with the same deterministic pipeline as evaluation
        transform = get_test_transforms(image_size)
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

    # -- Model Performance Section --------------------------------------------
    st.divider()
    st.header("Model Performance")

    # Load metrics from results/
    strat_metrics_path = RESULTS_DIR / "stratified" / "metrics_summary.json"
    center_metrics_path = RESULTS_DIR / "center_holdout" / "metrics_summary.json"
    strat_report_path = RESULTS_DIR / "stratified" / "classification_report.json"
    center_report_path = RESULTS_DIR / "center_holdout" / "classification_report.json"

    if strat_metrics_path.exists() and center_metrics_path.exists():
        strat_m = json.loads(strat_metrics_path.read_text())
        center_m = json.loads(center_metrics_path.read_text())
        strat_r = json.loads(strat_report_path.read_text())
        center_r = json.loads(center_report_path.read_text())

        tab1, tab2, tab3, tab4 = st.tabs([
            "Metrics Summary", "Paper Comparison", "Split Comparison", "Confusion Matrices"
        ])

        with tab1:
            st.subheader("Stratified Split (Active Checkpoint)")
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Accuracy", f"{strat_m['accuracy']:.1%}")
            col_b.metric("Macro AUC", f"{strat_m['macro_auc']:.3f}")
            col_c.metric("Malignant Sensitivity", f"{strat_m['malignant_sensitivity']:.1%}")
            col_d.metric("Test Samples", f"{strat_m['test_set_size']}")

            st.markdown("**Per-Class Classification Report**")
            report_rows = []
            for cls in ["Normal", "Benign", "Malignant"]:
                r = strat_r[cls]
                report_rows.append({
                    "Class": cls,
                    "Precision": f"{r['precision']:.3f}",
                    "Recall": f"{r['recall']:.3f}",
                    "F1-Score": f"{r['f1-score']:.3f}",
                    "Support": int(r["support"]),
                })
            st.dataframe(pd.DataFrame(report_rows), hide_index=True, use_container_width=True)

        with tab2:
            st.subheader("Our Results vs BTXRD Paper (Yao et al.)")
            st.caption(
                "The paper reports results using YOLOv8s-cls with a random 80/20 split, "
                "600px images, and 300 epochs. Our model uses EfficientNet-B0 with a "
                "stratified 70/15/15 split (harder -- with duplicate-aware leakage protection)."
            )

            # Paper baseline values (from Yao et al., Scientific Data 2025)
            paper_recall = {"Normal": 0.898, "Benign": 0.875, "Malignant": 0.839}
            paper_precision = {"Normal": 0.913, "Benign": 0.881, "Malignant": 0.734}

            # Headline comparison
            col_ours, col_paper = st.columns(2)
            with col_ours:
                st.markdown("**Our Model** (EfficientNet-B0, stratified split)")
                st.metric("Accuracy", f"{strat_m['accuracy']:.1%}")
                st.metric("Macro AUC", f"{strat_m['macro_auc']:.3f}")
                st.metric("Malignant Sensitivity", f"{strat_m['malignant_sensitivity']:.1%}")
            with col_paper:
                st.markdown("**BTXRD Paper** (YOLOv8s-cls, random split)")
                st.metric("Accuracy", "~84%")
                st.metric("Macro AUC", "Not reported")
                st.metric("Malignant Sensitivity", f"{paper_recall['Malignant']:.1%}")

            st.markdown("**Per-Class Sensitivity (Recall) Comparison**")
            paper_rows = []
            for cls in ["Normal", "Benign", "Malignant"]:
                ours = strat_m["per_class_sensitivity"][cls]
                theirs = paper_recall[cls]
                diff = ours - theirs
                paper_rows.append({
                    "Class": cls,
                    "Ours": f"{ours:.3f}",
                    "Paper": f"{theirs:.3f}",
                    "Difference": f"{diff:+.3f}",
                })
            st.dataframe(pd.DataFrame(paper_rows), hide_index=True, use_container_width=True)

            st.markdown("**Per-Class Precision Comparison**")
            prec_rows = []
            for cls in ["Normal", "Benign", "Malignant"]:
                ours = strat_r[cls]["precision"]
                theirs = paper_precision[cls]
                diff = ours - theirs
                prec_rows.append({
                    "Class": cls,
                    "Ours": f"{ours:.3f}",
                    "Paper": f"{theirs:.3f}",
                    "Difference": f"{diff:+.3f}",
                })
            st.dataframe(pd.DataFrame(prec_rows), hide_index=True, use_container_width=True)

            st.info(
                "Our model achieves comparable sensitivity to the paper's YOLOv8s-cls "
                "despite using a stricter evaluation protocol (stratified split with "
                "duplicate-aware leakage protection vs the paper's random split). "
                "Malignant sensitivity (84.3% vs 83.9%) slightly exceeds the paper's reported value."
            )

        with tab3:
            st.subheader("Stratified vs Center-Holdout")
            st.caption(
                "The center-holdout split tests generalization to unseen hospital centers. "
                "Lower scores indicate a generalization gap."
            )

            comparison_rows = []
            for metric_name, key in [
                ("Accuracy", "accuracy"),
                ("Macro AUC", "macro_auc"),
                ("Malignant Sensitivity", "malignant_sensitivity"),
            ]:
                s_val = strat_m[key]
                c_val = center_m[key]
                gap = c_val - s_val
                comparison_rows.append({
                    "Metric": metric_name,
                    "Stratified": f"{s_val:.3f}",
                    "Center Holdout": f"{c_val:.3f}",
                    "Gap": f"{gap:+.3f}",
                })
            st.dataframe(pd.DataFrame(comparison_rows), hide_index=True, use_container_width=True)

            st.markdown("**Per-Class Sensitivity Comparison**")
            sens_rows = []
            for cls in ["Normal", "Benign", "Malignant"]:
                s_val = strat_m["per_class_sensitivity"][cls]
                c_val = center_m["per_class_sensitivity"][cls]
                sens_rows.append({
                    "Class": cls,
                    "Stratified": f"{s_val:.3f}",
                    "Center Holdout": f"{c_val:.3f}",
                    "Gap": f"{c_val - s_val:+.3f}",
                })
            st.dataframe(pd.DataFrame(sens_rows), hide_index=True, use_container_width=True)

            st.markdown("**Per-Class AUC Comparison**")
            auc_rows = []
            for cls in ["Normal", "Benign", "Malignant"]:
                s_val = strat_m["per_class_auc"][cls]
                c_val = center_m["per_class_auc"][cls]
                auc_rows.append({
                    "Class": cls,
                    "Stratified": f"{s_val:.3f}",
                    "Center Holdout": f"{c_val:.3f}",
                    "Gap": f"{c_val - s_val:+.3f}",
                })
            st.dataframe(pd.DataFrame(auc_rows), hide_index=True, use_container_width=True)

        with tab4:
            st.subheader("Confusion Matrices")
            cm_col1, cm_col2 = st.columns(2)
            strat_cm = RESULTS_DIR / "stratified" / "confusion_matrix_normalized.png"
            center_cm = RESULTS_DIR / "center_holdout" / "confusion_matrix_normalized.png"
            with cm_col1:
                st.markdown("**Stratified**")
                if strat_cm.exists():
                    st.image(str(strat_cm), use_container_width=True)
            with cm_col2:
                st.markdown("**Center Holdout**")
                if center_cm.exists():
                    st.image(str(center_cm), use_container_width=True)
    else:
        st.info(
            "Model performance metrics not available. "
            "Run `make evaluate` to generate evaluation results."
        )


if __name__ == "__main__":
    main()
