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

import csv  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402

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
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
IMAGES_DIR = PROJECT_ROOT / "data_raw" / "images"
CHALLENGE_IMAGES_DIR = Path(__file__).resolve().parent / "challenge_images"
CHALLENGE_SEED = 99
CHALLENGE_PER_CLASS = 10


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


@st.cache_data
def load_challenge_images():
    """Select 30 test images (10 per class) in a fixed random order."""
    test_csv = SPLITS_DIR / "stratified_test.csv"
    if not test_csv.exists():
        return None
    rows_by_class = {"Normal": [], "Benign": [], "Malignant": []}
    with open(test_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"]
            if label in rows_by_class:
                rows_by_class[label].append(row["image_id"])
    rng = random.Random(CHALLENGE_SEED)
    selected = []
    for label in ["Normal", "Benign", "Malignant"]:
        imgs = rows_by_class[label]
        chosen = rng.sample(imgs, min(CHALLENGE_PER_CLASS, len(imgs)))
        for img_id in chosen:
            selected.append({"image_id": img_id, "true_label": label})
    rng.shuffle(selected)
    return selected


def get_ai_prediction(model, class_names, device, image_size, image_path):
    """Run the AI model on a single image and return the predicted class."""
    pil_img = Image.open(image_path).convert("RGB")
    image_np = np.array(pil_img)
    transform = get_test_transforms(image_size)
    transformed = transform(image=image_np)
    input_tensor = transformed["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
    pred_idx = probs.argmax(dim=1).item()
    return class_names[pred_idx]


def page_classifier(model, class_names, device, image_size):
    """Main classifier page with upload and model performance."""
    # -- File upload ----------------------------------------------------------
    uploaded_file = st.file_uploader(
        "Upload a bone radiograph",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
    )

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(pil_image)

        transform = get_test_transforms(image_size)
        transformed = transform(image=image_np)
        input_tensor = transformed["image"].unsqueeze(0)

        with st.spinner("Analyzing image..."):
            result = run_inference(model, input_tensor, class_names, device)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Uploaded Image")
            st.image(pil_image, use_container_width=True)
        with col2:
            st.subheader("Grad-CAM Overlay")
            st.image(result["overlay"], use_container_width=True)

        st.subheader(f"Prediction: {result['pred_class']}")
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


def page_specialist_challenge(model, class_names, device, image_size):
    """Specialist Challenge: human vs AI on 30 X-rays."""
    st.header("Specialist Challenge")
    st.markdown(
        "**Can you beat the AI?** You will be shown 30 bone X-rays "
        "(10 Normal, 10 Benign, 10 Malignant) in random order. "
        "Classify each one, then see how you compare against the AI model."
    )

    challenge_images = load_challenge_images()
    if challenge_images is None:
        st.error("Test split CSV not found. Cannot load challenge images.")
        return

    # Check that images directory exists
    if not CHALLENGE_IMAGES_DIR.exists():
        st.error(
            "Challenge images not found. Expected folder: app/challenge_images/"
        )
        return

    total = len(challenge_images)

    # Initialise session state
    if "challenge_answers" not in st.session_state:
        st.session_state.challenge_answers = {}
    if "challenge_submitted" not in st.session_state:
        st.session_state.challenge_submitted = False
    if "participant_registered" not in st.session_state:
        st.session_state.participant_registered = False
    if "participant_info" not in st.session_state:
        st.session_state.participant_info = {}

    # Participant details form
    if not st.session_state.participant_registered:
        st.subheader("Participant Details")
        st.markdown(
            "Your details will be used for authorship on any resulting publication."
        )
        with st.form("participant_form"):
            name = st.text_input("Full Name *")
            unit = st.text_input("Unit / Hospital / Institution *")
            years = st.number_input(
                "Years of Experience in Orthopaedic Oncology *",
                min_value=0, max_value=60, value=None, step=1,
            )
            submitted = st.form_submit_button("Start Challenge")
            if submitted:
                if not name or not unit or years is None:
                    st.error("Please fill in all fields.")
                else:
                    st.session_state.participant_info = {
                        "name": name,
                        "unit": unit,
                        "years_experience": years,
                    }
                    st.session_state.participant_registered = True
                    st.rerun()
        return

    if st.session_state.challenge_submitted:
        _show_challenge_results(
            challenge_images, model, class_names, device, image_size
        )
        if st.button("Restart Challenge"):
            st.session_state.challenge_answers = {}
            st.session_state.challenge_submitted = False
            st.session_state.participant_registered = False
            st.session_state.participant_info = {}
            st.rerun()
        return

    # Show images in a grid, 3 per row
    for i in range(0, total, 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= total:
                break
            item = challenge_images[idx]
            img_path = CHALLENGE_IMAGES_DIR / item["image_id"]
            with col:
                st.markdown(f"**Image {idx + 1}**")
                if img_path.exists():
                    st.image(str(img_path), use_container_width=True)
                else:
                    st.warning(f"Image not found: {item['image_id']}")
                answer = st.radio(
                    f"Your diagnosis for Image {idx + 1}",
                    options=["Normal", "Benign", "Malignant"],
                    index=None,
                    key=f"challenge_{idx}",
                )
                if answer is not None:
                    st.session_state.challenge_answers[idx] = answer

    # Count answers AFTER all radio buttons have been rendered
    answered = len(st.session_state.challenge_answers)
    st.progress(answered / total, text=f"{answered} / {total} answered")

    st.divider()
    if answered < total:
        st.info(f"Please classify all {total} images before submitting.")
    if st.button("Submit Answers", disabled=(answered < total)):
        st.session_state.challenge_submitted = True
        st.rerun()


def _save_response(participant_info, detail_rows, human_correct, ai_correct, total):
    """Append this participant's response to a CSV log."""
    import datetime

    responses_path = PROJECT_ROOT / "app" / "challenge_responses.csv"
    is_new = not responses_path.exists()
    with open(responses_path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow([
                "timestamp", "name", "unit", "years_experience",
                "human_score", "ai_score", "total",
            ] + [f"image_{i+1}" for i in range(total)])
        row = [
            datetime.datetime.now().isoformat(),
            participant_info.get("name", ""),
            participant_info.get("unit", ""),
            participant_info.get("years_experience", ""),
            human_correct,
            ai_correct,
            total,
        ] + [r["Your Answer"] for r in detail_rows]
        writer.writerow(row)


def _show_challenge_results(challenge_images, model, class_names, device, image_size):
    """Display results comparing the specialist's answers to the AI."""
    answers = st.session_state.challenge_answers
    total = len(challenge_images)

    human_correct = 0
    ai_correct = 0
    detail_rows = []

    for idx, item in enumerate(challenge_images):
        true_label = item["true_label"]
        human_answer = answers.get(idx, "")

        img_path = CHALLENGE_IMAGES_DIR / item["image_id"]
        if img_path.exists():
            ai_answer = get_ai_prediction(
                model, class_names, device, image_size, img_path
            )
        else:
            ai_answer = "N/A"

        h_ok = human_answer == true_label
        a_ok = ai_answer == true_label
        if h_ok:
            human_correct += 1
        if a_ok:
            ai_correct += 1

        detail_rows.append({
            "Image": idx + 1,
            "True Label": true_label,
            "Your Answer": human_answer,
            "You": "Correct" if h_ok else "Wrong",
            "AI Answer": ai_answer,
            "AI": "Correct" if a_ok else "Wrong",
        })

    # Participant info
    info = st.session_state.get("participant_info", {})
    if info:
        st.markdown(
            f"**Participant:** {info.get('name', 'N/A')} | "
            f"**Unit:** {info.get('unit', 'N/A')} | "
            f"**Experience:** {info.get('years_experience', 'N/A')} years"
        )

    # Save response to CSV
    _save_response(info, detail_rows, human_correct, ai_correct, total)

    # Headline scores
    st.subheader("Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Your Score", f"{human_correct} / {total} ({human_correct/total:.0%})")
    col2.metric("AI Score", f"{ai_correct} / {total} ({ai_correct/total:.0%})")
    diff = human_correct - ai_correct
    if diff > 0:
        verdict = "You beat the AI!"
    elif diff < 0:
        verdict = "The AI wins this round."
    else:
        verdict = "It's a tie!"
    col3.metric("Verdict", verdict)

    # Per-class breakdown
    st.subheader("Per-Class Breakdown")
    for cls in ["Normal", "Benign", "Malignant"]:
        cls_items = [r for r in detail_rows if r["True Label"] == cls]
        h_cls = sum(1 for r in cls_items if r["You"] == "Correct")
        a_cls = sum(1 for r in cls_items if r["AI"] == "Correct")
        n_cls = len(cls_items)
        st.markdown(
            f"**{cls}:** You {h_cls}/{n_cls} -- AI {a_cls}/{n_cls}"
        )

    # Full detail table
    st.subheader("Detailed Breakdown")
    df = pd.DataFrame(detail_rows)
    st.dataframe(df, hide_index=True, use_container_width=True)


def main() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="AssureXRay",
        page_icon=":material/radiology:",
        layout="wide",
        menu_items={
            "Get help": None,
            "Report a Bug": None,
            "About": None,
        },
    )

    # Hide GitHub icon, footer, and deploy button + keep-alive ping
    st.markdown(
        """
        <style>
        .stAppDeployButton {display: none;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header [data-testid="stStatusWidget"] {display: none;}
        </style>
        <script>
        // Ping the app every 5 minutes to prevent Streamlit Cloud sleep
        setInterval(function() {
            fetch(window.location.href, {method: 'HEAD', cache: 'no-store'});
        }, 5 * 60 * 1000);
        </script>
        """,
        unsafe_allow_html=True,
    )

    st.title("AssureXRay -- Bone Tumor Classification")
    st.warning(
        "**NOT FOR CLINICAL USE -- Research Prototype Only.** "
        "This tool is for demonstration purposes only and has not been "
        "validated for clinical decision-making."
    )

    model, class_names, device, image_size = load_model(str(DEFAULT_CHECKPOINT))

    page_classifier(model, class_names, device, image_size)

    st.divider()
    page_specialist_challenge(model, class_names, device, image_size)


if __name__ == "__main__":
    main()
