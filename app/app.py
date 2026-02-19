#!/usr/bin/env python3
"""AssureXRay Streamlit Demo Application.

Provides an interactive web interface for uploading bone X-ray images
and viewing classification predictions with Grad-CAM overlays.

Usage:
    streamlit run app/app.py

Implemented in: Phase 7
"""

from __future__ import annotations


def main() -> None:
    """Launch the Streamlit demo (placeholder)."""
    try:
        import streamlit as st
    except ImportError:
        raise ImportError(
            "Streamlit is required for the demo app. "
            "Install it with: pip install streamlit"
        )

    st.set_page_config(page_title="AssureXRay", page_icon="🦴", layout="wide")
    st.title("AssureXRay — Bone Tumor Classification")
    st.info(
        "This is a placeholder. The full demo will be implemented in Phase 7."
    )
    st.markdown(
        """
        **Planned features:**
        - Upload a bone X-ray image
        - View 3-class prediction (Normal / Benign / Malignant)
        - Inspect Grad-CAM heatmap overlay
        - Compare predictions across models
        """
    )


if __name__ == "__main__":
    main()
