#!/usr/bin/env python3
"""Profile dataset quality and generate audit report.

Reads images from ``data.raw_dir``, computes per-class statistics
(counts, dimensions, channel means), and writes a Markdown
audit report to ``paths.docs_dir``.

Usage:
    python scripts/audit.py --config configs/default.yaml

Implemented in: Phase 2
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Headless rendering before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from tqdm import tqdm

import imagehash

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.utils.reproducibility import set_seed

# ── Constants ───────────────────────────────────────────────────────
CLASS_ORDER = ["Normal", "Benign", "Malignant"]
CLASS_COLORS = {"Normal": "#2ecc71", "Benign": "#3498db", "Malignant": "#e74c3c"}

ANATOMICAL_SITE_COLS = [
    "hand",
    "ulna",
    "radius",
    "humerus",
    "foot",
    "tibia",
    "fibula",
    "femur",
    "hip bone",
    "ankle-joint",
    "knee-joint",
    "hip-joint",
    "wrist-joint",
    "elbow-joint",
    "shoulder-joint",
]

TUMOR_SUBTYPE_COLS = [
    "osteochondroma",
    "multiple osteochondromas",
    "simple bone cyst",
    "giant cell tumor",
    "osteofibroma",
    "synovial osteochondroma",
    "other bt",
    "osteosarcoma",
    "other mt",
]

BODY_REGION_COLS = ["upper limb", "lower limb", "pelvis"]
SHOOTING_ANGLE_COLS = ["frontal", "lateral", "oblique"]


# ── Helpers ─────────────────────────────────────────────────────────


def save_figure(fig: plt.Figure, name: str, figures_dir: Path) -> str:
    """Save a matplotlib figure as PNG and return the relative path."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return f"figures/{name}.png"


def glob_images(images_dir: Path) -> list[Path]:
    """Return a sorted list of JPEG files (case-insensitive)."""
    jpg = list(images_dir.glob("*.[jJ][pP][gG]"))
    jpeg = list(images_dir.glob("*.[jJ][pP][eE][gG]"))
    combined = list(set(jpg + jpeg))
    combined.sort()
    return combined


def derive_label(row: pd.Series) -> str:
    """Derive the 3-class label from binary columns."""
    if row["malignant"] == 1:
        return "Malignant"
    elif row["benign"] == 1:
        return "Benign"
    else:
        return "Normal"


def get_anatomical_site(row: pd.Series) -> str:
    """Return the anatomical site name where the binary column equals 1."""
    for col in ANATOMICAL_SITE_COLS:
        if row.get(col, 0) == 1:
            return col
    return "unknown"


# ── Audit Section Functions ─────────────────────────────────────────


def audit_class_distribution(
    df: pd.DataFrame, figures_dir: Path
) -> tuple[str | None, str]:
    """Bar chart and table of Normal/Benign/Malignant counts."""
    counts = df["label"].value_counts().reindex(CLASS_ORDER).fillna(0).astype(int)
    total = len(df)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        CLASS_ORDER,
        [counts[c] for c in CLASS_ORDER],
        color=[CLASS_COLORS[c] for c in CLASS_ORDER],
        edgecolor="white",
        linewidth=1.2,
    )
    for bar, label in zip(bars, CLASS_ORDER):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 20,
            str(int(height)),
            ha="center",
            fontweight="bold",
            fontsize=12,
        )
    ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12)
    ax.set_xlabel("Class", fontsize=12)

    fig_path = save_figure(fig, "class_distribution", figures_dir)

    md = "| Class | Count | Percentage |\n"
    md += "|-------|------:|----------:|\n"
    for cls in CLASS_ORDER:
        pct = counts[cls] / total * 100
        md += f"| {cls} | {counts[cls]:,} | {pct:.1f}% |\n"
    md += f"| **Total** | **{total:,}** | **100%** |\n"
    md += f"\nImbalance ratio (largest / smallest): {counts.max() / counts.min():.1f}x\n"

    return fig_path, md


def audit_image_dimensions(
    images_dir: Path, figures_dir: Path
) -> tuple[str | None, str]:
    """Histogram of image widths and heights."""
    image_paths = glob_images(images_dir)
    widths, heights = [], []

    for path in tqdm(image_paths, desc="Reading image dimensions", unit="img"):
        with Image.open(path) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)

    widths_arr = np.array(widths)
    heights_arr = np.array(heights)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(widths_arr, bins=50, color="#3498db", edgecolor="white", alpha=0.85)
    axes[0].set_title("Image Widths", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Pixels", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].axvline(
        widths_arr.mean(), color="#e74c3c", linestyle="--", label=f"Mean: {widths_arr.mean():.0f}"
    )
    axes[0].legend(fontsize=10)

    axes[1].hist(heights_arr, bins=50, color="#2ecc71", edgecolor="white", alpha=0.85)
    axes[1].set_title("Image Heights", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Pixels", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].axvline(
        heights_arr.mean(), color="#e74c3c", linestyle="--", label=f"Mean: {heights_arr.mean():.0f}"
    )
    axes[1].legend(fontsize=10)

    fig.suptitle("Image Dimension Distribution", fontsize=14, fontweight="bold")
    fig.tight_layout()

    fig_path = save_figure(fig, "dimension_histogram", figures_dir)

    md = "| Metric | Width (px) | Height (px) |\n"
    md += "|--------|----------:|-----------:|\n"
    md += f"| Min | {widths_arr.min()} | {heights_arr.min()} |\n"
    md += f"| Max | {widths_arr.max()} | {heights_arr.max()} |\n"
    md += f"| Mean | {widths_arr.mean():.1f} | {heights_arr.mean():.1f} |\n"
    md += f"| Median | {int(np.median(widths_arr))} | {int(np.median(heights_arr))} |\n"
    md += f"| Std Dev | {widths_arr.std():.1f} | {heights_arr.std():.1f} |\n"
    md += f"\nTotal images measured: {len(widths_arr):,}\n"

    return fig_path, md


def audit_per_center(
    df: pd.DataFrame, figures_dir: Path
) -> tuple[str | None, str]:
    """Stacked bar chart showing class distribution per center."""
    cross = pd.crosstab(df["center"], df["label"])
    cross = cross.reindex(columns=CLASS_ORDER, fill_value=0)
    cross = cross.sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    cross.plot.bar(
        ax=ax,
        stacked=True,
        color=[CLASS_COLORS[c] for c in CLASS_ORDER],
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_title("Class Distribution by Center", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12)
    ax.set_xlabel("Center", fontsize=12)
    ax.legend(title="Class", fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()

    fig_path = save_figure(fig, "center_breakdown", figures_dir)

    total = len(df)
    md = "| Center | Normal | Benign | Malignant | Total | % of Dataset |\n"
    md += "|-------:|-------:|-------:|----------:|------:|-----------:|\n"
    for center in cross.index:
        row_total = cross.loc[center].sum()
        pct = row_total / total * 100
        md += (
            f"| {center} "
            f"| {cross.loc[center, 'Normal']:,} "
            f"| {cross.loc[center, 'Benign']:,} "
            f"| {cross.loc[center, 'Malignant']:,} "
            f"| {row_total:,} "
            f"| {pct:.1f}% |\n"
        )
    md += (
        f"| **Total** "
        f"| **{cross['Normal'].sum():,}** "
        f"| **{cross['Benign'].sum():,}** "
        f"| **{cross['Malignant'].sum():,}** "
        f"| **{total:,}** "
        f"| **100%** |\n"
    )

    return fig_path, md


def audit_missing_values(
    df: pd.DataFrame, figures_dir: Path
) -> tuple[str | None, str]:
    """Horizontal bar chart and table of missing values per column."""
    missing = df.isna().sum()
    total = len(df)
    missing_any = missing[missing > 0].sort_values(ascending=True)

    fig_path = None
    if len(missing_any) > 0:
        fig, ax = plt.subplots(figsize=(10, max(4, len(missing_any) * 0.5)))
        missing_any.plot.barh(ax=ax, color="#e67e22", edgecolor="white")
        ax.set_title("Missing Values by Column", fontsize=14, fontweight="bold")
        ax.set_xlabel("Count", fontsize=12)
        for i, (col, val) in enumerate(missing_any.items()):
            ax.text(val + 1, i, str(val), va="center", fontsize=10)
        fig.tight_layout()
        fig_path = save_figure(fig, "missing_values", figures_dir)

    md = "| Column | Missing Count | Percentage |\n"
    md += "|--------|-------------:|----------:|\n"
    if len(missing_any) > 0:
        for col in missing_any.index[::-1]:  # descending order
            pct = missing[col] / total * 100
            md += f"| `{col}` | {missing[col]} | {pct:.2f}% |\n"
    else:
        md += "| *(none)* | 0 | 0.00% |\n"

    total_missing = missing.sum()
    total_cells = total * len(df.columns)
    md += (
        f"\n**Summary:** {total_missing:,} missing values out of "
        f"{total_cells:,} total cells ({total_missing / total_cells * 100:.2f}%).\n"
    )

    if total_missing == 0:
        md += "\nThe dataset has **no missing values** across all 37 columns.\n"

    return fig_path, md


def audit_annotation_coverage(
    df: pd.DataFrame,
    images_dir: Path,
    annot_dir: Path,
    figures_dir: Path,
) -> tuple[str | None, str]:
    """Bar chart and summary of annotation coverage for tumor images."""
    # Get tumor image IDs from CSV
    tumor_df = df[df["tumor"] == 1]
    tumor_ids_raw = set(tumor_df["image_id"].astype(str).values)

    # Normalize: strip extensions if present
    tumor_ids = set()
    for tid in tumor_ids_raw:
        stem = Path(tid).stem if "." in tid else tid
        tumor_ids.add(stem)

    # Get annotation file stems
    annot_files = set(p.stem for p in annot_dir.glob("*.json"))

    covered = tumor_ids & annot_files
    missing_annotations = tumor_ids - annot_files
    extra_annotations = annot_files - tumor_ids

    normal_count = len(df[df["tumor"] == 0])

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 5))
    categories = ["With Annotation", "Without Annotation"]
    values = [len(covered), len(missing_annotations)]
    colors = ["#2ecc71", "#e74c3c"]
    bars = ax.bar(categories, values, color=colors, edgecolor="white", linewidth=1.2)
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 5,
            str(int(height)),
            ha="center",
            fontweight="bold",
            fontsize=12,
        )
    ax.set_title("Annotation Coverage (Tumor Images Only)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12)
    fig.tight_layout()

    fig_path = save_figure(fig, "annotation_coverage", figures_dir)

    total_tumor = len(tumor_ids)
    coverage_pct = len(covered) / total_tumor * 100 if total_tumor > 0 else 0.0

    md = "| Metric | Count |\n"
    md += "|--------|------:|\n"
    md += f"| Tumor images in dataset.csv (`tumor=1`) | {total_tumor:,} |\n"
    md += f"| JSON annotation files found | {len(annot_files):,} |\n"
    md += f"| Tumor images with annotations | {len(covered):,} ({coverage_pct:.1f}%) |\n"
    md += f"| Tumor images missing annotations | {len(missing_annotations):,} |\n"
    md += f"| Extra annotation files (no matching tumor row) | {len(extra_annotations):,} |\n"
    md += f"| Normal images (`tumor=0`, no annotations expected) | {normal_count:,} |\n"

    if len(missing_annotations) > 0 and len(missing_annotations) <= 20:
        md += "\n**Missing annotation IDs:**\n"
        for mid in sorted(missing_annotations):
            md += f"- `{mid}`\n"
    elif len(missing_annotations) > 20:
        md += f"\n**Missing annotation IDs:** {len(missing_annotations)} images (too many to list).\n"

    if len(extra_annotations) > 0 and len(extra_annotations) <= 20:
        md += "\n**Extra annotation files (no matching tumor image in CSV):**\n"
        for eid in sorted(extra_annotations):
            md += f"- `{eid}`\n"
    elif len(extra_annotations) > 20:
        md += f"\n**Extra annotation files:** {len(extra_annotations)} files (too many to list).\n"

    return fig_path, md


def audit_duplicate_detection(
    images_dir: Path, figures_dir: Path
) -> tuple[str | None, str]:
    """Detect exact and near-duplicate images using perceptual hashing."""
    image_paths = glob_images(images_dir)

    # Compute perceptual hashes
    hashes: dict[str, imagehash.ImageHash] = {}
    for path in tqdm(image_paths, desc="Computing perceptual hashes", unit="img"):
        with Image.open(path) as img:
            h = imagehash.phash(img, hash_size=8)
            hashes[path.name] = h

    # Find exact duplicates (distance == 0)
    hash_groups: dict[str, list[str]] = defaultdict(list)
    for name, h in hashes.items():
        hash_groups[str(h)].append(name)

    exact_groups = {k: v for k, v in hash_groups.items() if len(v) > 1}
    exact_pair_count = sum(
        len(v) * (len(v) - 1) // 2 for v in exact_groups.values()
    )

    # Find near-duplicates (distance 1-5) via pairwise comparison
    names = list(hashes.keys())
    near_duplicates: list[tuple[str, str, int]] = []
    distances: list[int] = []

    n = len(names)
    for i in range(n):
        for j in range(i + 1, n):
            dist = hashes[names[i]] - hashes[names[j]]
            if dist <= 5 and dist > 0:
                near_duplicates.append((names[i], names[j], dist))
            if dist <= 20:
                distances.append(dist)

    # Create figure: histogram of pairwise distances (for close pairs only)
    fig, ax = plt.subplots(figsize=(8, 5))
    if distances:
        bins = list(range(0, 22))
        ax.hist(distances, bins=bins, color="#9b59b6", edgecolor="white", alpha=0.85)
        ax.set_title("Pairwise Hash Distance Distribution (d <= 20)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Hamming Distance", fontsize=12)
        ax.set_ylabel("Number of Pairs", fontsize=12)
        ax.axvline(5.5, color="#e74c3c", linestyle="--", linewidth=1.5, label="Near-duplicate threshold (d=5)")
        ax.legend(fontsize=10)
    else:
        ax.text(
            0.5, 0.5, "No close pairs found (d <= 20)",
            ha="center", va="center", fontsize=14, transform=ax.transAxes,
        )
        ax.set_title("Pairwise Hash Distance Distribution", fontsize=14, fontweight="bold")
    fig.tight_layout()

    fig_path = save_figure(fig, "duplicate_detection", figures_dir)

    md = "| Metric | Count |\n"
    md += "|--------|------:|\n"
    md += f"| Total images hashed | {len(hashes):,} |\n"
    md += f"| Exact duplicate groups (distance = 0) | {len(exact_groups):,} |\n"
    md += f"| Exact duplicate pairs | {exact_pair_count:,} |\n"
    md += f"| Near-duplicate pairs (distance 1-5) | {len(near_duplicates):,} |\n"

    if exact_groups:
        md += "\n### Exact Duplicate Groups\n\n"
        shown = 0
        for hash_val, group in sorted(exact_groups.items(), key=lambda x: -len(x[1])):
            if shown >= 20:
                md += f"\n*... and {len(exact_groups) - 20} more groups*\n"
                break
            md += f"- **Hash `{hash_val}`** ({len(group)} images): "
            md += ", ".join(f"`{name}`" for name in sorted(group)[:10])
            if len(group) > 10:
                md += f" ... ({len(group) - 10} more)"
            md += "\n"
            shown += 1

    if near_duplicates:
        md += "\n### Near-Duplicate Pairs (Top 20)\n\n"
        md += "| Image A | Image B | Distance |\n"
        md += "|---------|---------|--------:|\n"
        for a, b, d in sorted(near_duplicates, key=lambda x: x[2])[:20]:
            md += f"| `{a}` | `{b}` | {d} |\n"
        if len(near_duplicates) > 20:
            md += f"\n*... and {len(near_duplicates) - 20} more near-duplicate pairs*\n"

    return fig_path, md


def audit_leakage_risk(df: pd.DataFrame) -> str:
    """Text-only analysis of data leakage risks from multi-angle images."""
    # Shooting angle distribution
    angle_counts = {}
    for col in SHOOTING_ANGLE_COLS:
        if col in df.columns:
            angle_counts[col] = int(df[col].sum())

    # Derive anatomical site per row for proxy grouping analysis
    df_copy = df.copy()
    df_copy["_site"] = df_copy.apply(get_anatomical_site, axis=1)

    # Proxy grouping uniqueness analysis
    proxy_1 = df_copy.groupby(["center", "age", "gender"]).ngroups
    proxy_2 = df_copy.groupby(["center", "age", "gender", "_site"]).ngroups

    total = len(df)

    md = "### Same-Lesion Multi-Angle Images\n\n"
    md += (
        "The BTXRD dataset includes radiographs taken from multiple shooting angles "
        "(frontal, lateral, oblique) for the same patient visit. Since **no `patient_id` "
        "column exists** in `dataset.csv`, there is no way to definitively group images "
        "by patient.\n\n"
    )

    md += "**Risk:** If images of the same lesion from different angles are split across "
    md += "train and test sets, the model may learn patient-specific features (bone shape, "
    md += "implant presence, unique anatomy) rather than tumor characteristics, inflating "
    md += "test performance.\n\n"

    md += "### Shooting Angle Distribution\n\n"
    md += "| Angle | Count | Percentage |\n"
    md += "|-------|------:|----------:|\n"
    for angle, count in sorted(angle_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        md += f"| {angle.title()} | {count:,} | {pct:.1f}% |\n"
    multi_angle_sum = sum(angle_counts.values())
    md += f"| **Total** | **{multi_angle_sum:,}** | - |\n"
    if multi_angle_sum > total:
        md += (
            f"\n*Note: Total exceeds image count ({total:,}), indicating "
            f"some images have multiple angle flags set.*\n"
        )
    md += "\n"

    md += "### Available Proxy Grouping Columns\n\n"
    md += (
        "Since no `patient_id` exists, the following column combinations can serve as "
        "proxy patient groups. However, each combination has limitations:\n\n"
    )
    md += "| Column Combination | Unique Groups | Avg Images/Group | Limitation |\n"
    md += "|--------------------|--------------:|-----------------:|------------|\n"
    md += (
        f"| `center` + `age` + `gender` | {proxy_1:,} | "
        f"{total / proxy_1:.1f} | Many patients share demographics |\n"
    )
    md += (
        f"| `center` + `age` + `gender` + anatomical site | {proxy_2:,} | "
        f"{total / proxy_2:.1f} | Better but still not unique per patient |\n"
    )
    md += "\n"

    md += "### Mitigation Recommendation\n\n"
    md += (
        "**Do NOT fabricate patient groupings** from proxy columns. Acknowledge this "
        "limitation honestly. The **center-holdout split** (Phase 3) partially mitigates "
        "the leakage risk by testing on entirely different data sources (e.g., train on "
        "Center 1, test on Centers 2+3). This ensures no same-patient images leak across "
        "the train/test boundary for the holdout evaluation.\n\n"
    )
    md += (
        "For the **stratified split**, the leakage risk is inherent and should be reported "
        "alongside results as a known limitation.\n"
    )

    return md


# ── Report Assembly ─────────────────────────────────────────────────


def write_audit_report(
    sections: list[tuple[str, str | None, str]],
    output_path: Path,
) -> None:
    """Assemble sections into a complete markdown audit report.

    Parameters
    ----------
    sections : list of (title, figure_relative_path_or_None, markdown_text)
    output_path : Path to write the report
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(output_path, "w") as f:
        f.write("# BTXRD Data Audit Report\n\n")
        f.write(f"*Auto-generated by `scripts/audit.py` on {timestamp}*\n\n")

        # Table of contents
        f.write("## Table of Contents\n\n")
        for i, (title, _, _) in enumerate(sections, 1):
            anchor = title.lower().replace(" ", "-").replace("(", "").replace(")", "")
            f.write(f"{i}. [{title}](#{anchor})\n")
        f.write("\n---\n\n")

        # Sections
        for title, fig_path, text in sections:
            f.write(f"## {title}\n\n")
            if fig_path is not None:
                f.write(f"![{title}]({fig_path})\n\n")
            f.write(text)
            f.write("\n\n---\n\n")


# ── Main ────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile dataset quality and generate audit report."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Config overrides in key.subkey=value format",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    set_seed(cfg.get("seed", 42))

    logger = setup_logging(name="assurexray.audit", config=cfg)
    sns.set_theme(style="whitegrid", palette="muted")

    # ── Resolve paths ───────────────────────────────────────────────
    raw_dir = PROJECT_ROOT / cfg["data"]["raw_dir"]
    images_dir = raw_dir / "images"
    csv_path = raw_dir / "dataset.csv"
    docs_dir = PROJECT_ROOT / cfg["paths"]["docs_dir"]
    figures_dir = docs_dir / "figures"

    # Try both capitalizations for annotations
    annot_dir = raw_dir / "Annotations"
    if not annot_dir.is_dir():
        annot_dir = raw_dir / "annotations"

    # ── Validate data exists ────────────────────────────────────────
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}\n"
            "Run `make download` first to fetch the BTXRD dataset."
        )
    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Images directory not found: {images_dir}\n"
            "Run `make download` first to fetch the BTXRD dataset."
        )
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset CSV not found: {csv_path}\n"
            "Run `make download` first to fetch the BTXRD dataset."
        )

    # ── Load CSV ────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    logger.info(f"CSV columns ({len(df.columns)}): {df.columns.tolist()}")
    logger.info(f"CSV shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # Add derived label column
    df["label"] = df.apply(derive_label, axis=1)
    label_counts = df["label"].value_counts()
    logger.info(f"Label distribution: {dict(label_counts)}")

    # ── Run audit sections ──────────────────────────────────────────
    sections: list[tuple[str, str | None, str]] = []

    logger.info("Auditing class distribution...")
    fig_path, md = audit_class_distribution(df, figures_dir)
    sections.append(("Class Distribution", fig_path, md))

    logger.info("Auditing image dimensions...")
    fig_path, md = audit_image_dimensions(images_dir, figures_dir)
    sections.append(("Image Dimension Distribution", fig_path, md))

    logger.info("Auditing per-center breakdown...")
    fig_path, md = audit_per_center(df, figures_dir)
    sections.append(("Per-Center Class Breakdown", fig_path, md))

    logger.info("Auditing missing values...")
    fig_path, md = audit_missing_values(df, figures_dir)
    sections.append(("Missing Values", fig_path, md))

    if annot_dir.is_dir():
        logger.info("Auditing annotation coverage...")
        fig_path, md = audit_annotation_coverage(df, images_dir, annot_dir, figures_dir)
        sections.append(("Annotation Coverage", fig_path, md))
    else:
        logger.warning(
            f"Annotations directory not found ({annot_dir}). "
            "Skipping annotation coverage section."
        )

    logger.info("Auditing duplicate detection (this may take a few minutes)...")
    fig_path, md = audit_duplicate_detection(images_dir, figures_dir)
    sections.append(("Duplicate Detection", fig_path, md))

    logger.info("Auditing leakage risk...")
    md = audit_leakage_risk(df)
    sections.append(("Data Leakage Risk Assessment", None, md))

    # ── Write report ────────────────────────────────────────────────
    report_path = docs_dir / "data_audit_report.md"
    write_audit_report(sections, report_path)

    n_figures = sum(1 for _, fp, _ in sections if fp is not None)
    logger.info(f"Audit report written to {report_path} with {n_figures} figures")
    logger.info(f"Figures saved to {figures_dir}/")


if __name__ == "__main__":
    main()
