#!/usr/bin/env python3
"""Generate train/val/test split manifests.

Creates stratified and (optionally) center-based split CSV manifests
from the raw dataset metadata, writing them to ``data.splits_dir``.

Usage:
    python scripts/split.py --config configs/default.yaml

Implemented in: Phase 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.split_utils import (
    center_holdout_split,
    compute_duplicate_groups,
    derive_label,
    random_split,
    save_split_csv,
    stratified_split,
)
from src.utils.config import load_config
from src.utils.reproducibility import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _print_class_distribution(df: pd.DataFrame, split_name: str, label_col: str = "label") -> None:
    """Print class distribution for a split."""
    counts = df[label_col].value_counts().sort_index()
    total = len(df)
    print(f"  {split_name:>8s}: ", end="")
    parts = []
    for cls in ["Benign", "Malignant", "Normal"]:
        c = counts.get(cls, 0)
        pct = 100.0 * c / total if total > 0 else 0.0
        parts.append(f"{cls}={c} ({pct:.1f}%)")
    print(", ".join(parts) + f"  [total={total}]")


def _validate_splits(
    splits: dict[str, pd.DataFrame],
    strategy: str,
    expected_total: int,
    df_full: pd.DataFrame,
    group_col: str = "dup_group",
) -> None:
    """Run validation checks on a set of splits."""
    print(f"\n--- Validation: {strategy} ---")

    # Check 1: No overlapping image_ids across splits
    split_names = list(splits.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            overlap = set(splits[s1]["image_id"]) & set(splits[s2]["image_id"])
            assert len(overlap) == 0, (
                f"FAIL: {overlap} image_ids overlap between {s1} and {s2}"
            )
    print("  [PASS] No image_id overlaps between splits")

    # Check 2: All images accounted for
    total = sum(len(s) for s in splits.values())
    assert total == expected_total, (
        f"FAIL: Expected {expected_total} images, got {total}"
    )
    print(f"  [PASS] All {expected_total} images accounted for")

    # Check 3: Malignant count > 0 in every split
    for name, sdf in splits.items():
        mal_count = (sdf["label"] == "Malignant").sum()
        assert mal_count > 0, (
            f"FAIL: Malignant count is 0 in {strategy}/{name}"
        )
        print(f"  [PASS] {name} has {mal_count} Malignant images")

    # Check 4: Duplicate pairs are in the same split
    # Get groups that have more than 1 member across the full dataset
    group_sizes = df_full[group_col].value_counts()
    dup_groups = group_sizes[group_sizes > 1].index.tolist()

    if dup_groups:
        all_splits = pd.concat(splits.values())
        for gid in dup_groups:
            members = all_splits[all_splits[group_col] == gid]
            unique_splits = members["split"].unique()
            assert len(unique_splits) == 1, (
                f"FAIL: Duplicate group {gid} spans splits: {unique_splits}"
            )
        print(f"  [PASS] All {len(dup_groups)} duplicate groups stay in same split")
    else:
        print("  [INFO] No duplicate groups to check")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate train/val/test split manifests."
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

    # ── Load dataset.csv ────────────────────────────────────
    raw_dir = Path(PROJECT_ROOT / cfg["data"]["raw_dir"])
    csv_path = raw_dir / "dataset.csv"
    logger.info("Loading dataset from %s", csv_path)

    df = pd.read_csv(csv_path)

    # Derive 3-class labels
    df["label"] = df.apply(derive_label, axis=1)

    expected_count = cfg["data"]["expected_image_count"]
    assert len(df) == expected_count, (
        f"Expected {expected_count} images, got {len(df)}"
    )

    print("\n=== DATASET OVERVIEW ===")
    print(f"Total images: {len(df)}")
    print("\nClass distribution:")
    for cls in ["Normal", "Benign", "Malignant"]:
        c = (df["label"] == cls).sum()
        pct = 100.0 * c / len(df)
        print(f"  {cls:>10s}: {c:5d} ({pct:.1f}%)")

    print("\nCenter distribution:")
    for center in sorted(df["center"].unique()):
        c = (df["center"] == center).sum()
        print(f"  Center {center}: {c} images")

    # ── Compute duplicate groups ────────────────────────────
    images_dir = raw_dir / "images"
    image_ids = df["image_id"].tolist()
    id_to_group = compute_duplicate_groups(images_dir, image_ids)
    df["dup_group"] = df["image_id"].map(id_to_group)

    # Count duplicate groups
    group_sizes = df["dup_group"].value_counts()
    dup_groups_count = (group_sizes > 1).sum()

    print(f"\n=== DUPLICATE DETECTION ===")
    print(f"Duplicate groups found: {dup_groups_count}")
    print(f"WARNING: {dup_groups_count} exact duplicate image pairs detected.")
    print("Duplicate groups are forced to the same split side to prevent data leakage.")
    print("NOTE: No patient_id column exists in dataset. Same-lesion multi-angle images")
    print("cannot be reliably grouped. This is a known leakage risk documented in the")
    print("audit report (docs/data_audit_report.md). Proxy grouping (center+age+gender)")
    print("produces 295 groups which is too coarse for reliable patient-level splitting.")

    # ── Create output directory ─────────────────────────────
    splits_dir = Path(PROJECT_ROOT / cfg["data"]["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    # ── Stratified split ────────────────────────────────────
    seed = cfg.get("seed", 42)
    train_ratio = cfg["data"]["train_split"]
    val_ratio = cfg["data"]["val_split"]
    test_ratio = cfg["data"]["test_split"]

    print("\n=== STRATIFIED SPLIT (70/15/15) ===")
    df_s_train, df_s_val, df_s_test = stratified_split(
        df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    save_split_csv(df_s_train, "train", "stratified", splits_dir)
    save_split_csv(df_s_val, "val", "stratified", splits_dir)
    save_split_csv(df_s_test, "test", "stratified", splits_dir)

    print("\nStratified split class distribution:")
    _print_class_distribution(df_s_train, "train")
    _print_class_distribution(df_s_val, "val")
    _print_class_distribution(df_s_test, "test")

    # Validate stratified
    _validate_splits(
        {"train": df_s_train, "val": df_s_val, "test": df_s_test},
        "stratified",
        expected_count,
        df,
    )

    # ── Center-holdout split ────────────────────────────────
    print("=== CENTER-HOLDOUT SPLIT ===")
    print("Train/Val: Center 1 | Test: Centers 2+3")

    df_c_train, df_c_val, df_c_test = center_holdout_split(
        df,
        train_centers=[1],
        test_centers=[2, 3],
        val_ratio=val_ratio,
        seed=seed,
    )

    save_split_csv(df_c_train, "train", "center", splits_dir)
    save_split_csv(df_c_val, "val", "center", splits_dir)
    save_split_csv(df_c_test, "test", "center", splits_dir)

    print("\nCenter-holdout split class distribution:")
    _print_class_distribution(df_c_train, "train")
    _print_class_distribution(df_c_val, "val")
    _print_class_distribution(df_c_test, "test")

    print("\nCenter breakdown per split:")
    combined = pd.concat([df_c_train, df_c_val, df_c_test])
    ct = pd.crosstab(combined["split"], combined["center"])
    print(ct.to_string())

    # Validate center-holdout
    _validate_splits(
        {"train": df_c_train, "val": df_c_val, "test": df_c_test},
        "center",
        expected_count,
        df,
    )

    # ── Random split (paper replication) ───────────────────
    print("=== RANDOM SPLIT (80/20 — paper replication) ===")
    print("WARNING: No stratification or duplicate grouping (matches paper protocol)")

    df_r_train, df_r_val, df_r_test = random_split(
        df,
        val_ratio=0.20,
        seed=seed,
    )

    save_split_csv(df_r_train, "train", "random", splits_dir)
    save_split_csv(df_r_val, "val", "random", splits_dir)
    save_split_csv(df_r_test, "test", "random", splits_dir)

    print("\nRandom split class distribution:")
    _print_class_distribution(df_r_train, "train")
    _print_class_distribution(df_r_val, "val")

    # Simplified validation: overlaps + totals only (no dup group check)
    print(f"\n--- Validation: random ---")
    overlap = set(df_r_train["image_id"]) & set(df_r_val["image_id"])
    assert len(overlap) == 0, f"FAIL: {len(overlap)} image_ids overlap between train and val"
    print("  [PASS] No image_id overlaps between train and val")
    total = len(df_r_train) + len(df_r_val)
    assert total == expected_count, f"FAIL: Expected {expected_count} images, got {total}"
    print(f"  [PASS] All {expected_count} images accounted for")
    print("  [SKIP] Duplicate group check (deliberately skipped for paper replication)")
    print()

    # ── Summary ─────────────────────────────────────────────
    print("=== SUMMARY ===")
    print(f"Total images: {expected_count}")
    print(f"Strategies: stratified, center-holdout, random")
    print(f"Output directory: {splits_dir}")
    print(f"Files generated: 9 CSV manifests")
    print()
    print("Stratified splits:")
    print(f"  train: {len(df_s_train)}, val: {len(df_s_val)}, test: {len(df_s_test)}")
    print("Center-holdout splits:")
    print(f"  train: {len(df_c_train)}, val: {len(df_c_val)}, test: {len(df_c_test)}")
    print("Random splits (paper replication):")
    print(f"  train: {len(df_r_train)}, val: {len(df_r_val)} (test=copy of val)")

    # ── Leakage risk documentation ──────────────────────────
    print()
    print("=== LEAKAGE RISK DOCUMENTATION ===")
    print("- No patient_id available: cannot guarantee train/test images are from different patients")
    print(f"- {dup_groups_count} exact duplicate pairs: grouped to same split (MITIGATED)")
    print("- Same-lesion multi-angle images: NOT mitigated (no reliable grouping metadata)")
    print("- Proxy grouping (center+age+gender) too coarse: 295 groups for 3746 images")
    print("- Recommendation: treat evaluation metrics as optimistic upper bounds")

    logger.info("Split generation complete.")


if __name__ == "__main__":
    main()
