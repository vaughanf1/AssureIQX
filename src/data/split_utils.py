"""Stratified and center-holdout split strategies for BTXRD dataset.

Provides pure functions for:
- Label derivation from binary indicator columns
- Perceptual hash-based duplicate detection
- Stratified 70/15/15 splitting with duplicate-aware grouping
- Center-holdout splitting (Center 1 train/val, Centers 2+3 test)
- CSV manifest export

All split functions ensure duplicate image pairs land on the same
side of every split boundary to prevent data leakage.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def derive_label(row: pd.Series) -> str:
    """Derive a 3-class label from binary indicator columns.

    Logic:
        malignant == 1  ->  "Malignant"
        benign == 1     ->  "Benign"
        otherwise       ->  "Normal"

    Parameters
    ----------
    row : pd.Series
        A row from dataset.csv containing 'malignant' and 'benign' columns.

    Returns
    -------
    str
        One of "Malignant", "Benign", or "Normal".
    """
    if row["malignant"] == 1:
        return "Malignant"
    elif row["benign"] == 1:
        return "Benign"
    else:
        return "Normal"


def compute_duplicate_groups(
    images_dir: Path, image_ids: list[str]
) -> dict[str, int]:
    """Compute perceptual hash groups for duplicate detection.

    Uses imagehash.phash with hash_size=8 (64-bit hash). Images with
    identical hashes (hamming distance = 0) are assigned the same
    group ID.

    Parameters
    ----------
    images_dir : Path
        Directory containing the image files.
    image_ids : list[str]
        List of image filenames (e.g. ``["IMG000001.jpeg", ...]``).

    Returns
    -------
    dict[str, int]
        Mapping of image_id -> group_id. Every image gets a group_id;
        duplicates share the same group_id.
    """
    logger.info("Computing perceptual hashes for %d images...", len(image_ids))

    # Compute hashes
    hash_map: dict[str, str] = {}
    total = len(image_ids)
    for i, img_id in enumerate(image_ids):
        if (i + 1) % 500 == 0 or (i + 1) == total:
            logger.info("  Hashing progress: %d/%d", i + 1, total)
        img_path = images_dir / img_id
        img = Image.open(img_path)
        h = imagehash.phash(img, hash_size=8)
        hash_map[img_id] = str(h)

    # Group by identical hash
    hash_to_ids: dict[str, list[str]] = defaultdict(list)
    for img_id, h in hash_map.items():
        hash_to_ids[h].append(img_id)

    # Assign group IDs
    id_to_group: dict[str, int] = {}
    group_id = 0
    dup_count = 0
    for h, members in hash_to_ids.items():
        for m in members:
            id_to_group[m] = group_id
        if len(members) > 1:
            dup_count += 1
        group_id += 1

    logger.info(
        "Found %d duplicate groups (groups with >1 member) out of %d total groups",
        dup_count,
        group_id,
    )

    return id_to_group


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    label_col: str = "label",
    group_col: str = "dup_group",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split with duplicate-aware grouping.

    Ensures that:
    - Class proportions are preserved across splits
    - All members of a duplicate group land in the same split
    - Split ratios are approximately train_ratio/val_ratio/test_ratio

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with columns ``label_col`` and ``group_col``.
    train_ratio : float
        Fraction for training set (default 0.70).
    val_ratio : float
        Fraction for validation set (default 0.15).
    test_ratio : float
        Fraction for test set (default 0.15).
    seed : int
        Random state for reproducibility.
    label_col : str
        Column name for class labels.
    group_col : str
        Column name for duplicate group IDs.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (df_train, df_val, df_test) each with a "split" column added.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    )

    df = df.copy()

    # Identify duplicate groups (groups with more than 1 member)
    group_sizes = df[group_col].value_counts()
    dup_groups = set(group_sizes[group_sizes > 1].index)

    # For splitting: use representatives (first member of each dup group)
    # Non-duplicate images are their own representative
    if dup_groups:
        # Mark first occurrence per group as representative
        df["_is_rep"] = ~df.duplicated(subset=[group_col], keep="first")
        # For non-dup groups, all are representatives
        df.loc[~df[group_col].isin(dup_groups), "_is_rep"] = True
        df_reps = df[df["_is_rep"]].copy()
    else:
        df["_is_rep"] = True
        df_reps = df.copy()

    # Two-step split on representatives
    # Step 1: 70% train, 30% remainder
    remainder_ratio = val_ratio + test_ratio
    df_train_reps, df_remainder_reps = train_test_split(
        df_reps,
        test_size=remainder_ratio,
        stratify=df_reps[label_col],
        random_state=seed,
    )

    # Step 2: Split remainder 50/50 into val and test
    val_fraction_of_remainder = val_ratio / remainder_ratio
    df_val_reps, df_test_reps = train_test_split(
        df_remainder_reps,
        test_size=1.0 - val_fraction_of_remainder,
        stratify=df_remainder_reps[label_col],
        random_state=seed,
    )

    # Assign splits to all images (including duplicate partners)
    train_groups = set(df_train_reps[group_col])
    val_groups = set(df_val_reps[group_col])
    test_groups = set(df_test_reps[group_col])

    df.loc[df[group_col].isin(train_groups), "split"] = "train"
    df.loc[df[group_col].isin(val_groups), "split"] = "val"
    df.loc[df[group_col].isin(test_groups), "split"] = "test"

    # Clean up temporary column
    df.drop(columns=["_is_rep"], inplace=True)

    df_train = df[df["split"] == "train"].copy()
    df_val = df[df["split"] == "val"].copy()
    df_test = df[df["split"] == "test"].copy()

    return df_train, df_val, df_test


def center_holdout_split(
    df: pd.DataFrame,
    train_centers: list[int] | None = None,
    test_centers: list[int] | None = None,
    val_ratio: float = 0.15,
    seed: int = 42,
    label_col: str = "label",
    group_col: str = "dup_group",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Center-holdout split: train/val from one center, test from others.

    Center 1 images are split into train (85%) and val (15%).
    Centers 2+3 images form the test set.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with columns ``label_col``, ``group_col``, and ``"center"``.
    train_centers : list[int] | None
        Centers for train+val (default [1]).
    test_centers : list[int] | None
        Centers for test (default [2, 3]).
    val_ratio : float
        Fraction of train-center data reserved for validation (default 0.15).
    seed : int
        Random state for reproducibility.
    label_col : str
        Column name for class labels.
    group_col : str
        Column name for duplicate group IDs.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (df_train, df_val, df_test) each with a "split" column added.
    """
    if train_centers is None:
        train_centers = [1]
    if test_centers is None:
        test_centers = [2, 3]

    df = df.copy()

    # Separate by center
    df_trainval = df[df["center"].isin(train_centers)].copy()
    df_test = df[df["center"].isin(test_centers)].copy()

    # Handle duplicate groups: keep representatives for splitting trainval
    group_sizes = df_trainval[group_col].value_counts()
    dup_groups = set(group_sizes[group_sizes > 1].index)

    if dup_groups:
        df_trainval["_is_rep"] = ~df_trainval.duplicated(
            subset=[group_col], keep="first"
        )
        df_trainval.loc[
            ~df_trainval[group_col].isin(dup_groups), "_is_rep"
        ] = True
        df_reps = df_trainval[df_trainval["_is_rep"]].copy()
    else:
        df_trainval["_is_rep"] = True
        df_reps = df_trainval.copy()

    # Split Center 1 representatives into train/val
    df_train_reps, df_val_reps = train_test_split(
        df_reps,
        test_size=val_ratio,
        stratify=df_reps[label_col],
        random_state=seed,
    )

    # Assign splits to all trainval images (including duplicate partners)
    train_groups = set(df_train_reps[group_col])
    val_groups = set(df_val_reps[group_col])

    df_trainval.loc[df_trainval[group_col].isin(train_groups), "split"] = "train"
    df_trainval.loc[df_trainval[group_col].isin(val_groups), "split"] = "val"

    if "_is_rep" in df_trainval.columns:
        df_trainval.drop(columns=["_is_rep"], inplace=True)

    # Test set is all images from test centers
    df_test["split"] = "test"

    df_train = df_trainval[df_trainval["split"] == "train"].copy()
    df_val = df_trainval[df_trainval["split"] == "val"].copy()

    return df_train, df_val, df_test


def save_split_csv(
    df: pd.DataFrame,
    split_name: str,
    strategy: str,
    output_dir: Path,
) -> Path:
    """Save a split DataFrame as a CSV manifest.

    Writes columns [image_id, split, label] to
    ``output_dir/{strategy}_{split_name}.csv``.

    Parameters
    ----------
    df : pd.DataFrame
        Split DataFrame with columns ``image_id``, ``split``, ``label``.
    split_name : str
        Name of the split (e.g. "train", "val", "test").
    strategy : str
        Strategy name (e.g. "stratified", "center").
    output_dir : Path
        Directory to write the CSV file to.

    Returns
    -------
    Path
        Path to the written CSV file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"{strategy}_{split_name}.csv"
    df[["image_id", "split", "label"]].to_csv(out_path, index=False)

    logger.info("Saved %d rows to %s", len(df), out_path)
    return out_path
