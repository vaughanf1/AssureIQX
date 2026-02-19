#!/usr/bin/env python3
"""Download BTXRD dataset from figshare.

Downloads the Bone Tumor X-Ray Dataset archive, extracts it to
``data.raw_dir``, and verifies file counts and integrity.

Usage:
    python scripts/download.py --config configs/default.yaml
    python scripts/download.py --config configs/default.yaml --override data.raw_dir=data/raw_v2

Implemented in: Phase 2
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root on sys.path so ``from src.…`` imports work when called as a
# script from any working directory.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import requests
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)


def _count_jpegs(directory: Path) -> int:
    """Count JPEG files in a directory (case-insensitive)."""
    return len(
        list(directory.glob("*.[jJ][pP][gG]"))
        + list(directory.glob("*.[jJ][pP][eE][gG]"))
    )


def _count_jsons(directory: Path) -> int:
    """Count JSON files in a directory."""
    return len(list(directory.glob("*.json")))


def check_existing_data(raw_dir: Path, expected_count: int) -> bool:
    """Check whether data already exists at raw_dir.

    Returns True if images/, Annotations/ (or annotations/), and
    dataset.csv are present with at least some content.

    Parameters
    ----------
    raw_dir : Path
        Root data directory to check.
    expected_count : int
        Expected number of images (used only for reporting).

    Returns
    -------
    bool
        True if data already exists and looks valid.
    """
    csv_path = raw_dir / "dataset.csv"
    images_dir = raw_dir / "images"
    # Check both cases for annotations directory
    annot_dir = raw_dir / "Annotations"
    if not annot_dir.is_dir():
        annot_dir = raw_dir / "annotations"

    if not csv_path.is_file():
        return False
    if not images_dir.is_dir():
        return False
    if not annot_dir.is_dir():
        return False

    n_images = _count_jpegs(images_dir)
    if n_images < 1:
        return False

    n_annotations = _count_jsons(annot_dir)

    logger.info(
        "Existing data found at %s: %d images, %d annotations, dataset.csv present",
        raw_dir,
        n_images,
        n_annotations,
    )
    return True


def download_with_verification(url: str, dest: Path, expected_md5: str) -> None:
    """Stream-download a file with progress bar and MD5 verification.

    Parameters
    ----------
    url : str
        Download URL (follows redirects automatically).
    dest : Path
        Local file path to save the download.
    expected_md5 : str
        Expected MD5 hex digest for integrity verification.

    Raises
    ------
    ValueError
        If the computed MD5 does not match *expected_md5*.
    requests.HTTPError
        If the HTTP request fails.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s -> %s", url, dest)
    response = requests.get(url, stream=True, allow_redirects=True, timeout=600)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    md5_hash = hashlib.md5()

    with open(dest, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc=dest.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            md5_hash.update(chunk)
            pbar.update(len(chunk))

    actual_md5 = md5_hash.hexdigest()
    if actual_md5 != expected_md5:
        dest.unlink(missing_ok=True)
        raise ValueError(
            f"MD5 mismatch: expected {expected_md5}, got {actual_md5}. "
            "Corrupt download has been deleted."
        )

    logger.info("MD5 verified: %s", actual_md5)


def extract_and_organize(zip_path: Path, dest_dir: Path) -> None:
    """Extract ZIP and flatten a single top-level wrapper directory if present.

    After extraction, verifies the expected directory structure:
    images/, Annotations/ (or annotations/), dataset.csv.

    Parameters
    ----------
    zip_path : Path
        Path to the ZIP archive.
    dest_dir : Path
        Destination directory for extracted contents.
    """
    logger.info("Extracting %s -> %s", zip_path, dest_dir)

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Detect single top-level wrapper directory
        top_levels = {
            name.split("/")[0] for name in zf.namelist() if "/" in name
        }
        has_wrapper = len(top_levels) == 1

        zf.extractall(dest_dir)

    # If there is a single wrapper directory (e.g., BTXRD/), move contents up
    if has_wrapper:
        wrapper_name = top_levels.pop()
        wrapper = dest_dir / wrapper_name
        if wrapper.is_dir():
            logger.info("Flattening wrapper directory: %s", wrapper_name)
            for item in wrapper.iterdir():
                target = dest_dir / item.name
                # Use shutil.move for robustness (handles non-empty dirs)
                shutil.move(str(item), str(target))
            wrapper.rmdir()

    # Verify expected structure (case-insensitive checks)
    images_dir = dest_dir / "images"
    if not images_dir.is_dir():
        images_dir = dest_dir / "Images"
    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Expected images directory not found in {dest_dir}. "
            f"Contents: {[p.name for p in dest_dir.iterdir()]}"
        )

    annot_dir = dest_dir / "Annotations"
    if not annot_dir.is_dir():
        annot_dir = dest_dir / "annotations"
    if not annot_dir.is_dir():
        raise FileNotFoundError(
            f"Expected Annotations directory not found in {dest_dir}. "
            f"Contents: {[p.name for p in dest_dir.iterdir()]}"
        )

    csv_path = dest_dir / "dataset.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Expected dataset.csv not found in {dest_dir}. "
            f"Contents: {[p.name for p in dest_dir.iterdir()]}"
        )

    # Print counts
    n_images = _count_jpegs(images_dir)
    n_annotations = _count_jsons(annot_dir)

    import pandas as pd

    df = pd.read_csv(csv_path)
    n_rows = len(df)
    n_cols = len(df.columns)

    logger.info(
        "Extraction complete: %d images, %d annotations, CSV has %d rows x %d columns",
        n_images,
        n_annotations,
        n_rows,
        n_cols,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download BTXRD dataset from figshare."
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
    setup_logging(config=cfg)

    raw_dir = PROJECT_ROOT / cfg["data"]["raw_dir"]
    figshare_url = cfg["data"]["figshare_url"]
    expected_md5 = cfg["data"]["expected_md5"]
    expected_count = cfg["data"].get("expected_image_count", 3746)

    # Check if data already exists
    if check_existing_data(raw_dir, expected_count):
        logger.info("Data already exists at %s. Skipping download.", raw_dir)
        return

    # Download
    zip_path = raw_dir / "BTXRD.zip"
    download_with_verification(figshare_url, zip_path, expected_md5)

    # Extract and organize
    extract_and_organize(zip_path, raw_dir)

    # Clean up ZIP to save disk space
    logger.info("Removing ZIP archive: %s", zip_path)
    zip_path.unlink()

    # Final summary
    images_dir = raw_dir / "images"
    annot_dir = raw_dir / "Annotations"
    if not annot_dir.is_dir():
        annot_dir = raw_dir / "annotations"

    n_images = _count_jpegs(images_dir)
    n_annotations = _count_jsons(annot_dir)
    csv_path = raw_dir / "dataset.csv"

    logger.info(
        "Download complete! %d images, %d annotations, CSV: %s",
        n_images,
        n_annotations,
        csv_path,
    )


if __name__ == "__main__":
    main()
