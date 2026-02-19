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
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root on sys.path so ``from src.…`` imports work when called as a
# script from any working directory.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.reproducibility import set_seed


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

    raise NotImplementedError(
        "Download logic is implemented in Phase 2. "
        "See scripts/download.py for the placeholder."
    )


if __name__ == "__main__":
    main()
