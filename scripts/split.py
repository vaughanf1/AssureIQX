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
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.reproducibility import set_seed


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

    raise NotImplementedError(
        "Split logic is implemented in Phase 3. "
        "See scripts/split.py for the placeholder."
    )


if __name__ == "__main__":
    main()
