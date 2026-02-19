#!/usr/bin/env python3
"""Train EfficientNet-B0 classifier on BTXRD dataset.

Loads split manifests, builds data loaders with augmentations,
fine-tunes an EfficientNet-B0 backbone, and saves the best
checkpoint to ``paths.checkpoints_dir``.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --override training.epochs=10

Implemented in: Phase 4
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
        description="Train EfficientNet-B0 classifier on BTXRD dataset."
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
        "Training logic is implemented in Phase 4. "
        "See scripts/train.py for the placeholder."
    )


if __name__ == "__main__":
    main()
