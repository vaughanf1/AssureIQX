#!/usr/bin/env python3
"""Run single-image or batch inference with Grad-CAM overlay.

Accepts either a single ``--image`` path or a directory via
``--input-dir`` and produces predicted class labels with optional
Grad-CAM overlay images.

Usage:
    python scripts/infer.py --config configs/default.yaml --image path/to/xray.png
    python scripts/infer.py --config configs/default.yaml --input-dir data/raw/test/

Implemented in: Phase 6
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
        description="Run single-image or batch inference with Grad-CAM overlay."
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
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single image for inference",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Path to directory of images for batch inference",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, overrides=args.override)
    set_seed(cfg.get("seed", 42))

    raise NotImplementedError(
        "Inference logic is implemented in Phase 6. "
        "See scripts/infer.py for the placeholder."
    )


if __name__ == "__main__":
    main()
