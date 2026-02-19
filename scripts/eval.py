#!/usr/bin/env python3
"""Evaluate trained model on both split strategies.

Loads the best checkpoint and runs inference on the test set for
each split strategy, computing metrics (accuracy, macro-F1,
per-class precision/recall, confusion matrix, bootstrap CIs).

Usage:
    python scripts/eval.py --config configs/default.yaml

Implemented in: Phase 5
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
        description="Evaluate trained model on both split strategies."
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
        "Evaluation logic is implemented in Phase 5. "
        "See scripts/eval.py for the placeholder."
    )


if __name__ == "__main__":
    main()
