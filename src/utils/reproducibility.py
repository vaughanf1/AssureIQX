"""Deterministic seed setting for full reproducibility.

Sets seeds for random, numpy, torch (CPU + CUDA) and enables
deterministic CuDNN / cuBLAS behaviour.

Usage:
    from src.utils.reproducibility import set_seed
    set_seed(42)
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds across all random number generators.

    Parameters
    ----------
    seed : int
        The seed value to use everywhere (default ``42``).
    """
    # Python built-in
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA (all GPUs)
    torch.cuda.manual_seed_all(seed)

    # CuDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # cuBLAS workspace config for deterministic reductions
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Deterministic algorithms (warn rather than error for ops without
    # deterministic implementation)
    torch.use_deterministic_algorithms(True, warn_only=True)
