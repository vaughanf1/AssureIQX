"""Configuration loading with YAML parsing and CLI override support.

Usage:
    from src.utils.config import load_config

    # Basic load
    cfg = load_config("configs/default.yaml")

    # With CLI overrides (dot-notation)
    cfg = load_config("configs/default.yaml", overrides=["training.batch_size=64", "seed=123"])
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML configuration file and apply optional CLI overrides.

    Parameters
    ----------
    config_path : str
        Path to the YAML config file.
    overrides : list[str] | None
        List of "key.subkey=value" strings for nested overrides.
        Values are coerced via ``yaml.safe_load`` so that
        ``"true"`` becomes ``True``, ``"42"`` becomes ``42``, etc.

    Returns
    -------
    dict[str, Any]
        The loaded (and optionally overridden) configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If *config_path* does not exist.
    ValueError
        If an override string is malformed or targets a non-existent
        parent key.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    if overrides:
        for override in overrides:
            if "=" not in override:
                raise ValueError(
                    f"Malformed override (expected 'key.subkey=value'): {override}"
                )
            key_path, value_str = override.split("=", 1)
            keys = key_path.strip().split(".")
            # Coerce value through yaml.safe_load for type inference
            coerced_value = yaml.safe_load(value_str)

            # Walk the config tree to the parent of the target key
            node = cfg
            for k in keys[:-1]:
                if k not in node or not isinstance(node[k], dict):
                    raise ValueError(
                        f"Override key path invalid — '{k}' is not a dict in config: {key_path}"
                    )
                node = node[k]
            node[keys[-1]] = coerced_value

    return cfg
