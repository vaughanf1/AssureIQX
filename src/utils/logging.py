"""Logging configuration for consistent formatting across all scripts.

Usage:
    from src.utils.logging import setup_logging

    logger = setup_logging()                   # INFO to console
    logger = setup_logging(level="DEBUG")      # DEBUG to console
    logger = setup_logging(log_file="logs/train.log")  # also write to file
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging(
    name: str = "assurexray",
    level: str | None = None,
    log_file: str | None = None,
    config: dict[str, Any] | None = None,
) -> logging.Logger:
    """Configure and return a logger instance.

    Parameters
    ----------
    name : str
        Logger name (default ``"assurexray"``).
    level : str | None
        Logging level string (e.g. ``"DEBUG"``, ``"INFO"``).
        Falls back to ``config["logging"]["level"]`` if provided,
        otherwise defaults to ``"INFO"``.
    log_file : str | None
        Optional path to a log file.  Parent directories are created
        automatically if they do not exist.
    config : dict | None
        Full project config dict.  If supplied, ``logging.level`` and
        ``paths.logs_dir`` are read as fallback defaults.

    Returns
    -------
    logging.Logger
        Configured logger ready for use.
    """
    # Resolve level: explicit arg > config > INFO
    if level is None and config is not None:
        level = config.get("logging", {}).get("level", "INFO")
    if level is None:
        level = "INFO"

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers on repeated calls
    if not logger.handlers:
        # Console handler
        console = logging.StreamHandler()
        console.setLevel(numeric_level)
        console.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(console)

    # File handler (optional)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path))
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(file_handler)

    return logger
