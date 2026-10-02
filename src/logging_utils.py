"""Logging helper: one consistent format for console and optional log file.

    from src.logging_utils import get_logger
    log = get_logger(__name__, log_file="outputs/logs/run.log")
    log.info("median drift %.2f%%", 9.91)
"""
from __future__ import annotations

import logging
import os
import sys

FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "avirat", level: int | str = logging.INFO, log_file: str | None = None) -> logging.Logger:
    """Return a logger writing to stderr (and `log_file`, if given). Safe to call repeatedly:
    handlers are added once per destination, so messages are never duplicated."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter(FORMAT, DATEFMT)
    if not any(getattr(h, "_avirat", None) == "console" for h in logger.handlers):
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(fmt)
        h._avirat = "console"
        logger.addHandler(h)
    if log_file:
        path = os.path.abspath(log_file)
        if not any(getattr(h, "_avirat", None) == path for h in logger.handlers):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            h = logging.FileHandler(path)
            h.setFormatter(fmt)
            h._avirat = path
            logger.addHandler(h)
    return logger
