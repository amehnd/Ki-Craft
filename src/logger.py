"""Logging for the API, ingestion and scripts: one format, to stdout and ``logs/pipeline.log``.

Every module gets its logger with ``get_logger(__name__)``. The first call
sets up the root logger; later calls find it set up and leave it alone. If
something else configured the root logger first (a test runner, for
example), that configuration is kept.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMATTER = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", "%H:%M:%S")
_LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "pipeline.log"


def _setup_root_logger() -> None:
    """Send INFO and above to stdout and the log file, once per process."""
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.INFO)
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(_LOG_FILE, encoding="utf-8")):
        handler.setFormatter(_FORMATTER)
        root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return the logger for a module, setting up logging on first use.

    Args:
        name: Usually the calling module's ``__name__``.

    Returns:
        A logger that writes through the root logger's handlers.
    """
    _setup_root_logger()
    return logging.getLogger(name)
