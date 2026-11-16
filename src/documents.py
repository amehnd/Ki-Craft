"""Read a stored chunk's original text, tables and figures from its metadata.

A chunk's ``page_content`` is its summary; its ``metadata`` holds the original
content: ``raw_text``, and ``tables_html`` and ``image_paths`` as JSON strings.
Figures are files under ``IMAGES_DIR``; ``image_paths`` lists them in order.

Functions:

* :func:`chunk_text_and_tables` - text and tables, no figure read.
* :func:`stored_image_paths`, :func:`stored_image_count` - which figures a
  chunk has, without reading them.
* :func:`image_file` - the file behind one figure, kept inside ``IMAGES_DIR``;
  the image endpoint and the model prompt both find figures through it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from config.settings import IMAGES_DIR
from src.logger import get_logger

logger = get_logger(__name__)

# Repository root: <repo>/src/documents.py -> <repo>
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_json_value(raw: Any, default: Any) -> Any:
    """Parse ``raw`` as JSON when it is a string, otherwise return it as-is."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return default
    return raw if raw is not None else default


def chunk_text_and_tables(chunk: Document) -> dict:
    """Return a chunk's original text and tables without reading its figures.

    Args:
        chunk: A retrieved LangChain ``Document``.

    Returns:
        Dict with ``raw_text`` (str) and ``tables_html`` (list[str]).
    """
    metadata = chunk.metadata or {}
    return {
        "raw_text": metadata.get("raw_text", ""),
        "tables_html": _parse_json_value(metadata.get("tables_html"), []),
    }


def stored_image_count(chunk: Document) -> int:
    """Count a chunk's figures without reading any."""
    return len(stored_image_paths(chunk.metadata or {}))


def stored_image_paths(metadata: dict) -> list[str]:
    """Return a record's figure paths in stored order."""
    paths = _parse_json_value(metadata.get("image_paths"), [])
    return [str(path) for path in paths] if isinstance(paths, list) else []


def image_file(metadata: dict, position: int) -> Path | None:
    """Find the file of one figure, as ``/v1/images/{record_id}/{position}`` names it.

    The path comes from the record's metadata and must resolve to a file inside
    ``IMAGES_DIR``; anything else is treated as missing.

    Args:
        metadata: The record's metadata, as Chroma stores it.
        position: 0-based index into the record's ``image_paths``.

    Returns:
        The absolute path of the figure, or None when the position is out of
        range, the path points outside ``IMAGES_DIR``, or the file is missing.
    """
    paths = stored_image_paths(metadata)
    if not 0 <= position < len(paths):
        return None
    images_root = (_PROJECT_ROOT / IMAGES_DIR).resolve()
    full_path = (_PROJECT_ROOT / paths[position]).resolve()
    if not full_path.is_relative_to(images_root) or not full_path.is_file():
        logger.warning("Stored figure path is outside %s or missing: %s", IMAGES_DIR, paths[position])
        return None
    return full_path
