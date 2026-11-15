"""Document helpers shared across the retrieval and UI layers.

The pipeline stores the original multimodal content of every chunk inside its
``metadata``.  Over time two storage schemas have been used:

* **Nested** -- ``metadata["original_content"]`` is a JSON string (or dict)
  containing ``raw_text``, ``tables_html`` and ``images_base64``.
* **Flattened** -- ``metadata`` holds ``raw_text`` and ``tables_html`` directly,
  while images are referenced by file path in ``metadata["image_paths"]``.

This module normalises both schemas into a single dict so callers never need to
know which one a document uses. Figure URLs (``/v1/images/{record_id}/{position}``)
are served from files only, so :func:`image_file` reads the flattened schema;
the nested one is what ``scripts/ingest.py`` still writes, and its figures have
no URL.
"""

from __future__ import annotations

import base64
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


def extract_original_data(chunk: Document) -> dict:
    """Return the chunk's original multimodal content as a dict.

    The returned dict always contains the keys ``raw_text`` (str),
    ``tables_html`` (list[str]) and ``images_base64`` (list[str]), so callers
    can rely on a stable shape regardless of the underlying metadata schema.

    Args:
        chunk: A retrieved LangChain ``Document``.

    Returns:
        Dict with ``raw_text``, ``tables_html`` and ``images_base64`` keys.
    """
    metadata = chunk.metadata or {}

    # Preferred schema: the full original payload stored inline.
    raw = metadata.get("original_content")
    if raw is not None:
        parsed = _parse_json_value(raw, {})
        if isinstance(parsed, dict):
            return {
                "raw_text": parsed.get("raw_text", ""),
                "tables_html": _parse_json_value(parsed.get("tables_html"), []),
                "images_base64": parsed.get("images_base64", []),
            }

    # Fallback schema: fields are flattened onto the metadata dict directly,
    # and images live on disk (referenced by path).
    tables_html = _parse_json_value(metadata.get("tables_html"), [])
    image_paths = _parse_json_value(metadata.get("image_paths"), [])

    images_base64: list[str] = []
    for path in image_paths:
        images_base64.append(_read_image_as_base64(path))

    return {
        "raw_text": metadata.get("raw_text", ""),
        "tables_html": tables_html,
        "images_base64": images_base64,
    }


def stored_image_paths(metadata: dict) -> list[str]:
    """Return a record's figure paths, in stored order; empty when it has none.

    Only the flattened schema has paths; figures stored inline as base64 have
    none, so they get no URL.
    """
    paths = _parse_json_value(metadata.get("image_paths"), [])
    return [str(path) for path in paths] if isinstance(paths, list) else []


def image_file(metadata: dict, position: int) -> Path | None:
    """Find the file of one figure of a stored record.

    The path comes from the record's own metadata, never from the caller, and
    must resolve to a file inside ``IMAGES_DIR``; anything else (a stored path
    with ``..`` or an absolute path elsewhere) is treated as missing.

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


def _read_image_as_base64(image_path: str) -> str:
    """Resolve a stored image path to a base64-encoded JPEG string.

    Paths saved in metadata are relative to the repository root.  If the file
    cannot be read (missing or corrupt) an empty string is returned and the
    problem is logged instead of raising.
    """
    clean = image_path.lstrip("./")
    full_path = _PROJECT_ROOT / clean
    try:
        return base64.b64encode(full_path.read_bytes()).decode()
    except OSError as exc:
        logger.warning("Could not read image %s: %s", full_path, exc)
        return ""
