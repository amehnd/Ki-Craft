"""Short descriptions of retrieved chunks for logs and LangSmith traces.

:func:`summarize_chunks` turns each chunk into a small dict of lengths,
counts and a 300-character preview. No figure is read.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.documents import Document

from src.documents import chunk_text_and_tables, stored_image_count


def _extract_chunk_summary(chunk: Document) -> dict[str, Any]:
    """Build a compact summary of one chunk's multimodal content.

    The returned dict describes the chunk (lengths and presence of text,
    tables, images) rather than its full payload, which keeps trace metadata
    small and useful for debugging retrieval quality.
    """
    enhanced = chunk.page_content or ""
    # Counts only: no figure is read or encoded for a log line.
    original = chunk_text_and_tables(chunk)
    tables = original["tables_html"]
    image_count = stored_image_count(chunk)

    return {
        "enhanced_length": len(enhanced),
        "enhanced_preview": enhanced[:300],
        "raw_text_length": len(original["raw_text"]),
        "table_count": len(tables),
        "image_count": image_count,
        "has_table": len(tables) > 0,
        "has_image": image_count > 0,
    }


def summarize_chunks(chunks: list[Document], max_preview: int = 5) -> list[dict]:
    """Summarise up to ``max_preview`` chunks for inclusion in trace metadata.

    Args:
        chunks: Retrieved documents to describe.
        max_preview: Upper bound on the number of chunks summarised.

    Returns:
        List of compact chunk summaries (see :func:`_extract_chunk_summary`).
    """
    return [_extract_chunk_summary(c) for c in chunks[:max_preview]]
