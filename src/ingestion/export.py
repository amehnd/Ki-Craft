"""Export processed chunks to JSON.

Writes the summarised chunks to ``json/<source>.json``: each chunk's summary
and its original text, tables and figures (base64). Summarising is the slow,
paid step, so the export keeps its result: ``src/ingestion/pipeline.py:store_export``
stores records from it, and it can rebuild the store without summarising again
(``python -m scripts.ingest json/<source>.json``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.logger import get_logger

logger = get_logger(__name__)


def export_chunks_to_json(chunks: list[Document], filename: str | Path) -> list[dict]:
    """Export processed LangChain Documents to a clean JSON file.

    Each entry contains ``chunk_id``, ``enhanced_content`` and metadata with
    the original ``raw_text``, ``tables_html`` and ``images_base64``.

    Args:
        chunks: List of LangChain Documents.
        filename: Output JSON file path, ``json/<source>.json`` in the pipeline.

    Returns:
        The exported data as a list of dicts.
    """
    export_data: list[dict[str, Any]] = []

    for i, doc in enumerate(chunks):
        raw_meta = doc.metadata.get("original_content", "{}")
        if isinstance(raw_meta, str):
            original = json.loads(raw_meta)
        else:
            original = raw_meta

        export_data.append(
            {
                "chunk_id": i + 1,
                "enhanced_content": doc.page_content,
                "metadata": {"original_content": original},
            }
        )

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    logger.info("Exported %d chunks to %s", len(export_data), filename)
    return export_data
