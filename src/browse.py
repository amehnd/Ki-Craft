"""Browse the stored corpus: which documents are indexed, and their chunks in order.

Sits beside retrieval, not inside it: these reads use no embedding and no
vector search. ``api/service.py`` calls them for ``GET /v1/documents`` and
``GET /v1/documents/{document_id}/chunks``, which the dashboard's Chunk
Explorer, Compare and Analytics pages use instead of reading ``json/``.

A document is identified by its ``source``, the PDF's file name without
``.pdf`` (``lm317``). Chunks come back in reading order, by the ``chunk_id``
stored at ingestion, which counts from 1 within each document.

The store is Chroma today. When Postgres and pgvector replace it
(Checkpoint 6), these two functions become SQL queries (``GROUP BY source``,
and ``ORDER BY chunk_id LIMIT ... OFFSET ...``) and their callers do not change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from src.documents import chunk_text_and_tables, stored_image_paths


@dataclass(frozen=True)
class DocumentSummary:
    """One indexed document and how much it holds.

    Attributes:
        id: The document's ``source`` name, such as ``lm317``.
        chunk_count: Stored chunks.
        image_count: Figures across those chunks.
        table_count: Tables across those chunks.
    """

    id: str
    chunk_count: int
    image_count: int
    table_count: int


def list_documents(db: Any) -> list[DocumentSummary]:
    """Summarise every indexed document, sorted by name.

    Reads every record's metadata once; no text, summary or vector is loaded.

    Args:
        db: The loaded vector store (anything with Chroma's ``get``).

    Returns:
        One summary per distinct ``source``. Records without a ``source``
        are left out, since no document can be named for them.
    """
    found = db.get(include=["metadatas"])
    counts: dict[str, list[int]] = {}
    for metadata in found.get("metadatas") or []:
        metadata = metadata or {}
        source = metadata.get("source")
        if not source:
            continue
        tables = chunk_text_and_tables(Document(page_content="", metadata=metadata))["tables_html"]
        chunks, images, table_total = counts.get(source, [0, 0, 0])
        counts[source] = [chunks + 1, images + len(stored_image_paths(metadata)), table_total + len(tables)]
    return [DocumentSummary(source, *counts[source]) for source in sorted(counts)]


def document_chunks(db: Any, document_id: str, offset: int, limit: int) -> tuple[list[Document], int]:
    """Return one page of a document's chunks, in reading order.

    Chroma can filter by metadata but cannot sort, so every chunk of the
    document is read, sorted by ``chunk_id``, then sliced. The largest
    document holds under 100 chunks, so this stays cheap.

    Args:
        db: The loaded vector store (anything with Chroma's ``get``).
        document_id: The document's ``source`` name.
        offset: Chunks to skip, counted from the start of the document.
        limit: Most chunks to return.

    Returns:
        ``(chunks, total)``: the page of chunks as ``Document`` objects (id,
        summary as ``page_content``, stored metadata), and how many chunks
        the document holds. ``total`` is 0 for an unknown document.
    """
    found = db.get(where={"source": document_id}, include=["documents", "metadatas"])
    chunks = [
        Document(id=record_id, page_content=summary or "", metadata=metadata or {})
        for record_id, summary, metadata in zip(
            found.get("ids") or [], found.get("documents") or [], found.get("metadatas") or []
        )
    ]
    chunks.sort(key=lambda chunk: chunk.metadata.get("chunk_id") or 0)
    return chunks[offset : offset + limit], len(chunks)
