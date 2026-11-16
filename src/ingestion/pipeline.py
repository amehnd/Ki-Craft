"""Turn chunk exports into stored records, and run the PDF-to-Chroma pipeline.

A chunk export (``json/<source>.json``) holds each chunk's summary and its
original text, tables and figures (base64). :func:`documents_from_export`
writes each figure to a file under the images folder and returns one
LangChain ``Document`` per chunk with flat metadata: ``source``, ``chunk_id``,
``raw_text``, ``tables_html``, ``image_paths``, ``has_table``, ``has_image``.
:func:`create_vector_store` embeds those Documents into Chroma.
:func:`run_complete_ingestion_pipeline` runs every step for one PDF.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langsmith import traceable

from config.settings import (
    CHROMA_COLLECTION_METADATA,
    CHROMA_PERSIST_DIR,
    ENHANCEMENT_BACKEND,
    IMAGES_DIR,
    JSON_EXPORT_DIR,
)
from src.embed import load_embedding_model
from src.ingestion.chunk import create_chunks_by_title
from src.ingestion.enrich import summarise_chunks
from src.ingestion.export import export_chunks_to_json
from src.ingestion.extract import partition_document
from src.logger import get_logger

logger = get_logger(__name__)


def documents_from_export(
    export: list[dict],
    source: str,
    images_dir: str | Path = IMAGES_DIR,
) -> list[Document]:
    """Write an export's figures to files and build one Document per chunk.

    Figure ``n`` of chunk ``c`` is written to ``<images_dir>/<source>_c<c>_img<n>.jpg``
    and its path, as ``images_dir`` was given, is stored in ``image_paths``.

    Args:
        export: Entries with ``chunk_id``, ``enhanced_content`` and
            ``metadata.original_content`` (``raw_text``, ``tables_html``,
            ``images_base64``), as :func:`export_chunks_to_json` writes them.
        source: Document name stored on every record and used in file names.
        images_dir: Folder the figure files are written to.

    Returns:
        Documents with the summary as ``page_content`` and flat metadata.
    """
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    documents: list[Document] = []
    for chunk in export:
        original = chunk["metadata"]["original_content"]
        chunk_id = chunk["chunk_id"]
        tables = original.get("tables_html", [])
        image_paths = []
        for idx, b64 in enumerate(original.get("images_base64", [])):
            path = images_dir / f"{source}_c{chunk_id}_img{idx}.jpg"
            path.write_bytes(base64.b64decode(b64))
            image_paths.append(str(path))
        documents.append(
            Document(
                page_content=chunk["enhanced_content"],
                metadata={
                    "source": source,
                    "chunk_id": chunk_id,
                    "raw_text": original.get("raw_text", ""),
                    "tables_html": json.dumps(tables, ensure_ascii=False),
                    "image_paths": json.dumps(image_paths, ensure_ascii=False),
                    "has_table": bool(tables),
                    "has_image": bool(image_paths),
                },
            )
        )
    return documents


@traceable(run_type="chain", name="CreateVectorStore")
def create_vector_store(
    documents: list[Document],
    persist_directory: str = CHROMA_PERSIST_DIR,
) -> Chroma:
    """Embed Documents and add them to the Chroma store in ``persist_directory``.

    Args:
        documents: Documents to embed; ``page_content`` is what gets embedded.
        persist_directory: Folder of the Chroma store; created if missing.

    Returns:
        The Chroma store holding the new records.
    """
    logger.info("Embedding %d chunks into %s", len(documents), persist_directory)
    return Chroma.from_documents(
        documents=documents,
        embedding=load_embedding_model(),
        persist_directory=persist_directory,
        collection_metadata=CHROMA_COLLECTION_METADATA,
    )


@traceable(run_type="chain", name="FullIngestionPipeline")
def run_complete_ingestion_pipeline(
    pdf_path: str | Path,
    persist_directory: str = CHROMA_PERSIST_DIR,
    images_dir: str | Path = IMAGES_DIR,
    export_dir: str | Path = JSON_EXPORT_DIR,
    backend: str = ENHANCEMENT_BACKEND,
) -> Chroma:
    """Ingest one PDF: partition, chunk, summarise, export, write figures, embed.

    The document's name (``source``) is the PDF file name without its suffix.

    Args:
        pdf_path: The PDF to ingest.
        persist_directory: Folder of the Chroma store.
        images_dir: Folder the figure files are written to.
        export_dir: Folder the chunk export ``<source>.json`` is written to.
        backend: Summary backend, ``huggingface`` or ``local``.

    Returns:
        The Chroma store holding the new records.

    Raises:
        FileExistsError: If ``<export_dir>/<source>.json`` already exists, which
            means the document was ingested before.
        SummaryError: If a chunk's summary fails; nothing is written then.
    """
    source = Path(pdf_path).stem
    export_path = Path(export_dir) / f"{source}.json"
    if export_path.exists():
        raise FileExistsError(f"{export_path} exists; {source} looks ingested already")

    elements = partition_document(pdf_path)
    chunks = create_chunks_by_title(elements)
    summarised = summarise_chunks(chunks, backend)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export = export_chunks_to_json(summarised, filename=str(export_path))
    db = create_vector_store(documents_from_export(export, source, images_dir), persist_directory)
    logger.info("Stored %d chunks from %s", len(export), pdf_path)
    return db
