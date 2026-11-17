"""Vector store loading and document retrieval.

Loads the persisted Chroma store and builds the retriever used by the answer
generation layer. The search type (similarity, MMR, or score threshold) and
its parameters come from ``config/settings.py``.  ``retrieve_chunks`` is a traced run so retrieval
shows up as its own child run inside the ``AnswerQuery`` trace.
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langsmith import traceable

from config.settings import (
    CHROMA_PERSIST_DIR,
    RETRIEVAL_FETCH_K,
    RETRIEVAL_K,
    RETRIEVAL_LAMBDA_MULT,
    RETRIEVAL_SCORE_THRESHOLD,
    RETRIEVAL_SEARCH_TYPE,
)
from src.documents import chunk_text_and_tables, stored_image_count
from src.logger import get_logger

logger = get_logger(__name__)


def load_vector_store(embedding_model: Embeddings) -> Chroma:
    """Load the persisted Chroma vector store.

    Args:
        embedding_model: The embedding model to query with (must match the
            one used during ingestion).

    Returns:
        The loaded Chroma vector store.
    """
    logger.info("Loading vector store from %s", CHROMA_PERSIST_DIR)
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embedding_model,
    )


def search_kwargs(search_type: str, k: int) -> dict:
    """Build the ``search_kwargs`` LangChain needs for one search type.

    Each search type takes different parameters, and Chroma rejects ones that
    do not belong to it, so only the matching ones are included.

    Args:
        search_type: One of ``RETRIEVAL_SEARCH_TYPES`` in ``config/settings.py``.
        k: How many chunks to return.

    Returns:
        The keyword arguments for ``as_retriever(search_kwargs=...)``.

    Raises:
        ValueError: If ``search_type`` is not a known search type.
    """
    if search_type == "similarity":
        return {"k": k}
    if search_type == "mmr":
        # The pool must be at least k, or MMR has fewer candidates than it returns.
        return {"k": k, "fetch_k": max(k, RETRIEVAL_FETCH_K), "lambda_mult": RETRIEVAL_LAMBDA_MULT}
    if search_type == "similarity_score_threshold":
        return {"k": k, "score_threshold": RETRIEVAL_SCORE_THRESHOLD}
    raise ValueError(f"Unknown search type {search_type!r}")


def build_retriever(db: Chroma, k: int = RETRIEVAL_K):
    """Build a retriever with the configured search type.

    Args:
        db: The loaded Chroma vector store.
        k: How many chunks each query returns.

    Returns:
        A LangChain retriever whose ``invoke(question)`` returns ``k`` chunks
        (fewer for ``similarity_score_threshold``).
    """
    kwargs = search_kwargs(RETRIEVAL_SEARCH_TYPE, k)
    logger.debug("Building retriever (search_type=%s, %s)", RETRIEVAL_SEARCH_TYPE, kwargs)
    return db.as_retriever(search_type=RETRIEVAL_SEARCH_TYPE, search_kwargs=kwargs)


@traceable(run_type="retriever", name="RetrieveChunks")
def retrieve_chunks(retriever, query: str) -> list[Document]:
    """Run retrieval and return matching chunks.

    Args:
        retriever: The configured retriever.
        query: The user's question.

    Returns:
        List of retrieved document chunks.
    """
    chunks = retriever.invoke(query)
    total_images = sum(stored_image_count(chunk) for chunk in chunks)
    total_tables = sum(len(chunk_text_and_tables(chunk)["tables_html"]) for chunk in chunks)
    logger.info(
        "Retrieved %d chunks (%d images, %d tables)",
        len(chunks),
        total_images,
        total_tables,
    )
    return chunks
