"""Backing service for the FastAPI endpoints.

Wraps the existing retrieval stack (``src.retrieval``) behind a lazy-loaded,
thread-safe service so endpoints stay thin.  All heavy work (embedding model
load, Chroma load, LLM calls) runs in a worker thread via ``asyncio.to_thread``
so FastAPI's event loop is never blocked.

Every query, with the default model or a caller-chosen one, goes through
``stream_answer`` and so through the one generation path
(``answer_query_stream``). The LangSmith UI shows one ``AnswerQuery`` run with
``RetrieveChunks`` and ``GenerateAnswerStream`` nested underneath.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel

from config.settings import (
    GENERATION_MAX_IMAGES,
    GENERATION_MODEL,
    GENERATION_PROVIDER,
    RETRIEVAL_K,
)
from src.documents import chunk_text_and_tables, image_file, stored_image_paths
from src.embed import load_embedding_model
from src.llm import accepts_images
from src.logger import get_logger
from src.retrieval.generate import StreamEvent, answer_query_stream, default_llm
from src.retrieval.search import build_retriever, load_vector_store, retrieve_chunks

from api.schemas import ChunkPayload, RetrieveResponse

logger = get_logger(__name__)

# Metadata keys that are excluded from the API payload's ``metadata`` dict to
# keep responses small (the full content is exposed via dedicated fields).
_EXCLUDED_META_KEYS = {"original_content", "raw_text", "tables_html", "images_base64", "image_paths"}


class StoreUnavailableError(RuntimeError):
    """The embedding client or the vector store could not be loaded.

    ``api/main.py`` maps it to HTTP 503: the service cannot answer right now,
    and ``/health`` reports the same.
    """


class RAGService:
    """Lazily loads the vector store and exposes retrieve/answer operations."""

    def __init__(self):
        self._db = None
        self._retriever = None
        self._load_lock = asyncio.Lock()

    # -- lifecycle ----------------------------------------------------------

    async def ensure_ready(self) -> None:
        """Load the vector store on first use (idempotent, thread-safe).

        Raises:
            StoreUnavailableError: If the embedding client or the store fails
                to load. The next call tries again.
        """
        if self._db is None:
            async with self._load_lock:
                if self._db is None:
                    try:
                        await asyncio.to_thread(self._load)
                    except Exception as exc:
                        raise StoreUnavailableError(f"Vector store unavailable: {exc}") from exc

    def _load(self) -> None:
        logger.info("Loading embedding model and vector store...")
        model = load_embedding_model()
        self._db = load_vector_store(model)
        self._retriever = build_retriever(self._db)
        logger.info("Vector store ready.")

    def get_retriever(self, top_k: int | None = None):
        """Return a retriever, honouring a per-request ``top_k`` override."""
        if self._db is None:
            raise RuntimeError("Vector store is not loaded")
        if top_k and top_k != RETRIEVAL_K:
            return build_retriever(self._db, top_k)
        return self._retriever

    # -- operations ---------------------------------------------------------

    async def retrieve(self, query: str, top_k: int) -> RetrieveResponse:
        """Retrieve chunks for a query without generating an answer."""
        await self.ensure_ready()
        retriever = self.get_retriever(top_k)
        chunks = await asyncio.to_thread(retrieve_chunks, retriever, query)
        return self.to_retrieve_response(query, chunks)

    def stream_answer(
        self,
        query: str,
        top_k: int,
        max_images: int = GENERATION_MAX_IMAGES,
        llm: BaseChatModel | None = None,
        provider_id: str | None = None,
        model_id: str | None = None,
    ) -> Iterator[StreamEvent]:
        """Retrieve chunks and return the answer as a stream of events.

        Blocking: call it from a worker thread, after ``ensure_ready``. This is
        the one place that picks the default model when no ``llm`` is given.

        Args:
            query: The user's question.
            top_k: How many chunks to retrieve.
            max_images: Most images to send to the model (0 = all).
            llm: A chat model from ``src/llm.py:create_llm``; the default
                Gemini model when omitted.
            provider_id: The catalogue provider of ``llm``. Ignored when
                ``llm`` is omitted.
            model_id: The catalogue id of ``llm``, recorded in usage. Ignored
                when ``llm`` is omitted. With ``provider_id`` it decides
                whether figures are sent: a text-only model gets none.

        Returns:
            The generator from ``answer_query_stream``: ``("chunks", ...)``,
            then ``("token", ...)`` per answer piece, then ``("usage", ...)``.
            Nothing runs until it is iterated.
        """
        if llm is None:
            llm, provider_id, model_id = default_llm(), GENERATION_PROVIDER, GENERATION_MODEL
        return answer_query_stream(
            self.get_retriever(top_k),
            query,
            llm,
            model_id,
            max_images=max_images,
            send_images=accepts_images(provider_id, model_id),
        )

    async def answer(
        self,
        query: str,
        top_k: int,
        max_images: int = GENERATION_MAX_IMAGES,
        llm: BaseChatModel | None = None,
        provider_id: str | None = None,
        model_id: str | None = None,
    ) -> tuple[str, RetrieveResponse, dict]:
        """Retrieve chunks and generate a full answer.

        Consumes every event from :meth:`stream_answer`, so the arguments are
        the same and retrieval and generation stay under a single trace.

        Returns:
            A tuple of ``(answer, retrieve_response, usage)`` where ``usage``
            carries token counts, model, latency and cost estimate.
        """
        await self.ensure_ready()

        def _run() -> tuple[str, list, dict]:
            chunks: list = []
            tokens: list[str] = []
            usage: dict = {}
            for kind, value in self.stream_answer(query, top_k, max_images, llm, provider_id, model_id):
                if kind == "chunks":
                    chunks = value
                elif kind == "token":
                    tokens.append(value)
                else:
                    usage = value
            return "".join(tokens), chunks, usage

        answer, chunks, usage = await asyncio.to_thread(_run)
        return answer, self.to_retrieve_response(query, chunks), usage

    async def image(self, record_id: str, position: int) -> Path | None:
        """Find one figure of a stored record, for ``/v1/images/{record_id}/{position}``.

        The file path is read from the record's metadata in the store, never
        taken from the caller.

        Args:
            record_id: The Chroma record id (a UUID string).
            position: 0-based index into the record's figures.

        Returns:
            The figure's file, or None when the record, the position or the
            file does not exist.

        Raises:
            StoreUnavailableError: If the store cannot be loaded.
        """
        await self.ensure_ready()
        found = await asyncio.to_thread(self._db.get, ids=[record_id], include=["metadatas"])
        metadatas = found.get("metadatas") or []
        if not metadatas:
            return None
        return image_file(metadatas[0] or {}, position)

    # -- payload assembly ---------------------------------------------------

    def to_retrieve_response(self, query: str, chunks: list) -> RetrieveResponse:
        """Build the API response for a set of retrieved chunks.

        Each figure stored as a file gets a stable URL,
        ``/v1/images/{record_id}/{position}``, where ``record_id`` is the
        chunk's Chroma record id and ``position`` counts from 0 within the
        chunk. No image file is read: the response carries URLs, not bytes,
        so ``image_count`` and ``total_images`` count those URLs.
        """
        payloads: list[ChunkPayload] = []
        total_images = 0
        total_tables = 0
        for i, chunk in enumerate(chunks):
            raw = chunk_text_and_tables(chunk)
            tables = raw["tables_html"]
            image_urls = (
                [
                    f"/v1/images/{chunk.id}/{position}"
                    for position in range(len(stored_image_paths(chunk.metadata or {})))
                ]
                if chunk.id
                else []
            )
            metadata = {
                k: v
                for k, v in (chunk.metadata or {}).items()
                if k not in _EXCLUDED_META_KEYS
            }
            payloads.append(
                ChunkPayload(
                    chunk_id=i + 1,
                    enhanced_content=chunk.page_content or "",
                    raw_text=raw["raw_text"],
                    tables_html=tables,
                    image_count=len(image_urls),
                    record_id=chunk.id,
                    image_urls=image_urls,
                    has_table=bool(tables),
                    has_image=bool(image_urls),
                    source=metadata.get("source"),
                    metadata=metadata,
                )
            )
            total_images += len(image_urls)
            total_tables += len(tables)

        return RetrieveResponse(
            query=query,
            num_chunks=len(chunks),
            total_images=total_images,
            total_tables=total_tables,
            chunks=payloads,
        )
