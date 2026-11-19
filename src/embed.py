"""Embedding model factory.

Provides a single function to build the embedding model used by both the
ingestion pipeline (when creating the vector store) and the retrieval path
(when loading it back), guaranteeing both sides use the same configuration.

Embeddings are served by IBM watsonx.ai, so the store must be re-indexed
whenever ``EMBEDDING_MODEL`` changes: vectors from different models are not
comparable and usually differ in dimension.

The stored vectors were made by ``ibm/granite-embedding-278m-multilingual`` on
watsonx. A query must be embedded into that same vector space, or its distances
to the stored vectors mean nothing. ``EMBEDDING_BACKEND=local`` therefore runs a
GGUF build of that same model with llama.cpp on this machine, so the store can be
queried and extended without watsonx credentials and without torch. Another
embedder works only if it produces vectors in the same space; two separately
trained models almost never do, even with 768 dimensions. To check a candidate,
embed some stored summaries with it and compare each with its stored vector: a
cosine near 1.0 means the same space, anything much lower means the store must
be re-indexed. The Q8_0 file scores 0.9999 against the stored vectors.

Both backends give unit-length vectors and read at most
``EMBEDDING_MAX_INPUT_TOKENS`` tokens per text, so local vectors are
interchangeable with watsonx ones for both ingestion and retrieval.
"""

from __future__ import annotations

import threading
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_ibm import WatsonxEmbeddings

from config.settings import (
    EMBEDDING_BACKEND,
    EMBEDDING_MAX_INPUT_TOKENS,
    EMBEDDING_MODEL,
    LOCAL_EMBEDDING_GGUF_PATH,
    LOCAL_EMBEDDING_THREADS,
    WATSONX_API_KEY,
    WATSONX_PROJECT_ID,
    WATSONX_URL,
)
from src.logger import get_logger

logger = get_logger(__name__)


class GGUFEmbeddings(Embeddings):
    """Granite embeddings from a GGUF file, computed in-process by llama.cpp.

    Matches the watsonx model's output: CLS pooling (as Granite's own
    sentence-transformers config uses), at most ``EMBEDDING_MAX_INPUT_TOKENS``
    tokens per text, and unit-length vectors. Calls are serialised with a lock:
    the API embeds each request in its own worker thread, and llama.cpp
    crashes (segmentation fault) when two threads use one model at once.

    Args:
        model_path: Path to the ``.gguf`` file.
        threads: CPU threads llama.cpp may use.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.
    """

    def __init__(self, model_path: str, threads: int) -> None:
        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"GGUF embedding model not found at {model_path}; download it as "
                "described in CONTRIBUTING.md or set LOCAL_EMBEDDING_GGUF_PATH"
            )
        # Imported here so the watsonx deployment never needs llama.cpp installed.
        import llama_cpp

        self._llm = llama_cpp.Llama(
            model_path=model_path,
            embedding=True,
            pooling_type=llama_cpp.LLAMA_POOLING_TYPE_CLS,
            # embed() truncates each text to n_batch tokens, so both are the
            # model's input limit, like watsonx's truncate_input_tokens.
            n_ctx=EMBEDDING_MAX_INPUT_TOKENS,
            n_batch=EMBEDDING_MAX_INPUT_TOKENS,
            n_threads=threads,
            verbose=False,
        )
        self._lock = threading.Lock()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed each text; returns one unit-length 768-number vector per text."""
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed one text; returns a unit-length 768-number vector."""
        with self._lock:
            return self._llm.embed(text, normalize=True, truncate=True)


def _load_local_embedding_model() -> Embeddings:
    """Load the GGUF Granite model configured in settings."""
    logger.info(
        "Loading local embedding model: %s (%d thread(s))",
        LOCAL_EMBEDDING_GGUF_PATH,
        LOCAL_EMBEDDING_THREADS,
    )
    return GGUFEmbeddings(LOCAL_EMBEDDING_GGUF_PATH, LOCAL_EMBEDDING_THREADS)


def load_embedding_model() -> Embeddings:
    """Create the embedding model configured in settings.

    With ``EMBEDDING_BACKEND=local`` this loads the GGUF model on this machine
    instead (see :class:`GGUFEmbeddings`).

    Inputs longer than ``EMBEDDING_MAX_INPUT_TOKENS`` are truncated by the
    service; without this, watsonx rejects the whole request with a 400.

    Returns:
        A configured embeddings object (:class:`WatsonxEmbeddings` or
        :class:`GGUFEmbeddings`) ready to embed text or load a vector store.

    Raises:
        RuntimeError: If ``WATSONX_API_KEY`` or ``WATSONX_PROJECT_ID`` is not
            set, so the failure names the missing variable instead of
            surfacing as an opaque authentication error.
        FileNotFoundError: If the local backend is chosen and the GGUF file
            is missing.
    """
    if EMBEDDING_BACKEND == "local":
        return _load_local_embedding_model()

    missing = [
        name
        for name, value in (
            ("WATSONX_API_KEY", WATSONX_API_KEY),
            ("WATSONX_PROJECT_ID", WATSONX_PROJECT_ID),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing watsonx.ai credentials: {', '.join(missing)}")

    logger.info("Loading embedding model: %s (%s)", EMBEDDING_MODEL, WATSONX_URL)
    return WatsonxEmbeddings(
        model_id=EMBEDDING_MODEL,
        url=WATSONX_URL,
        project_id=WATSONX_PROJECT_ID,
        api_key=WATSONX_API_KEY,
        params={"truncate_input_tokens": EMBEDDING_MAX_INPUT_TOKENS},
    )
