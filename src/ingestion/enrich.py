"""Content-type separation and AI-enhanced summary generation.

Each unstructured chunk is split into its text, tables and images, and a
vision model writes a searchable summary of all three. The model is served
over the OpenAI chat protocol by one of two backends, chosen by
``ENHANCEMENT_BACKEND``: the Hugging Face router, or a local
``mlx_vlm.server``. Each chunk becomes a LangChain ``Document`` with the
summary as ``page_content`` and the original content in
``metadata["original_content"]``.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from langchain_core.documents import Document
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import APIConnectionError, OpenAI, OpenAIError

from config.settings import (
    ENHANCEMENT_BACKEND,
    ENHANCEMENT_BACKENDS,
    ENHANCEMENT_MAX_IMAGES,
    ENHANCEMENT_MAX_RETRIES,
    ENHANCEMENT_MAX_TOKENS,
    ENHANCEMENT_TEMPERATURE,
    ENHANCEMENT_TIMEOUT_SECONDS,
    HF_TOKEN,
    LOCAL_ENHANCEMENT_MAX_PIXELS,
    LOCAL_ENHANCEMENT_REPETITION_PENALTY,
    LOCAL_ENHANCEMENT_THINKING,
)
from src.logger import get_logger

logger = get_logger(__name__)


def separate_content_types(chunk: Any) -> dict:
    """Analyse which content types are present in a chunk.

    Inspects the ``orig_elements`` attached by Unstructured to split a chunk
    into its textual content, HTML tables and base64-encoded images.

    Args:
        chunk: An Unstructured chunk with a ``text`` and ``metadata``.

    Returns:
        Dict with keys ``text`` (str), ``tables`` (list[str]),
        ``images`` (list[str]) and ``types`` (set of type names).
    """
    content_data: dict = {
        "text": chunk.text,
        "tables": [],
        "images": [],
        "types": ["text"],
    }

    orig_elements = getattr(chunk.metadata, "orig_elements", None)
    if orig_elements is None:
        return content_data

    for element in orig_elements:
        element_type = type(element).__name__

        if element_type == "Table":
            content_data["types"].append("table")
            table_html = getattr(element.metadata, "text_as_html", element.text)
            content_data["tables"].append(table_html)

        elif element_type == "Image":
            img_b64 = getattr(element.metadata, "image_base64", None)
            if img_b64:
                content_data["types"].append("image")
                content_data["images"].append(img_b64)

    content_data["types"] = list(set(content_data["types"]))
    return content_data


class SummaryError(RuntimeError):
    """The summary model failed or returned no text for a chunk."""


def create_enhancement_client(backend: str = ENHANCEMENT_BACKEND) -> OpenAI:
    """Create the OpenAI-protocol client for one summary backend.

    The client retries timeouts, connection errors, 429 and 5xx responses with
    exponential backoff, and is wrapped with ``wrap_openai`` so LangSmith
    records each call with its images.

    Args:
        backend: A key of ``ENHANCEMENT_BACKENDS``: ``huggingface`` or ``local``.

    Returns:
        The wrapped client.

    Raises:
        KeyError: If ``backend`` is not a known backend.
    """
    config = ENHANCEMENT_BACKENDS[backend]
    # mlx_vlm.server checks a key only when MLX_VLM_SERVER_API_KEY is set for it;
    # the client still needs a non-empty value.
    api_key = HF_TOKEN if backend == "huggingface" else os.getenv("MLX_VLM_SERVER_API_KEY", "local")
    raw = OpenAI(
        api_key=api_key,
        base_url=config["base_url"],
        timeout=ENHANCEMENT_TIMEOUT_SECONDS,
        max_retries=ENHANCEMENT_MAX_RETRIES,
    )
    return wrap_openai(raw, chat_name="EnhancementVisionLLM")


def check_enhancement_backend(backend: str = ENHANCEMENT_BACKEND) -> None:
    """Fail fast, before any PDF is parsed, if the backend cannot be reached.

    Raises:
        SummaryError: If the server does not answer, or the local server does
            not serve the configured model.
    """
    config = ENHANCEMENT_BACKENDS[backend]
    if backend == "huggingface":
        if not HF_TOKEN:
            raise SummaryError("HF_TOKEN is not set; the huggingface backend needs it")
        return
    try:
        served = [m.id for m in create_enhancement_client(backend).models.list()]
    except APIConnectionError as exc:
        raise SummaryError(
            f"No mlx_vlm server at {config['base_url']}. Start one with: "
            f"python -m mlx_vlm.server --model {config['model']} --port 8080"
        ) from exc
    if served and config["model"] not in served:
        logger.warning("The local server lists %s, not %s; it will load the model on first call", served, config["model"])


def _extra_request_fields(backend: str) -> dict:
    """Request fields only mlx_vlm.server understands; none for Hugging Face."""
    if backend != "local":
        return {}
    return {
        "max_pixels": LOCAL_ENHANCEMENT_MAX_PIXELS,
        "repetition_penalty": LOCAL_ENHANCEMENT_REPETITION_PENALTY,
        "enable_thinking": LOCAL_ENHANCEMENT_THINKING,
    }


def _build_enhancement_prompt(text: str, tables: list[str], document: str = "") -> str:
    """Build the summary prompt for one chunk of ``document``."""
    prompt = "You are writing a search index entry for one chunk of a technical document.\n\n"
    if document:
        prompt += f"DOCUMENT: {document}\n\n"
    prompt += f"CHUNK TEXT:\n{text}\n\n"
    for i, table in enumerate(tables, start=1):
        prompt += f"TABLE {i}:\n{table}\n\n"
    prompt += (
        "Describe what this chunk contains: its specific facts, numbers and conditions,\n"
        "and what each attached table or figure shows. Name the part number where it\n"
        "helps a search.\n"
        "Do not repeat facts about the whole document (manufacturer, document number,\n"
        "revision dates, the list of part variants) unless this chunk is about them.\n"
        "Use only facts present in the content; do not guess.\n"
        "Then list up to 3 questions this chunk answers and up to 3 other search terms,\n"
        "each specific to this chunk.\n"
        "Plain text, no headings, at most about 200 words."
    )
    return prompt


def create_ai_enhanced_summary(
    client: OpenAI,
    text: str,
    tables: list[str],
    images: list[str],
    backend: str = ENHANCEMENT_BACKEND,
    document: str = "",
) -> str:
    """Ask the summary model for a searchable description of one chunk.

    Args:
        client: From :func:`create_enhancement_client` for the same ``backend``.
        text: Raw OCR text of the chunk.
        tables: HTML tables of the chunk.
        images: Base64 JPEG figures of the chunk; at most
            ``ENHANCEMENT_MAX_IMAGES`` are sent (0 = all).
        backend: Which entry of ``ENHANCEMENT_BACKENDS`` names the model.
        document: What the chunk belongs to, from
            :func:`src.ingestion.extract.document_context`.

    Returns:
        The summary text. A summary cut off by ``ENHANCEMENT_MAX_TOKENS`` is
        kept, with a warning.

    Raises:
        SummaryError: If the call still fails after the client's retries, or
            the model returns no text.
    """
    sent = images[:ENHANCEMENT_MAX_IMAGES] if ENHANCEMENT_MAX_IMAGES else images
    content: list[dict] = [{"type": "text", "text": _build_enhancement_prompt(text, tables, document)}]
    content.extend(
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}} for b64 in sent
    )
    try:
        response = client.chat.completions.create(
            model=ENHANCEMENT_BACKENDS[backend]["model"],
            messages=[{"role": "user", "content": content}],
            temperature=ENHANCEMENT_TEMPERATURE,
            max_tokens=ENHANCEMENT_MAX_TOKENS,
            extra_body=_extra_request_fields(backend) or None,
        )
    except OpenAIError as exc:
        raise SummaryError(f"summary call failed: {type(exc).__name__}: {exc}") from exc

    choice = response.choices[0]
    summary = (choice.message.content or "").strip()
    if not summary:
        raise SummaryError(f"summary model returned no text (finish_reason={choice.finish_reason})")
    if choice.finish_reason == "length":
        logger.warning("Summary cut off at ENHANCEMENT_MAX_TOKENS=%d", ENHANCEMENT_MAX_TOKENS)
    return summary


@traceable(run_type="chain", name="SummariseChunks")
def summarise_chunks(chunks: list, backend: str = ENHANCEMENT_BACKEND, document: str = "") -> list[Document]:
    """Summarise every chunk and wrap each as a LangChain ``Document``.

    A chunk with tables or images gets a model summary; a text-only chunk
    keeps its text as the summary, with no model call.

    Args:
        chunks: Unstructured chunks, in document order.
        backend: ``huggingface`` or ``local``; see ``ENHANCEMENT_BACKENDS``.
        document: What the chunks belong to; named in every summary prompt.

    Returns:
        One Document per chunk: the summary as ``page_content``, and
        ``metadata["original_content"]`` holding ``raw_text``, ``tables_html``
        and ``images_base64``.

    Raises:
        SummaryError: If any chunk's summary fails; no partial list is returned.
    """
    client = create_enhancement_client(backend)
    documents: list[Document] = []
    started = time.monotonic()

    for i, chunk in enumerate(chunks, start=1):
        content = separate_content_types(chunk)
        if content["tables"] or content["images"]:
            t0 = time.monotonic()
            summary = create_ai_enhanced_summary(
                client, content["text"], content["tables"], content["images"], backend, document
            )
            logger.info(
                "  Chunk %d/%d: summarised (%d tables, %d images) in %.1fs",
                i, len(chunks), len(content["tables"]), len(content["images"]), time.monotonic() - t0,
            )
        else:
            summary = content["text"]
            logger.info("  Chunk %d/%d: text only, kept as is", i, len(chunks))

        documents.append(
            Document(
                page_content=summary,
                metadata={
                    "original_content": json.dumps(
                        {
                            "raw_text": content["text"],
                            "tables_html": content["tables"],
                            "images_base64": content["images"],
                        }
                    )
                },
            )
        )

    logger.info("Summarised %d chunks with %s in %.0fs", len(documents), backend, time.monotonic() - started)
    return documents
