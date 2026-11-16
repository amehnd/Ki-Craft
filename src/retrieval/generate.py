"""Answer generation using a multimodal LLM.

Builds the prompt from retrieved chunks (enhanced summaries + HTML tables +
base64 images), streams an answer from the chat model it is given, and wraps
both retrieval and generation under a single LangSmith trace so the whole
query shows up as one run in the UI. The default model and a user-chosen
model share this one path; only the ``llm`` passed in differs.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Iterator
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langsmith import traceable

from config.settings import (
    GEMINI_API_KEY,
    GENERATION_INPUT_PRICE_PER_1M,
    GENERATION_MAX_IMAGES,
    GENERATION_MAX_TOKENS,
    GENERATION_MODEL,
    GENERATION_OUTPUT_PRICE_PER_1M,
    GENERATION_PROVIDER,
    GENERATION_TEMPERATURE,
)
from src.documents import chunk_text_and_tables, extract_original_data
from src.llm import LLMProviderError, create_llm
from src.logger import get_logger
from src.retrieval.search import retrieve_chunks
from src.tracing import summarize_chunks

logger = get_logger(__name__)

# One item from answer_query_stream: ("chunks", list[Document]),
# ("token", str) or ("usage", dict).
StreamEvent = tuple[str, Any]


def _capture_usage(obj: Any, usage: dict[str, Any]) -> None:
    """Merge token counts reported by the LLM into ``usage``.

    ``obj`` is the AIMessage/AIMessageChunk returned by the provider.  Usage
    may arrive on ``usage_metadata`` or nested in ``response_metadata``; for
    streams it is typically present on every chunk (cumulative), so later
    chunks overwrite earlier ones.
    """
    raw = getattr(obj, "usage_metadata", None)
    if not raw:
        raw = (getattr(obj, "response_metadata", None) or {}).get("usage_metadata")
    if not raw:
        return
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        if key in raw and isinstance(raw[key], int):
            usage[key] = raw[key]


def _estimate_cost(usage: dict[str, Any]) -> float | None:
    """Estimate generation cost (USD) from token usage and configured pricing."""
    if not GENERATION_INPUT_PRICE_PER_1M and not GENERATION_OUTPUT_PRICE_PER_1M:
        return None
    inp = usage.get("input_tokens") or 0
    out = usage.get("output_tokens") or 0
    cost = (
        inp / 1_000_000 * GENERATION_INPUT_PRICE_PER_1M
        + out / 1_000_000 * GENERATION_OUTPUT_PRICE_PER_1M
    )
    return round(cost, 6)


def _prompt_stats(message_content: list[dict]) -> tuple[int, int]:
    """Return ``(text_characters, image_count)`` of a message content list."""
    text = next((x["text"] for x in message_content if x["type"] == "text"), "")
    images = sum(1 for x in message_content if x["type"] == "image_url")
    return len(text), images


def _extract_text(content: Any) -> str:
    """Extract plain text from LLM response content (string or list of parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def _build_text_prompt(chunks: list[Document], query: str, with_images: bool = True) -> str:
    """Build the text portion of the prompt from retrieved chunks.

    Sends enhanced summaries as primary context, with HTML tables appended,
    for every chunk given; the prompt size is set by how many chunks were
    retrieved (``top_k``). Images are sent separately via
    :func:`_collect_images`. When ``with_images`` is false (a text-only
    model), the instructions do not mention figures.
    """
    if with_images:
        contents = "- optional HTML tables,\n- optional attached figures/images.\n\n"
        use = "Use both the text summaries and the attached images when answering.\n"
    else:
        contents = "- optional HTML tables.\n\n"
        use = "Use the text summaries and tables when answering.\n"
    parts = [
        "You are given retrieved chunks from a technical document.\n"
        "Each chunk contains:\n"
        "- a searchable summary of the original content,\n"
        + contents
        + "Treat the summaries as authoritative representations of the document.\n"
        "Do NOT claim information is absent unless you have examined ALL provided chunks.\n"
        "If a chunk explicitly contains the answer, answer directly using that chunk.\n"
        + use,
        f"QUESTION: {query}\n",
        "RETRIEVED CONTEXT:",
        "",
    ]

    for i, chunk in enumerate(chunks):
        enhanced = chunk.page_content
        if enhanced:
            parts.append(f"--- Chunk {i + 1} ---")
            parts.append(f"SUMMARY:\n{enhanced.strip()}\n")

        tables_html = chunk_text_and_tables(chunk)["tables_html"]
        if tables_html:
            parts.append("TABLES:")
            for j, table in enumerate(tables_html):
                parts.append(f"Table {j + 1}:\n{table}\n")

        parts.append("")

    if with_images:
        parts.append(
            "The attached images are figures from the document. "
            "Use the summaries, tables, and images together to provide a detailed answer."
        )
    else:
        parts.append("Use the summaries and tables together to provide a detailed answer.")

    return "\n".join(parts)


def _collect_images(chunks: list[Document], max_images: int) -> list[dict]:
    """Collect base64 images from chunks into message content blocks.

    Images are taken in chunk order (all of chunk 1's, then chunk 2's, ...)
    until ``max_images`` is reached. 0 means no limit (send all).
    """
    image_blocks: list[dict] = []
    for chunk in chunks:
        for b64 in extract_original_data(chunk)["images_base64"]:
            if max_images and len(image_blocks) >= max_images:
                return image_blocks
            image_blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )
    return image_blocks


def _build_message_content(
    chunks: list[Document], query: str, max_images: int, send_images: bool = True
) -> list[dict]:
    """Build the full multimodal message content list (text + images).

    ``max_images`` caps how many images are sent to the model (0 = all).
    ``send_images`` is false for a text-only model: no image blocks at all.
    """
    content: list[dict] = [{"type": "text", "text": _build_text_prompt(chunks, query, send_images)}]
    if send_images:
        content.extend(_collect_images(chunks, max_images))
    return content


# What providers report when a reply stopped at max_tokens: OpenAI-style
# APIs (OpenAI, Groq, Hugging Face) say "length", Gemini says "MAX_TOKENS".
_OUT_OF_BUDGET = {"length", "MAX_TOKENS"}


def _empty_answer_message(finish_reason: str | None, usage: dict[str, Any]) -> str:
    """Explain why the model returned no answer text."""
    if finish_reason in _OUT_OF_BUDGET:
        return (
            f"The model used its whole output budget ({usage.get('output_tokens', '?')} "
            "tokens, reasoning included) before writing any answer text. "
            "Raise max_tokens and try again."
        )
    return f"The model returned no answer text (finish_reason: {finish_reason or 'not reported'})."


def default_llm() -> BaseChatModel:
    """Create the default chat model configured in ``config/settings.py``.

    Its model id is ``GENERATION_MODEL``, which callers pass alongside it as
    the ``model_id`` that labels usage.
    """
    return create_llm(
        GENERATION_PROVIDER,
        GENERATION_MODEL,
        GEMINI_API_KEY,
        temperature=GENERATION_TEMPERATURE,
        max_tokens=GENERATION_MAX_TOKENS,
    )


def _collect_image_attachments(chunks: list[Document]) -> dict:
    """Gather images from chunks as binary attachments for LangSmith traces.

    Uses :func:`extract_original_data` to resolve either storage schema
    (file-path or inline base64) to image bytes.
    """
    attachments: dict = {}
    img_idx = 0
    for chunk in chunks:
        for b64_str in extract_original_data(chunk)["images_base64"]:
            try:
                attachments[f"chunk_img_{img_idx}"] = ("image/jpeg", base64.b64decode(b64_str))
                img_idx += 1
            except Exception:
                logger.warning("Could not decode image attachment for trace")
    return attachments


def _attach_images_to_trace(run_tree, chunks: list[Document]) -> None:
    """Attach retrieved chunk images to ``run_tree`` if any are available."""
    if run_tree is None:
        return
    attachments = _collect_image_attachments(chunks)
    if attachments:
        run_tree.attachments = attachments


def _trace_inputs(inputs: dict) -> dict:
    """Choose what LangSmith records as a generation run's inputs.

    The retriever, the chunks and the chat model object are left out: they
    serialise to long reprs, and the model object carries the caller's key.
    The question is the only text input, so the LangSmith runs table previews
    it; the model id goes into run metadata instead (see
    :func:`answer_query_stream`).
    """
    recorded = {
        "question": inputs.get("query"),
        "max_images": inputs.get("max_images"),
        "send_images": inputs.get("send_images"),
    }
    if "chunks" in inputs:
        recorded["num_chunks"] = len(inputs["chunks"])
    return recorded


def _answer_output(tokens: list[str]) -> dict:
    """Turn the tokens a generation run yielded into its recorded output."""
    return {"answer": "".join(tokens)}


def _query_output(events: list[StreamEvent]) -> dict:
    """Turn the events a query run yielded into its recorded output.

    The chunks are left out: the nested ``RetrieveChunks`` run records them.
    """
    return {
        "answer": "".join(value for kind, value in events if kind == "token"),
        "usage": next((value for kind, value in events if kind == "usage"), {}),
    }


@traceable(
    run_type="chain",
    name="AnswerQuery",
    process_inputs=_trace_inputs,
    reduce_fn=_query_output,
    dangerously_allow_filesystem=True,
)
def answer_query_stream(
    retriever,
    query: str,
    llm: BaseChatModel,
    model_id: str,
    *,
    max_images: int = GENERATION_MAX_IMAGES,
    send_images: bool = True,
    run_tree=None,
) -> Iterator[StreamEvent]:
    """Retrieve chunks then stream an answer under a single LangSmith trace.

    This is the one generation path: the default model and a user-chosen
    model both come through here, differing only in the ``llm`` passed in.
    It is a generator, so LangSmith keeps the ``AnswerQuery`` run open until
    the last event and records the full answer as its output. ``run_tree`` is
    injected by LangSmith.

    Args:
        retriever: Vector store retriever.
        query: The user's question.
        llm: The chat model to answer with (see ``src/llm.py``).
        model_id: The model's catalogue id, recorded in ``usage``.
        max_images: Most images to send to the model (0 = all).
        send_images: False for a text-only model; no figures are sent.

    Yields:
        ``("chunks", list[Document])`` once, as soon as retrieval finishes;
        then ``("token", str)`` for each piece of the answer; then
        ``("usage", dict)`` once, with token counts, latency and prompt size.

    Raises:
        LLMProviderError: From ``generate_answer_stream``, if the provider
            fails or writes no answer text.
    """
    if run_tree is not None:
        run_tree.add_metadata({"model": model_id})
    chunks = retrieve_chunks(retriever, query)
    _attach_images_to_trace(run_tree, chunks)
    yield "chunks", chunks

    usage: dict[str, Any] = {}
    for token in generate_answer_stream(
        chunks,
        query,
        llm,
        model_id,
        usage=usage,
        max_images=max_images,
        send_images=send_images,
    ):
        yield "token", token
    yield "usage", usage


@traceable(
    run_type="chain",
    name="GenerateAnswerStream",
    process_inputs=_trace_inputs,
    reduce_fn=_answer_output,
)
def generate_answer_stream(
    chunks: list[Document],
    query: str,
    llm: BaseChatModel,
    model_id: str,
    *,
    usage: dict[str, Any] | None = None,
    max_images: int = GENERATION_MAX_IMAGES,
    send_images: bool = True,
) -> Iterator[str]:
    """Stream an answer from ``llm`` for already retrieved chunks.

    Traced as a ``chain`` run that stays open until the last token. The model
    call itself appears under it as LangChain's own ``llm`` run, which carries
    the provider, model name and token counts.

    Args:
        chunks: Retrieved document chunks.
        query: The original user query.
        llm: The chat model to answer with.
        model_id: The model's catalogue id, recorded in ``usage``.
        usage: Optional shared dict filled in place with token counts,
            latency and prompt size as the stream runs; a fresh one is used
            when omitted.
        max_images: Most images to send to the model (0 = all).
        send_images: False for a text-only model; no figures are sent.

    Yields:
        Answer token strings.

    Raises:
        LLMProviderError: If the provider call fails, or the model finishes
            without writing any answer text (for example, it spent its whole
            ``max_tokens`` budget on reasoning).
    """
    if usage is None:
        usage = {}

    chunk_summaries = summarize_chunks(chunks)
    total_images = sum(c.get("image_count", 0) for c in chunk_summaries)
    total_tables = sum(c.get("table_count", 0) for c in chunk_summaries)

    logger.info("  %d chunks received by generation", len(chunks))
    for ci, (c, summary) in enumerate(zip(chunks, summarize_chunks(chunks, max_preview=len(chunks)))):
        logger.debug(
            "    Chunk %d: summary=%d chars, tables=%d, images=%d",
            ci + 1,
            len(c.page_content or ""),
            summary["table_count"],
            summary["image_count"],
        )

    message_content = _build_message_content(chunks, query, max_images, send_images)

    prompt_chars, prompt_images = _prompt_stats(message_content)
    usage.update(
        {
            "model": model_id,
            "prompt_chars": prompt_chars,
            "prompt_images": prompt_images,
            "chunks_used": len(chunks),
        }
    )

    logger.info(
        "LangSmith trace: %d chunks, %d images, %d tables",
        len(chunks),
        total_images,
        total_tables,
    )

    message = HumanMessage(content=message_content)
    start = time.perf_counter()
    wrote_text = False
    finish_reason = None
    try:
        for chunk in llm.stream([message]):
            token_text = _extract_text(chunk.content)
            if token_text:
                wrote_text = True
                yield token_text
            _capture_usage(chunk, usage)
            finish_reason = (chunk.response_metadata or {}).get("finish_reason") or finish_reason
    except Exception as exc:
        logger.warning("Model provider failed: %s", type(exc).__name__)
        raise LLMProviderError(f"The model provider failed: {exc}") from exc
    usage["latency_ms"] = round((time.perf_counter() - start) * 1000, 1)
    usage["estimated_cost_usd"] = _estimate_cost(usage)

    if not wrote_text:
        raise LLMProviderError(_empty_answer_message(finish_reason, usage))

