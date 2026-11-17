"""Answer prompt: the messages sent to the chat model for one question.

Sits between retrieval and generation. ``src/retrieval/generate.py`` passes
the retrieved chunks here and gets back a system message with the answering
rules and a user message holding every chunk (its document, summary, original
text and tables as Markdown), each chosen figure right after its chunk, and
the question last. Nothing here calls a model or the network; figure files
are read from disk only when the message is built.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from src.documents import chunk_text_and_tables, image_file, stored_image_paths
from src.tables import table_to_markdown


@dataclass(frozen=True)
class PromptFigure:
    """One figure sent to the model, labelled ``F<n>`` in prompt order.

    Attributes:
        label: ``F1``, ``F2``, ... as the prompt names it.
        chunk_number: 1-based position of its chunk in the retrieved list.
        position: 0-based index into that chunk's stored figures.
        record_id: The chunk's store record id, or None if it has none.
        source: The chunk's document name.
        path: The figure file, inside ``IMAGES_DIR``. Its bytes are read only
            for the model request and the trace attachment.
    """

    label: str
    chunk_number: int
    position: int
    record_id: str | None
    source: str
    path: Path


def prompt_stats(message_content: list[dict]) -> tuple[int, int]:
    """Return ``(text_characters, image_count)`` of a message content list."""
    text_characters = sum(len(x["text"]) for x in message_content if x["type"] == "text")
    images = sum(1 for x in message_content if x["type"] == "image_url")
    return text_characters, images


_RULES = (
    "You answer questions about electronics documents using only the context provided. "
    "If the context does not contain the answer, say that the documents provided do not state it. "
    "Each chunk names its document; never combine values from different documents "
    "unless the question asks for a comparison. "
    "Give values with their units and test conditions exactly as stated."
)


_FIGURE_RULE = " When you use a figure, cite its label in square brackets, like [F1]."


def _system_prompt(with_figures: bool) -> str:
    """The fixed rules; the figure rule only when figures are sent."""
    return _RULES + (_FIGURE_RULE if with_figures else "")


def _chunk_text(chunk: Document, chunk_number: int) -> str:
    """One chunk as prompt text: its document, summary, original text and tables.

    Tables are sent as Markdown (see ``src/tables.py``).

    A chunk with no tables or figures was stored with its original text as
    its summary, so the text is written once.
    """
    source = (chunk.metadata or {}).get("source") or "unknown document"
    content = chunk_text_and_tables(chunk)
    summary = (chunk.page_content or "").strip()
    raw_text = (content["raw_text"] or "").strip()
    lines = [f"[Chunk {chunk_number}] Document: {source}"]
    if summary and summary != raw_text:
        lines.append(f"Summary:\n{summary}")
    if raw_text:
        lines.append(f"Text:\n{raw_text}")
    tables = [markdown for markdown in map(table_to_markdown, content["tables_html"]) if markdown]
    for number, table in enumerate(tables, start=1):
        lines.append(f"Table {number}:\n{table}")
    return "\n\n".join(lines)


def select_figures(chunks: list[Document], max_images: int) -> list[PromptFigure]:
    """Choose the figures to send to the model, in chunk order.

    Figures are numbered in prompt order. No figure file is read here. A
    figure whose file is missing, or outside ``IMAGES_DIR``, takes no label
    and does not count toward ``max_images``.

    Args:
        chunks: Retrieved document chunks.
        max_images: Most figures to choose; 0 chooses every readable one.

    Returns:
        The chosen figures, labelled ``F1``, ``F2``, ...
    """
    figures: list[PromptFigure] = []
    for chunk_number, chunk in enumerate(chunks, start=1):
        metadata = chunk.metadata or {}
        source = metadata.get("source") or "unknown document"
        for position in range(len(stored_image_paths(metadata))):
            path = image_file(metadata, position)
            if path is None:
                continue
            figures.append(
                PromptFigure(f"F{len(figures) + 1}", chunk_number, position, chunk.id, source, path)
            )
            if len(figures) == max_images:
                return figures
    return figures


def _figure_blocks(figure: PromptFigure) -> list[dict]:
    """A figure's label, then its image read from disk as base64."""
    b64 = base64.b64encode(figure.path.read_bytes()).decode()
    return [
        {"type": "text", "text": f"Figure {figure.label}, from {figure.source}, chunk {figure.chunk_number}"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ]


def _build_message_content(chunks: list[Document], query: str, figures: list[PromptFigure]) -> list[dict]:
    """Build the user message: every chunk with its figures right after it, then the question.

    The context comes first and the question last, which suits long
    multimodal prompts. Every retrieved chunk is included (``top_k`` sets
    the context size); only the figures in ``figures`` are attached.
    """
    content: list[dict] = [{"type": "text", "text": "CONTEXT"}]
    for chunk_number, chunk in enumerate(chunks, start=1):
        content.append({"type": "text", "text": _chunk_text(chunk, chunk_number)})
        for figure in figures:
            if figure.chunk_number == chunk_number:
                content.extend(_figure_blocks(figure))
    content.append({"type": "text", "text": f"QUESTION: {query}"})
    return content


def build_messages(chunks: list[Document], query: str, figures: list[PromptFigure]) -> list[BaseMessage]:
    """The system rules and the user message sent to the model."""
    return [
        SystemMessage(content=_system_prompt(bool(figures))),
        HumanMessage(content=_build_message_content(chunks, query, figures)),
    ]
