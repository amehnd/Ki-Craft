"""How an answer's evidence is drawn on the Query page: chunks, figures and usage.

The data comes from the ``/v1/query/stream`` events: ``retrieval`` carries
the chunks, ``done`` the answer, its ``figures``, ``citations`` and
``usage``. Each function draws one section into whatever container the
caller opens, so the page can fill sections in place while the answer
streams, and draw them again from session state on later reruns.
"""

from __future__ import annotations

import streamlit as st

from components.chunk_view import render_retrieved_chunk


def render_chunks(chunks: list[dict], cited_chunks: list[int]) -> None:
    """Draw the retrieved chunks in one collapsed panel, each marked when the answer cites it.

    Args:
        chunks: The API's ``chunks`` list, most relevant first.
        cited_chunks: ``citations.chunks``, the ``chunk_id`` of each cited
            chunk; empty while the answer is still being written.
    """
    cited_count = sum(1 for chunk in chunks if chunk.get("chunk_id") in cited_chunks)
    label = f"Retrieved chunks ({len(chunks)})" + (f" · {cited_count} cited" if cited_count else "")
    # One collapsed panel holds the chunks, each its own panel inside it, so
    # the answer stays near the top and you open only the chunks you want.
    with st.expander(label):
        for chunk in chunks:
            render_retrieved_chunk(chunk, cited=chunk.get("chunk_id") in cited_chunks)


def cited_first(figures: list[dict], citations: dict) -> list[tuple[dict, bool]]:
    """Order figures for display: cited ones first, then the rest.

    Args:
        figures: The response's ``figures``, in prompt order.
        citations: The response's ``citations``; its ``figures`` lists the
            cited labels in the order the answer first cites them.

    Returns:
        ``(figure, cited)`` pairs: cited figures in citation order, then the
        uncited ones in prompt order.
    """
    cited_labels = (citations or {}).get("figures") or []
    by_label = {figure["label"]: figure for figure in figures}
    cited = [(by_label[label], True) for label in cited_labels if label in by_label]
    rest = [(figure, False) for figure in figures if figure["label"] not in cited_labels]
    return cited + rest


def render_figure_row(title: str, figures: list[dict]) -> None:
    """Draw figures three to a row under a heading; nothing when there are none.

    Each caption gives the label the answer uses, the document and the chunk.
    """
    if not figures:
        return
    st.markdown(f"#### {title}")
    columns = st.columns(3)
    for index, figure in enumerate(figures):
        with columns[index % 3]:
            st.image(
                figure["url"],
                caption=f"{figure['label']}: {figure['source']}, chunk C{figure['chunk_id']}",
                width="stretch",
            )


def render_unknown_citations(citations: dict) -> None:
    """Warn about cited labels that match no chunk or figure the model was given."""
    unknown = (citations or {}).get("unknown") or []
    if unknown:
        st.warning(
            f"The answer cites {', '.join(unknown)}, which match no chunk or "
            "figure the model was given."
        )


def render_usage(usage: dict | None, session: dict) -> None:
    """Draw this answer's model, tokens, latency and cost, then the session's totals.

    Args:
        usage: The ``done`` event's ``usage``, or None before an answer.
        session: Running totals for this browser tab: ``answers``,
            ``input_tokens``, ``output_tokens`` and ``cost``.
    """
    if not usage:
        return
    latency = usage.get("latency_ms")
    cost = usage.get("estimated_cost_usd")
    cols = st.columns(4)
    cols[0].metric("Input tokens", f"{usage.get('input_tokens', 0):,}")
    cols[1].metric("Output tokens", f"{usage.get('output_tokens', 0):,}")
    cols[2].metric("Latency", f"{latency / 1000:.1f} s" if isinstance(latency, (int, float)) else "-")
    cols[3].metric("Est. cost", f"${cost:.5f}" if isinstance(cost, (int, float)) else "-")
    st.caption(
        f"Answered by {usage.get('model', 'the default model')}. "
        f"This session: {session['answers']} answer{'s' if session['answers'] != 1 else ''}, "
        f"{session['input_tokens'] + session['output_tokens']:,} tokens, "
        f"about ${session['cost']:.5f}."
    )
