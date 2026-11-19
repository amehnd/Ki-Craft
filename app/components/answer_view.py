"""How an answer's evidence is drawn: retrieved chunks, figures and usage.

Used by the Query page. The data comes from the API's ``retrieval`` and
``done`` stream events (or a ``POST /v1/query`` response).
"""

from __future__ import annotations

from typing import Optional

import streamlit as st

from components.chunk_view import render_chunk_payload

def render_retrieval(retrieval: dict) -> None:
    """Render the ``chunks`` of a retrieve/query API response."""
    chunks = retrieval.get("chunks", [])
    with st.expander(f"**Referenced Chunks** — {len(chunks)} chunks", expanded=True):
        for i, payload in enumerate(chunks):
            render_chunk_payload(payload, i)


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


def render_figures(figures: list[dict], citations: dict) -> None:
    """Show the figures the model saw under the answer, cited ones first.

    A citation that matches no chunk or sent figure (``citations.unknown``)
    is shown as a warning, since the answer points at evidence that was
    never in the prompt.
    """
    unknown = (citations or {}).get("unknown") or []
    if unknown:
        st.warning(
            f"The answer cites {', '.join(unknown)}, which match no chunk or "
            "figure the model was given."
        )
    if not figures:
        return
    st.markdown("### Figures the model saw")
    ordered = cited_first(figures, citations)
    columns = st.columns(min(len(ordered), 3))
    for index, (figure, cited) in enumerate(ordered):
        state = "cited" if cited else "not cited"
        caption = f"{figure['label']} ({state}): {figure['source']}, chunk C{figure['chunk_id']}"
        with columns[index % len(columns)]:
            st.image(figure["url"], caption=caption, width="stretch")


def remember_answer(result: dict) -> None:
    """Keep a finished answer's text, figures, citations and usage for later reruns."""
    st.session_state.chat_answer = result.get("answer", "")
    st.session_state.chat_figures = result.get("figures") or []
    st.session_state.chat_citations = result.get("citations") or {}
    st.session_state.chat_usage = result.get("usage")


def render_usage(usage: Optional[dict]) -> None:
    """Render token usage and generation stats returned by the backend."""
    if not usage:
        return

    st.markdown("#### Usage")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Model", usage.get("model", "—"))
    col_b.metric("Input tokens", usage.get("input_tokens", 0))
    col_c.metric("Output tokens", usage.get("output_tokens", 0))
    col_d.metric("Total tokens", usage.get("total_tokens", 0))

    latency = usage.get("latency_ms")
    latency_label = f"{latency:,.0f} ms" if isinstance(latency, (int, float)) else "—"
    cost = usage.get("estimated_cost_usd")
    cost_label = f"${cost:.6f}" if isinstance(cost, (int, float)) else "—"

    col_e, col_f, col_g, col_h = st.columns(4)
    col_e.metric("Latency", latency_label)
    col_f.metric("Prompt", f"{usage.get('prompt_chars', 0)} chars")
    col_g.metric("Images", usage.get("prompt_images", 0))
    col_h.metric("Est. cost", cost_label)


