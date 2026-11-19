"""Compare Chunks page (``/compare``): two chunks of one document side by side."""

from __future__ import annotations

import streamlit as st

from components.chunk_view import render_chunk_card
from components.dataset import require_dataset

_PURPOSE = (
    "Put any two chunks side by side &mdash; useful to see how one visual region "
    "got summarised, and how a multimodal chunk differs from a text-only one."
)


def render() -> None:
    """Draw the page: document picker in the sidebar, two chunks in the body."""
    data = require_dataset("Compare Chunks", _PURPOSE)
    if data is not None:
        render_compare(data)

def render_compare(data: list[dict]) -> None:
    """Draw two chunk pickers and the chosen chunks side by side."""
    st.header("Compare Chunks")

    ids = [c.get("chunk_id") for c in data]

    col_a, col_b = st.columns(2)
    with col_a:
        a_id = st.selectbox("Chunk A", ids, key="cmp_a")
    with col_b:
        b_id = st.selectbox("Chunk B", ids, key="cmp_b")

    chunk_a = next((c for c in data if c.get("chunk_id") == a_id), None)
    chunk_b = next((c for c in data if c.get("chunk_id") == b_id), None)

    if not chunk_a or not chunk_b:
        st.warning("Select two chunks to compare.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Chunk {a_id}")
        render_chunk_card(chunk_a)
    with col2:
        st.subheader(f"Chunk {b_id}")
        render_chunk_card(chunk_b)


