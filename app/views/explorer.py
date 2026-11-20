"""Chunk Explorer page (``/explorer``): one document's chunks, filtered, one opened in detail."""

from __future__ import annotations

import streamlit as st

from chunk_data import build_chunk_df
from components.chunk_view import render_detail
from components.dataset import render_totals, require_dataset
from components.filters import apply_filters, render_filters

_PURPOSE = (
    "The document's totals, then open any chunk on its own: the text extracted "
    "from the PDF, the summary written at ingestion, its figures and tables, its "
    "metadata and a 0-100 health score. Filter by search text, figures, tables, "
    "original text, summary or chunk ID."
)


def render() -> None:
    """Draw the page: document picker and filters in the sidebar, chunks in the body."""
    data = require_dataset("Chunk Explorer", _PURPOSE)
    if data is None:
        return
    render_filters()
    st.header("Chunk Explorer")
    page_explorer(data)


def _open_selected_row(data: list[dict]) -> None:
    """Open the chunk of the table row just selected (an ``on_select`` callback)."""
    rows = st.session_state.chunk_table.selection.rows
    if rows:
        st.session_state.selected_chunk_id = data[rows[0]].get("chunk_id")


def _open_picked_chunk() -> None:
    """Open the chunk just picked in the "Pick a chunk" box (an ``on_change`` callback)."""
    if st.session_state.chunk_selector is not None:
        st.session_state.selected_chunk_id = st.session_state.chunk_selector


def render_chunk_list(data: list[dict]) -> None:
    """Draw the chunk table; selecting a row opens that chunk in the detail panel.

    Both ways to open a chunk, a table row and the "Pick a chunk" box, set
    ``selected_chunk_id`` in a callback. Streamlit runs a callback before the
    rerun the click causes, so the detail panel shows the chunk in that same
    run, and whichever was used last wins.
    """
    st.subheader("Chunks")
    st.dataframe(
        build_chunk_df(data),
        width="stretch",
        hide_index=True,
        column_config={
            "Chunk ID": st.column_config.NumberColumn(width="small"),
            "Images": st.column_config.NumberColumn(width="small"),
            "Tables": st.column_config.NumberColumn(width="small"),
            "Raw Len": st.column_config.NumberColumn(width="medium"),
            "Enh Len": st.column_config.NumberColumn(width="medium"),
            "Health": st.column_config.ProgressColumn(
                format="$score / 100",
                min_value=0,
                max_value=100,
                width="small",
            ),
        },
        key="chunk_table",
        # st.dataframe passes no arguments to its callback, so the rows are bound here.
        on_select=lambda: _open_selected_row(data),
        selection_mode="single-row",
    )


def page_explorer(data: list[dict]) -> None:
    """Draw the totals, the filtered chunk list and the opened chunk side by side."""
    render_totals(build_chunk_df(data))

    filtered = apply_filters(data)
    st.info(f"Showing {len(filtered)} of {len(data)} chunks")

    if not filtered:
        st.warning("No chunks match the current filters.")
        return

    list_col, detail_col = st.columns([1, 1.8])

    with list_col:
        render_chunk_list(filtered)
        cid_filter = st.session_state.chunk_id_filter.strip()
        if not cid_filter:
            st.selectbox(
                "Select chunk",
                options=[c.get("chunk_id") for c in filtered],
                format_func=lambda x: f"Chunk {x}",
                key="chunk_selector",
                label_visibility="collapsed",
                index=None,
                placeholder="Pick a chunk...",
                on_change=_open_picked_chunk,
            )

    with detail_col:
        selected_id = st.session_state.selected_chunk_id
        if selected_id is not None:
            chunk = next(
                (c for c in data if c.get("chunk_id") == selected_id), None
            )
            if chunk:
                with st.container(border=True):
                    render_detail(chunk)
            else:
                st.info("Select a chunk from the list.")


