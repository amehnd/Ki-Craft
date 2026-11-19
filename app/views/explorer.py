"""Chunk Explorer page (``/explorer``): one document's chunks, filtered, one opened in detail."""

from __future__ import annotations

import streamlit as st

from chunk_data import build_chunk_df, get_enhanced_content, get_image_urls, get_raw_text, get_tables_html
from components.chunk_view import render_detail
from components.dataset import require_dataset
from components.filters import apply_filters, render_filters

_PURPOSE = (
    "Dataset-wide metrics, then open any chunk on its own: raw OCR text, the "
    "AI-generated summary, decoded images, parsed tables, full metadata and a "
    "0-100 health score. Filter by search text, images, tables, raw text, "
    "enhanced content or chunk ID."
)


def render() -> None:
    """Draw the page: document picker and filters in the sidebar, chunks in the body."""
    data = require_dataset("Chunk Explorer", _PURPOSE)
    if data is None:
        return
    render_filters()
    st.header("Chunk Explorer")
    page_explorer(data)

def render_dashboard(data: list[dict]) -> None:
    total = len(data)
    total_images = sum(len(get_image_urls(c)) for c in data)
    total_tables = sum(len(get_tables_html(c)) for c in data)
    chunks_with_images = sum(1 for c in data if get_image_urls(c))
    chunks_with_tables = sum(1 for c in data if get_tables_html(c))
    avg_raw = (
        sum(len(get_raw_text(c)) for c in data) / total if total else 0
    )
    avg_enh = (
        sum(len(get_enhanced_content(c)) for c in data) / total if total else 0
    )

    cols = st.columns(7)
    metrics = [
        ("Total Chunks", total),
        ("Total Images", total_images),
        ("Total Tables", total_tables),
        ("Chunks w/ Images", chunks_with_images),
        ("Chunks w/ Tables", chunks_with_tables),
        ("Avg Raw Length", f"{avg_raw:.0f}"),
        ("Avg Enhanced Length", f"{avg_enh:.0f}"),
    ]
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)


def render_chunk_list(data: list[dict]) -> None:
    st.subheader("Chunks")
    df = build_chunk_df(data)
    selection = st.dataframe(
        df,
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
        on_select="rerun",
        selection_mode="single-row",
    )
    if selection and selection.selection.rows:
        idx = selection.selection.rows[0]
        st.session_state.selected_chunk_id = data[idx].get("chunk_id")


def page_explorer(data: list[dict]) -> None:
    render_dashboard(data)

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
            sel = st.selectbox(
                "Select chunk",
                options=[c.get("chunk_id") for c in filtered],
                format_func=lambda x: f"Chunk {x}",
                key="chunk_selector",
                label_visibility="collapsed",
                index=None,
                placeholder="Pick a chunk...",
            )
            if sel is not None:
                st.session_state.selected_chunk_id = sel
                st.rerun()

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


