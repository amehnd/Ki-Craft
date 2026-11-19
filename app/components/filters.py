"""Sidebar filters for the Chunk Explorer, and the function that applies them.

The filter widgets write to ``st.session_state``; ``apply_filters`` reads the
same keys, so the two always agree.
"""

from __future__ import annotations

import streamlit as st

from chunk_data import get_enhanced_content, get_image_urls, get_raw_text, get_tables_html

def apply_filters(data: list[dict]) -> list[dict]:
    """Return the chunks that pass every sidebar filter, in their original order."""
    result = list(data)

    q = st.session_state.search_query.strip().lower()
    if q:
        filtered = []
        for c in result:
            haystack = (
                get_enhanced_content(c) + " " + get_raw_text(c)
            ).lower()
            if q in haystack:
                filtered.append(c)
        result = filtered

    if st.session_state.has_images:
        result = [c for c in result if get_image_urls(c)]
    if st.session_state.has_tables:
        result = [c for c in result if get_tables_html(c)]
    if st.session_state.has_raw_text:
        result = [c for c in result if get_raw_text(c)]
    if st.session_state.has_enhanced:
        result = [c for c in result if get_enhanced_content(c)]

    cid = st.session_state.chunk_id_filter.strip()
    if cid:
        try:
            num = int(cid)
            result = [c for c in result if c.get("chunk_id") == num]
        except ValueError:
            pass

    min_l = st.session_state.min_len
    max_l = st.session_state.max_len
    result = [
        c for c in result if min_l <= len(get_raw_text(c)) <= max_l
    ]

    return result


def on_filter_change() -> None:
    """Close the opened chunk: it may no longer be in the filtered list."""
    st.session_state.selected_chunk_id = None


def render_filters() -> None:
    """Draw the search box and filters in the sidebar (Chunk Explorer only)."""
    with st.sidebar:
        st.divider()
        st.text_input(
            "Search",
            key="search_query",
            placeholder="Search enhanced / raw text ...",
            on_change=on_filter_change,
        )
        st.subheader("Filters")
        col1, col2 = st.columns(2)
        with col1:
            st.checkbox("Has Images", key="has_images", on_change=on_filter_change)
            st.checkbox("Has Tables", key="has_tables", on_change=on_filter_change)
        with col2:
            st.checkbox("Has Raw Text", key="has_raw_text", on_change=on_filter_change)
            st.checkbox("Has Enhanced", key="has_enhanced", on_change=on_filter_change)
        st.text_input("Chunk ID", key="chunk_id_filter", on_change=on_filter_change)
        st.number_input(
            "Min raw text length", min_value=0, max_value=100_000, key="min_len", on_change=on_filter_change
        )
        st.number_input(
            "Max raw text length", min_value=0, max_value=100_000, key="max_len", on_change=on_filter_change
        )


