"""The document that the Explorer, Compare and Analytics pages work on.

The picker sits in the sidebar of those three pages. ``require_dataset`` is
what each of them calls first: it draws the picker and returns the loaded
document's chunks, or draws an empty state with one button per document and
returns None. ``render_totals`` draws the document's totals, the same on the
Explorer and Analytics pages.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from api_client import API_BASE_URL
from chunk_data import display_name, document_ids_or_error
from state import load_dataset


def render_dataset_picker() -> None:
    """Draw the document picker in the sidebar and load the chosen document."""
    # A document loaded from a page button is shown in the picker. It must be
    # set before the selectbox is created; Streamlit forbids changing a
    # widget's value after it is drawn in the same run.
    pending = st.session_state.pop("pending_document", None)
    if pending:
        st.session_state["document_selectbox"] = pending

    with st.sidebar:
        st.header("Datasheet")
        document_ids, _ = document_ids_or_error()
        if not document_ids:
            st.caption("No documents: the API is unreachable or holds none.")
            return
        selected = st.selectbox(
            "Select a datasheet",
            options=[""] + document_ids,
            format_func=lambda x: "Choose a document..." if x == "" else display_name(x),
            key="document_selectbox",
        )
        if selected and selected != st.session_state.get("document_id"):
            load_dataset(selected, sync_sidebar=False)
            st.rerun()
        data = st.session_state.data
        if data is not None:
            st.caption(f"Loaded {display_name(st.session_state.document_id)}: {len(data)} chunks")


def _render_no_dataset(title: str, purpose: str) -> None:
    """Empty state: what the page does, and one button per indexed document."""
    st.header(title)
    st.markdown(purpose)

    document_ids, error = document_ids_or_error()
    if document_ids:
        st.markdown("#### Select a datasheet")
        cols = st.columns(len(document_ids))
        for col, document_id in zip(cols, document_ids):
            if col.button(display_name(document_id), width="stretch", key=f"load_{document_id}"):
                load_dataset(document_id)
                st.rerun()
    else:
        st.error(
            f"{error or 'The API holds no documents.'} A 404 here usually means the server at "
            f"`{API_BASE_URL}` runs older code than this dashboard; set API_BASE_URL to the right one."
        )
    st.caption("You can also pick a datasheet from the sidebar.")


def require_dataset(title: str, purpose: str) -> list[dict] | None:
    """Draw the sidebar picker and return the loaded document's chunks.

    Args:
        title: The page's title, shown in the empty state.
        purpose: One or two sentences on what the page shows.

    Returns:
        The chunks of the loaded document, or None when no document is
        loaded yet (the empty state has been drawn instead).
    """
    render_dataset_picker()
    data = st.session_state.data
    if data is None:
        _render_no_dataset(title, purpose)
    return data


def render_totals(df: pd.DataFrame) -> None:
    """Draw a document's totals: chunks, figures, tables and average text lengths.

    Args:
        df: The document's per-chunk table from ``chunk_data.build_chunk_df``.
    """
    if df.empty:
        return
    metrics = [
        ("Total Chunks", len(df)),
        ("Total Images", int(df["Images"].sum())),
        ("Total Tables", int(df["Tables"].sum())),
        ("Chunks w/ Images", int((df["Images"] > 0).sum())),
        ("Chunks w/ Tables", int((df["Tables"] > 0).sum())),
        ("Avg Raw Length", f"{df['Raw Len'].mean():.0f}"),
        ("Avg Enhanced Length", f"{df['Enh Len'].mean():.0f}"),
    ]
    for col, (label, value) in zip(st.columns(len(metrics)), metrics):
        col.metric(label, value)
