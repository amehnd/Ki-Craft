"""Session state shared by every page: defaults, kept widget values, and the loaded document.

Streamlit reruns the whole script on every interaction; ``st.session_state``
is the dictionary that survives those reruns for one browser tab. Each page
reads and writes the keys set up here.
"""

from __future__ import annotations

import copy

import streamlit as st

from chunk_data import fetch_document_chunks

# Starting values. Widgets with a key take their first value from here, so
# no widget also passes ``value=`` (Streamlit warns when both are set).
_DEFAULTS: dict = {
    "data": None,
    "document_id": None,
    "selected_chunk_id": None,
    "search_query": "",
    "has_images": False,
    "has_tables": False,
    "has_raw_text": False,
    "has_enhanced": False,
    "chunk_id_filter": "",
    "min_len": 0,
    "max_len": 100_000,
    "llm_custom_model": "",
    "llm_api_key": "",
    "llm_temperature": 0.0,
    "llm_max_tokens": 0,
    "llm_max_images": 3,
    "llm_top_k": 5,
    "use_custom_provider": False,
    "chat_query": "",
    "chat_retrieval": None,
    "chat_answer": None,
    "chat_done": None,
    "chat_error": None,
    "chat_session_usage": {"answers": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0},
}

# Sidebar widgets drawn on only some pages. Streamlit deletes a widget's value
# from session state at the end of any run that does not draw the widget, so
# without this, visiting Home would erase the filters and the API key.
_KEPT_WIDGETS: tuple[str, ...] = (
    "document_selectbox",
    "search_query",
    "has_images",
    "has_tables",
    "has_raw_text",
    "has_enhanced",
    "chunk_id_filter",
    "min_len",
    "max_len",
    "llm_provider",
    "llm_model",
    "llm_custom_model",
    "llm_api_key",
    "llm_temperature",
    "llm_max_tokens",
    "llm_max_images",
    "llm_top_k",
    "use_custom_provider",
    "chat_query",
)


def init_state() -> None:
    """Set every missing session key to its default; existing values are left alone."""
    for key, value in _DEFAULTS.items():
        if key not in st.session_state:
            # A copy: the defaults are shared by every browser tab on this
            # server, and the session usage totals are changed in place.
            st.session_state[key] = copy.deepcopy(value)


def keep_widget_values() -> None:
    """Keep the sidebar widgets' values across pages that do not draw them.

    Assigning a key to itself turns a widget's value into an ordinary session
    value, which Streamlit does not delete; the widget picks it up again the
    next time a page draws it. This is Streamlit's documented way to keep
    widget values across a multipage app. Call it once per run, before any
    widget is drawn.
    """
    for key in _KEPT_WIDGETS:
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]


def load_dataset(document_id: str, sync_sidebar: bool = True) -> None:
    """Load one document's chunks from the API into the session.

    A failed request is shown as an error and leaves the session unchanged.

    Args:
        document_id: The document's name, such as ``lm317``.
        sync_sidebar: True when the load came from a page button, so the
            sidebar picker is set to match on the next run.
    """
    try:
        chunks = fetch_document_chunks(document_id)
    except Exception as exc:
        st.error(f"Could not load {document_id} from the API: {exc}")
        return
    st.session_state.data = chunks
    st.session_state.document_id = document_id
    st.session_state.selected_chunk_id = None
    if sync_sidebar:
        # Applied on the next run, before the sidebar selectbox is created, so
        # the sidebar stays in sync with what was loaded from a page button.
        st.session_state.pending_document = document_id
