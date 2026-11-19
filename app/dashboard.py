"""KI-CRAFT dashboard: the entry point that sets up the app and routes to its pages.

Run with ``streamlit run app/dashboard.py`` (Streamlit Community Cloud uses
this file as its main file). Streamlit runs this script from the top on every
click. Each run sets the page config and styles, prepares session state, and
hands over to ``st.navigation``, which reads the URL and runs one page:

    /            views/home.py       what KI-CRAFT is and how it works
    /explorer    views/explorer.py   one document's chunks, filtered
    /compare     views/compare.py    two chunks side by side
    /analytics   views/analytics.py  distributions over a document
    /chat        views/chat.py       ask a question, stream the answer

The pages reach the backend only over HTTP, through ``api_client.py``.
"""

from __future__ import annotations

import logging
import os
import sys
from functools import partial
from pathlib import Path

import streamlit as st

# Only app/ is on the path, so pages import api_client, chunk_data, state,
# components and views; config/, src/ and api/ stay out of reach.
_APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_APP_DIR))


def _merge_streamlit_secrets() -> None:
    """Copy Streamlit secrets into environment variables that are not set yet.

    Locally ``api_client`` reads ``API_BASE_URL`` from ``.env``. On Streamlit
    Cloud the value lives in ``st.secrets``; copying it into ``os.environ``
    before ``api_client`` is imported lets that one code path serve both.
    """
    try:
        for key, value in st.secrets.items():
            os.environ.setdefault(str(key), str(value))
    except Exception:
        pass


_merge_streamlit_secrets()

from state import init_state, keep_widget_values  # noqa: E402  (needs the secrets above)
from views import analytics, chat, compare, explorer, home  # noqa: E402

# The dashboard logs to the console only: Streamlit Community Cloud keeps no
# files between restarts. The format matches the backend's src/logger.py.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

PAGE_TITLE = "KI-CRAFT"

# Each page's url_path is the address it is served at, so a page can be
# bookmarked and linked. The default page is served at "/".
EXPLORER_PAGE = st.Page(explorer.render, title="Chunk Explorer", url_path="explorer")
CHAT_PAGE = st.Page(chat.render, title="Query & Retrieve", url_path="chat")
# The home page links to two other pages, and a link needs the page object,
# so they are passed in rather than imported (which would be circular).
HOME_PAGE = st.Page(
    partial(home.render, chat_page=CHAT_PAGE, explorer_page=EXPLORER_PAGE),
    title="Home",
    url_path="home",
    default=True,
)
PAGES = [
    HOME_PAGE,
    EXPLORER_PAGE,
    st.Page(compare.render, title="Compare Chunks", url_path="compare"),
    st.Page(analytics.render, title="Dataset Analytics", url_path="analytics"),
    CHAT_PAGE,
]


def main() -> None:
    """Set up one run of the app and draw the page the URL names."""
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.markdown(f"<style>{(_APP_DIR / 'styles.css').read_text()}</style>", unsafe_allow_html=True)
    init_state()
    keep_widget_values()
    st.navigation(PAGES).run()


if __name__ == "__main__":
    main()
