"""Chunk data for the dashboard: cached API reads and plain helpers over chunk dicts.

Every page reads documents and chunks through here. The reads call
``api_client`` (HTTP only) and are cached with ``st.cache_data`` so that
Streamlit's rerun on every click does not hit the API again. The helpers
below them take one chunk dict, as ``GET /v1/documents/{id}/chunks`` returns
it, and compute nothing that needs the network.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

import pandas as pd
import streamlit as st

from api_client import API_BASE_URL
from api_client import document_chunks as api_document_chunks
from api_client import documents as api_documents
from api_client import providers as api_providers

logger = logging.getLogger(__name__)

DATASET_LABELS: dict[str, str] = {
    "lm555": "LM555 Timer",
}


def display_name(document_id: str) -> str:
    """Human-friendly label for a document: its name, upper case."""
    return DATASET_LABELS.get(document_id) or document_id.upper()


@st.cache_data(ttl=60, show_spinner=False)
def list_document_summaries() -> list[dict]:
    """The indexed documents with their chunk, figure and table counts, from the API.

    Cached for a minute: Streamlit reruns the script on every click, and the
    list changes only when a document is ingested. A failure raises, and
    Streamlit never caches an exception, so the next run asks the API again.
    """
    return api_documents()


def list_documents() -> list[str]:
    """Ids of the indexed documents, such as ``lm317``. A failure raises."""
    return [document["id"] for document in list_document_summaries()]


@st.cache_data(ttl=300, show_spinner=False)
def list_providers() -> list[dict]:
    """The providers and models the API accepts. A failure raises and is not cached."""
    return api_providers()


def document_ids_or_error() -> tuple[list[str], str | None]:
    """The indexed document ids, or an empty list and why the API could not list them."""
    try:
        return list_documents(), None
    except Exception as exc:
        logger.warning("Could not list documents from %s: %s", API_BASE_URL, exc)
        return [], f"`GET {API_BASE_URL}/v1/documents` failed: {exc}"


@st.cache_data(ttl=60, show_spinner="Loading chunks from the API ...")
def fetch_document_chunks(document_id: str) -> list[dict]:
    """Every chunk of one document, from the API, in reading order."""
    return api_document_chunks(document_id)


def get_image_urls(chunk: dict) -> list[str]:
    """Absolute URLs of the chunk's figures, served by the API."""
    return chunk.get("image_urls") or []


def get_tables_html(chunk: dict) -> list[str]:
    """The chunk's tables as HTML strings."""
    return chunk.get("tables_html") or []


def get_raw_text(chunk: dict) -> str:
    """The chunk's text as extracted from the PDF."""
    return chunk.get("raw_text") or ""


def chunk_fields(chunk: dict) -> dict:
    """The chunk's identifying fields, without its long text, tables and figures."""
    return {k: v for k, v in chunk.items() if k not in {"enhanced_content", "raw_text", "tables_html"}}


def get_enhanced_content(chunk: dict) -> str:
    return chunk.get("enhanced_content") or ""


def word_count(text: str) -> int:
    return len(text.split())


def health_score(chunk: dict) -> tuple[int, str]:
    score = 0
    if get_enhanced_content(chunk):
        score += 25
    if get_raw_text(chunk):
        score += 25
    if get_image_urls(chunk):
        score += 25
    if get_tables_html(chunk):
        score += 25

    if score >= 75:
        label = "Pass"
    elif score >= 50:
        label = "Partial"
    else:
        label = "Fail"
    return score, label


def parse_table_html(html: str) -> Optional[pd.DataFrame]:
    try:
        tables = pd.read_html(io.StringIO(html))
        if tables:
            return tables[0]
    except Exception:
        pass
    return None


def build_chunk_df(data: list[dict]) -> pd.DataFrame:
    rows = []
    for c in data:
        cid = c.get("chunk_id", "?")
        n_img = len(get_image_urls(c))
        n_tbl = len(get_tables_html(c))
        rlen = len(get_raw_text(c))
        elen = len(get_enhanced_content(c))
        score, _ = health_score(c)
        rows.append(
            {
                "Chunk ID": cid,
                "Images": n_img,
                "Tables": n_tbl,
                "Raw Len": rlen,
                "Enh Len": elen,
                "Health": score,
            }
        )
    return pd.DataFrame(rows)


