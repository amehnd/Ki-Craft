"""Thin HTTP client for the FastAPI backend (``api.main``).

Lets the Streamlit dashboard talk to the API instead of loading the vector
store and calling the LLM itself.  Image URLs returned by the backend (each
chunk's ``image_urls`` and each ``figures`` entry's ``url``) are relative,
``/v1/images/{record_id}/{position}``; this client rewrites them to absolute
URLs so ``st.image`` can load them directly.

The backend's address is the client's one setting, read here from the
environment rather than from ``config/settings.py``: the UI shares no code
with the backend, only the HTTP API.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterator

import requests
from dotenv import load_dotenv

# Fills os.environ from the repository's .env without overwriting variables
# that are already set (a shell export, or Streamlit secrets copied earlier).
load_dotenv()

API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")

_REQUEST_TIMEOUT = 300


def _url(path: str) -> str:
    """The full backend URL for an API path such as ``/v1/query``."""
    return f"{API_BASE_URL.rstrip('/')}{path}"


def _raise_for_status(response: requests.Response) -> None:
    """Raise ``requests.HTTPError`` carrying the server's ``detail`` message.

    ``response.raise_for_status`` alone says only "502 Server Error: Bad
    Gateway for url ...". Every API error body is ``{"detail": "..."}``, so the
    message here tells the user why the request failed.
    """
    if response.ok:
        return
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    raise requests.HTTPError(f"{response.status_code}: {detail}", response=response)


def _absolutize_image_urls(payload: dict[str, Any], key: str = "chunks") -> dict[str, Any]:
    """Turn relative ``/v1/images/...`` URLs into absolute backend URLs.

    ``key`` names the list of chunks: ``chunks`` in query and retrieve
    responses, ``items`` in a page of a document's chunks.
    """
    for chunk in payload.get(key, []):
        chunk["image_urls"] = [_url(u) for u in chunk.get("image_urls", [])]
    return payload


def _absolutize_figure_urls(payload: dict[str, Any]) -> dict[str, Any]:
    """Turn each ``figures`` entry's relative ``url`` into an absolute backend URL."""
    for figure in payload.get("figures", []):
        figure["url"] = _url(figure["url"])
    return payload


def providers() -> list[dict]:
    """List the providers a query's ``llm`` object accepts, each with its ``models``."""
    response = requests.get(_url("/v1/providers"), timeout=30)
    _raise_for_status(response)
    return response.json()["items"]


def documents() -> list[dict]:
    """List the indexed documents, each with ``id`` and chunk, figure and table counts."""
    response = requests.get(_url("/v1/documents"), timeout=30)
    _raise_for_status(response)
    return response.json()["items"]


# The API's largest page; fewer requests for the same chunks.
_PAGE_SIZE = 200


def document_chunks(document_id: str) -> list[dict]:
    """Return every chunk of one document, in reading order.

    Reads ``GET /v1/documents/{document_id}/chunks`` page by page until
    ``total`` chunks have arrived, so a document larger than one page is
    still read whole.

    Raises:
        requests.HTTPError: 404 for an unknown document, or any other failure.
    """
    chunks: list[dict] = []
    while True:
        response = requests.get(
            _url(f"/v1/documents/{document_id}/chunks"),
            params={"offset": len(chunks), "limit": _PAGE_SIZE},
            timeout=_REQUEST_TIMEOUT,
        )
        _raise_for_status(response)
        page = _absolutize_image_urls(response.json(), key="items")
        chunks.extend(page["items"])
        if not page["items"] or len(chunks) >= page["total"]:
            return chunks


def health() -> dict:
    """Return the backend health payload, raising if the backend is unreachable.

    A 503 still returns its body (``status: "error"`` with a ``detail``), so
    the caller can tell "running but unhealthy" from "not reachable".
    """
    response = requests.get(_url("/health"), timeout=10)
    if response.status_code != 503:
        _raise_for_status(response)
    return response.json()


def retrieve(query: str, top_k: int = 5) -> dict:
    """Retrieve chunks only (no answer generation)."""
    response = requests.post(
        _url("/v1/retrieve"),
        json={"query": query, "top_k": top_k},
        timeout=_REQUEST_TIMEOUT,
    )
    _raise_for_status(response)
    return _absolutize_image_urls(response.json())


def query(
    query: str,
    top_k: int = 5,
    max_images: int = 3,
    llm: dict | None = None,
) -> dict:
    """Retrieve chunks and generate the full answer in one call.

    Args:
        query: The question.
        top_k: How many chunks to retrieve.
        max_images: Most figures to send to the model (0 = all).
        llm: Optional ``{"provider", "model", "api_key", "temperature",
            "max_tokens"}`` to answer with that model and key; the server's
            default model when omitted.

    Raises:
        requests.HTTPError: On any 4xx or 5xx, with the server's detail.
    """
    body: dict = {"query": query, "top_k": top_k, "max_images": max_images}
    if llm is not None:
        body["llm"] = llm
    response = requests.post(_url("/v1/query"), json=body, timeout=_REQUEST_TIMEOUT)
    _raise_for_status(response)
    return _absolutize_figure_urls(_absolutize_image_urls(response.json()))


def stream(
    query: str,
    top_k: int = 5,
    max_images: int | None = None,
    llm: dict | None = None,
) -> Iterator[tuple[str, dict]]:
    """Stream SSE events for the same question and model options as :func:`query`.

    Yields ``(event, data)`` pairs where ``event`` is one of
    ``retrieval``, ``token``, ``done`` or ``error``.
    Raises ``requests.HTTPError`` for failures before the stream starts.
    """
    body: dict = {"query": query, "top_k": top_k}
    if max_images is not None:
        body["max_images"] = max_images
    if llm is not None:
        body["llm"] = llm
    with requests.post(
        _url("/v1/query/stream"),
        json=body,
        stream=True,
        timeout=_REQUEST_TIMEOUT,
    ) as response:
        _raise_for_status(response)
        event: str | None = None
        # chunk_size=None hands over data as it arrives; the default (512
        # bytes) would hold small token frames back until more data came.
        for line in response.iter_lines(chunk_size=None, decode_unicode=True):
            if not line:
                continue
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1].strip())
                if event == "retrieval":
                    data = _absolutize_image_urls(data)
                elif event == "done":
                    data = _absolutize_figure_urls(data)
                yield event, data
