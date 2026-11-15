"""Thin HTTP client for the FastAPI backend (``api.main``).

Lets the Streamlit dashboard talk to the API instead of loading the vector
store and calling the LLM itself.  Image URLs returned by the backend are
relative (``/v1/images/{response_id}/{index}``); this client rewrites them to
absolute URLs so ``st.image`` can load them directly.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

import requests

from config.settings import API_BASE_URL

_REQUEST_TIMEOUT = 300


def _url(path: str) -> str:
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


def _absolutize_image_urls(payload: dict[str, Any]) -> dict[str, Any]:
    """Turn relative ``/v1/images/...`` URLs into absolute backend URLs."""
    for chunk in payload.get("chunks", []):
        chunk["image_urls"] = [_url(u) for u in chunk.get("image_urls", [])]
    return payload


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
    return _absolutize_image_urls(response.json())


def stream(query: str, top_k: int = 5) -> Iterator[tuple[str, dict]]:
    """Stream SSE events from ``/v1/query/stream``.

    Yields ``(event, data)`` pairs where ``event`` is one of
    ``retrieval``, ``token``, ``done`` or ``error``.
    """
    with requests.get(
        _url("/v1/query/stream"),
        params={"query": query, "top_k": top_k},
        stream=True,
        timeout=_REQUEST_TIMEOUT,
    ) as response:
        _raise_for_status(response)
        event: str | None = None
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = json.loads(line.split(":", 1)[1].strip())
                if event == "retrieval":
                    data = _absolutize_image_urls(data)
                yield event, data
