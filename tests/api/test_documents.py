"""Offline tests for ``GET /v1/documents`` and ``GET /v1/documents/{document_id}/chunks``.

The app runs with fakes for the store and the models; see ``conftest.py``.
"""

from __future__ import annotations

import pytest

from tests.api.fakes import JPEG_BYTES, TLV1117_ID, LM317_FIRST_ID


def test_stream_replaces_get_and_documents_the_shared_body(client):
    assert client.get("/v1/query/stream", params={"query": "x"}).status_code == 405
    paths = client.get("/openapi.json").json()["paths"]
    stream = paths["/v1/query/stream"]["post"]
    assert stream["requestBody"] == paths["/v1/query"]["post"]["requestBody"]
    assert set(stream["responses"]["200"]["content"]) == {"text/event-stream"}

def test_documents_lists_each_source_with_counts(client):
    """A record with no source names no document, so it is left out."""
    response = client.get("/v1/documents")
    assert response.status_code == 200
    assert response.json() == {
        "items": [
            {"id": "lm317", "chunk_count": 2, "image_count": 0, "table_count": 1},
            {"id": "tlv1117", "chunk_count": 1, "image_count": 1, "table_count": 0},
        ],
        "total": 2,
    }

def test_document_chunks_come_in_reading_order(client):
    body = client.get("/v1/documents/lm317/chunks").json()
    assert body["total"] == 2 and body["offset"] == 0 and body["limit"] == 50
    first, second = body["items"]
    assert [first["chunk_id"], second["chunk_id"]] == [1, 2]
    assert first["record_id"] == LM317_FIRST_ID
    assert second["raw_text"] == "Pin 1 ADJ, pin 2 VOUT, pin 3 VIN."
    assert second["tables_html"] == ["<table><tr><td>ADJ</td></tr></table>"]
    assert second["has_table"] and not second["has_image"]

def test_document_chunks_carry_figure_urls_that_serve(client):
    chunk = client.get("/v1/documents/tlv1117/chunks").json()["items"][0]
    assert chunk["image_urls"] == [f"/v1/images/{TLV1117_ID}/0"]
    assert client.get(chunk["image_urls"][0]).content == JPEG_BYTES

def test_document_chunks_page_with_offset_and_limit(client):
    page = client.get("/v1/documents/lm317/chunks", params={"offset": 1, "limit": 1}).json()
    assert [c["chunk_id"] for c in page["items"]] == [2]
    assert (page["total"], page["offset"], page["limit"]) == (2, 1, 1)
    past_end = client.get("/v1/documents/lm317/chunks", params={"offset": 5}).json()
    assert past_end["items"] == [] and past_end["total"] == 2

def test_unknown_document_is_404(client):
    response = client.get("/v1/documents/nope/chunks")
    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found: nope"}

@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 201}, {"offset": -1}])
def test_document_chunks_reject_bad_paging(client, params):
    assert client.get("/v1/documents/lm317/chunks", params=params).status_code == 422

def test_browsing_with_broken_store_is_503(client, backend):
    backend.store_error = "disk gone"
    assert client.get("/v1/documents").status_code == 503
    assert client.get("/v1/documents/lm317/chunks").status_code == 503
