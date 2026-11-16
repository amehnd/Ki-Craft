"""Offline tests for the HTTP API in ``api/main.py``.

They check each endpoint's status codes and response shape, so a change to
the API layer shows exactly which parts of the HTTP contract it alters.

The real FastAPI app and its lifespan run, but the heavy things are replaced
before the app starts:

* the watsonx embedding client and the Chroma store (patched in
  ``api.service``, where ``RAGService._load`` looks them up);
* the two chat client classes inside ``src/llm.py``. The real ``create_llm``
  still runs, so its catalogue checks are part of every test.

Guards make any attempt to open the real Chroma store or build a real watsonx
client fail the test, and LangSmith tracing is switched off, so no test
touches ``dbv2/`` or the network.
"""

from __future__ import annotations

import json
import logging

import langsmith
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk

import api.service
import src.documents
import src.embed
import src.llm
import src.retrieval.search
from api.main import app

# Starts with the JPEG magic bytes so it looks like a real figure.
JPEG_BYTES = b"\xff\xd8\xff\xe0fake-jpeg-for-tests"

LM317_ID = "11111111-1111-4111-8111-111111111111"
TLV1117_ID = "22222222-2222-4222-8222-222222222222"
# A record whose stored path climbs out of the images folder.
ESCAPING_ID = "33333333-3333-4333-8333-333333333333"


def _fake_chunks() -> list[Document]:
    """Two stored chunks: one with a table and no figure, one with a figure file."""
    lm317 = Document(
        id=LM317_ID,
        page_content="LM317 summary: adjustable regulator with ADJ, VOUT, VIN pins.",
        metadata={
            "source": "lm317",
            "raw_text": "Pin 1 ADJ, pin 2 VOUT, pin 3 VIN.",
            "tables_html": json.dumps(["<table><tr><td>ADJ</td></tr></table>"]),
            "image_paths": "[]",
        },
    )
    tlv1117 = Document(
        id=TLV1117_ID,
        page_content="TLV1117 summary: fixed 3.3 V LDO.",
        metadata={
            "source": "tlv1117",
            "raw_text": "Dropout 1.1 V at 800 mA.",
            "tables_html": "[]",
            "image_paths": json.dumps(["dbv2/images/tlv1117_c1_img0.jpg"]),
        },
    )
    return [lm317, tlv1117]


class _FakeRetriever:
    """Returns the first ``k`` fake chunks for any question."""

    def __init__(self, k: int):
        self.k = k

    def invoke(self, query: str) -> list[Document]:
        """Mimic a LangChain retriever's ``invoke``."""
        return _fake_chunks()[: self.k]


class _FakeStore:
    """Stands in for the Chroma store: ``as_retriever`` and ``get`` are used."""

    def as_retriever(self, search_type: str = "mmr", search_kwargs: dict | None = None):
        """Mimic ``Chroma.as_retriever``."""
        return _FakeRetriever((search_kwargs or {}).get("k", 10))

    def get(self, ids: list[str], include: list[str]) -> dict:
        """Mimic ``Chroma.get``: the metadata of the records that exist."""
        records = {doc.id: doc.metadata for doc in _fake_chunks()}
        records[ESCAPING_ID] = {"image_paths": json.dumps(["dbv2/images/../../secret.txt"])}
        found = [i for i in ids if i in records]
        return {"ids": found, "metadatas": [records[i] for i in found]}


class FakeBackend:
    """Switches the tests flip to make the fake store or LLM fail.

    ``prompts`` records every message list sent to the fake LLM so a test can
    check what the model would have seen.
    """

    def __init__(self):
        self.store_error: str | None = None
        self.llm_error: str | None = None
        self.llm_error_after_first_token: str | None = None
        self.finish_reason: str | None = None
        self.prompts: list = []
        self.client_kwargs: list[dict] = []

    def load_store(self, embedding_model) -> _FakeStore:
        """Replacement for ``load_vector_store``."""
        if self.store_error:
            raise RuntimeError(self.store_error)
        return _FakeStore()

    def make_llm(self, *args, **kwargs) -> "_FakeLLM":
        """Replacement for both chat client classes; records what it was given."""
        self.client_kwargs.append(kwargs)
        return _FakeLLM(self)


class _FakeLLM:
    """Streams two tokens, then reports usage on the last chunk like Gemini.

    With ``backend.finish_reason`` set it writes no text at all and stops with
    that reason, like a reasoning model that spent its whole budget.
    """

    def __init__(self, backend: FakeBackend):
        self._backend = backend

    def stream(self, messages):
        """Mimic ``BaseChatModel.stream``."""
        self._backend.prompts.append(messages)
        if self._backend.llm_error:
            raise RuntimeError(self._backend.llm_error)
        if self._backend.finish_reason:
            yield AIMessageChunk(
                content="",
                usage_metadata={"input_tokens": 10, "output_tokens": 512, "total_tokens": 522},
                response_metadata={"finish_reason": self._backend.finish_reason},
            )
            return
        yield AIMessageChunk(content="Hello")
        if self._backend.llm_error_after_first_token:
            raise RuntimeError(self._backend.llm_error_after_first_token)
        yield AIMessageChunk(
            content=" world",
            usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        )


def _forbidden(name: str):
    """Build a stand-in that fails the test if the real ``name`` is constructed."""

    def _raise(*args, **kwargs):
        raise AssertionError(f"offline tests must not construct {name}")

    return _raise


@pytest.fixture
def backend(monkeypatch, tmp_path) -> FakeBackend:
    """Patch every heavy dependency with fakes and turn tracing off.

    Figure files live under ``tmp_path``, which stands in for the repository
    root, so no test reads the real ``dbv2/images``.
    """
    fake = FakeBackend()
    (tmp_path / "dbv2" / "images").mkdir(parents=True)
    (tmp_path / "dbv2" / "images" / "tlv1117_c1_img0.jpg").write_bytes(JPEG_BYTES)
    (tmp_path / "secret.txt").write_text("outside the images folder")
    monkeypatch.setattr(src.documents, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(api.service, "load_embedding_model", lambda: "fake-embeddings")
    monkeypatch.setattr(api.service, "load_vector_store", fake.load_store)
    monkeypatch.setattr(src.llm, "ChatGoogleGenerativeAI", fake.make_llm)
    monkeypatch.setattr(src.llm, "ChatOpenAI", fake.make_llm)
    monkeypatch.setattr(src.retrieval.search, "Chroma", _forbidden("Chroma"))
    monkeypatch.setattr(src.embed, "WatsonxEmbeddings", _forbidden("WatsonxEmbeddings"))

    # A process-wide switch: it also reaches the threads TestClient and
    # asyncio.to_thread run the app in, which a context manager would not.
    langsmith.configure(enabled=False)
    yield fake
    langsmith.configure(enabled=None)


@pytest.fixture
def client(backend: FakeBackend):
    """A TestClient whose ``with`` block runs the app's lifespan."""
    with TestClient(app) as test_client:
        yield test_client


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    """Split a text/event-stream body into ``(event, data)`` pairs."""
    events = []
    for frame in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in frame.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_root_lists_endpoints(client):
    body = client.get("/").json()
    assert body["docs"] == "/docs"
    assert "POST /v1/query" in body["endpoints"]
    assert "POST /v1/query/stream" in body["endpoints"]


def test_api_is_versioned_but_root_and_health_are_not(client):
    """Every endpoint but ``/`` and ``/health`` moved under ``/v1``; old paths are gone."""
    assert client.get("/").status_code == 200
    assert client.get("/health").status_code == 200
    assert client.get("/v1/health").status_code == 404
    assert client.post("/retrieve", json={"query": "x"}).status_code == 404
    assert client.post("/query", json={"query": "x"}).status_code == 404
    assert client.get("/query/stream", params={"query": "x"}).status_code == 404
    assert client.get("/images/any-id/0").status_code == 404
    paths = client.get("/openapi.json").json()["paths"]
    assert {p for p in paths if not p.startswith("/v1/")} == {"/", "/health"}


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "vector_store": "loaded"}


def test_health_reports_store_failure_as_503(client, backend):
    """A broken store is a 503, so an uptime probe can read the code alone."""
    backend.store_error = "store missing"
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "error"
    assert response.json()["detail"] == "Vector store unavailable: store missing"


def test_query_with_broken_store_is_503(client, backend):
    backend.store_error = "store missing"
    response = client.post("/v1/query", json={"query": "LM317 pins"})
    assert response.status_code == 503
    assert response.json() == {"detail": "Vector store unavailable: store missing"}


def test_retrieve_returns_chunks_with_tables_and_images(client):
    response = client.post("/v1/retrieve", json={"query": "LM317 pins", "top_k": 2})
    assert response.status_code == 200
    body = response.json()

    assert body["query"] == "LM317 pins"
    assert body["num_chunks"] == 2
    assert body["total_images"] == 1
    assert body["total_tables"] == 1
    assert "response_id" not in body

    first, second = body["chunks"]
    assert first["chunk_id"] == 1
    assert first["source"] == "lm317"
    assert first["raw_text"] == "Pin 1 ADJ, pin 2 VOUT, pin 3 VIN."
    assert first["has_table"] and not first["has_image"]
    assert first["record_id"] == LM317_ID
    assert "images_base64" not in first
    assert first["image_urls"] == []
    assert "original_content" not in first["metadata"]
    assert second["record_id"] == TLV1117_ID
    assert second["has_image"]
    assert second["image_urls"] == [f"/v1/images/{TLV1117_ID}/0"]


def test_retrieve_rejects_bad_input(client):
    assert client.post("/v1/retrieve", json={"query": ""}).status_code == 422
    assert client.post("/v1/retrieve", json={"query": "x", "top_k": 31}).status_code == 422


def test_building_a_response_reads_no_image_file(client, monkeypatch):
    """Responses carry URLs, so retrieval alone never opens a figure."""

    def opened(path):
        raise AssertionError(f"a response build read the image {path}")

    monkeypatch.setattr(src.documents, "_read_image_as_base64", opened)
    assert client.post("/v1/retrieve", json={"query": "x", "top_k": 2}).status_code == 200


def test_image_url_serves_the_file(client):
    body = client.post("/v1/retrieve", json={"query": "LM317 pins", "top_k": 2}).json()
    response = client.get(body["chunks"][1]["image_urls"][0])
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "public, max-age=86400"
    assert response.content == JPEG_BYTES


def test_image_url_is_stable_across_restarts(backend):
    """The same URL works from a fresh app: nothing is kept between requests."""
    with TestClient(app) as first:
        url = first.post("/v1/retrieve", json={"query": "x", "top_k": 2}).json()["chunks"][1]["image_urls"][0]
    with TestClient(app) as second:
        assert second.get(url).content == JPEG_BYTES


def test_unknown_image_is_404(client):
    assert client.get(f"/v1/images/{TLV1117_ID}/1").status_code == 404, "no second figure"
    assert client.get("/v1/images/99999999-9999-4999-8999-999999999999/0").status_code == 404
    assert client.get(f"/v1/images/{LM317_ID}/0").status_code == 404, "a chunk without figures"


def test_image_url_must_name_a_record_and_position(client):
    assert client.get("/v1/images/not-a-uuid/0").status_code == 422
    assert client.get(f"/v1/images/{TLV1117_ID}/-1").status_code == 422
    assert client.get("/v1/images/..%2F..%2Fsecret.txt/0").status_code in (404, 422)


def test_stored_path_outside_the_images_folder_is_not_served(client):
    response = client.get(f"/v1/images/{ESCAPING_ID}/0")
    assert response.status_code == 404
    assert b"outside the images folder" not in response.content


def test_image_with_broken_store_is_503(client, backend):
    backend.store_error = "store missing"
    assert client.get(f"/v1/images/{TLV1117_ID}/0").status_code == 503


def test_query_returns_answer_and_usage(client, backend):
    response = client.post("/v1/query", json={"query": "LM317 pins", "top_k": 2})
    assert response.status_code == 200
    body = response.json()

    assert body["answer"] == "Hello world"
    assert body["answer_characters"] == len("Hello world")
    assert body["num_chunks"] == 2
    assert body["usage"]["input_tokens"] == 10
    assert body["usage"]["prompt_images"] == 1

    content = backend.prompts[0][0].content
    assert "QUESTION: LM317 pins" in content[0]["text"]
    assert content[1] == {"type": "text", "text": "Figure F1, from tlv1117, chunk 2"}
    assert content[2]["type"] == "image_url"


def test_query_defaults_come_from_settings(client, backend):
    """Without top_k or max_images the settings decide, and every chunk is used."""
    from config.settings import GENERATION_MAX_IMAGES, RETRIEVAL_K

    body = client.post("/v1/query", json={"query": "LM317 pins"}).json()
    assert body["num_chunks"] == min(RETRIEVAL_K, 2)
    assert body["usage"]["chunks_used"] == body["num_chunks"]
    assert body["usage"]["prompt_images"] <= GENERATION_MAX_IMAGES

    prompt = backend.prompts[0][0].content[0]["text"]
    assert "--- Chunk 2 ---" in prompt


def test_query_llm_failure_is_502(client, backend):
    """A failing provider is a 502 whose detail carries the provider's message."""
    backend.llm_error = "quota exceeded"
    response = client.post("/v1/query", json={"query": "LM317 pins", "top_k": 2})
    assert response.status_code == 502
    assert response.json() == {"detail": "The model provider failed: quota exceeded"}


@pytest.mark.parametrize("path", ["/v1/query", "/v1/query/stream"])
@pytest.mark.parametrize("finish_reason", ["length", "MAX_TOKENS"])
def test_answer_cut_off_before_any_text_is_502(client, backend, finish_reason, path):
    """OpenAI-style and Gemini names for "stopped at max_tokens" both count."""
    backend.finish_reason = finish_reason
    response = client.post(path, json={"query": "LM317 pins", "top_k": 2})
    assert response.status_code == 502
    assert "whole output budget (512 tokens" in response.json()["detail"]
    assert "Raise max_tokens" in response.json()["detail"]


def test_empty_answer_for_another_reason_is_502(client, backend):
    backend.finish_reason = "content_filter"
    response = client.post("/v1/query", json={"query": "LM317 pins", "top_k": 2})
    assert response.status_code == 502
    assert response.json()["detail"] == "The model returned no answer text (finish_reason: content_filter)."


def test_unexpected_error_is_500_without_internals(backend, monkeypatch):
    """The caller sees a generic message; the real one stays in the server log."""

    def broken(retriever, query):
        raise RuntimeError("secret internal path /srv/app")

    monkeypatch.setattr(api.service, "retrieve_chunks", broken)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        response = test_client.post("/v1/retrieve", json={"query": "LM317 pins"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}


def test_stream_emits_retrieval_tokens_done(client):
    response = client.post("/v1/query/stream", json={"query": "LM317 pins", "top_k": 2})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(response.text)
    assert [name for name, _ in events] == ["retrieval", "token", "token", "done"]
    assert events[0][1]["num_chunks"] == 2
    assert [data["delta"] for name, data in events if name == "token"] == ["Hello", " world"]
    done = events[-1][1]
    assert done["answer"] == "Hello world"
    assert done["num_chunks"] == 2
    assert "response_id" not in done and "response_id" not in events[0][1]


def test_stream_reports_llm_failure_as_error_event(client, backend):
    """A provider failure after the first token ends the stream with an error."""
    backend.llm_error_after_first_token = "quota exceeded"
    response = client.post("/v1/query/stream", json={"query": "LM317 pins", "top_k": 2})
    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert [name for name, _ in events] == ["retrieval", "token", "error"]
    assert events[-1][1] == {"status": 502, "message": "The model provider failed: quota exceeded"}


def test_stream_provider_failure_before_first_token_is_502(client, backend):
    backend.llm_error = "quota exceeded"
    response = client.post("/v1/query/stream", json={"query": "LM317 pins", "top_k": 2})
    assert response.status_code == 502
    assert response.json() == {"detail": "The model provider failed: quota exceeded"}


def test_stream_replaces_get_and_documents_the_shared_body(client):
    assert client.get("/v1/query/stream", params={"query": "x"}).status_code == 405
    paths = client.get("/openapi.json").json()["paths"]
    stream = paths["/v1/query/stream"]["post"]
    assert stream["requestBody"] == paths["/v1/query"]["post"]["requestBody"]
    assert set(stream["responses"]["200"]["content"]) == {"text/event-stream"}


def test_stream_uses_requested_llm_and_body_options(client, backend):
    response = client.post(
        "/v1/query/stream",
        json={"query": "LM317 pins", "top_k": 1, "max_images": 0, "llm": _llm(max_tokens=720)},
    )
    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert events[0][1]["num_chunks"] == 1
    assert events[-1][1]["usage"]["model"] == "openai/gpt-oss-20b"
    assert backend.client_kwargs[-1]["api_key"] == "test-key"
    assert backend.client_kwargs[-1]["max_tokens"] == 720


def test_stream_with_broken_store_is_503(client, backend):
    backend.store_error = "store missing"
    response = client.post("/v1/query/stream", json={"query": "LM317 pins"})
    assert response.status_code == 503


def _llm(**overrides) -> dict:
    """An ``llm`` object for a request body, Groq GPT-OSS 20B unless overridden."""
    return {"provider": "groq", "model": "openai/gpt-oss-20b", "api_key": "test-key", **overrides}


def test_query_with_llm_uses_chosen_model_and_key(client, backend):
    response = client.post(
        "/v1/query",
        json={"query": "LM317 pins", "top_k": 2,
              "llm": _llm(provider="gemini", model="gemini-3.5-flash-lite")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Hello world"
    assert body["usage"]["model"] == "gemini-3.5-flash-lite"
    # The SecretStr is unwrapped once, so the client gets the plain key.
    assert backend.client_kwargs[-1]["google_api_key"] == "test-key"


def test_query_with_text_only_model_gets_no_figures(client, backend):
    """A model the catalogue marks text-only is sent the text and tables only."""
    response = client.post("/v1/query", json={"query": "LM317 pins", "top_k": 2, "llm": _llm()})
    assert response.status_code == 200
    assert response.json()["usage"]["prompt_images"] == 0
    assert [part["type"] for part in backend.prompts[0][0].content] == ["text"]
    assert response.json()["total_images"] == 1, "the response still lists the chunk's figure URL"


def test_query_with_llm_without_max_tokens_uses_catalogue_default(client, backend):
    response = client.post(
        "/v1/query", json={"query": "LM317 pins", "top_k": 2, "llm": _llm(max_tokens=None)}
    )
    assert response.status_code == 200
    assert backend.client_kwargs[-1]["max_tokens"] == 2048


def test_llm_without_api_key_is_422(client, backend):
    """The server's own key is never used for a model the caller picks."""
    llm = _llm()
    del llm["api_key"]
    response = client.post("/v1/query", json={"query": "x", "llm": llm})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "llm", "api_key"]
    assert backend.prompts == []


@pytest.mark.parametrize("path", ["/v1/query", "/v1/query/stream"])
def test_validation_error_does_not_echo_the_key(client, path):
    """A missing ``llm.model`` would echo the whole ``llm`` object as ``input``."""
    llm = _llm(api_key="sk-must-not-leak")
    del llm["model"]
    response = client.post(path, json={"query": "x", "llm": llm})
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"] == ["body", "llm", "model"]
    assert set(error) == {"type", "loc", "msg"}
    assert "sk-must-not-leak" not in response.text


@pytest.mark.parametrize("path", ["/v1/query", "/v1/query/stream"])
def test_query_unknown_model_is_422(client, backend, path):
    """An unknown model is the caller's mistake, rejected before retrieval."""
    response = client.post(
        path, json={"query": "x", "llm": _llm(provider="gemini", model="no-such-model")}
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "Model no-such-model not found for provider gemini"}
    assert backend.prompts == []


def test_query_with_llm_provider_failure_is_502(client, backend):
    backend.llm_error = "invalid API key"
    response = client.post("/v1/query", json={"query": "x", "llm": _llm()})
    assert response.status_code == 502
    assert response.json() == {"detail": "The model provider failed: invalid API key"}


@pytest.mark.parametrize("llm_error", [None, "invalid API key"])
def test_api_key_never_reaches_the_logs(client, backend, caplog, llm_error):
    backend.llm_error = llm_error
    caplog.set_level(logging.DEBUG)
    client.post("/v1/query", json={"query": "x", "llm": _llm(api_key="sk-must-not-leak")})
    assert caplog.records, "the request should have logged something"
    assert "sk-must-not-leak" not in caplog.text


def test_custom_route_is_gone(client):
    response = client.post("/v1/query/custom", json={"query": "x", **_llm()})
    assert response.status_code == 404


def test_retrieve_does_not_take_an_llm(client):
    """Retrieval never calls a model, so its body has no ``llm`` field."""
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    assert "llm" not in schemas["RetrieveRequest"]["properties"]
    assert "llm" in schemas["QueryRequest"]["properties"]
