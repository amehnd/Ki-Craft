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

import base64
import json

import langsmith
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk

import api.service
import src.embed
import src.llm
import src.retrieval.search
from api.main import app

# Starts with the JPEG magic bytes so it looks like a real figure.
JPEG_BYTES = b"\xff\xd8\xff\xe0fake-jpeg-for-tests"


def _fake_chunks() -> list[Document]:
    """Two chunks, one per metadata schema that ``src/documents.py`` reads."""
    nested = Document(
        page_content="LM317 summary: adjustable regulator with ADJ, VOUT, VIN pins.",
        metadata={
            "source": "lm317",
            "original_content": json.dumps(
                {
                    "raw_text": "Pin 1 ADJ, pin 2 VOUT, pin 3 VIN.",
                    "tables_html": ["<table><tr><td>ADJ</td></tr></table>"],
                    "images_base64": [base64.b64encode(JPEG_BYTES).decode()],
                }
            ),
        },
    )
    flat = Document(
        page_content="TLV1117 summary: fixed 3.3 V LDO.",
        metadata={
            "source": "tlv1117",
            "raw_text": "Dropout 1.1 V at 800 mA.",
            "tables_html": "[]",
            "image_paths": "[]",
        },
    )
    return [nested, flat]


class _FakeRetriever:
    """Returns the first ``k`` fake chunks for any question."""

    def __init__(self, k: int):
        self.k = k

    def invoke(self, query: str) -> list[Document]:
        """Mimic a LangChain retriever's ``invoke``."""
        return _fake_chunks()[: self.k]


class _FakeStore:
    """Stands in for the Chroma store: only ``as_retriever`` is used."""

    def as_retriever(self, search_type: str = "mmr", search_kwargs: dict | None = None):
        """Mimic ``Chroma.as_retriever``."""
        return _FakeRetriever((search_kwargs or {}).get("k", 10))


class FakeBackend:
    """Switches the tests flip to make the fake store or LLM fail.

    ``prompts`` records every message list sent to the fake LLM so a test can
    check what the model would have seen.
    """

    def __init__(self):
        self.store_error: str | None = None
        self.llm_error: str | None = None
        self.finish_reason: str | None = None
        self.prompts: list = []

    def load_store(self, embedding_model) -> _FakeStore:
        """Replacement for ``load_vector_store``."""
        if self.store_error:
            raise RuntimeError(self.store_error)
        return _FakeStore()

    def make_llm(self, *args, **kwargs) -> "_FakeLLM":
        """Replacement for both chat client classes; ignores model and key."""
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
def backend(monkeypatch) -> FakeBackend:
    """Patch every heavy dependency with fakes and turn tracing off."""
    fake = FakeBackend()
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
    assert "POST /query" in body["endpoints"]


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
    response = client.post("/query", json={"query": "LM317 pins"})
    assert response.status_code == 503
    assert response.json() == {"detail": "Vector store unavailable: store missing"}


def test_retrieve_returns_chunks_with_tables_and_images(client):
    response = client.post("/retrieve", json={"query": "LM317 pins", "top_k": 2})
    assert response.status_code == 200
    body = response.json()

    assert body["query"] == "LM317 pins"
    assert body["num_chunks"] == 2
    assert body["total_images"] == 1
    assert body["total_tables"] == 1

    first, second = body["chunks"]
    assert first["chunk_id"] == 1
    assert first["source"] == "lm317"
    assert first["raw_text"] == "Pin 1 ADJ, pin 2 VOUT, pin 3 VIN."
    assert first["has_table"] and first["has_image"]
    assert first["images_base64"] == [base64.b64encode(JPEG_BYTES).decode()]
    assert first["image_urls"] == [f"/images/{body['response_id']}/0"]
    assert "original_content" not in first["metadata"]
    assert second["image_urls"] == [] and not second["has_image"]


def test_retrieve_rejects_bad_input(client):
    assert client.post("/retrieve", json={"query": ""}).status_code == 422
    assert client.post("/retrieve", json={"query": "x", "top_k": 31}).status_code == 422


def test_image_url_serves_jpeg(client):
    body = client.post("/retrieve", json={"query": "LM317 pins", "top_k": 2}).json()
    response = client.get(body["chunks"][0]["image_urls"][0])
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == JPEG_BYTES


def test_unknown_image_is_404(client):
    body = client.post("/retrieve", json={"query": "LM317 pins", "top_k": 2}).json()
    assert client.get(f"/images/{body['response_id']}/5").status_code == 404
    assert client.get("/images/not-a-response-id/0").status_code == 404


def test_query_returns_answer_and_usage(client, backend):
    response = client.post("/query", json={"query": "LM317 pins", "top_k": 2})
    assert response.status_code == 200
    body = response.json()

    assert body["answer"] == "Hello world"
    assert body["answer_characters"] == len("Hello world")
    assert body["num_chunks"] == 2
    assert body["usage"]["input_tokens"] == 10
    assert body["usage"]["prompt_images"] == 1

    content = backend.prompts[0][0].content
    assert "QUESTION: LM317 pins" in content[0]["text"]
    assert content[1]["type"] == "image_url"


def test_query_defaults_come_from_settings(client, backend):
    """Without top_k or max_images the settings decide, and every chunk is used."""
    from config.settings import GENERATION_MAX_IMAGES, RETRIEVAL_K

    body = client.post("/query", json={"query": "LM317 pins"}).json()
    assert body["num_chunks"] == min(RETRIEVAL_K, 2)
    assert body["usage"]["chunks_used"] == body["num_chunks"]
    assert body["usage"]["prompt_images"] <= GENERATION_MAX_IMAGES

    prompt = backend.prompts[0][0].content[0]["text"]
    assert "--- Chunk 2 ---" in prompt


def test_query_llm_failure_is_502(client, backend):
    """A failing provider is a 502 whose detail carries the provider's message."""
    backend.llm_error = "quota exceeded"
    response = client.post("/query", json={"query": "LM317 pins", "top_k": 2})
    assert response.status_code == 502
    assert response.json() == {"detail": "The model provider failed: quota exceeded"}


@pytest.mark.parametrize("finish_reason", ["length", "MAX_TOKENS"])
def test_answer_cut_off_before_any_text_is_502(client, backend, finish_reason):
    """OpenAI-style and Gemini names for "stopped at max_tokens" both count."""
    backend.finish_reason = finish_reason
    response = client.post("/query", json={"query": "LM317 pins", "top_k": 2})
    assert response.status_code == 502
    assert "whole output budget (512 tokens" in response.json()["detail"]
    assert "Raise max_tokens" in response.json()["detail"]


def test_empty_answer_for_another_reason_is_502(client, backend):
    backend.finish_reason = "content_filter"
    response = client.post("/query", json={"query": "LM317 pins", "top_k": 2})
    assert response.status_code == 502
    assert response.json()["detail"] == "The model returned no answer text (finish_reason: content_filter)."


def test_unexpected_error_is_500_without_internals(backend, monkeypatch):
    """The caller sees a generic message; the real one stays in the server log."""

    def broken(retriever, query):
        raise RuntimeError("secret internal path /srv/app")

    monkeypatch.setattr(api.service, "retrieve_chunks", broken)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        response = test_client.post("/retrieve", json={"query": "LM317 pins"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}


def test_stream_emits_retrieval_tokens_done(client):
    response = client.get("/query/stream", params={"query": "LM317 pins", "top_k": 2})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(response.text)
    assert [name for name, _ in events] == ["retrieval", "token", "token", "done"]
    assert events[0][1]["num_chunks"] == 2
    assert [data["delta"] for name, data in events if name == "token"] == ["Hello", " world"]
    done = events[-1][1]
    assert done["answer"] == "Hello world"
    assert done["response_id"] == events[0][1]["response_id"]


def test_stream_reports_llm_failure_as_error_event(client, backend):
    """Headers are already sent, so a mid-stream failure is an ``error`` event."""
    backend.llm_error = "quota exceeded"
    response = client.get("/query/stream", params={"query": "LM317 pins", "top_k": 2})
    events = _parse_sse(response.text)
    assert [name for name, _ in events] == ["retrieval", "error"]
    assert events[-1][1] == {"message": "The model provider failed: quota exceeded"}


def test_stream_with_broken_store_is_503(client, backend):
    backend.store_error = "store missing"
    response = client.get("/query/stream", params={"query": "LM317 pins"})
    assert response.status_code == 503


def test_custom_query_uses_chosen_model(client):
    response = client.post(
        "/query/custom",
        json={
            "query": "LM317 pins",
            "top_k": 2,
            "provider": "gemini",
            "model": "gemini-3.5-flash-lite",
            "api_key": "test-key",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Hello world"
    assert body["usage"]["model"] == "gemini-3.5-flash-lite"


def test_custom_query_text_only_model_gets_no_figures(client, backend):
    """A model the catalogue marks text-only is sent the text and tables only."""
    response = client.post(
        "/query/custom",
        json={
            "query": "LM317 pins",
            "top_k": 2,
            "provider": "groq",
            "model": "openai/gpt-oss-20b",
            "api_key": "test-key",
        },
    )
    assert response.status_code == 200
    assert response.json()["usage"]["prompt_images"] == 0
    assert [part["type"] for part in backend.prompts[0][0].content] == ["text"]
    assert response.json()["total_images"] == 1, "the response still lists the chunk's figure"


def test_custom_query_without_max_tokens_is_accepted(client):
    """max_tokens is optional; the catalogue default fills it in."""
    response = client.post(
        "/query/custom",
        json={"query": "LM317 pins", "top_k": 2, "provider": "groq",
              "model": "openai/gpt-oss-20b", "api_key": "k", "max_tokens": None},
    )
    assert response.status_code == 200


def test_custom_query_unknown_model_is_422(client, backend):
    """An unknown model is the caller's mistake, rejected before retrieval."""
    response = client.post(
        "/query/custom",
        json={"query": "x", "provider": "gemini", "model": "no-such-model", "api_key": "k"},
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "Model no-such-model not found for provider gemini"}
    assert backend.prompts == []


def test_custom_query_provider_failure_is_502(client, backend):
    backend.llm_error = "invalid API key"
    response = client.post(
        "/query/custom",
        json={"query": "x", "provider": "groq", "model": "openai/gpt-oss-20b", "api_key": "k"},
    )
    assert response.status_code == 502
    assert response.json() == {"detail": "The model provider failed: invalid API key"}
