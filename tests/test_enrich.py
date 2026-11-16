"""Tests for src/ingestion/enrich.py: the summary backends, with no network.

A real ``OpenAI`` client is given an httpx ``MockTransport``, so each test sees
the exact request body the server would get.
"""

import functools
import json
from types import SimpleNamespace

import httpx
import langsmith
import pytest
from openai import OpenAI

import src.ingestion.enrich as enrich
from src.ingestion.enrich import SummaryError, check_enhancement_backend, create_ai_enhanced_summary


@pytest.fixture(autouse=True)
def _no_tracing():
    langsmith.configure(enabled=False)
    yield
    langsmith.configure(enabled=None)


def _server(monkeypatch, handler, retries: int = 0) -> list[dict]:
    """Route every OpenAI client enrich builds to ``handler``; return the request bodies."""
    bodies: list[dict] = []

    def record(request: httpx.Request) -> httpx.Response:
        if request.content:
            bodies.append(json.loads(request.content))
        return handler(request)

    transport = httpx.MockTransport(record)
    monkeypatch.setattr(enrich, "OpenAI", functools.partial(OpenAI, http_client=httpx.Client(transport=transport)))
    monkeypatch.setattr(enrich, "ENHANCEMENT_MAX_RETRIES", retries)
    return bodies


def _reply(text: str | None, finish_reason: str = "stop") -> httpx.Response:
    return httpx.Response(200, json={
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": {"role": "assistant", "content": text}}],
    })


def _summary(backend: str, images: list[str]) -> str:
    client = enrich.create_enhancement_client(backend)
    return create_ai_enhanced_summary(client, "pin 1 GND", ["<table/>"], images, backend)


def test_local_backend_sends_its_model_and_mlx_fields(monkeypatch):
    bodies = _server(monkeypatch, lambda r: _reply("NE555 pinout"))
    assert _summary("local", ["AAA", "BBB"]) == "NE555 pinout"

    body = bodies[0]
    assert body["model"] == enrich.ENHANCEMENT_BACKENDS["local"]["model"]
    assert body["max_pixels"] == enrich.LOCAL_ENHANCEMENT_MAX_PIXELS
    assert body["repetition_penalty"] == enrich.LOCAL_ENHANCEMENT_REPETITION_PENALTY
    assert body["enable_thinking"] is False
    images = [part for part in body["messages"][0]["content"] if part["type"] == "image_url"]
    assert [part["image_url"]["url"] for part in images] == ["data:image/jpeg;base64,AAA", "data:image/jpeg;base64,BBB"]


def test_huggingface_backend_sends_no_mlx_fields(monkeypatch):
    bodies = _server(monkeypatch, lambda r: _reply("summary"))
    _summary("huggingface", [])
    assert bodies[0]["model"] == "zai-org/GLM-4.5V"
    assert not {"max_pixels", "repetition_penalty", "enable_thinking"} & set(bodies[0])


def test_image_limit_caps_the_figures_sent(monkeypatch):
    bodies = _server(monkeypatch, lambda r: _reply("summary"))
    monkeypatch.setattr(enrich, "ENHANCEMENT_MAX_IMAGES", 1)
    _summary("local", ["AAA", "BBB", "CCC"])
    assert sum(part["type"] == "image_url" for part in bodies[0]["messages"][0]["content"]) == 1


def test_empty_summary_is_an_error(monkeypatch):
    _server(monkeypatch, lambda r: _reply(None, finish_reason="length"))
    with pytest.raises(SummaryError, match="no text"):
        _summary("local", [])


def test_server_error_is_retried_then_raised(monkeypatch):
    monkeypatch.setattr("openai._base_client.BaseClient._calculate_retry_timeout", lambda *a, **k: 0)
    bodies = _server(monkeypatch, lambda r: httpx.Response(500, json={"error": "boom"}), retries=2)
    with pytest.raises(SummaryError, match="InternalServerError"):
        _summary("local", [])
    assert len(bodies) == 3, "one call and two retries"


def test_cut_off_summary_is_kept(monkeypatch):
    _server(monkeypatch, lambda r: _reply("partial summ", finish_reason="length"))
    assert _summary("local", []) == "partial summ"


def test_text_only_chunks_make_no_call(monkeypatch):
    bodies = _server(monkeypatch, lambda r: _reply("unused"))
    chunk = SimpleNamespace(text="plain text", metadata=SimpleNamespace(orig_elements=[]))
    docs = enrich.summarise_chunks([chunk], backend="local")
    assert docs[0].page_content == "plain text"
    assert bodies == []


def test_unreachable_local_server_names_the_start_command(monkeypatch):
    def refuse(request):
        raise httpx.ConnectError("refused")

    _server(monkeypatch, refuse)
    with pytest.raises(SummaryError, match="python -m mlx_vlm.server --model"):
        check_enhancement_backend("local")


def test_huggingface_backend_needs_a_token(monkeypatch):
    monkeypatch.setattr(enrich, "HF_TOKEN", "")
    with pytest.raises(SummaryError, match="HF_TOKEN"):
        check_enhancement_backend("huggingface")

