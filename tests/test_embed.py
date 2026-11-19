"""Tests for the watsonx.ai embedding factory (no network calls)."""

import pytest

import src.embed as embed


@pytest.fixture(autouse=True)
def watsonx_backend(monkeypatch):
    """Default to watsonx so the tests do not depend on EMBEDDING_BACKEND in .env."""
    monkeypatch.setattr(embed, "EMBEDDING_BACKEND", "watsonx")


def test_builds_watsonx_embeddings_from_settings(monkeypatch):
    captured = {}
    monkeypatch.setattr(embed, "WATSONX_API_KEY", "key")
    monkeypatch.setattr(embed, "WATSONX_PROJECT_ID", "project")
    monkeypatch.setattr(embed, "WatsonxEmbeddings", lambda **kw: captured.update(kw) or "model")

    assert embed.load_embedding_model() == "model"
    assert captured == {
        "model_id": embed.EMBEDDING_MODEL,
        "url": embed.WATSONX_URL,
        "project_id": "project",
        "api_key": "key",
        "params": {"truncate_input_tokens": embed.EMBEDDING_MAX_INPUT_TOKENS},
    }


def test_missing_credentials_are_named(monkeypatch):
    monkeypatch.setattr(embed, "WATSONX_API_KEY", "")
    monkeypatch.setattr(embed, "WATSONX_PROJECT_ID", "")

    with pytest.raises(RuntimeError, match="WATSONX_API_KEY, WATSONX_PROJECT_ID"):
        embed.load_embedding_model()


def test_local_backend_normalises_and_skips_watsonx(monkeypatch):
    import sys
    import types

    captured = {}
    fake = types.SimpleNamespace(HuggingFaceEmbeddings=lambda **kw: captured.update(kw) or "local")
    monkeypatch.setitem(sys.modules, "langchain_huggingface", fake)
    monkeypatch.setattr(embed, "EMBEDDING_BACKEND", "local")
    monkeypatch.setattr(embed, "WATSONX_API_KEY", "")

    assert embed.load_embedding_model() == "local"
    assert captured == {
        "model_name": embed.LOCAL_EMBEDDING_MODEL,
        "encode_kwargs": {"normalize_embeddings": True},
    }
