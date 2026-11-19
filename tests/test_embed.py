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


class _FakeLlama:
    """Stands in for llama_cpp.Llama: records its arguments and the embed calls."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def embed(self, text, normalize, truncate):
        self.calls.append((text, normalize, truncate))
        return [0.6, 0.8]


def _fake_llama_cpp(monkeypatch):
    """Install a fake llama_cpp module and return it."""
    import sys
    import types

    fake = types.SimpleNamespace(LLAMA_POOLING_TYPE_CLS=2, Llama=_FakeLlama)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)
    return fake


def test_local_backend_loads_gguf_and_skips_watsonx(monkeypatch, tmp_path):
    _fake_llama_cpp(monkeypatch)
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"")
    monkeypatch.setattr(embed, "EMBEDDING_BACKEND", "local")
    monkeypatch.setattr(embed, "LOCAL_EMBEDDING_GGUF_PATH", str(model_file))
    monkeypatch.setattr(embed, "WATSONX_API_KEY", "")

    model = embed.load_embedding_model()

    assert isinstance(model, embed.GGUFEmbeddings)
    assert model._llm.kwargs == {
        "model_path": str(model_file),
        "embedding": True,
        "pooling_type": 2,
        "n_ctx": embed.EMBEDDING_MAX_INPUT_TOKENS,
        "n_batch": embed.EMBEDDING_MAX_INPUT_TOKENS,
        "n_threads": embed.LOCAL_EMBEDDING_THREADS,
        "verbose": False,
    }


def test_gguf_embeddings_normalise_and_truncate(monkeypatch, tmp_path):
    _fake_llama_cpp(monkeypatch)
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"")
    model = embed.GGUFEmbeddings(str(model_file), threads=1)

    assert model.embed_documents(["a", "b"]) == [[0.6, 0.8], [0.6, 0.8]]
    assert model._llm.calls == [("a", True, True), ("b", True, True)]


def test_missing_gguf_file_is_named(tmp_path):
    missing = tmp_path / "absent.gguf"

    with pytest.raises(FileNotFoundError, match="absent.gguf"):
        embed.GGUFEmbeddings(str(missing), threads=1)


def test_gguf_embeddings_never_run_two_calls_at_once(monkeypatch, tmp_path):
    import threading
    import time

    class _OverlapLlama(_FakeLlama):
        """Records how many embed calls ran at the same time."""

        active = 0
        most = 0

        def embed(self, text, normalize, truncate):
            type(self).active += 1
            type(self).most = max(type(self).most, type(self).active)
            time.sleep(0.01)
            type(self).active -= 1
            return [1.0]

    fake = _fake_llama_cpp(monkeypatch)
    fake.Llama = _OverlapLlama
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"")
    model = embed.GGUFEmbeddings(str(model_file), threads=1)

    threads = [threading.Thread(target=model.embed_query, args=("q",)) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert _OverlapLlama.most == 1
