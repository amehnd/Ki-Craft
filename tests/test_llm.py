"""Tests for the chat model factory in ``src/llm.py`` (no network calls).

Both client classes are replaced with recorders, so each test checks the exact
arguments the factory would construct the real client with.
"""

from __future__ import annotations

import pytest

import src.llm as llm


@pytest.fixture
def built(monkeypatch) -> dict:
    """Record which client class was built and with which arguments."""
    record: dict = {}

    def recorder(name: str):
        def build(**kwargs):
            record.update(client=name, kwargs=kwargs)
            return name

        return build

    monkeypatch.setattr(llm, "ChatGoogleGenerativeAI", recorder("gemini"))
    monkeypatch.setattr(llm, "ChatOpenAI", recorder("openai"))
    return record


def test_gemini_protocol_builds_google_client(built):
    llm.create_llm("gemini", "gemini-2.5-flash", "key", temperature=0.2, max_tokens=300)
    assert built == {
        "client": "gemini",
        "kwargs": {
            "model": "gemini-2.5-flash",
            "google_api_key": "key",
            "max_output_tokens": 300,
            "temperature": 0.2,
        },
    }


def test_fixed_sampling_gemini_model_gets_no_temperature(built):
    llm.create_llm("gemini", "gemini-3.5-flash-lite", "key", temperature=0.0)
    assert "temperature" not in built["kwargs"]


@pytest.mark.parametrize(
    ("provider", "model", "base_url"),
    [
        ("groq", "openai/gpt-oss-20b", "https://api.groq.com/openai/v1"),
        ("huggingface", "deepseek-ai/DeepSeek-V3", "https://router.huggingface.co/v1"),
        ("openai", "gpt-4o-mini", None),
    ],
)
def test_openai_protocol_uses_the_provider_base_url(built, provider, model, base_url):
    """The client is chosen by provider, even when a model id starts with 'openai/'."""
    llm.create_llm(provider, model, "key")
    assert built["client"] == "openai"
    assert built["kwargs"]["base_url"] == base_url
    assert built["kwargs"]["model"] == model


def test_o_series_model_gets_no_temperature(built):
    llm.create_llm("openai", "o3-mini", "key", temperature=0.0)
    assert "temperature" not in built["kwargs"]


@pytest.mark.parametrize(
    ("provider", "model"),
    [("groq", "openai/gpt-oss-20b"), ("groq", "qwen/qwen3.8-27b"), ("openai", "gpt-4o")],
)
def test_no_reasoning_setting_is_sent(built, provider, model):
    """Each provider's own reasoning default applies.

    Groq GPT-OSS accepts only low, medium, or high, so the "none" this
    factory used to send to every Groq model was not a valid value for it.
    """
    llm.create_llm(provider, model, "key")
    assert "model_kwargs" not in built["kwargs"]
    assert "reasoning_effort" not in built["kwargs"]


def test_unknown_provider_is_rejected(built):
    with pytest.raises(ValueError, match="Unknown provider nope"):
        llm.create_llm("nope", "gpt-4o", "key")
    assert built == {}


def test_unknown_model_is_rejected(built):
    with pytest.raises(ValueError, match="Model gpt-9 not found for provider openai"):
        llm.create_llm("openai", "gpt-9", "key")
    assert built == {}


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        ("gemini", "gemini-3.5-flash-lite", True),
        ("openai", "gpt-4o-mini", True),
        ("groq", "openai/gpt-oss-20b", False),
        ("huggingface", "deepseek-ai/DeepSeek-V3", False),
        ("gemini", "no-such-model", False),
    ],
)
def test_accepts_images_follows_the_catalogue(provider, model, expected):
    assert llm.accepts_images(provider, model) is expected


@pytest.mark.parametrize(
    ("provider", "model", "budget"),
    [
        ("groq", "openai/gpt-oss-20b", 2048),
        ("openai", "o3-mini", 2048),
        ("gemini", "gemini-3.5-flash-lite", 512),
        ("openai", "gpt-4o-mini", 512),
    ],
)
def test_unset_max_tokens_uses_the_model_default(built, provider, model, budget):
    """Reasoning models need room to think before answering."""
    llm.create_llm(provider, model, "key")
    sent = built["kwargs"].get("max_tokens", built["kwargs"].get("max_output_tokens"))
    assert sent == budget


def test_explicit_max_tokens_wins_over_the_model_default(built):
    llm.create_llm("groq", "openai/gpt-oss-20b", "key", max_tokens=300)
    assert built["kwargs"]["max_tokens"] == 300
