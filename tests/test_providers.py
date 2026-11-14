"""Tests for the provider/model registry (pure dataclasses, no deps)."""

from config.providers import (
    ALL_PROVIDERS,
    PROVIDER_BY_ID,
    get_model,
    get_provider,
    list_models_for_provider,
)


def test_expected_providers_present():
    ids = {p.id for p in ALL_PROVIDERS}
    assert {"gemini", "groq", "huggingface", "openai"} <= ids


def test_duplicate_provider_ids():
    assert len(PROVIDER_BY_ID) == len(ALL_PROVIDERS)


def test_get_provider_unknown_returns_none():
    assert get_provider("nope") is None


def test_get_model_known():
    model = get_model("gemini", "gemini-3.5-flash-lite")
    assert model is not None
    assert model.context_window > 0


def test_get_model_unknown_returns_none():
    assert get_model("gemini", "does-not-exist") is None
    assert get_model("nope", "x") is None


def test_every_model_has_unique_id():
    seen = set()
    for provider in ALL_PROVIDERS:
        for model in provider.models:
            key = (provider.id, model.id)
            assert key not in seen
            seen.add(key)


def test_list_models_for_provider():
    assert len(list_models_for_provider("gemini")) > 0


def test_openai_compatible_providers_have_an_endpoint():
    """Only OpenAI itself may rely on the client's default endpoint."""
    for provider in ALL_PROVIDERS:
        if provider.protocol == "openai" and provider.id != "openai":
            assert provider.base_url, provider.id


def test_groq_models_that_returned_404_are_gone():
    removed = {
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "qwen/qwen3.6-27b",
        "groq/compound",
        "groq/compound-mini",
    }
    assert not removed & {m.id for m in list_models_for_provider("groq")}
