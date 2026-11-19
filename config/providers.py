"""Provider and model registry for LLM selection.

Central place to define available providers, their models, and metadata
(pricing and capabilities).  Served to clients by ``GET /v1/providers``
(the dashboard's model picker), and checked by the chat model factory in
``src/llm.py``, which builds only models listed here for a query's ``llm`` object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ModelInfo:
    """Metadata for a single model."""
    id: str                    # model identifier used in API calls
    display_name: str          # human-readable name for UI
    input_price_per_1m: float  # USD per 1M input tokens (0 if unknown/free)
    output_price_per_1m: float # USD per 1M output tokens (0 if unknown/free)
    supports_vision: bool = False
    # False for models that reject a custom temperature (OpenAI o-series) or
    # ignore it (Gemini fixed-sampling and GPT-5 models); src/llm.py then
    # leaves it out of the request.
    accepts_temperature: bool = True
    # Output token budget when the caller does not set max_tokens. Reasoning
    # models spend output tokens thinking before they answer, so 512 can be
    # used up before any answer text (Groq GPT-OSS returned an empty answer
    # at 512 on 2026-11-15). Models that reason by default get 2048.
    default_max_tokens: int = 512


@dataclass(frozen=True)
class ProviderInfo:
    """Metadata for a provider."""
    id: str                    # short key: "gemini", "groq", "huggingface", "openai"
    display_name: str          # "Google Gemini", "Groq", "Hugging Face", "OpenAI"
    # Wire protocol, which picks the LangChain client in src/llm.py: "gemini"
    # for Google's API, "openai" for any OpenAI-compatible chat API.
    protocol: Literal["gemini", "openai"]
    base_url: str | None = None # endpoint for OpenAI-compatible providers; None means OpenAI itself
    models: tuple[ModelInfo, ...] = ()


# Google Gemini
GEMINI_MODELS = (
    ModelInfo(
        id="gemini-3.7-flash",
        display_name="Gemini 3.7 Flash (New)",
        input_price_per_1m=0.75,
        output_price_per_1m=3.75,
        supports_vision=True,
    ),
    ModelInfo(
        id="gemini-3.6-flash",
        default_max_tokens=2048,
        display_name="Gemini 3.6 Flash",
        input_price_per_1m=1.50,
        output_price_per_1m=7.50,
        supports_vision=True,
        accepts_temperature=False,
    ),
    ModelInfo(
        id="gemini-3.5-flash",
        display_name="Gemini 3.5 Flash",
        input_price_per_1m=1.50,
        output_price_per_1m=9.00,
        supports_vision=True,
    ),
    ModelInfo(
        id="gemini-3.5-flash-lite",
        display_name="Gemini 3.5 Flash-Lite",
        input_price_per_1m=0.30,
        output_price_per_1m=2.50,
        supports_vision=True,
        accepts_temperature=False,
    ),
    ModelInfo(
        id="gemini-3.1-pro-preview",
        default_max_tokens=2048,
        display_name="Gemini 3.1 Pro (Preview)",
        input_price_per_1m=2.00,
        output_price_per_1m=12.00,
        supports_vision=True,
    ),
    ModelInfo(
        id="gemini-3.1-flash-lite",
        display_name="Gemini 3.1 Flash-Lite",
        input_price_per_1m=0.25,
        output_price_per_1m=1.50,
        supports_vision=True,
    ),
    ModelInfo(
        id="gemini-3-flash-preview",
        display_name="Gemini 3 Flash (Preview)",
        input_price_per_1m=0.50,
        output_price_per_1m=3.00,
        supports_vision=True,
    ),
    ModelInfo(
        id="gemini-2.5-pro",
        default_max_tokens=2048,
        display_name="Gemini 2.5 Pro",
        input_price_per_1m=1.25,
        output_price_per_1m=10.00,
        supports_vision=True,
    ),
    ModelInfo(
        id="gemini-2.5-flash",
        default_max_tokens=2048,
        display_name="Gemini 2.5 Flash",
        input_price_per_1m=0.30,
        output_price_per_1m=2.50,
        supports_vision=True,
    ),
    ModelInfo(
        id="gemini-2.5-flash-lite",
        display_name="Gemini 2.5 Flash-Lite",
        input_price_per_1m=0.10,
        output_price_per_1m=0.40,
        supports_vision=True,
    ),
)

# Groq (models confirmed to answer on 2026-11-13)
GROQ_MODELS = (
    ModelInfo(
        id="openai/gpt-oss-120b",
        default_max_tokens=2048,
        display_name="OpenAI GPT OSS 120B",
        input_price_per_1m=0.15,
        output_price_per_1m=0.60,
    ),
    ModelInfo(
        id="openai/gpt-oss-20b",
        default_max_tokens=2048,
        display_name="OpenAI GPT OSS 20B",
        input_price_per_1m=0.075,
        output_price_per_1m=0.30,
    ),
    ModelInfo(
        id="qwen/qwen3.8-27b",
        display_name="Qwen 3.8 27B",
        input_price_per_1m=0.30,
        output_price_per_1m=0.30,
    ),
    ModelInfo(
        id="allam-2-7b",
        display_name="Allam 2 7B",
        input_price_per_1m=0.10,
        output_price_per_1m=0.10,
    ),
)

# Hugging Face Inference Providers (OpenAI-compatible router)
# These use the HF router at https://router.huggingface.co/v1 with HF token
HF_MODELS = (
    ModelInfo(
        id="meta-llama/Llama-3.3-70B-Instruct",
        display_name="Llama 3.3 70B (HF Router)",
        input_price_per_1m=0.59,
        output_price_per_1m=0.79,
    ),
    ModelInfo(
        id="meta-llama/Llama-3.1-8B-Instruct",
        display_name="Llama 3.1 8B (HF Router)",
        input_price_per_1m=0.05,
        output_price_per_1m=0.08,
    ),
    ModelInfo(
        id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        display_name="Mixtral 8x7B (HF Router)",
        input_price_per_1m=0.24,
        output_price_per_1m=0.24,
    ),
    ModelInfo(
        id="google/gemma-2-9b-it",
        display_name="Gemma 2 9B IT (HF Router)",
        input_price_per_1m=0.20,
        output_price_per_1m=0.20,
    ),
    ModelInfo(
        id="deepseek-ai/DeepSeek-V3",
        display_name="DeepSeek V3 (HF Router)",
        input_price_per_1m=0.60,
        output_price_per_1m=1.10,
    ),
    ModelInfo(
        id="Qwen/Qwen2.5-72B-Instruct",
        display_name="Qwen 2.5 72B (HF Router)",
        input_price_per_1m=0.60,
        output_price_per_1m=3.00,
    ),
    ModelInfo(
        id="openai/gpt-oss-120b",
        default_max_tokens=2048,
        display_name="GPT-OSS 120B (HF Router)",
        input_price_per_1m=0.15,
        output_price_per_1m=0.60,
    ),
)

# OpenAI
OPENAI_MODELS = (
    ModelInfo(
        id="gpt-4o",
        display_name="GPT-4o",
        input_price_per_1m=2.50,
        output_price_per_1m=10.00,
        supports_vision=True,
    ),
    ModelInfo(
        id="gpt-4o-mini",
        display_name="GPT-4o mini",
        input_price_per_1m=0.15,
        output_price_per_1m=0.60,
        supports_vision=True,
    ),
    ModelInfo(
        id="o1",
        default_max_tokens=2048,
        display_name="o1 (Reasoning)",
        input_price_per_1m=15.00,
        output_price_per_1m=60.00,
        accepts_temperature=False,
    ),
    ModelInfo(
        id="o1-mini",
        default_max_tokens=2048,
        display_name="o1-mini (Reasoning)",
        input_price_per_1m=3.00,
        output_price_per_1m=12.00,
        accepts_temperature=False,
    ),
    ModelInfo(
        id="o3-mini",
        default_max_tokens=2048,
        display_name="o3-mini (Reasoning)",
        input_price_per_1m=1.10,
        output_price_per_1m=4.40,
        accepts_temperature=False,
    ),
    ModelInfo(
        id="gpt-4.1",
        display_name="GPT-4.1",
        input_price_per_1m=2.00,
        output_price_per_1m=8.00,
        supports_vision=True,
    ),
    ModelInfo(
        id="gpt-4.1-mini",
        display_name="GPT-4.1 mini",
        input_price_per_1m=0.40,
        output_price_per_1m=1.60,
        supports_vision=True,
    ),
    ModelInfo(
        id="gpt-5.4",
        default_max_tokens=2048,
        display_name="GPT-5.4 (Latest Flagship)",
        input_price_per_1m=2.50,
        output_price_per_1m=15.00,
        supports_vision=True,
        accepts_temperature=False,
    ),
    ModelInfo(
        id="gpt-5.4-mini",
        default_max_tokens=2048,
        display_name="GPT-5.4 Mini",
        input_price_per_1m=0.75,
        output_price_per_1m=4.50,
        supports_vision=True,
        accepts_temperature=False,
    ),
    ModelInfo(
        id="gpt-5.4-nano",
        default_max_tokens=2048,
        display_name="GPT-5.4 Nano",
        input_price_per_1m=0.20,
        output_price_per_1m=1.25,
        accepts_temperature=False,
    ),
)

# Registry
ALL_PROVIDERS: tuple[ProviderInfo, ...] = (
    ProviderInfo(
        id="gemini",
        display_name="Google Gemini",
        protocol="gemini",
        models=GEMINI_MODELS,
    ),
    ProviderInfo(
        id="groq",
        display_name="Groq",
        protocol="openai",
        base_url="https://api.groq.com/openai/v1",
        models=GROQ_MODELS,
    ),
    ProviderInfo(
        id="huggingface",
        display_name="Hugging Face (Inference Providers)",
        protocol="openai",
        base_url="https://router.huggingface.co/v1",
        models=HF_MODELS,
    ),
    ProviderInfo(
        id="openai",
        display_name="OpenAI",
        protocol="openai",
        models=OPENAI_MODELS,
    ),
)

PROVIDER_BY_ID = {p.id: p for p in ALL_PROVIDERS}

def get_provider(provider_id: str) -> ProviderInfo | None:
    """Look up a provider by its short ID."""
    return PROVIDER_BY_ID.get(provider_id)

def get_model(provider_id: str, model_id: str) -> ModelInfo | None:
    """Look up a model by provider ID and model ID."""
    provider = get_provider(provider_id)
    if not provider:
        return None
    for m in provider.models:
        if m.id == model_id:
            return m
    return None