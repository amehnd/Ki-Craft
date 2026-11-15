"""Chat model factory: builds the LangChain client for a catalogue model.

Generation asks for a model by provider id and model id, both default
(``src/retrieval/generate.py:default_llm``) and user-chosen (the ``llm``
object of ``POST /v1/query``, in ``api/routers/query.py``).
This module looks both up in ``config/providers.py`` and returns a ready chat
model. The provider's ``protocol``, not the shape of the model's name,
decides which client class is used, and anything not in the catalogue is
rejected.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from config.providers import get_model, get_provider
from src.logger import get_logger

logger = get_logger(__name__)


class UnknownModelError(ValueError):
    """The provider or model asked for is not in ``config/providers.py``."""


class LLMProviderError(RuntimeError):
    """The model provider failed, or returned no answer text.

    Raised during generation (``src/retrieval/generate.py``). The API maps it
    to HTTP 502: the provider is the upstream server and it did not deliver.
    """


def accepts_images(provider_id: str, model_id: str) -> bool:
    """Say whether a catalogue model can read images in the prompt.

    Args:
        provider_id: A provider id from ``config/providers.py``.
        model_id: A model id listed under that provider.

    Returns:
        The model's ``supports_vision`` flag; False for a model not in the
        catalogue.
    """
    model = get_model(provider_id, model_id)
    return bool(model and model.supports_vision)


def create_llm(
    provider_id: str,
    model_id: str,
    api_key: str,
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> BaseChatModel:
    """Build the chat model for one provider and model from the catalogue.

    Args:
        provider_id: A provider id from ``config/providers.py`` (``gemini``,
            ``groq``, ``huggingface``, ``openai``).
        model_id: A model id listed under that provider.
        api_key: The key to authenticate with. Never logged.
        temperature: Sampling temperature. Sent only when the model's
            catalogue entry accepts it; otherwise the provider default applies.
        max_tokens: Maximum output tokens, reasoning included. None uses the
            model's ``default_max_tokens`` from the catalogue.

    Returns:
        A LangChain chat model ready for ``invoke`` or ``stream``.

    Raises:
        UnknownModelError: If the provider or the model is not in the
            catalogue.
    """
    provider = get_provider(provider_id)
    if provider is None:
        raise UnknownModelError(f"Unknown provider {provider_id}")
    model = get_model(provider_id, model_id)
    if model is None:
        raise UnknownModelError(f"Model {model_id} not found for provider {provider_id}")

    # Some models reject a temperature (OpenAI o-series) or ignore it with a
    # warning (Gemini fixed-sampling models), so it is sent only when accepted.
    sampling = {"temperature": temperature} if model.accepts_temperature else {}
    if max_tokens is None:
        max_tokens = model.default_max_tokens
    logger.info("Creating chat model %s/%s (max_tokens=%d)", provider_id, model_id, max_tokens)

    if provider.protocol == "gemini":
        return ChatGoogleGenerativeAI(
            model=model_id,
            google_api_key=api_key,
            max_output_tokens=max_tokens,
            **sampling,
        )

    return ChatOpenAI(
        model=model_id,
        api_key=api_key,
        base_url=provider.base_url,
        max_tokens=max_tokens,
        **sampling,
    )
