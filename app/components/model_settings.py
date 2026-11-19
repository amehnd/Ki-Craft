"""Settings in the Query page's sidebar: how much to retrieve, and which model answers.

Retrieved Chunks and Max Images apply to every question. The provider, model
and key apply only when "Use my own model" is ticked; otherwise the server's
default model answers. The provider and model lists come from
``GET /v1/providers``. Every value is kept in session state (``llm_*``
keys), where ``views/chat.py`` reads it when a question is sent.
"""

from __future__ import annotations

import logging

import streamlit as st

from api_client import API_BASE_URL
from chunk_data import list_providers

logger = logging.getLogger(__name__)


def _render_own_model() -> None:
    """Provider, model, key and limits for answering with the caller's own model."""
    try:
        catalogue = list_providers()
    except Exception as exc:
        logger.warning("Could not list providers from %s: %s", API_BASE_URL, exc)
        st.caption(f"Providers unavailable: `GET {API_BASE_URL}/v1/providers` failed ({exc}).")
        return
    provider_names = {p["id"]: p["display_name"] for p in catalogue}
    models_by_provider = {p["id"]: p["models"] for p in catalogue}

    provider = st.selectbox(
        "Provider",
        options=list(provider_names),
        format_func=lambda x: provider_names.get(x, x),
        key="llm_provider",
    )
    if not provider:
        return
    model_names = {m["id"]: m["display_name"] for m in models_by_provider[provider]}
    st.selectbox("Model", options=list(model_names), format_func=lambda x: model_names.get(x, x), key="llm_model")
    st.text_input(
        "Custom Model ID",
        key="llm_custom_model",
        placeholder="e.g. qwen/qwen3.8-27b (overrides dropdown)",
        help="Paste any model ID available on the provider. Leave empty to use dropdown.",
    )
    st.text_input(
        "API Key",
        type="password",
        key="llm_api_key",
        placeholder=f"Enter your {provider_names[provider]} API key",
        help="Sent with each question to the provider you chose. Not stored.",
    )
    st.number_input("Temperature", min_value=0.0, max_value=2.0, step=0.1, key="llm_temperature")
    st.number_input(
        "Max Tokens",
        min_value=0,
        max_value=8192,
        key="llm_max_tokens",
        help="0 uses the model's default (2048 for models that reason by default, otherwise 512).",
    )


def render_model_settings() -> None:
    """Draw the question and model settings in the sidebar."""
    with st.sidebar:
        st.header("Question settings")
        st.number_input(
            "Retrieved Chunks",
            min_value=1,
            max_value=30,
            step=1,
            key="llm_top_k",
            help="How many chunks to retrieve; all of them are sent to the model.",
        )
        st.number_input(
            "Max Images",
            min_value=0,
            max_value=50,
            step=1,
            key="llm_max_images",
            help="Most images to send to the model, in chunk order. 0 = all.",
        )
        st.header("Model")
        st.checkbox(
            "Use my own model",
            key="use_custom_provider",
            help="Answer with the provider, model and API key below instead of the server's default model.",
        )
        if st.session_state.use_custom_provider:
            _render_own_model()
        else:
            st.caption("The server's default model answers.")
