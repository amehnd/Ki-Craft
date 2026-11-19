"""Model settings in the Query page's sidebar: provider, model, key and limits.

The provider and model lists come from ``GET /v1/providers``. The values are
kept in session state (``llm_*`` keys); ``views/chat.py`` reads them when a
question is sent.
"""

from __future__ import annotations

import logging

import streamlit as st

from api_client import API_BASE_URL
from chunk_data import list_providers

logger = logging.getLogger(__name__)


def render_model_settings() -> None:
    """Draw the model settings in the sidebar."""
    with st.sidebar:
        st.header("Model")
        try:
            catalogue = list_providers()
        except Exception as exc:
            logger.warning("Could not list providers from %s: %s", API_BASE_URL, exc)
            st.caption(f"Providers unavailable: `GET {API_BASE_URL}/v1/providers` failed ({exc}).")
            catalogue = []
        provider_ids = [p["id"] for p in catalogue]
        provider_names = {p["id"]: p["display_name"] for p in catalogue}
        models_by_provider = {p["id"]: p["models"] for p in catalogue}

        selected_provider = st.selectbox(
            "Provider",
            options=provider_ids,
            format_func=lambda x: provider_names.get(x, x),
            key="llm_provider",
            help="Select LLM provider. API key required for custom queries.",
        )

        if selected_provider:
            models = models_by_provider[selected_provider]
            model_ids = [m["id"] for m in models]
            model_names = {m["id"]: m["display_name"] for m in models}

            st.selectbox(
                "Model",
                options=model_ids,
                format_func=lambda x: model_names.get(x, x),
                key="llm_model",
                help="Select specific model from the provider",
            )
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
                placeholder=f"Enter your {provider_names[selected_provider]} API key",
                help="Your API key for the selected provider. Not stored.",
            )
            st.number_input("Temperature", min_value=0.0, max_value=2.0, step=0.1, key="llm_temperature")
            st.number_input(
                "Max Tokens",
                min_value=0,
                max_value=8192,
                key="llm_max_tokens",
                help="0 uses the model's default (2048 for models that reason by default, otherwise 512).",
            )
            st.number_input(
                "Max Images",
                min_value=0,
                max_value=50,
                step=1,
                key="llm_max_images",
                help="Most images to send to the model, in chunk order. 0 = all.",
            )
            st.number_input(
                "Retrieved Chunks",
                min_value=1,
                max_value=30,
                step=1,
                key="llm_top_k",
                help="How many chunks to retrieve; all of them are sent to the model.",
            )
            st.checkbox(
                "Use Custom Provider",
                key="use_custom_provider",
                help="When checked, queries use the selected provider/model/key instead of the default backend",
            )
