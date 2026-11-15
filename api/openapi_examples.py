"""Text and example payloads shown in the interactive API docs (``/docs``).

Shared by the retrieval and query routers. Nothing here affects how a request
is handled; it only shapes the OpenAPI schema FastAPI generates.
"""

from __future__ import annotations

# Named examples for request bodies, in the shape FastAPI's
# ``Body(openapi_examples=...)`` expects: each key is one entry in the
# Examples dropdown on /docs, ``summary`` is its label, ``description`` the
# note under it, and ``value`` the payload. Every example spells out every
# setting, so a reader sees the whole body without opening the schema.
_LM2596 = "Tell me about the pin configuration of the LM2596."

RETRIEVE_BODY_EXAMPLES = {
    "lm2596_pins": {
        "summary": "LM2596 pin configuration",
        "value": {"query": _LM2596, "top_k": 5},
    },
    "lm317_pins": {
        "summary": "LM317, 10 chunks",
        "value": {"query": "Tell me about the pin configuration of the LM317.", "top_k": 10},
    },
}

QUERY_BODY_EXAMPLES = {
    "default_model": {
        "summary": "Server's default model (no llm)",
        "description": (
            "Leave out ``llm`` and the server's default model (Gemini 3.5 "
            "Flash-Lite) answers with the server's key. This is how /v1/query "
            "behaved before the ``llm`` object existed."
        ),
        "value": {"query": _LM2596, "top_k": 5, "max_images": 3},
    },
    "default_model_every_image": {
        "summary": "Server's default model, 10 chunks and every image",
        "description": "``max_images`` 0 sends every retrieved figure, which costs far more tokens.",
        "value": {"query": "Tell me about the pin configuration of the LM317.", "top_k": 10, "max_images": 0},
    },
    "gemini": {
        "summary": "Gemini 3.5 Flash-Lite with your key",
        "description": (
            "This model uses fixed sampling, so ``temperature`` is not sent to it."
        ),
        "value": {
            "query": _LM2596,
            "top_k": 5,
            "max_images": 3,
            "llm": {
                "provider": "gemini",
                "model": "gemini-3.5-flash-lite",
                "api_key": "YOUR_GEMINI_API_KEY",
                "temperature": 0.0,
                "max_tokens": 512,
            },
        },
    },
    "groq_gpt_oss": {
        "summary": "Groq GPT-OSS 120B",
        "description": (
            "A text-only model: it gets the chunks' text and tables, no figures, "
            "whatever ``max_images`` says. It reasons before answering, and the "
            "reasoning counts against ``max_tokens``, so its default is 2048; a "
            "budget that runs out before any answer text returns 502."
        ),
        "value": {
            "query": _LM2596,
            "top_k": 5,
            "max_images": 3,
            "llm": {
                "provider": "groq",
                "model": "openai/gpt-oss-120b",
                "api_key": "YOUR_GROQ_API_KEY",
                "temperature": 0.0,
                "max_tokens": 2048,
            },
        },
    },
    "openai_gpt_4o_mini": {
        "summary": "OpenAI GPT-4o mini",
        "description": "Reads images, so up to ``max_images`` figures are sent.",
        "value": {
            "query": _LM2596,
            "top_k": 5,
            "max_images": 3,
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "YOUR_OPENAI_API_KEY",
                "temperature": 0.0,
                "max_tokens": 512,
            },
        },
    },
    "huggingface_llama": {
        "summary": "Hugging Face router, Llama 3.3 70B",
        "description": "A text-only model: no figures are sent. The key is a Hugging Face token.",
        "value": {
            "query": _LM2596,
            "top_k": 5,
            "max_images": 3,
            "llm": {
                "provider": "huggingface",
                "model": "meta-llama/Llama-3.3-70B-Instruct",
                "api_key": "YOUR_HF_TOKEN",
                "temperature": 0.0,
                "max_tokens": 512,
            },
        },
    },
}

QUERY_PARAM_EXAMPLES = [
    "Tell me about the pin configuration of the LM2596.",
    "Tell me about the pin configuration of the LM317.",
]

RICH_DESCRIPTION_COMMON = """
Ask a question about the indexed PDFs. The service:

1. embeds the question with the same model used at ingest time,
2. retrieves the `top_k` most relevant chunks from Chroma,
3. (for `/v1/query` and `/v1/query/stream`) asks the multimodal LLM to answer
   using the chunk text, tables and figures.

### Try it

```json
{
  "query": "Tell me about the pin configuration of the LM317.",
  "top_k": 5
}
```

### Response

Every chunk carries its `record_id`, and each of its figures can be loaded
over HTTP at a stable URL, `GET /v1/images/{record_id}/{position}`.
"""



def error_responses(*codes: int) -> dict[int, dict]:
    """Describe a route's error status codes for ``/docs``.

    Every error body has one shape, ``{"detail": "..."}``; the handlers that
    produce them are in ``api/main.py``. 422 is not listed: FastAPI already
    documents it, with its schema, on every route that takes a body.
    """
    descriptions = {
        500: "Unexpected server error; details are in the server log",
        502: "The model provider failed, or returned no answer text",
        503: "The vector store could not be loaded",
    }
    return {code: {"description": descriptions[code]} for code in codes}
