"""Pydantic request/response models for the FastAPI layer.

Every field carries a description and a sample value, and every response
model an example, so the OpenAPI docs (``/docs``) explain themselves. Whole
request-body examples live with the routes, in ``api/openapi_examples.py``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from config.settings import GENERATION_MAX_IMAGES, RETRIEVAL_K


class RetrieveRequest(BaseModel):
    """A retrieval request: the question and how much context to fetch.

    ``top_k`` overrides the default retrieval size for this call without
    rebuilding the vector store.
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description=(
            "The user's question about the documents. The model retrieves the "
            "most relevant chunks and answers using their text, tables and images."
        ),
        examples=[
            "Tell me about the pin configuration of the LM317.",
            "Tell me about the pin configuration of the LM2596.",
        ],
    )
    top_k: int = Field(
        RETRIEVAL_K,
        ge=1,
        le=30,
        description=(
            "How many chunks to retrieve. Every retrieved chunk's text and "
            "tables are sent to the model, so this also sets the context size."
        ),
        examples=[RETRIEVAL_K],
    )
    max_images: int = Field(
        GENERATION_MAX_IMAGES,
        ge=0,
        le=50,
        description=(
            "Most images to send to the model, taken in chunk order. "
            "0 sends every retrieved image. Images cost far more tokens than "
            "text, so this is the main cost control."
        ),
        examples=[GENERATION_MAX_IMAGES],
    )


class LLMSettings(BaseModel):
    """The model to answer with, and the caller's own key for it.

    Sent as the optional ``llm`` object of a query. ``api_key`` is required:
    without it a caller could run the server's own key against any model.
    It is a ``SecretStr``, so printing, logging or dumping the request shows
    ``**********``; only ``get_secret_value()`` returns the real key.
    """

    provider: str = Field(
        ...,
        description="Provider ID: 'gemini', 'groq', 'huggingface', or 'openai'",
        examples=["gemini"],
    )
    model: str = Field(
        ...,
        description="Model ID, as listed for that provider in config/providers.py",
        examples=["gemini-3.5-flash-lite"],
    )
    api_key: SecretStr = Field(
        ...,
        min_length=1,
        description="Your API key for the provider. Used for this request only; never stored, logged or traced.",
    )
    temperature: float = Field(
        0.0,
        ge=0.0,
        le=2.0,
        description=(
            "Sampling temperature. Not sent to models that reject or ignore "
            "it (Gemini 3.5 Flash-Lite and 3.6 Flash, OpenAI o-series and "
            "GPT-5.4); they use the provider's default."
        ),
        examples=[0.0],
    )
    max_tokens: int | None = Field(
        None,
        ge=1,
        le=8192,
        description=(
            "Maximum output tokens, including any tokens the model spends "
            "reasoning. Leave out to use the model's default from the "
            "catalogue: 512, or 2048 for models that reason by default "
            "(Groq GPT-OSS, OpenAI o-series and GPT-5.4, some Gemini models)."
        ),
        examples=[512],
    )


class QueryRequest(RetrieveRequest):
    """A retrieval plus generation request.

    Without ``llm`` the server's default model answers with the server's key.
    """

    llm: LLMSettings | None = Field(
        None,
        description=(
            "Answer with this provider and model, using your own API key. "
            "Leave out to use the server's default model."
        ),
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------
class ChunkPayload(BaseModel):
    """One retrieved chunk with its full multimodal content.

    Figures are not inline: ``image_urls`` lists one stable URL per figure,
    ``/v1/images/{record_id}/{position}``, built from the chunk's Chroma
    record id and the figure's 0-based position within the chunk.
    ``image_count`` counts those URLs; figures stored inline have none.
    """

    chunk_id: int = Field(..., description="1-based position in this response")
    record_id: str | None = Field(
        None,
        description="The chunk's id in the vector store; stable, and part of its image URLs",
    )
    enhanced_content: str = Field(
        ..., description="AI-enhanced searchable summary of the chunk"
    )
    raw_text: str = Field("", description="Original OCR text extracted from the PDF")
    tables_html: list[str] = Field(default_factory=list, description="HTML table markup")
    image_count: int = Field(0, description="Number of figures in image_urls")
    image_urls: list[str] = Field(
        default_factory=list,
        description="Stable URL of each figure, /v1/images/{record_id}/{position}",
    )
    has_table: bool = False
    has_image: bool = False
    source: str | None = Field(None, description="Source document, if known")
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chunk_id": 1,
                "record_id": "004ca22b-a528-444e-9c0e-27c8b5858ff5",
                "enhanced_content": (
                    "The LM317 is an adjustable linear voltage regulator. "
                    "Its three pins are: ADJ (adjust), VOUT (output) and "
                    "VIN (input). The output voltage is set by the resistor "
                    "divider on the ADJ pin."
                ),
                "raw_text": (
                    "Pin 1 (ADJ): adjustment pin used to set output voltage via "
                    "a resistor divider. Pin 2 (VOUT): regulated output. "
                    "Pin 3 (VIN): unregulated input."
                ),
                "tables_html": [
                    "<table><tr><th>Pin</th><th>Name</th><th>Function</th></tr>"
                    "<tr><td>1</td><td>ADJ</td><td>Adjust</td></tr></table>"
                ],
                "image_count": 2,
                "image_urls": [
                    "/v1/images/004ca22b-a528-444e-9c0e-27c8b5858ff5/0",
                    "/v1/images/004ca22b-a528-444e-9c0e-27c8b5858ff5/1",
                ],
                "has_table": True,
                "has_image": True,
                "source": "lm317_datasheet.pdf",
                "metadata": {"has_table": True, "has_image": True},
            }
        }
    )


class RetrieveResponse(BaseModel):
    """Result of a retrieve-only request (no answer generation)."""

    query: str
    num_chunks: int = Field(..., description="Number of chunks returned")
    total_images: int = Field(..., description="Figures (image URLs) across all chunks")
    total_tables: int = Field(..., description="Tables across all chunks")
    chunks: list[ChunkPayload] = Field(default_factory=list)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Tell me about the pin configuration of the LM317.",
                "num_chunks": 2,
                "total_images": 2,
                "total_tables": 1,
                "chunks": [
                    {
                        "chunk_id": 1,
                        "record_id": "004ca22b-a528-444e-9c0e-27c8b5858ff5",
                        "enhanced_content": "The LM317 is an adjustable linear voltage regulator...",
                        "raw_text": "Pin 1 (ADJ): ...",
                        "tables_html": ["<table>..."],
                        "image_count": 2,
                        "image_urls": [
                            "/v1/images/004ca22b-a528-444e-9c0e-27c8b5858ff5/0",
                            "/v1/images/004ca22b-a528-444e-9c0e-27c8b5858ff5/1",
                        ],
                        "has_table": True,
                        "has_image": True,
                        "source": "lm317_datasheet.pdf",
                        "metadata": {"has_table": True, "has_image": True},
                    }
                ],
            }
        }
    )


class UsageMetadata(BaseModel):
    """Token usage and generation stats for a single LLM call.

    ``input_tokens``/``output_tokens`` come straight from the provider's usage
    metadata (Gemini reports them on the final response/stream chunk).
    ``estimated_cost_usd`` is derived from per-1M-token pricing in
    ``config.settings`` and is ``None`` when pricing is disabled.
    """

    model: str = Field(..., description="Model used for generation")
    input_tokens: int = Field(0, description="Prompt (input) tokens sent to the model")
    output_tokens: int = Field(0, description="Tokens generated by the model")
    total_tokens: int = Field(0, description="input_tokens + output_tokens")
    latency_ms: float | None = Field(None, description="Generation wall time in milliseconds")
    prompt_chars: int = Field(0, description="Length of the text prompt in characters")
    prompt_images: int = Field(0, description="Number of images attached to the prompt")
    chunks_used: int = Field(0, description="Number of retrieved chunks sent to the model")
    estimated_cost_usd: float | None = Field(
        None, description="Estimated generation cost in USD, when pricing is configured"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model": "gemini-2.5-flash-lite",
                "input_tokens": 1024,
                "output_tokens": 256,
                "total_tokens": 1280,
                "latency_ms": 1840.5,
                "prompt_chars": 8200,
                "prompt_images": 4,
                "chunks_used": 5,
                "estimated_cost_usd": 0.0002,
            }
        }
    )


class QueryResponse(RetrieveResponse):
    """Result of a full query: retrieved chunks plus the generated answer."""

    answer: str = Field(..., description="The LLM-generated answer")
    answer_characters: int = Field(..., description="Length of the answer in characters")
    usage: "UsageMetadata | None" = Field(
        None,
        description=(
            "Token usage and generation stats from the LLM call, when the "
            "provider reported them."
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Tell me about the pin configuration of the LM317.",
                "num_chunks": 2,
                "total_images": 2,
                "total_tables": 1,
                "answer": (
                    "The LM317 is an adjustable linear voltage regulator with "
                    "three pins. Pin 1 is the adjust (ADJ) pin, used to set the "
                    "output voltage with a resistor divider. Pin 2 is the "
                    "regulated output (VOUT), and pin 3 is the unregulated input "
                    "(VIN). The output voltage is approximately 1.25 V times "
                    "(1 + R2/R1)."
                ),
                "answer_characters": 340,
                "usage": {
                    "model": "gemini-2.5-flash-lite",
                    "input_tokens": 1024,
                    "output_tokens": 256,
                    "total_tokens": 1280,
                    "latency_ms": 1840.5,
                    "prompt_chars": 8200,
                    "prompt_images": 4,
                    "chunks_used": 5,
                    "estimated_cost_usd": 0.0002,
                },
                "chunks": [],
            }
        }
    )
