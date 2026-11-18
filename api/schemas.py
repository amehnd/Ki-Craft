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
    """A question and how many chunks to retrieve for it."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question to answer from the indexed documents.",
        examples=["Tell me about the pin configuration of the LM317."],
    )
    top_k: int = Field(
        RETRIEVAL_K,
        ge=1,
        le=30,
        description=(
            "Number of chunks to retrieve. For /v1/query every retrieved chunk's "
            "text and tables go to the model, so this also sets the context size."
        ),
        examples=[RETRIEVAL_K],
    )


class LLMSettings(BaseModel):
    """The model that answers, and your API key for its provider."""

    provider: str = Field(
        ...,
        description="Provider ID: `gemini`, `groq`, `huggingface` or `openai`.",
        examples=["gemini"],
    )
    model: str = Field(
        ...,
        description="Model ID from the server's catalogue for that provider.",
        examples=["gemini-3.5-flash-lite"],
    )
    # Required, so a caller can never run the server's own key against a model
    # of their choice. SecretStr prints as ********** in logs and dumps.
    api_key: SecretStr = Field(
        ...,
        min_length=1,
        description="Your API key for the provider. Used for this request only; never stored or logged.",
    )
    temperature: float = Field(
        0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Ignored by models with fixed sampling.",
        examples=[0.0],
    )
    max_tokens: int | None = Field(
        None,
        ge=1,
        le=8192,
        description=(
            "Maximum output tokens, reasoning included. Defaults to the model's "
            "catalogue value: 512, or 2048 for models that reason by default."
        ),
        examples=[512],
    )


class QueryRequest(RetrieveRequest):
    """A question to answer, with optional figure limit and model choice."""

    max_images: int = Field(
        GENERATION_MAX_IMAGES,
        ge=0,
        le=50,
        description=(
            "Maximum figures sent to the model, taken in chunk order; 0 sends all. "
            "Figures cost far more tokens than text. Text-only models get none."
        ),
        examples=[GENERATION_MAX_IMAGES],
    )
    llm: LLMSettings | None = Field(
        None,
        description="Answer with this provider and model using your own key. Omit to use the server's default model.",
    )


class ValidationIssue(BaseModel):
    """One reason a request was rejected."""

    type: str = Field(..., description="Error type.", examples=["missing"])
    loc: list[str | int] = Field(..., description="Where the problem is.", examples=[["body", "llm", "api_key"]])
    msg: str = Field(..., description="What is wrong.", examples=["Field required"])
    ctx: dict[str, Any] | None = Field(None, description="Extra detail for some error types.")


class ValidationErrorResponse(BaseModel):
    """The body of a 422 response. The rejected value is never echoed back."""

    detail: list[ValidationIssue]


class ErrorResponse(BaseModel):
    """The body of an error response. A 422 has its own schema."""

    detail: str = Field(..., description="What went wrong.", examples=["The model provider failed: quota exceeded"])


class ChunkPayload(BaseModel):
    """One retrieved chunk: its summary, original text, tables and figure URLs."""

    chunk_id: int = Field(..., description="1-based position of the chunk in this response.")
    record_id: str | None = Field(
        None,
        description="Stable ID of the chunk in the store; part of its figure URLs.",
    )
    enhanced_content: str = Field(
        ..., description="Summary written at ingestion; the text that is searched."
    )
    raw_text: str = Field("", description="Text extracted from the PDF.")
    tables_html: list[str] = Field(default_factory=list, description="Each table as an HTML `<table>` string.")
    image_count: int = Field(0, description="Number of entries in `image_urls`.")
    image_urls: list[str] = Field(
        default_factory=list,
        description="Relative URL of each figure, `/v1/images/{record_id}/{position}`.",
    )
    has_table: bool = Field(False, description="Whether the chunk has tables.")
    has_image: bool = Field(False, description="Whether the chunk has figures.")
    source: str | None = Field(None, description="Name of the source document.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Other stored fields of the chunk.")

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
                "source": "lm317",
                "metadata": {"has_table": True, "has_image": True},
            }
        }
    )


class RetrieveResponse(BaseModel):
    """The chunks retrieved for a question."""

    query: str = Field(..., description="The question, as received.")
    num_chunks: int = Field(..., description="Number of chunks returned.")
    total_images: int = Field(..., description="Figures across all chunks.")
    total_tables: int = Field(..., description="Tables across all chunks.")
    chunks: list[ChunkPayload] = Field(default_factory=list, description="The chunks, most relevant first.")

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
                        "source": "lm317",
                        "metadata": {"has_table": True, "has_image": True},
                    }
                ],
            }
        }
    )


class UsageMetadata(BaseModel):
    """Token counts and timing of the model call.

    Token counts are as the provider reports them. The cost estimate uses
    the answering model's prices in ``config/providers.py`` and is null when
    the catalogue has no price for it.
    """

    model: str = Field(..., description="Model ID that answered.")
    input_tokens: int = Field(0, description="Input tokens, as reported by the provider.")
    output_tokens: int = Field(0, description="Output tokens, reasoning included.")
    total_tokens: int = Field(0, description="Input plus output tokens.")
    latency_ms: float | None = Field(None, description="Time spent generating, in milliseconds.")
    prompt_chars: int = Field(0, description="Characters of text in the prompt.")
    prompt_images: int = Field(0, description="Figures sent in the prompt.")
    chunks_used: int = Field(0, description="Chunks sent in the prompt.")
    estimated_cost_usd: float | None = Field(
        None, description="Estimated cost in USD at the model's catalogue prices; null when they are unknown."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model": "gemini-3.5-flash-lite",
                "input_tokens": 1024,
                "output_tokens": 256,
                "total_tokens": 1280,
                "latency_ms": 1840.5,
                "prompt_chars": 8200,
                "prompt_images": 1,
                "chunks_used": 2,
                "estimated_cost_usd": 0.0002,
            }
        }
    )


class FigurePayload(BaseModel):
    """A figure sent to the model. The answer cites it by label, as `[F1]`."""

    label: str = Field(..., description="Label in the prompt: `F1`, `F2`, ...", examples=["F1"])
    url: str = Field(
        ...,
        description="Relative URL of the figure, `/v1/images/{record_id}/{position}`.",
        examples=["/v1/images/004ca22b-a528-444e-9c0e-27c8b5858ff5/0"],
    )
    chunk_id: int = Field(..., description="`chunk_id` of its chunk in this response.", examples=[1])
    source: str = Field(..., description="Name of the source document.", examples=["lm317"])


class DocumentSummaryPayload(BaseModel):
    """One indexed document and how much it holds."""

    id: str = Field(..., description="The document's name, its PDF file name without `.pdf`.", examples=["lm317"])
    chunk_count: int = Field(..., description="Stored chunks.", examples=[51])
    image_count: int = Field(..., description="Figures across its chunks.", examples=[40])
    table_count: int = Field(..., description="Tables across its chunks.", examples=[22])


class DocumentList(BaseModel):
    """Every indexed document."""

    items: list[DocumentSummaryPayload] = Field(..., description="The documents, sorted by `id`.")
    total: int = Field(..., description="Number of documents.")


class StoredChunk(BaseModel):
    """One stored chunk of a document, as the chunk explorer shows it."""

    record_id: str = Field(..., description="Stable ID of the chunk in the store; part of its figure URLs.")
    chunk_id: int = Field(..., description="Position of the chunk in its document, counting from 1.")
    source: str = Field(..., description="The document's name.")
    enhanced_content: str = Field(..., description="Summary written at ingestion; the text that is searched.")
    raw_text: str = Field("", description="Text extracted from the PDF.")
    tables_html: list[str] = Field(default_factory=list, description="Each table as an HTML `<table>` string.")
    image_urls: list[str] = Field(
        default_factory=list,
        description="Relative URL of each figure, `/v1/images/{record_id}/{position}`.",
    )
    has_table: bool = Field(False, description="Whether the chunk has tables.")
    has_image: bool = Field(False, description="Whether the chunk has figures.")


class ChunkPage(BaseModel):
    """One page of a document's chunks, in reading order."""

    items: list[StoredChunk] = Field(..., description="The chunks on this page.")
    total: int = Field(..., description="Chunks in the whole document.")
    offset: int = Field(..., description="Chunks skipped before this page.")
    limit: int = Field(..., description="Most chunks a page holds, as requested.")


class ModelPayload(BaseModel):
    """One model a request's `llm` object may name, with what it can do and costs."""

    id: str = Field(..., description="Model id to send as `llm.model`.", examples=["gemini-3.5-flash-lite"])
    display_name: str = Field(..., description="Name to show in a model picker.", examples=["Gemini 3.5 Flash-Lite"])
    supports_vision: bool = Field(..., description="Whether it reads images; a text-only model is sent no figures.")
    accepts_temperature: bool = Field(..., description="Whether `llm.temperature` reaches it; when false it is left out.")
    default_max_tokens: int = Field(..., description="Output budget used when `llm.max_tokens` is left out.")
    input_price_per_1m: float = Field(..., description="USD per million input tokens; 0 when unknown.")
    output_price_per_1m: float = Field(..., description="USD per million output tokens; 0 when unknown.")


class ProviderPayload(BaseModel):
    """One model provider and the models it offers."""

    id: str = Field(..., description="Provider id to send as `llm.provider`.", examples=["gemini"])
    display_name: str = Field(..., description="Name to show in a provider picker.", examples=["Google Gemini"])
    models: list[ModelPayload] = Field(..., description="Its models, in catalogue order.")


class ProviderList(BaseModel):
    """Every provider and model a request's `llm` object accepts."""

    items: list[ProviderPayload] = Field(..., description="The providers, in catalogue order.")
    total: int = Field(..., description="Number of providers.")


class QueryResponse(RetrieveResponse):
    """The retrieved chunks, the generated answer and the figures it drew on."""

    answer: str = Field(..., description="The generated answer.")
    answer_characters: int = Field(..., description="Length of the answer in characters.")
    figures: list[FigurePayload] = Field(
        default_factory=list,
        description="Figures sent to the model, in prompt order. Empty for a text-only model.",
    )
    usage: "UsageMetadata | None" = Field(None, description="Token counts and timing of the model call.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Tell me about the pin configuration of the LM317.",
                "num_chunks": 2,
                "total_images": 1,
                "total_tables": 1,
                "answer": (
                    "The LM317 is an adjustable linear voltage regulator with "
                    "three pins [F1]. Pin 1 is the adjust (ADJ) pin, used to set the "
                    "output voltage with a resistor divider. Pin 2 is the "
                    "regulated output (VOUT), and pin 3 is the unregulated input "
                    "(VIN). The output voltage is approximately 1.25 V times "
                    "(1 + R2/R1)."
                ),
                "answer_characters": 302,
                "figures": [
                    {
                        "label": "F1",
                        "url": "/v1/images/004ca22b-a528-444e-9c0e-27c8b5858ff5/0",
                        "chunk_id": 1,
                        "source": "lm317",
                    }
                ],
                "usage": {
                    "model": "gemini-3.5-flash-lite",
                    "input_tokens": 1024,
                    "output_tokens": 256,
                    "total_tokens": 1280,
                    "latency_ms": 1840.5,
                    "prompt_chars": 8200,
                    "prompt_images": 1,
                    "chunks_used": 2,
                    "estimated_cost_usd": 0.0002,
                },
                "chunks": [],
            }
        }
    )
