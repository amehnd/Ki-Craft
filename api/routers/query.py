"""Answer endpoints: retrieve chunks, then generate an answer from them.

``/v1/query`` returns the whole answer at once and ``/v1/query/stream`` sends
it token by token over Server-Sent Events. ``/v1/query`` answers with the
server's default model, or with the provider, model and API key in the body's
optional ``llm`` object. The paths below are short; ``api/main.py`` adds the
``/v1`` prefix.
"""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Body, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from api.openapi_examples import (
    QUERY_BODY_EXAMPLES,
    QUERY_PARAM_EXAMPLES,
    RICH_DESCRIPTION_COMMON,
    error_responses,
)
from api.schemas import QueryRequest, QueryResponse
from config.settings import GENERATION_MAX_IMAGES, RETRIEVAL_K
from src.llm import UnknownModelError, create_llm

router = APIRouter(tags=["generation"])


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Retrieve chunks and generate an answer",
    description=(
        "Everything /v1/retrieve returns, plus a complete answer generated "
        "from the retrieved chunks.\n\n"
        "Without ``llm`` the server's default model answers. With ``llm`` the "
        "answer comes from that provider and model, using the API key you "
        "send:\n\n"
        "- ``provider``: one of ``gemini``, ``groq``, ``huggingface``, ``openai``\n"
        "- ``model``: a model ID listed for that provider\n"
        "- ``api_key``: your key for the provider (required whenever ``llm`` is sent)\n"
        "- ``temperature``, ``max_tokens``: optional\n\n"
        "Pick a ready-made body from the Examples list below.\n\n"
        "A provider or model that is not in the catalogue is rejected with 422 "
        "before retrieval runs. A failing provider (bad key, quota, timeout) "
        "returns 502 with the provider's message. Answers are grounded in the "
        "retrieved chunks; no web search is used."
    )
    + RICH_DESCRIPTION_COMMON,
    response_description="Retrieved chunks plus the generated answer",
    responses=error_responses(502, 503),
)
async def query(
    request: Request,
    req: Annotated[QueryRequest, Body(openapi_examples=QUERY_BODY_EXAMPLES)],
) -> QueryResponse:
    """Retrieve chunks and generate a full answer.

    Raises:
        HTTPException: 422 when ``llm`` names a provider or model that is
            not in the catalogue.
    """
    llm = None
    if req.llm is not None:
        try:
            # Built before retrieval, so an unknown provider or model fails fast.
            llm = create_llm(
                req.llm.provider,
                req.llm.model,
                req.llm.api_key.get_secret_value(),
                temperature=req.llm.temperature,
                max_tokens=req.llm.max_tokens,
            )
        except UnknownModelError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    answer, base, usage = await request.app.state.service.answer(
        req.query,
        req.top_k,
        req.max_images,
        llm=llm,
        provider_id=req.llm.provider if req.llm else None,
        model_id=req.llm.model if req.llm else None,
    )
    return QueryResponse(
        **base.model_dump(),
        answer=answer,
        answer_characters=len(answer),
        usage=usage,
    )


def _sse(event: str, data: dict) -> str:
    """Format one Server-Sent Events frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.get(
    "/query/stream",
    summary="Stream an answer over SSE",
    description=(
        "Streams the answer token-by-token over Server-Sent Events. Event flow:\n\n"
        "1. ``event: retrieval`` - full chunk payloads (like /v1/retrieve),\n"
        "2. ``event: token``    - one per answer fragment (field ``delta``),\n"
        "3. ``event: done``     - final answer, chunk count and usage,\n"
        "4. ``event: error``    - only on failure (``message`` says why).\n\n"
        "Open the URL directly in a browser to watch the stream, or consume it "
        "with any SSE client. Works with e.g. ``curl -N``."
    ),
    response_description="text/event-stream of retrieval + answer tokens",
    responses=error_responses(503),
)
async def query_stream(
    request: Request,
    query: str = Query(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's question about the documents.",
        examples=QUERY_PARAM_EXAMPLES,
    ),
    top_k: int = Query(
        RETRIEVAL_K,
        ge=1,
        le=30,
        description="How many chunks to retrieve; all of them are sent to the model.",
        examples=[RETRIEVAL_K],
    ),
    max_images: int = Query(
        GENERATION_MAX_IMAGES,
        ge=0,
        le=50,
        description="Most images to send to the model, taken in chunk order (0 = all).",
        examples=[GENERATION_MAX_IMAGES],
    ),
) -> StreamingResponse:
    """Stream an answer over SSE.

    Emits a ``retrieval`` event (full chunk payloads), then one ``token`` event
    per answer chunk, then a final ``done`` event.  On failure an ``error``
    event is emitted instead.
    """
    service = request.app.state.service
    await service.ensure_ready()

    def generate():
        try:
            collected: list[str] = []
            for kind, value in service.stream_answer(query, top_k, max_images):
                if kind == "chunks":
                    base = service.to_retrieve_response(query, value)
                    yield _sse("retrieval", base.model_dump())
                elif kind == "token":
                    collected.append(value)
                    yield _sse("token", {"delta": value})
                else:
                    answer = "".join(collected)
                    yield _sse(
                        "done",
                        {
                            "answer": answer,
                            "answer_characters": len(answer),
                            "num_chunks": base.num_chunks,
                            "usage": value,
                        },
                    )
        except Exception as exc:
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(generate(), media_type="text/event-stream")
