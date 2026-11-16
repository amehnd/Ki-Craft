"""Answer endpoints: retrieve chunks, then generate an answer from them.

``/v1/query`` returns the whole answer at once and ``/v1/query/stream`` sends
it token by token over Server-Sent Events. Both use the server's default
model, or the provider, model and API key in the body's
optional ``llm`` object. The paths below are short; ``api/main.py`` adds the
``/v1`` prefix.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Generator
from contextlib import closing
from typing import Annotated

from anyio import CancelScope
from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.language_models.chat_models import BaseChatModel
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from api.openapi_examples import (
    QUERY_BODY_EXAMPLES,
    RICH_DESCRIPTION_COMMON,
    error_responses,
)
from api.schemas import LLMSettings, QueryRequest, QueryResponse
from api.service import RAGService
from src.llm import LLMProviderError, UnknownModelError, create_llm
from src.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["generation"])


def _requested_llm(llm: LLMSettings | None) -> BaseChatModel | None:
    """Build the requested model, or leave model selection to the service."""
    if llm is None:
        return None
    try:
        return create_llm(
            llm.provider,
            llm.model,
            llm.api_key.get_secret_value(),
            temperature=llm.temperature,
            max_tokens=llm.max_tokens,
        )
    except UnknownModelError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


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
    answer, base, usage = await request.app.state.service.answer(
        req.query,
        req.top_k,
        req.max_images,
        llm=_requested_llm(req.llm),
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


def _stream_frames(
    service: RAGService,
    req: QueryRequest,
    llm: BaseChatModel | None,
) -> Generator[str, None, None]:
    """Format the service events, raising early failures before the first frame."""
    events = service.stream_answer(
        req.query,
        req.top_k,
        req.max_images,
        llm=llm,
        provider_id=req.llm.provider if req.llm else None,
        model_id=req.llm.model if req.llm else None,
    )
    with closing(events):
        _, chunks = next(events)
        _, first_token = next(events)
        retrieval = service.to_retrieve_response(req.query, chunks)
        yield _sse("retrieval", retrieval.model_dump())
        yield _sse("token", {"delta": first_token})
        tokens = [first_token]
        try:
            for kind, value in events:
                if kind == "token":
                    tokens.append(value)
                    yield _sse("token", {"delta": value})
                elif kind == "usage":
                    answer = "".join(tokens)
                    yield _sse(
                        "done",
                        {
                            "answer": answer,
                            "answer_characters": len(answer),
                            "num_chunks": retrieval.num_chunks,
                            "usage": value,
                        },
                    )
        except LLMProviderError as exc:
            yield _sse("error", {"status": 502, "message": str(exc)})
        except Exception:
            logger.exception("Unexpected failure while streaming an answer")
            yield _sse("error", {"status": 500, "message": "Internal server error"})


@router.post(
    "/query/stream",
    response_class=StreamingResponse,
    summary="Stream an answer over SSE",
    description=(
        "Takes the same JSON body as POST /v1/query, including an optional ``llm``. "
        "Once the first answer fragment is ready, the event flow is:\n\n"
        "1. ``event: retrieval`` - full chunk payloads (like /v1/retrieve),\n"
        "2. ``event: token``    - one per answer fragment (field ``delta``),\n"
        "3. ``event: done``     - final answer, chunk count and usage,\n"
        "4. ``event: error``    - replaces ``done`` on mid-stream failure "
        "(``status`` and ``message``).\n\n"
        "A provider failure before the first fragment returns HTTP 502. "
        "Read the stream with browser fetch or curl -N."
    ),
    response_description="text/event-stream of retrieval + answer tokens",
    responses={200: {"content": {"text/event-stream": {}}}, **error_responses(500, 502, 503)},
)
async def query_stream(
    request: Request,
    req: Annotated[QueryRequest, Body(openapi_examples=QUERY_BODY_EXAMPLES)],
) -> StreamingResponse:
    """Take a question and model options and stream the answer over SSE.

    An invalid model raises HTTPException (422). Store and provider errors
    before the response starts reach the application's HTTP error handlers.
    """
    service = request.app.state.service
    llm = _requested_llm(req.llm)
    await service.ensure_ready()
    frames = _stream_frames(service, req, llm)
    # StreamingResponse sends HTTP headers before it reads the body iterator.
    first_frame = await run_in_threadpool(next, frames)

    async def generate() -> AsyncIterator[str]:
        """Send the prepared frame and continue the generator off the event loop."""
        try:
            yield first_frame
            async for frame in iterate_in_threadpool(frames):
                yield frame
        finally:
            with CancelScope(shield=True):
                await run_in_threadpool(frames.close)

    return StreamingResponse(generate(), media_type="text/event-stream")
