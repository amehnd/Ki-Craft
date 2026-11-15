"""Retrieve-only endpoint: the chunks for a question, with no generated answer."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Request

from api.openapi_examples import QUERY_BODY_EXAMPLES, RICH_DESCRIPTION_COMMON, error_responses
from api.schemas import RetrieveRequest, RetrieveResponse

router = APIRouter(tags=["retrieval"])


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    summary="Retrieve chunks (no answer)",
    description=(
        "Embed the question, fetch the most relevant chunks and return them "
        "with their text, tables and images.\n\n"
        "Use this when you only need context (e.g. to feed another model)."
    )
    + RICH_DESCRIPTION_COMMON,
    response_description="Retrieved chunks with multimodal content",
    responses=error_responses(503),
)
async def retrieve(
    request: Request,
    req: Annotated[RetrieveRequest, Body(openapi_examples=QUERY_BODY_EXAMPLES)],
) -> RetrieveResponse:
    """Retrieve chunks for a query; returns text, tables and images."""
    return await request.app.state.service.retrieve(req.query, req.top_k)
