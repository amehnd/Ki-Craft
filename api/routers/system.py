"""Index and health endpoints.

``/`` lists what the API exposes and ``/health`` reports whether the vector
store loads. Both stay unversioned so uptime probes have a fixed URL.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from api.service import StoreUnavailableError

router = APIRouter(tags=["meta"])


@router.get("/", summary="List endpoints")
def root(request: Request) -> dict:
    """Human-readable index of what this API exposes."""
    return {
        "name": "Multimodal RAG API",
        "version": request.app.version,
        "docs": "/docs",
        "endpoints": {
            "POST /v1/retrieve": "Retrieve chunks only (text, tables, images)",
            "POST /v1/query": "Retrieve chunks and generate a full answer",
            "GET /v1/query/stream": "SSE stream of the answer with retrieval summary",
            "POST /v1/query/custom": "The same answer from a provider and model chosen in the body",
            "GET /v1/images/{response_id}/{index}": "Fetch a single figure as JPEG",
            "GET /health": "Service health / vector store status",
        },
    }


@router.get(
    "/health",
    summary="Service health",
    response_description="The vector store is loaded",
    responses={503: {"description": "The vector store could not be loaded"}},
)
async def health(request: Request):
    """Report whether the vector store is loadable in this runtime.

    Returns 200 when it is and 503 when it is not, so an uptime probe can
    read the status code alone. The first call may take a few seconds while
    the embedding model and the Chroma store load into memory.
    """
    try:
        await request.app.state.service.ensure_ready()
    except StoreUnavailableError as exc:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "vector_store": "unavailable", "detail": str(exc)},
        )
    return {"status": "ok", "vector_store": "loaded"}
