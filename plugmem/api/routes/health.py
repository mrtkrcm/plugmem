"""Health check endpoint."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from plugmem import __version__
from plugmem.api.dependencies import get_config, get_embedder, get_graph_manager, get_llm
from plugmem.api.schemas import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    cfg = get_config()

    llm_ok = False
    try:
        llm = get_llm(cfg)
        if llm is not None:
            llm_ok = True
    except Exception:
        logger.debug("LLM health check failed", exc_info=True)

    embedding_ok = False
    try:
        embedder = get_embedder(cfg)
        if embedder is not None:
            embedding_ok = True
    except Exception:
        logger.debug("Embedding health check failed", exc_info=True)

    storage_ok = False
    try:
        gm = get_graph_manager(cfg)
        gm.list_graphs()
        storage_ok = True
    except Exception:
        logger.debug("Storage health check failed (backend=%s)", cfg.storage_backend, exc_info=True)

    return HealthResponse(
        status="ok" if (llm_ok and embedding_ok and storage_ok) else "degraded",
        version=__version__,
        llm_available=llm_ok,
        embedding_available=embedding_ok,
        chroma_available=storage_ok,
        storage_available=storage_ok,
        storage_backend=cfg.storage_backend,
    )
