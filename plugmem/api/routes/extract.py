"""Promotion-gate extraction endpoint.

Takes a list of candidates (failure_delta / correction text windows)
and asks the configured LLM to emit 0-N structured memory nodes.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from plugmem.api.auth import require_api_key
from plugmem.api.dependencies import get_llm
from plugmem.api.schemas import ExtractRequest, ExtractResponse, ExtractedMemory
from plugmem.inference.promotion import extract_coding_memories


logger = logging.getLogger(__name__)
router = APIRouter(tags=["extract"], dependencies=[Depends(require_api_key)])


@router.post("/extract", response_model=ExtractResponse)
async def extract(body: ExtractRequest) -> ExtractResponse:
    if not body.candidates:
        return ExtractResponse(memories=[])

    candidates = [{"kind": c.kind, "window": c.window} for c in body.candidates]
    llm = get_llm()
    raw = extract_coding_memories(llm, candidates)

    memories = []
    for m in raw:
        try:
            memories.append(ExtractedMemory(**m))
        except Exception:
            logger.info("extract: dropping invalid memory %r", m)

    return ExtractResponse(memories=memories)
