"""Promotion-gate extraction endpoint.

Takes a list of candidates (failure_delta / correction text windows)
and asks the configured LLM to emit 0-N structured memory nodes.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends

from plugmem.api.auth import require_api_key
from plugmem.api.dependencies import get_llm
from plugmem.api.schemas import ExtractRequest, ExtractResponse, ExtractedMemory, RejectedCandidate
from plugmem.inference.promotion import extract_coding_memories


logger = logging.getLogger(__name__)
router = APIRouter(tags=["extract"], dependencies=[Depends(require_api_key)])


@router.post("/extract", response_model=ExtractResponse)
async def extract(body: ExtractRequest) -> ExtractResponse:
    if not body.candidates:
        return ExtractResponse(memories=[])

    candidates = [{"kind": c.kind, "window": c.window} for c in body.candidates]
    llm = get_llm()
    memories_raw, rejections_raw = extract_coding_memories(llm, candidates)

    memories = []
    for m in memories_raw:
        try:
            memories.append(ExtractedMemory(**m))
        except Exception:
            logger.info("extract: dropping invalid memory %r", m)

    rejected: List[RejectedCandidate] = []
    for r in rejections_raw:
        try:
            rejected.append(RejectedCandidate(**r))
        except Exception:
            logger.info("extract: dropping invalid rejection %r", r)

    return ExtractResponse(memories=memories, rejected=rejected)
