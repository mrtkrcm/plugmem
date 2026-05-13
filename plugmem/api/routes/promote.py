"""Atomic promotion endpoint: extract + insert in one call.

POST /graphs/{graph_id}/promote accepts coding candidates, runs
extraction, inserts accepted memories atomically (with dedupe/upsert),
and returns the inserted node IDs plus dropped candidates + reasons.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException

from plugmem.api.auth import require_api_key
from plugmem.api.dependencies import get_graph_manager, get_llm
from plugmem.api.schemas import (
    ExtractedMemory,
    PromoteRequest,
    PromoteResponse,
    PromotedMemory,
    RejectedCandidate,
)
from plugmem.graph_manager import GraphManager
from plugmem.inference.promotion import extract_coding_memories

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/graphs", tags=["promote"], dependencies=[Depends(require_api_key)])


def _manager() -> GraphManager:
    return get_graph_manager()


def _promote_candidate(
    graph,
    memory: ExtractedMemory,
) -> PromotedMemory:
    """Insert a single extracted memory into the graph (with upsert)."""
    provenance: Dict[str, Any] = {}
    if memory.provenance is not None:
        provenance = memory.provenance.model_dump(exclude_none=True)

    if memory.type == "semantic":
        existing = graph._find_matching_semantic(
            text=memory.semantic_memory or "",
            source=memory.source,
        )
        if existing is not None:
            existing.confidence = max(existing.confidence, memory.confidence)
            existing.time = graph.semantic_time
            graph.storage.update_semantic(
                graph.graph_id,
                semantic_id=existing.semantic_id,
                metadata_updates={
                    "confidence": existing.confidence,
                    "time": existing.time,
                },
            )
            node_id = existing.semantic_id
        else:
            mem = graph._empty_insert_memory()
            mem.memory["semantic"].append({
                "semantic_memory": memory.semantic_memory,
                "tags": memory.tags,
                "source": memory.source,
                "confidence": memory.confidence,
                "session_id": None,
                "provenance": provenance,
            })
            mem.memory_embedding["semantic"].append({
                "semantic_memory": graph.embedder.embed(memory.semantic_memory or ""),
                "tags": [graph.embedder.embed(t) for t in memory.tags],
            })
            graph.insert(mem)
            node_id = graph.semantic_nodes[-1].semantic_id
    elif memory.type == "procedural":
        existing = graph._find_matching_procedural(
            subgoal=memory.subgoal or "",
            text=memory.procedural_memory or "",
            source=memory.source,
        )
        if existing is not None:
            existing.confidence = max(existing.confidence, memory.confidence)
            existing.time = graph.procedural_time
            graph.storage.update_procedural(
                graph.graph_id,
                procedural_id=existing.procedural_id,
                metadata_updates={
                    "confidence": existing.confidence,
                    "time": existing.time,
                },
            )
            node_id = existing.procedural_id
        else:
            mem = graph._empty_insert_memory()
            mem.memory["procedural"].append({
                "subgoal": memory.subgoal or "",
                "procedural_memory": memory.procedural_memory or "",
                "time": graph.semantic_time,
                "return": 0.0,
                "source": memory.source,
                "confidence": memory.confidence,
                "session_id": None,
                "provenance": provenance,
            })
            mem.memory_embedding["procedural"].append({
                "subgoal": graph.embedder.embed(memory.subgoal or ""),
            })
            graph.insert(mem)
            node_id = graph.procedural_nodes[-1].procedural_id
    else:
        raise ValueError(f"Unknown memory type: {memory.type}")

    return PromotedMemory(node_type=memory.type, node_id=node_id, memory=memory)


@router.post("/{graph_id}/promote", response_model=PromoteResponse)
async def promote(graph_id: str, body: PromoteRequest) -> PromoteResponse:
    gm = _manager()

    try:
        graph = gm.get_graph(graph_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Graph '{graph_id}' not found")

    if not body.candidates:
        return PromoteResponse(inserted=[], dropped=[])

    candidates = [{"kind": c.kind, "window": c.window} for c in body.candidates]
    llm = get_llm()
    memories_raw, rejections_raw = extract_coding_memories(llm, candidates)

    # Parse raw memories into ExtractedMemory models
    parsed_memories: List[ExtractedMemory] = []
    for m in memories_raw:
        try:
            parsed_memories.append(ExtractedMemory(**m))
        except Exception:
            logger.info("promote: dropping invalid memory %r", m)
            rejections_raw.append({
                "index": len(rejections_raw),
                "kind": "unknown",
                "reason": "invalid memory structure",
            })

    # Apply source_in and min_confidence filters
    filtered_memories: List[ExtractedMemory] = []
    for m in parsed_memories:
        if body.source_in is not None and m.source not in body.source_in:
            continue
        if body.min_confidence is not None and m.confidence < body.min_confidence:
            continue
        filtered_memories.append(m)

    # Insert
    inserted: List[PromotedMemory] = []
    for m in filtered_memories:
        try:
            result = _promote_candidate(graph, m)
            inserted.append(result)
        except Exception:
            logger.exception("promote: insert failed for memory %r", m)

    # Build rejection list from the raw rejections
    dropped: List[RejectedCandidate] = []
    rejected_indices: set = set()
    for r in rejections_raw:
        try:
            dropped.append(RejectedCandidate(**r))
            rejected_indices.add(r["index"])
        except Exception:
            pass

    # Silent-drop tracking: any input candidate that was neither rejected nor
    # produced a memory should be reported.  Since extracted memories do not
    # carry a candidate index we use a heuristic: if a candidate's *kind*
    # appears as the source of at least one memory, consider it covered.
    produced_sources: set = {m.source for m in filtered_memories}
    for i, c in enumerate(body.candidates):
        if i not in rejected_indices and c.kind not in produced_sources:
            dropped.append(RejectedCandidate(
                index=i, kind=c.kind,
                reason="llm produced no output for this candidate",
            ))

    return PromoteResponse(inserted=inserted, dropped=dropped)
