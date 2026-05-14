"""Graph CRUD endpoints."""
from __future__ import annotations

import logging

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

logger = logging.getLogger(__name__)

from plugmem.api.auth import require_api_key
from plugmem.api.dependencies import get_graph_manager
from plugmem.api.schemas import (
    GraphCreateRequest,
    GraphListResponse,
    GraphResponse,
    StatsResponse,
    NodeListResponse,
)
from plugmem.graph_manager import GraphManager

router = APIRouter(prefix="/graphs", tags=["graphs"], dependencies=[Depends(require_api_key)])


def _manager() -> GraphManager:
    return get_graph_manager()


def _serialize_storage_node(node_type: str, doc: str, meta: dict) -> dict:
    if node_type == "semantic":
        prov = meta.get("provenance")
        if prov is None:
            prov = {
                k.removeprefix("provenance_"): v
                for k, v in meta.items()
                if k.startswith("provenance_") and v is not None
            }
        return {
            "semantic_id": meta["semantic_id"],
            "semantic_memory": doc or "",
            "tags": meta.get("tags", []),
            "is_active": meta.get("is_active", 1),
            "credibility": meta.get("credibility", 10),
            "time": meta.get("time", 0),
            "source": meta.get("source"),
            "confidence": meta.get("confidence"),
            "provenance": prov or {},
        }
    return {
        "procedural_id": meta["procedural_id"],
        "procedural_memory": doc or "",
        "subgoals": [meta.get("subgoal", "")] if meta.get("subgoal") else [],
        "return": meta.get("return_value", meta.get("return", 0.0)),
        "session_id": meta.get("session_id"),
        "time": meta.get("time", 0),
        "source": meta.get("source"),
        "confidence": meta.get("confidence"),
        "provenance": meta.get("provenance") or {
            k.removeprefix("provenance_"): v
            for k, v in meta.items()
            if k.startswith("provenance_") and v is not None
        },
    }


@router.post("", response_model=GraphResponse, status_code=status.HTTP_201_CREATED)
async def create_graph(body: GraphCreateRequest) -> GraphResponse:
    gm = _manager()
    try:
        graph_id = gm.create_graph(graph_id=body.graph_id)
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    stats = gm.get_stats(graph_id)
    return GraphResponse(graph_id=graph_id, stats=stats)


@router.get("", response_model=GraphListResponse)
async def list_graphs() -> GraphListResponse:
    gm = _manager()
    try:
        graphs = gm.list_graphs()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to list graphs: {exc}")
    return GraphListResponse(graphs=graphs)


@router.get("/{graph_id}", response_model=GraphResponse)
async def get_graph(graph_id: str) -> GraphResponse:
    gm = _manager()
    try:
        gm.get_graph(graph_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Graph '{graph_id}' not found")
    stats = gm.get_stats(graph_id)
    return GraphResponse(graph_id=graph_id, stats=stats)


@router.delete("/{graph_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_graph(graph_id: str) -> None:
    gm = _manager()
    try:
        gm.delete_graph(graph_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{graph_id}/stats", response_model=StatsResponse)
async def get_stats(graph_id: str) -> StatsResponse:
    gm = _manager()
    try:
        stats = gm.get_stats(graph_id)
    except Exception as exc:
        logger.warning("get_stats(%s) failed: %s", graph_id, exc)
        raise HTTPException(status_code=404, detail=f"Graph '{graph_id}' not found") from exc
    return StatsResponse(graph_id=graph_id, stats=stats)


@router.get(
    "/{graph_id}/nodes",
    response_model=NodeListResponse,
    summary="List memories, optionally scoped by provenance / source / confidence",
    description=(
        "Agent-facing experience browser. Returns nodes filtered by metadata "
        "(language, repo, source, min_confidence) — never by content text. "
        "For content-aware retrieval use /retrieve or /reason."
    ),
)
async def browse_nodes(
    graph_id: str,
    node_type: str = "semantic",
    limit: int = 50,
    offset: int = 0,
    language: Optional[str] = None,
    repo: Optional[str] = None,
    component: Optional[str] = None,
    source_in: Optional[List[str]] = Query(default=None),
    min_confidence: Optional[float] = None,
) -> NodeListResponse:
    gm = _manager()
    if not gm.storage.graph_exists(graph_id):
        raise HTTPException(status_code=404, detail=f"Graph '{graph_id}' not found")

    if node_type in ("semantic", "procedural"):
        provenance_filters: dict[str, list[str]] = {}
        if language:
            provenance_filters["language"] = [language]
        if repo:
            provenance_filters["repo"] = [repo]
        if component:
            provenance_filters["component"] = [component]
        try:
            result = gm.storage.browse_nodes(
                graph_id,
                node_type=node_type,
                limit=limit,
                offset=offset,
                source_in=source_in,
                min_confidence=min_confidence,
                provenance_filters=provenance_filters or None,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Browse failed: {exc}")
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        nodes = [
            _serialize_storage_node(node_type, doc, meta)
            for doc, meta in zip(docs, metas)
        ]
        return NodeListResponse(
            graph_id=graph_id,
            node_type=node_type,
            count=int(result.get("count", len(nodes))),
            nodes=nodes,
        )

    try:
        graph = gm.get_graph(graph_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Graph '{graph_id}' not found")

    type_map = {
        "episodic": graph.episodic_nodes,
        "semantic": graph.semantic_nodes,
        "tag": graph.tag_nodes,
        "subgoal": graph.subgoal_nodes,
        "procedural": graph.procedural_nodes,
    }
    if node_type not in type_map:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid node_type '{node_type}'. Must be one of: {list(type_map)}",
        )

    nodes = type_map[node_type]

    page = nodes[offset : offset + limit]

    serialized = []
    for n in page:
        d: dict = {}
        if node_type == "episodic":
            d = {"episodic_id": n.episodic_id, "observation": n.observation, "action": n.action,
                 "subgoal": n.subgoal, "state": n.state, "reward": n.reward, "time": n.time}
        elif node_type == "semantic":
            d = {"semantic_id": n.semantic_id, "semantic_memory": n.get_semantic_memory(),
                 "tags": [t.tag for t in n.tag_nodes], "is_active": n.is_active,
                 "credibility": getattr(n, "credibility", 10), "time": n.time,
                 "source": getattr(n, "source", None),
                 "confidence": getattr(n, "confidence", None),
                 "provenance": getattr(n, "provenance", None) or {}}
        elif node_type == "tag":
            d = {"tag_id": n.tag_id, "tag": n.tag, "importance": n.importance, "time": n.time}
        elif node_type == "subgoal":
            d = {"subgoal_id": n.subgoal_id, "subgoal": n.subgoal, "time": n.time}
        elif node_type == "procedural":
            d = {"procedural_id": n.procedural_id,
                 "procedural_memory": n.get_procedural_memory(),
                 "subgoals": [s.subgoal for s in n.subgoal_nodes] or list(n.subgoals),
                 "return": n.return_value,
                 "session_id": n.session_id,
                 "time": n.time,
                 "source": getattr(n, "source", None),
                 "confidence": getattr(n, "confidence", None),
                 "provenance": getattr(n, "provenance", None) or {}}
        serialized.append(d)

    return NodeListResponse(
        graph_id=graph_id,
        node_type=node_type,
        count=len(nodes),
        nodes=serialized,
    )
