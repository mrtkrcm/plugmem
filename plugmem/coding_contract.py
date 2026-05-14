"""Shared request-shaping helpers for coding-oriented clients."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


PROVENANCE_FILTER_KEYS = (
    "repo", "branch", "commit", "language", "filepath",
    "package_manager", "tool_name", "tool_version", "os", "component",
)


def build_provenance_filters(args: Dict[str, Any]) -> Dict[str, List[str]]:
    prov: Dict[str, List[str]] = {}
    for key in PROVENANCE_FILTER_KEYS:
        value = args.get(key)
        if value:
            prov[key] = [str(value)]
    return prov


def build_recall_body(
    *,
    observation: str,
    goal: Optional[str] = None,
    subgoal: Optional[str] = None,
    state: Optional[str] = None,
    task_type: Optional[str] = None,
    time: Optional[str] = None,
    session_id: Optional[str] = None,
    mode: Optional[str] = None,
    source_in: Optional[List[str]] = None,
    min_confidence: Optional[float] = None,
    provenance_filters: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {"observation": observation}
    for key, value in (
        ("goal", goal),
        ("subgoal", subgoal),
        ("state", state),
        ("task_type", task_type),
        ("time", time),
        ("session_id", session_id),
        ("mode", mode),
    ):
        if value:
            body[key] = value
    if source_in:
        body["source_in"] = list(source_in)
    if min_confidence is not None:
        body["min_confidence"] = float(min_confidence)
    if provenance_filters:
        body["provenance_filters"] = provenance_filters
    return body


def build_promote_body(
    *,
    candidates: List[Dict[str, str]],
    source_in: Optional[List[str]] = None,
    min_confidence: Optional[float] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {"candidates": candidates}
    if source_in:
        body["source_in"] = list(source_in)
    if min_confidence is not None:
        body["min_confidence"] = float(min_confidence)
    return body
