"""Coding-agent promotion-gate extractor.

Given a list of detected promotion candidates (failure->success deltas,
user corrections, explicit memories, repeated lookups), prompt an LLM
to emit 0-N structured memory nodes with ``source`` / ``confidence``
and optional ``provenance`` metadata.

The extractor is intentionally conservative: when in doubt, return
fewer memories. Bad memories actively harm coding agents (a stale
"always use X" rule leads to wrong code), so the prompt biases toward
*omitting* uncertain extractions.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from plugmem.clients.llm import LLMClient, with_phase

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """You extract durable, reusable memory nodes from coding-agent session signals.

You receive a list of CANDIDATES, each a short context window from a session that may indicate something worth remembering across future sessions. Possible kinds:

- failure_delta: a tool call failed, then a follow-up succeeded. The fix may be a debugging recipe (procedural) or a configuration fact (semantic).
- correction: the user explicitly corrected the agent ("don't", "stop", "actually"). The rule is a semantic fact about preferences/conventions.
- explicit: a direct, unambiguous user instruction or rule meant to persist across sessions.
- repeated_lookup: the agent looked up the same information multiple times, suggesting it should be cached as a semantic fact.

For each candidate, decide whether to emit 0, 1, or rarely 2 memory nodes. Be conservative -- bad memories actively harm. Emit nothing if:
- The fix was trivial (typo, syntax error)
- The correction was ambiguous or session-specific
- You're not sure the rule generalizes beyond this session

For EACH candidate that did NOT produce a memory, provide a short explanation in the "rejections" array.

Optionally include provenance metadata when available from context:
- repo (string): git repository slug
- branch (string): git branch
- commit (string): commit hash
- language (string): programming language
- filepath (string): relevant file path
- package_manager (string): e.g. uv, npm, cargo
- tool_name (string): e.g. ruff, mypy, pytest
- tool_version (string): tool version
- os (string): operating system
- component (string): service or component name

Output JSON only, no prose. Schema:

{
  "memories": [
    {
      "candidate_index": 0,
      "type": "semantic",
      "semantic_memory": "short factual statement",
      "tags": ["tag1", "tag2"],
      "source": "correction" | "failure_delta" | "explicit" | "repeated_lookup",
      "confidence": 0.0-1.0,
      "provenance": {
        "repo": "org/repo",
        "language": "python",
        "tool_name": "ruff"
      }
    },
    {
      "candidate_index": 1,
      "type": "procedural",
      "subgoal": "what was being attempted",
      "procedural_memory": "the steps that worked",
      "source": "failure_delta",
      "confidence": 0.0-1.0,
      "provenance": {
        "repo": "org/repo"
      }
    }
  ],
  "rejections": [
    {
      "index": 0,
      "kind": "correction",
      "reason": "trivial fix - typo correction, no durable rule"
    },
    {
      "index": 2,
      "kind": "failure_delta",
      "reason": "ambiguous pattern - session-specific, not generalizable"
    }
  ]
}

Confidence guide: 0.9+ for explicit user rules, 0.7-0.8 for clear failure->success patterns, 0.5-0.6 for plausible but ambiguous. Below 0.5: don't emit at all.

Return {"memories": [], "rejections": []} if no candidate warrants a memory."""


def extract_coding_memories(
    llm: LLMClient,
    candidates: List[Dict[str, str]],
    *,
    max_tokens: int = 4096,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not candidates:
        return [], []

    user_msg = _format_candidates(candidates)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    with with_phase("extract"):
        raw = llm.complete(messages=messages, temperature=0, max_tokens=max_tokens)
    if not raw:
        logger.info("promotion extractor: empty LLM response")
        return [], _build_default_rejections(candidates)

    parsed = _parse_json_object(raw)
    if parsed is not None:
        memories_raw = parsed.get("memories", [])
        rejections_raw = parsed.get("rejections", [])
    else:
        legacy = _parse_json_array(raw)
        if legacy is not None:
            memories_raw = legacy
            rejections_raw = []
        else:
            logger.info("promotion extractor: unparseable LLM output: %r", raw[:200])
            return [], _build_default_rejections(candidates)

    memories = [m for m in memories_raw if _is_valid_memory(m)]
    rejections = [r for r in rejections_raw if _is_valid_rejection(r, candidates)]

    return memories, rejections


def _build_default_rejections(candidates: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Build generic rejection reasons when the LLM output was unparseable."""
    return [
        {"index": i, "kind": c.get("kind", "unknown"), "reason": "LLM output unparseable"}
        for i, c in enumerate(candidates)
    ]


def _format_candidates(candidates: List[Dict[str, str]]) -> str:
    out = ["CANDIDATES:"]
    for i, c in enumerate(candidates, 1):
        kind = c.get("kind", "unknown")
        window = c.get("window", "")
        out.append(f"\n--- Candidate {i} (kind: {kind}) ---\n{window}")
    return "\n".join(out)


def _parse_json_object(raw: str) -> Optional[Dict[str, Any]]:
    stripped = raw.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```\s*$", "", stripped)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _parse_json_array(raw: str) -> Optional[List[Any]]:
    """Legacy parser for the old-style LLM output (just an array)."""
    stripped = raw.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```\s*$", "", stripped)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    return parsed


_VALID_SOURCES = {"failure_delta", "correction", "merged", "repeated_lookup", "explicit"}


def _is_valid_memory(m: Any) -> bool:
    if not isinstance(m, dict):
        return False
    candidate_index = m.get("candidate_index")
    if candidate_index is not None and not isinstance(candidate_index, int):
        return False
    mtype = m.get("type")
    source = m.get("source")
    confidence = m.get("confidence")
    if source not in _VALID_SOURCES:
        return False
    if not isinstance(confidence, (int, float)) or not (0.0 <= float(confidence) <= 1.0):
        return False
    if mtype == "semantic":
        return isinstance(m.get("semantic_memory"), str) and m["semantic_memory"].strip() != ""
    if mtype == "procedural":
        return (
            isinstance(m.get("subgoal"), str)
            and m["subgoal"].strip() != ""
            and isinstance(m.get("procedural_memory"), str)
            and m["procedural_memory"].strip() != ""
        )
    return False


def _is_valid_rejection(r: Any, candidates: List[Dict[str, str]]) -> bool:
    if not isinstance(r, dict):
        return False
    idx = r.get("index")
    if not isinstance(idx, int):
        return False
    if idx < 0 or idx >= len(candidates):
        return False
    if not isinstance(r.get("reason"), str) or not r["reason"].strip():
        return False
    return True
