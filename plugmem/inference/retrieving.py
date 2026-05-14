"""Retrieval inference — accepts injected LLMClient and optional PromptRegistry."""
from __future__ import annotations

import ast
import json
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from plugmem.clients.llm import LLMClient

if TYPE_CHECKING:
    from plugmem.prompts.registry import PromptRegistry
from plugmem.inference._shared import render_messages, resolve
from plugmem.prompts.retrieving import (
    GetModePrompt,
    GetNewSemanticPrompt,
    GetNewSubgoalPrompt,
    GetPlanPrompt,
)


_ALLOWED_REL = {
    "UPDATE_SAME_FACT",
    "SAME_TOPIC_MERGE_WELL",
    "WEAK_RELATED_STITCH_RISK",
}
_REQUIRED_KEYS = {
    "merged_statement",
    "relationship",
    "deactivate_earlier",
    "deactivate_later",
    "simple_reasoning",
}


def _heuristic_mode(observation: str, task_type: str) -> Optional[str]:
    """Cheap mode resolver for obvious cases before invoking the LLM."""
    obs = (observation or "").casefold().strip()
    task = (task_type or "").casefold().strip()
    combined = f"{task}\n{obs}"

    episodic_markers = (
        "earlier conversation", "previous conversation", "conversation history",
        "chat history", "earlier session", "previous session", "what did i say",
        "what did we discuss", "historical conversation",
    )
    procedural_markers = (
        "workflow", "interactive environment", "web navigation", "browser",
        "debugging", "fix flaky", "fix failing", "run tests", "deploy",
        "install deps", "how to ", "steps to ", "procedure", "playbook",
    )

    if any(marker in combined for marker in episodic_markers):
        return "episodic_memory"
    if any(marker in combined for marker in procedural_markers):
        return "procedural_memory"
    if obs.startswith(("who ", "what ", "when ", "where ", "which ", "why ", "is ", "are ", "do ", "does ", "can ")):
        return "semantic_memory"
    if "?" in obs and "how" not in obs:
        return "semantic_memory"
    return None


def get_plan(
    llm: LLMClient, goal: str, subgoal: str, state: str, observation: str,
    *, prompts: Optional[PromptRegistry] = None, graph_id: Optional[str] = None,
) -> Tuple[str, List[str]]:
    prompt_obj = resolve("get_plan", GetPlanPrompt, prompts, graph_id)
    variables = {"goal": goal, "subgoal": subgoal, "state": state, "observation": observation}
    response = llm.complete(messages=render_messages(prompt_obj, variables))

    tags_pattern = r"\*\*Tags:\*\*\s*(.*)\n"
    tags_match = re.search(tags_pattern, response)
    tags: List[str] = []
    if tags_match:
        raw = tags_match.group(1).strip()
        try:
            tags = json.loads(raw)
        except json.JSONDecodeError:
            try:
                tags = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                tags = []

    subgoal_pattern = r"### Next Subgoal\n(.*)"
    subgoal_match = re.search(subgoal_pattern, response, re.S)
    next_subgoal = subgoal_match.group(1).strip() if subgoal_match else "<the next subgoal>"
    return next_subgoal, tags


def get_new_semantic(
    llm: LLMClient, old_semantic_memory: str, new_semantic_memory: str,
    *, prompts: Optional[PromptRegistry] = None, graph_id: Optional[str] = None,
) -> Dict[str, Any]:

    def _extract_json_object(text: str) -> Dict[str, Any]:
        text = text.strip()
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return json.loads(m.group(1))
        l = text.find("{")
        r = text.rfind("}")
        if l == -1 or r == -1 or r <= l:
            raise ValueError("No JSON object found in model output.")
        return json.loads(text[l : r + 1])

    def _to_bool(x: Any) -> bool:
        if isinstance(x, bool):
            return x
        if isinstance(x, str) and x.strip().lower() in {"true", "false"}:
            return x.strip().lower() == "true"
        raise TypeError(f"Expected boolean, got {x!r} ({type(x)})")

    def parse_merge_decision(model_text: str) -> Dict[str, Any]:
        obj = _extract_json_object(model_text)
        if not isinstance(obj, dict):
            raise TypeError("Parsed JSON is not an object/dict.")
        missing = _REQUIRED_KEYS - set(obj.keys())
        if missing:
            raise KeyError(f"Missing required keys: {sorted(missing)}")
        rel = obj.get("relationship")
        if rel not in _ALLOWED_REL:
            raise ValueError(f"Invalid relationship: {rel}. Allowed: {sorted(_ALLOWED_REL)}")
        merged = obj.get("merged_statement")
        if not isinstance(merged, str) or not merged.strip():
            raise ValueError("merged_statement must be a non-empty string.")
        obj["deactivate_earlier"] = _to_bool(obj["deactivate_earlier"])
        obj["deactivate_later"] = _to_bool(obj["deactivate_later"])
        return {
            "merged_statement": merged.strip(),
            "relationship": rel,
            "deactivate_earlier": obj["deactivate_earlier"],
            "deactivate_later": obj["deactivate_later"],
            "simple_reasoning": obj["simple_reasoning"],
        }

    prompt_obj = resolve("get_new_semantic", GetNewSemanticPrompt, prompts, graph_id)
    variables = {"memory_earlier": old_semantic_memory, "memory_later": new_semantic_memory}
    response = llm.complete(messages=render_messages(prompt_obj, variables))
    return parse_merge_decision(response)


def get_new_subgoal(
    llm: LLMClient, old_subgoal: str, new_subgoal: str,
    *, prompts: Optional[PromptRegistry] = None, graph_id: Optional[str] = None,
) -> str:
    prompt_obj = resolve("get_new_subgoal", GetNewSubgoalPrompt, prompts, graph_id)
    variables = {"goal_1": old_subgoal, "goal_2": new_subgoal}
    response = llm.complete(messages=render_messages(prompt_obj, variables))
    return response


def get_mode(
    llm: LLMClient, observation: str, task_type: str,
    *, prompts: Optional[PromptRegistry] = None, graph_id: Optional[str] = None,
) -> str:
    fast = _heuristic_mode(observation, task_type)
    if fast is not None:
        return fast
    prompt_obj = resolve("get_mode", GetModePrompt, prompts, graph_id)
    variables = {"observation": observation, "task_type": task_type}
    response = llm.complete(messages=render_messages(prompt_obj, variables))
    pattern = r"### Memory Type\n(.*)"
    match = re.search(pattern, response)
    return match.group(1).strip() if match else "semantic_memory"
