"""Shared helpers used by structuring and retrieving inference modules."""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from plugmem.prompts.registry import PromptRegistry


def render_messages(prompt_obj, variables: dict) -> List[Dict[str, str]]:
    messages = prompt_obj.render(variables)
    return [{"role": m.role, "content": m.content} for m in messages]


def resolve(
    name: str, fallback_cls: type,
    prompts: Optional[PromptRegistry], graph_id: Optional[str],
):
    if prompts is not None:
        return prompts.get(name, graph_id=graph_id)
    return fallback_cls()
