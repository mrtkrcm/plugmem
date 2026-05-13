"""Concrete value function implementations.

Source-aware scoring: memories with higher-confidence sources (explicit
corrections, user rules) receive a boost via ``compute_source_boost``.
"""
from __future__ import annotations

from typing import Optional

from plugmem.core.value_base import ValueBase

# Source boost multipliers: explicit user corrections > failure deltas >
# generic/legacy memories. Keys correspond to :class:`MemorySource` values.
_SOURCE_BOOST = {
    "explicit": 0.3,
    "correction": 0.25,
    "failure_delta": 0.1,
    "merged": 0.05,
    "repeated_lookup": 0.05,
}


class RelevanceValueFunc(ValueBase):
    """Value function that scores primarily on relevance, with all other
    dimensions returning zero by default. Subclasses override ``k`` and
    ``value_threshold`` only.

    ``confidence`` and ``source`` are passed through to
    ``compute_source_boost`` so coding-agent promotions rank higher.
    """

    def __init__(self, k: int = 1, value_threshold: float = 0.0):
        self.k = k
        self.value_threshold = value_threshold

    def compute_importance(self, Importance: float) -> float:
        return 0

    def compute_relevance(self, Relevance: float) -> float:
        return Relevance

    def compute_recency(self, Recency: float) -> float:
        return 0

    def compute_return(self, Return: float) -> float:
        return 0

    def compute_credibility(self, Credibility: float) -> float:
        return 0

    def compute_source_boost(self, Source: Optional[str] = None, Confidence: float = 0.0) -> float:
        """Apply a source- and confidence-aware boost.

        Explicit user corrections rank highest, inferred failure deltas
        rank lower, legacy nodes with no source get no boost.

        The boost is scaled by confidence so a low-confidence correction
        does not outrank a high-confidence failure delta.
        """
        if Source is None:
            return 0.0
        boost = _SOURCE_BOOST.get(Source, 0.0)
        return boost * Confidence


class TagEqual(RelevanceValueFunc):
    def __init__(self):
        super().__init__(k=1, value_threshold=0.9)


class TagRelevant(RelevanceValueFunc):
    def __init__(self):
        super().__init__(k=1, value_threshold=0.8)


class SemanticEqual(RelevanceValueFunc):
    def __init__(self):
        super().__init__(k=1, value_threshold=0.9)


class SemanticRelevant(RelevanceValueFunc):
    def __init__(self):
        super().__init__(k=10, value_threshold=0.0)


class SemanticRelevant4Episodic(RelevanceValueFunc):
    def __init__(self):
        super().__init__(k=30, value_threshold=0.0)


class SubgoalEqual(RelevanceValueFunc):
    def __init__(self):
        super().__init__(k=1, value_threshold=0.8)


class SubgoalRelevant(RelevanceValueFunc):
    def __init__(self):
        super().__init__(k=1, value_threshold=0.1)


class ProceduralEqual(RelevanceValueFunc):
    def __init__(self):
        super().__init__(k=1, value_threshold=0.8)


class ProceduralRelevant(RelevanceValueFunc):
    def __init__(self):
        super().__init__(k=1, value_threshold=0.1)
