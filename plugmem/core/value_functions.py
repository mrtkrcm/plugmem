"""Concrete value function implementations."""
from __future__ import annotations

from plugmem.core.value_base import ValueBase


class RelevanceValueFunc(ValueBase):
    """Value function that scores primarily on relevance, with all other
    dimensions returning zero by default. Subclasses override ``k`` and
    ``value_threshold`` only.
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
