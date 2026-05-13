"""Value function base class — unchanged from original."""
from __future__ import annotations

from abc import ABC, abstractmethod


class ValueBase(ABC):
    """
    Base class for value functions that score a memory item.
    Subclasses implement the four component scorers; the final value is their sum.
    """

    @abstractmethod
    def __init__(self):
        self.value_threshold = 0

    def evaluate(
        self,
        Importance: float = 0,
        Relevance: float = 0,
        Recency: float = 0,
        Return: float = 0,
        Credibility: float = 0,
    ) -> float:
        return float(
            self.compute_importance(Importance)
            + self.compute_relevance(Relevance)
            + self.compute_recency(Recency)
            + self.compute_return(Return)
            + self.compute_credibility(Credibility)
        )

    @abstractmethod
    def compute_importance(self, Importance: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def compute_relevance(self, Relevance: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def compute_recency(self, Recency: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def compute_return(self, Return: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def compute_credibility(self, Credibility: float) -> float:
        raise NotImplementedError
