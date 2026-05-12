"""Tests for the Memory class."""
from __future__ import annotations

from plugmem.core.memory import Memory
from plugmem.clients.llm import LLMClient
from plugmem.clients.embedding import EmbeddingClient
from tests.conftest import FakeEmbedder


def test_from_structured_creates_memory():
    embedder = FakeEmbedder()
    mem = Memory.from_structured(embedder=embedder, time=42, session_id="test-session")
    assert mem.time == 42
    assert mem.session_id == "test-session"
    assert mem.embedder is embedder
    assert mem.memory == {"goal": "", "episodic": [], "semantic": [], "procedural": []}
    assert mem.memory_embedding == {"semantic": [], "procedural": []}


def test_from_structured_memory_fills_correctly():
    embedder = FakeEmbedder()
    mem = Memory.from_structured(embedder=embedder)
    sem_text = "test semantic memory"
    mem.memory["semantic"].append({
        "semantic_memory": sem_text,
        "tags": ["test"],
    })
    mem.memory_embedding["semantic"].append({
        "semantic_memory": embedder.embed(sem_text),
        "tags": [embedder.embed("test")],
    })
    assert len(mem.memory["semantic"]) == 1
    assert len(mem.memory_embedding["semantic"]) == 1
    assert len(mem.memory_embedding["semantic"][0]["tags"]) == 1
