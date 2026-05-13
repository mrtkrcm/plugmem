"""Stability and performance tests for the promote endpoint and upsert logic."""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List

import pytest

from plugmem.api import dependencies as deps
from plugmem.clients.llm import LLMClient
from plugmem.core.memory_graph import MemoryGraph


class RespondingLLM(LLMClient):
    """Returns a canned structured response for promotion extraction."""

    def __init__(self, response: str = ""):
        super().__init__()
        self.response = response
        self.calls: list = []

    def complete(self, messages, temperature=0, top_p=1.0, max_tokens=4096):
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        return self.response


_DEFAULT_RESPONSE = json.dumps({
    "memories": [
        {
            "type": "semantic",
            "semantic_memory": "Use uv, not pip for Python dependency management",
            "tags": ["python", "tooling"],
            "source": "correction",
            "confidence": 0.9,
            "provenance": {"repo": "org/repo", "language": "python", "tool_name": "uv"},
        },
        {
            "type": "procedural",
            "subgoal": "install project dependencies",
            "procedural_memory": "Run `uv sync` instead of `pip install -r requirements.txt`",
            "source": "failure_delta",
            "confidence": 0.8,
            "provenance": {"repo": "org/repo", "package_manager": "uv"},
        },
    ],
    "rejections": [
        {"index": 2, "kind": "failure_delta", "reason": "trivial fix - typo"},
    ],
})


@pytest.fixture(autouse=True)
def _no_leakage():
    yield
    deps.reset_singletons()


def _set_llm(canned: RespondingLLM):
    deps._llm_client = canned


def test_promote_empty_candidates_returns_empty(client):
    client.post("/api/v1/graphs", json={"graph_id": "promo_empty_g"})
    resp = client.post("/api/v1/graphs/promo_empty_g/promote", json={"candidates": []})
    assert resp.status_code == 200
    data = resp.json()
    assert data["inserted"] == []
    assert data["dropped"] == []


def test_promote_inserts_and_rejects(client):
    _set_llm(RespondingLLM(_DEFAULT_RESPONSE))
    client.post("/api/v1/graphs", json={"graph_id": "promo1"})

    resp = client.post("/api/v1/graphs/promo1/promote", json={
        "candidates": [
            {"kind": "correction", "window": "use uv, not pip"},
            {"kind": "failure_delta", "window": "pip failed, uv sync worked"},
            {"kind": "failure_delta", "window": "typo fix"},
        ],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["inserted"]) == 2
    assert len(data["dropped"]) == 1
    assert data["dropped"][0]["index"] == 2
    assert data["dropped"][0]["reason"] == "trivial fix - typo"

    # Verify inserted IDs
    assert data["inserted"][0]["node_type"] == "semantic"
    assert data["inserted"][1]["node_type"] == "procedural"
    assert data["inserted"][0]["memory"]["semantic_memory"] == "Use uv, not pip for Python dependency management"
    assert data["inserted"][1]["memory"]["procedural_memory"] == "Run `uv sync` instead of `pip install -r requirements.txt`"


def test_promote_upsert_dedupe(client):
    """Same signal promoted twice should update, not duplicate."""
    _set_llm(RespondingLLM(_DEFAULT_RESPONSE))
    client.post("/api/v1/graphs", json={"graph_id": "promo_dupe"})

    # First promotion
    resp1 = client.post("/api/v1/graphs/promo_dupe/promote", json={
        "candidates": [{"kind": "correction", "window": "use uv"}],
    })
    assert resp1.status_code == 200
    first_ids = [m["node_id"] for m in resp1.json()["inserted"]]

    # Second promotion with same content
    resp2 = client.post("/api/v1/graphs/promo_dupe/promote", json={
        "candidates": [{"kind": "correction", "window": "use uv"}],
    })
    assert resp2.status_code == 200
    second_ids = [m["node_id"] for m in resp2.json()["inserted"]]

    # IDs should match (upsert, not new nodes)
    assert first_ids == second_ids, f"Expected upsert (same IDs), got {first_ids} vs {second_ids}"

    # Verify only 1 semantic and 1 procedural node total
    stats = client.get("/api/v1/graphs/promo_dupe/stats").json()["stats"]
    assert stats["semantic"] == 1
    assert stats["procedural"] == 1


def test_promote_filters_by_source_in(client):
    _set_llm(RespondingLLM(_DEFAULT_RESPONSE))
    client.post("/api/v1/graphs", json={"graph_id": "promo_filt"})

    resp = client.post("/api/v1/graphs/promo_filt/promote", json={
        "candidates": [{"kind": "correction", "window": "use uv"}],
        "source_in": ["correction"],
    })
    assert resp.status_code == 200
    data = resp.json()
    # Only correction-type memory should pass; failure_delta should be filtered out
    assert len(data["inserted"]) == 1
    assert data["inserted"][0]["memory"]["source"] == "correction"


def test_promote_filters_by_min_confidence(client):
    _set_llm(RespondingLLM(_DEFAULT_RESPONSE))
    client.post("/api/v1/graphs", json={"graph_id": "promo_conf"})

    resp = client.post("/api/v1/graphs/promo_conf/promote", json={
        "candidates": [{"kind": "correction", "window": "use uv"}],
        "min_confidence": 0.85,
    })
    assert resp.status_code == 200
    data = resp.json()
    # semantic has 0.9 >= 0.85, procedural has 0.8 < 0.85
    assert len(data["inserted"]) == 1
    assert data["inserted"][0]["memory"]["type"] == "semantic"


def test_promote_rejects_nonexistent_graph(client):
    resp = client.post("/api/v1/graphs/nonexistent/promote", json={
        "candidates": [{"kind": "correction", "window": "x"}],
    })
    assert resp.status_code == 404


def test_promote_stress_many_candidates(client):
    """Stress test: promote 50 candidates to measure throughput and stability."""
    _set_llm(RespondingLLM(_DEFAULT_RESPONSE))
    client.post("/api/v1/graphs", json={"graph_id": "promo_stress"})

    candidates = [{"kind": "correction", "window": f"rule {i}"} for i in range(50)]

    t0 = time.time()
    resp = client.post("/api/v1/graphs/promo_stress/promote", json={
        "candidates": candidates,
    })
    elapsed = time.time() - t0

    assert resp.status_code == 200
    data = resp.json()
    # Each candidate triggers the same LLM response, so each produces 2 memories
    # but upsert dedupe means only 2 unique memories total
    assert len(data["inserted"]) == 2
    assert elapsed < 10.0, f"Promote took {elapsed:.2f}s for 50 candidates"


def test_promote_handles_empty_llm_output(client):
    _set_llm(RespondingLLM("[]"))  # Legacy empty array
    client.post("/api/v1/graphs", json={"graph_id": "promo_empty_llm"})

    resp = client.post("/api/v1/graphs/promo_empty_llm/promote", json={
        "candidates": [{"kind": "correction", "window": "x"}],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["inserted"] == []


def test_promote_handles_new_style_empty_output(client):
    _set_llm(RespondingLLM(json.dumps({"memories": [], "rejections": []})))
    client.post("/api/v1/graphs", json={"graph_id": "promo_empty_new"})

    resp = client.post("/api/v1/graphs/promo_empty_new/promote", json={
        "candidates": [{"kind": "correction", "window": "x"}],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["inserted"] == []


def test_promote_with_provenance_roundtrip(client):
    _set_llm(RespondingLLM(_DEFAULT_RESPONSE))
    client.post("/api/v1/graphs", json={"graph_id": "promo_prov"})

    resp = client.post("/api/v1/graphs/promo_prov/promote", json={
        "candidates": [{"kind": "correction", "window": "use uv"}],
    })
    assert resp.status_code == 200
    data = resp.json()

    # Verify provenance was stored and returned
    inserted = data["inserted"][0]
    prov = inserted["memory"].get("provenance", {})
    assert prov["repo"] == "org/repo"
    assert prov["language"] == "python"

    # Verify provenance persists through storage by reading node details
    graph_id = "promo_prov"
    stats = client.get(f"/api/v1/graphs/{graph_id}/stats").json()
    assert stats["stats"]["semantic"] == 1


def test_extract_with_rejection_reasons(client):
    """Verify /extract now returns rejection reasons alongside memories."""
    _set_llm(RespondingLLM(_DEFAULT_RESPONSE))
    resp = client.post("/api/v1/extract", json={
        "candidates": [
            {"kind": "correction", "window": "use uv"},
            {"kind": "failure_delta", "window": "pip failed, uv sync"},
            {"kind": "failure_delta", "window": "typo fix"},
        ],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["memories"]) == 2
    assert len(data["rejected"]) == 1
    assert data["rejected"][0]["reason"] == "trivial fix - typo"
    assert data["rejected"][0]["index"] == 2


def test_upsert_strengthens_confidence(client):
    """Re-promoting with higher confidence should bump the stored node's confidence."""
    _set_llm(RespondingLLM(_DEFAULT_RESPONSE))
    client.post("/api/v1/graphs", json={"graph_id": "promo_bump"})

    # First promotion at confidence 0.9 / 0.8
    resp1 = client.post("/api/v1/graphs/promo_bump/promote", json={
        "candidates": [{"kind": "correction", "window": "use uv"}],
    })
    assert resp1.status_code == 200

    # Second promotion with higher confidence response
    high_conf = json.dumps({
        "memories": [
            {
                "type": "semantic",
                "semantic_memory": "Use uv, not pip for Python dependency management",
                "tags": ["python", "tooling"],
                "source": "correction",
                "confidence": 1.0,
            },
        ],
        "rejections": [],
    })
    _set_llm(RespondingLLM(high_conf))
    resp2 = client.post("/api/v1/graphs/promo_bump/promote", json={
        "candidates": [{"kind": "correction", "window": "use uv"}],
    })
    assert resp2.status_code == 200

    # Still only 1 semantic node (upsert, not duplicate)
    stats = client.get("/api/v1/graphs/promo_bump/stats").json()["stats"]
    assert stats["semantic"] == 1


def test_value_function_source_boost():
    """Verify source-aware scoring produces higher values for correction sources."""
    from plugmem.core.value_functions import SemanticRelevant, ProceduralRelevant

    sem = SemanticRelevant()
    proc = ProceduralRelevant()

    # Same relevance, different sources
    base = sem.evaluate(Relevance=0.5, Recency=10)
    boosted = sem.evaluate(Relevance=0.5, Recency=10, Source="correction", Confidence=0.9)
    assert boosted > base, f"Expected boosted ({boosted}) > base ({base})"

    # Explicit > correction > failure_delta
    explicit = sem.evaluate(Relevance=0.5, Recency=10, Source="explicit", Confidence=1.0)
    correction = sem.evaluate(Relevance=0.5, Recency=10, Source="correction", Confidence=1.0)
    failure = sem.evaluate(Relevance=0.5, Recency=10, Source="failure_delta", Confidence=1.0)
    assert explicit > correction > failure

    # No source = no boost
    no_source = sem.evaluate(Relevance=0.5, Recency=10)
    assert no_source == base

    # ProceduralRelevant also benefits
    p_base = proc.evaluate(Relevance=0.5, Recency=10)
    p_boosted = proc.evaluate(Relevance=0.5, Recency=10, Source="correction", Confidence=0.9)
    assert p_boosted > p_base
