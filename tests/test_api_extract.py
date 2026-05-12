"""Tests for the /api/v1/extract promotion-gate endpoint."""
from __future__ import annotations

import json

import pytest

from plugmem.api import dependencies as deps
from plugmem.clients.llm import LLMClient
from plugmem.inference import promotion as promo


class CannedLLM(LLMClient):
    def __init__(self, response: str = ""):
        super().__init__()
        self.response = response
        self.calls: list = []

    def complete(self, messages, temperature=0, top_p=1.0, max_tokens=4096):
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        return self.response


@pytest.fixture(autouse=True)
def _no_leakage():
    yield
    deps.reset_singletons()


def test_extract_no_candidates_returns_empty(client):
    resp = client.post("/api/v1/extract", json={"candidates": []})
    assert resp.status_code == 200
    assert resp.json() == {"memories": []}


def _set_llm(canned: CannedLLM):
    """Set the LLM singleton for one test."""
    deps._llm_client = canned


def test_extract_returns_parsed_memories(client):
    _set_llm(CannedLLM(json.dumps([
        {
            "type": "semantic",
            "semantic_memory": "Use httpx, not requests",
            "tags": ["python", "convention"],
            "source": "correction",
            "confidence": 0.9,
        },
        {
            "type": "procedural",
            "subgoal": "fix import error in tests",
            "procedural_memory": "pip install -e . then pytest",
            "source": "failure_delta",
            "confidence": 0.7,
        },
    ])))
    resp = client.post("/api/v1/extract", json={
        "candidates": [
            {"kind": "correction", "window": "user said: actually use httpx"},
            {"kind": "failure_delta", "window": "pytest failed; pip install fixed it"},
        ],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["memories"]) == 2
    assert data["memories"][0]["type"] == "semantic"
    assert data["memories"][0]["source"] == "correction"
    assert data["memories"][1]["type"] == "procedural"


def test_extract_drops_invalid_memories(client):
    _set_llm(CannedLLM(json.dumps([
        {
            "type": "semantic",
            "semantic_memory": "x",
            "confidence": 0.5,
        },
        {
            "type": "semantic",
            "semantic_memory": "y",
            "source": "correction",
            "confidence": 99,
        },
        {
            "type": "semantic",
            "semantic_memory": "z",
            "source": "correction",
            "confidence": 0.6,
        },
    ])))
    resp = client.post("/api/v1/extract", json={
        "candidates": [{"kind": "correction", "window": "..."}],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["memories"]) == 1
    assert data["memories"][0]["semantic_memory"] == "z"


def test_extract_handles_unparseable_llm_output(client):
    _set_llm(CannedLLM("this is not JSON at all"))
    resp = client.post("/api/v1/extract", json={
        "candidates": [{"kind": "correction", "window": "..."}],
    })
    assert resp.status_code == 200
    assert resp.json() == {"memories": []}


def test_extract_strips_code_fences(client):
    _set_llm(CannedLLM(
        "```json\n"
        + json.dumps([{
            "type": "semantic",
            "semantic_memory": "fenced",
            "source": "correction",
            "confidence": 0.7,
        }])
        + "\n```"
    ))
    resp = client.post("/api/v1/extract", json={
        "candidates": [{"kind": "correction", "window": "..."}],
    })
    assert resp.status_code == 200
    assert len(resp.json()["memories"]) == 1


def test_extract_rejects_invalid_kind(client):
    resp = client.post("/api/v1/extract", json={
        "candidates": [{"kind": "made_up_kind", "window": "x"}],
    })
    assert resp.status_code == 422


def test_extractor_returns_empty_for_no_candidates():
    canned = CannedLLM("")
    assert promo.extract_coding_memories(canned, []) == []
    assert canned.calls == []


def test_extractor_filters_invalid_source():
    canned = CannedLLM(json.dumps([
        {
            "type": "semantic",
            "semantic_memory": "x",
            "source": "not_a_real_source",
            "confidence": 0.7,
        },
    ]))
    out = promo.extract_coding_memories(canned, [{"kind": "correction", "window": "."}])
    assert out == []


def test_extractor_filters_missing_semantic_text():
    canned = CannedLLM(json.dumps([
        {
            "type": "semantic",
            "semantic_memory": "",
            "source": "correction",
            "confidence": 0.7,
        },
    ]))
    out = promo.extract_coding_memories(canned, [{"kind": "correction", "window": "."}])
    assert out == []


def test_extractor_filters_partial_procedural():
    canned = CannedLLM(json.dumps([
        {
            "type": "procedural",
            "subgoal": "x",
            "source": "failure_delta",
            "confidence": 0.7,
        },
    ]))
    out = promo.extract_coding_memories(canned, [{"kind": "failure_delta", "window": "."}])
    assert out == []
