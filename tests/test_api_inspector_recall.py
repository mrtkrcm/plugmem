"""Smoke tests for previously-uncovered endpoints:
- POST /inspector/{graph_id}/recall_trace
- GET  /inspector/{graph_id}/recalls
- GET  /graphs/{graph_id}/stats
"""
from __future__ import annotations


def _make_graph(client, graph_id="recall_graph"):
    client.post("/api/v1/graphs", json={"graph_id": graph_id})


def _seed(client, graph_id="recall_graph"):
    client.post(f"/api/v1/graphs/{graph_id}/memories", json={
        "mode": "structured",
        "semantic": [{"semantic_memory": "the sky is blue", "tags": ["color"]}],
    })


def test_stats_endpoint_returns_counts(client):
    _make_graph(client)
    _seed(client)
    resp = client.get("/api/v1/graphs/recall_graph/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["graph_id"] == "recall_graph"
    assert body["stats"]["semantic"] == 1


def test_stats_endpoint_404_for_unknown_graph(client):
    resp = client.get("/api/v1/graphs/does_not_exist/stats")
    assert resp.status_code == 404


def test_recall_trace_smoke(client):
    _make_graph(client)
    _seed(client)
    resp = client.post(
        "/api/v1/graphs/recall_graph/recall_trace",
        json={"observation": "what color is the sky?"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "mode" in body and "plan" in body and "trace" in body
    assert "selected" in body
    assert isinstance(body["rendered_prompt"], list)


def test_recalls_endpoint_lists_after_recall_trace(client):
    _make_graph(client)
    _seed(client)
    client.post(
        "/api/v1/graphs/recall_graph/recall_trace",
        json={"observation": "what color is the sky?", "session_id": "s1"},
    )
    resp = client.get("/api/v1/graphs/recall_graph/recalls")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["graph_id"] == "recall_graph"
    assert body["count"] >= 1
    assert body["recalls"][0]["endpoint"] == "recall_trace"


def test_recalls_endpoint_filters_by_session(client):
    _make_graph(client)
    _seed(client)
    client.post(
        "/api/v1/graphs/recall_graph/recall_trace",
        json={"observation": "q1", "session_id": "sA"},
    )
    client.post(
        "/api/v1/graphs/recall_graph/recall_trace",
        json={"observation": "q2", "session_id": "sB"},
    )
    resp = client.get(
        "/api/v1/graphs/recall_graph/recalls",
        params={"session_id": "sA"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "sA"
    assert all(r.get("session_id") == "sA" for r in body["recalls"])
