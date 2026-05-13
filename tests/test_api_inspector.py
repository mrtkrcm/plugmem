"""Smoke tests for inspector and demo routes."""
from __future__ import annotations


def _make_graph(client, graph_id="test_graph"):
    client.post("/api/v1/graphs", json={"graph_id": graph_id})


def _insert_semantic(client, graph_id="test_graph"):
    client.post(f"/api/v1/graphs/{graph_id}/memories", json={
        "mode": "structured",
        "semantic": [{"semantic_memory": "test fact", "tags": ["test"]}],
    })


def test_search_nodes_empty_graph(client):
    _make_graph(client)
    resp = client.get("/api/v1/graphs/test_graph/nodes", params={"node_type": "semantic"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_type"] == "semantic"
    assert data["count"] == 0


def test_search_nodes_with_data(client):
    _make_graph(client)
    _insert_semantic(client)
    resp = client.get("/api/v1/graphs/test_graph/nodes", params={"node_type": "semantic"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    assert data["nodes"][0]["semantic_memory"] == "test fact"


def test_search_nodes_invalid_type(client):
    _make_graph(client)
    resp = client.get("/api/v1/graphs/test_graph/nodes", params={"node_type": "invalid"})
    assert resp.status_code == 400


def test_get_node_detail(client):
    _make_graph(client)
    _insert_semantic(client)
    resp = client.get("/api/v1/graphs/test_graph/node/semantic/0")
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_type"] == "semantic"


def test_get_node_detail_includes_new_semantic_siblings_without_reload(client):
    _make_graph(client)
    resp = client.post("/api/v1/graphs/test_graph/memories", json={
        "mode": "structured",
        "semantic": [
            {"semantic_memory": "fact one", "tags": ["alpha"]},
            {"semantic_memory": "fact two", "tags": ["beta"]},
        ],
    })
    assert resp.status_code == 200, resp.text

    resp = client.get("/api/v1/graphs/test_graph/node/semantic/0")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["edges"]["bro_semantics"]) == 1
    assert data["edges"]["bro_semantics"][0]["semantic_id"] == 1


def test_get_node_detail_not_found(client):
    _make_graph(client)
    resp = client.get("/api/v1/graphs/test_graph/node/semantic/999")
    assert resp.status_code == 404


def test_get_topology(client):
    _make_graph(client)
    _insert_semantic(client)
    resp = client.get("/api/v1/graphs/test_graph/topology")
    assert resp.status_code == 200
    data = resp.json()
    assert "nodes" in data
    assert "edges" in data


def test_session_timeline(client):
    _make_graph(client)
    _insert_semantic(client)
    resp = client.get("/api/v1/graphs/test_graph/sessions")
    assert resp.status_code == 200


def test_demo_seed(client):
    resp = client.post("/api/v1/demo/seed", params={"graph_id": "demo_test"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["graph_id"] == "demo_test"


def test_demo_seed_twice_is_idempotent(client):
    resp1 = client.post("/api/v1/demo/seed", params={"graph_id": "demo_idem"})
    resp2 = client.post("/api/v1/demo/seed", params={"graph_id": "demo_idem"})
    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert resp1.json() == resp2.json()
