"""Tests for memory insertion endpoints."""

import plugmem.core.memory_graph as mg_module
from plugmem.clients.embedding import HTTPEmbeddingClient


def test_insert_structured_semantic(client):
    client.post("/api/v1/graphs", json={"graph_id": "mem_test"})
    resp = client.post("/api/v1/graphs/mem_test/memories", json={
        "mode": "structured",
        "semantic": [
            {"semantic_memory": "The capital of France is Paris", "tags": ["geography", "france"]},
            {"semantic_memory": "Python was created by Guido van Rossum", "tags": ["programming"]},
        ],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["stats"]["semantic"] == 2

    # Verify via stats endpoint
    resp = client.get("/api/v1/graphs/mem_test/stats")
    assert resp.json()["stats"]["semantic"] == 2
    assert resp.json()["stats"]["tag"] >= 2  # at least "geography", "france", "programming"


def test_insert_structured_procedural(client):
    client.post("/api/v1/graphs", json={"graph_id": "proc_test"})
    resp = client.post("/api/v1/graphs/proc_test/memories", json={
        "mode": "structured",
        "procedural": [
            {"subgoal": "deploy app", "procedural_memory": "run docker-compose up"},
        ],
    })
    assert resp.status_code == 200
    assert resp.json()["stats"]["procedural"] == 1


def test_insert_structured_procedural_consolidates_similar_subgoal(client, monkeypatch):
    client.post("/api/v1/graphs", json={"graph_id": "proc_merge_test"})

    first = client.post("/api/v1/graphs/proc_merge_test/memories", json={
        "mode": "structured",
        "procedural": [
            {"subgoal": "deploy app", "procedural_memory": "build container image"},
        ],
    })
    assert first.status_code == 200, first.text

    original_find = mg_module.MemoryGraph._find_matching_subgoal

    def fake_find(self, subgoal, subgoal_embedding):
        if subgoal == "ship app":
            return self.subgoal_nodes[0]
        return original_find(self, subgoal, subgoal_embedding)

    monkeypatch.setattr(mg_module.MemoryGraph, "_find_matching_subgoal", fake_find)
    monkeypatch.setattr(mg_module, "get_new_subgoal", lambda *args, **kwargs: "ship release")

    second = client.post("/api/v1/graphs/proc_merge_test/memories", json={
        "mode": "structured",
        "procedural": [
            {"subgoal": "ship app", "procedural_memory": "roll out canary deploy"},
        ],
    })
    assert second.status_code == 200, second.text

    resp = client.get("/api/v1/graphs/proc_merge_test/nodes", params={"node_type": "subgoal"})
    assert resp.status_code == 200, resp.text
    nodes = resp.json()["nodes"]
    assert len(nodes) == 1
    assert nodes[0]["subgoal"] == "ship release"


def test_insert_structured_episodic(client):
    client.post("/api/v1/graphs", json={"graph_id": "epi_test"})
    resp = client.post("/api/v1/graphs/epi_test/memories", json={
        "mode": "structured",
        "episodic": [[
            {"observation": "saw login page", "action": "clicked login"},
            {"observation": "saw dashboard", "action": "clicked settings"},
        ]],
    })
    assert resp.status_code == 200
    assert resp.json()["stats"]["episodic"] == 2


def test_insert_graph_not_found(client):
    resp = client.post("/api/v1/graphs/nope/memories", json={
        "mode": "structured",
        "semantic": [{"semantic_memory": "test", "tags": []}],
    })
    assert resp.status_code == 404


def test_insert_trajectory_missing_goal(client):
    client.post("/api/v1/graphs", json={"graph_id": "traj_err"})
    resp = client.post("/api/v1/graphs/traj_err/memories", json={
        "mode": "trajectory",
        "steps": [{"observation": "a", "action": "b"}],
    })
    assert resp.status_code == 422


def test_insert_stamps_session_id_on_all_node_types(client):
    """An insert with session_id should stamp it on every newly-created node."""
    client.post("/api/v1/graphs", json={"graph_id": "sess_test"})
    resp = client.post("/api/v1/graphs/sess_test/memories", json={
        "mode": "structured",
        "session_id": "run-2026-05-01",
        "episodic": [[{"observation": "obs", "action": "act"}]],
        "semantic": [{"semantic_memory": "fact", "tags": ["t"]}],
        "procedural": [{"subgoal": "do", "procedural_memory": "how"}],
    })
    assert resp.status_code == 200

    # Read each node type back via the inspector and check session_id field
    for node_type, expected_count in [("episodic", 1), ("semantic", 1), ("procedural", 1)]:
        r = client.get(
            f"/api/v1/graphs/sess_test/search?node_type={node_type}&limit=10"
        )
        assert r.status_code == 200, r.text
        nodes = r.json()["nodes"]
        assert len(nodes) == expected_count
        assert nodes[0]["session_id"] == "run-2026-05-01", (
            f"{node_type} node missing session_id: {nodes[0]}"
        )


def test_insert_without_session_id_leaves_field_null(client):
    client.post("/api/v1/graphs", json={"graph_id": "no_sess_test"})
    resp = client.post("/api/v1/graphs/no_sess_test/memories", json={
        "mode": "structured",
        "semantic": [{"semantic_memory": "fact", "tags": []}],
    })
    assert resp.status_code == 200

    r = client.get("/api/v1/graphs/no_sess_test/search?node_type=semantic")
    nodes = r.json()["nodes"]
    assert nodes[0]["session_id"] is None


def test_batch_insert_structured_items(client):
    client.post("/api/v1/graphs", json={"graph_id": "batch_test"})
    resp = client.post("/api/v1/graphs/batch_test/memories/batch", json={
        "items": [
            {
                "mode": "structured",
                "session_id": "run-A",
                "semantic": [{"semantic_memory": "fact A", "tags": ["shared"]}],
                "procedural": [{"subgoal": "do A", "procedural_memory": "how A"}],
            },
            {
                "mode": "structured",
                "session_id": "run-B",
                "episodic": [[{"observation": "obs B", "action": "act B"}]],
                "semantic": [{"semantic_memory": "fact B", "tags": ["shared"]}],
            },
        ]
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["stats"]["semantic"] == 2
    assert data["stats"]["procedural"] == 1
    assert data["stats"]["episodic"] == 1

    sessions = client.get("/api/v1/graphs/batch_test/sessions").json()["sessions"]
    assert "run-A" in sessions
    assert "run-B" in sessions


def test_batch_insert_rejects_trajectory_items(client):
    client.post("/api/v1/graphs", json={"graph_id": "batch_err"})
    resp = client.post("/api/v1/graphs/batch_err/memories/batch", json={
        "items": [
            {"mode": "structured", "semantic": [{"semantic_memory": "ok", "tags": []}]},
            {"mode": "trajectory", "goal": "g", "steps": [{"observation": "o", "action": "a"}]},
        ]
    })
    assert resp.status_code == 422


def test_batch_insert_empty_is_noop(client):
    client.post("/api/v1/graphs", json={"graph_id": "batch_empty"})
    resp = client.post("/api/v1/graphs/batch_empty/memories/batch", json={"items": []})
    assert resp.status_code == 200
    assert resp.json()["stats"]["semantic"] == 0


def test_http_embedding_client_batches_requests(monkeypatch):
    captured = {}

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [
                    {"embedding": [1.0, 0.0]},
                    {"embedding": [0.0, 1.0]},
                ]
            }

    def fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr("plugmem.clients.embedding.requests.post", fake_post)

    client = HTTPEmbeddingClient(
        base_url="https://example.test/v1/embeddings",
        model="test-model",
        api_key="secret",
        timeout=7,
    )
    out = client.embed_batch(["alpha", "beta"])

    assert out == [[1.0, 0.0], [0.0, 1.0]]
    assert captured["url"] == "https://example.test/v1/embeddings"
    assert captured["json"] == {"model": "test-model", "input": ["alpha", "beta"]}
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["timeout"] == 7
