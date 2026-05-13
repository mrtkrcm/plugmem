"""Unit test for MemoryGraph.merge_semantic deactivation flags."""
from __future__ import annotations

import plugmem.core.memory_graph as mg_module


def test_merge_semantic_deactivates_per_decision(client, graph_manager, monkeypatch):
    """merge_semantic must surface the LLM's deactivate flags verbatim and
    persist the merged node with a new semantic_id."""
    client.post("/api/v1/graphs", json={"graph_id": "merge_test"})
    client.post("/api/v1/graphs/merge_test/memories", json={
        "mode": "structured",
        "semantic": [
            {"semantic_memory": "apples are red", "tags": []},
            {"semantic_memory": "apples are green", "tags": []},
        ],
    })
    g = graph_manager.get_graph("merge_test")
    assert len(g.semantic_nodes) == 2
    id1, id2 = g.semantic_nodes[0].semantic_id, g.semantic_nodes[1].semantic_id

    # Stub the LLM-driven merge decision; verify graph-level wiring only.
    def fake_get_new_semantic(*args, **kwargs):
        return {
            "merged_statement": "apples come in many colors",
            "deactivate_earlier": True,
            "deactivate_later": False,
        }

    monkeypatch.setattr(mg_module, "get_new_semantic", fake_get_new_semantic)

    merged_node, del_1, del_2 = g.merge_semantic(id1, id2)

    assert del_1 is True
    assert del_2 is False
    assert merged_node.semantic_memory_str == "apples come in many colors"
    # Merged node gets a fresh id, distinct from inputs.
    assert merged_node.semantic_id not in (id1, id2)
    # son linkage records both inputs.
    son_ids = {s.semantic_id for s in merged_node.son_semantic}
    assert son_ids == {id1, id2}
