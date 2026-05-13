from __future__ import annotations

from plugmem.core.memory import Memory


def _seed_semantics(graph_manager, graph_id: str = "retrieval_graph"):
    graph_manager.create_graph(graph_id)
    graph = graph_manager.get_graph(graph_id)
    mem = Memory.from_structured(embedder=graph_manager.embedder)
    mem.memory["semantic"] = [
        {"semantic_memory": "Water boils at 100 degrees Celsius", "tags": ["physics", "water"]},
        {"semantic_memory": "The Earth orbits the Sun", "tags": ["astronomy"]},
    ]
    mem.memory_embedding["semantic"] = [
        {
            "semantic_memory": graph_manager.embedder.embed("Water boils at 100 degrees Celsius"),
            "tags": [graph_manager.embedder.embed("physics"), graph_manager.embedder.embed("water")],
        },
        {
            "semantic_memory": graph_manager.embedder.embed("The Earth orbits the Sun"),
            "tags": [graph_manager.embedder.embed("astronomy")],
        },
    ]
    graph.insert(mem)
    return graph


def test_semantic_retrieval_prefers_storage_query(graph_manager, monkeypatch):
    graph = _seed_semantics(graph_manager, "retrieval_query")
    seen = {"called": False}

    original = graph.storage.query_semantic

    def wrapped(*args, **kwargs):
        seen["called"] = True
        return original(*args, **kwargs)

    monkeypatch.setattr(graph.storage, "query_semantic", wrapped)

    query_embedding = graph_manager.embedder.embed("What temperature does water boil?")
    nodes, sim_list = graph._semantic_similarity_candidates(
        query_embedding=query_embedding,
        n_results=5,
    )

    assert seen["called"] is True
    assert nodes
    assert sim_list


def test_semantic_retrieval_falls_back_when_storage_query_fails(graph_manager, monkeypatch):
    graph = _seed_semantics(graph_manager, "retrieval_fallback")

    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(graph.storage, "query_semantic", broken)

    query_embedding = graph_manager.embedder.embed("What temperature does water boil?")
    nodes, sim_list = graph._semantic_similarity_candidates(
        query_embedding=query_embedding,
        n_results=5,
    )

    assert nodes
    assert sim_list
    assert any("Water boils" in node.get_semantic_memory() for node in nodes)
