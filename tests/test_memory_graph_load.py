from __future__ import annotations

from plugmem.core.memory import Memory


def test_load_is_idempotent(graph_manager):
    graph_id = graph_manager.create_graph("reload_test")
    graph = graph_manager.get_graph(graph_id)

    mem = Memory.from_structured(embedder=graph_manager.embedder, session_id="s1")
    mem.memory["episodic"] = [[{"observation": "obs", "action": "act"}]]
    mem.memory["semantic"] = [{"semantic_memory": "fact", "tags": ["t"]}]
    mem.memory["procedural"] = [{"subgoal": "do", "procedural_memory": "how"}]
    mem.memory_embedding["semantic"] = [{
        "semantic_memory": graph_manager.embedder.embed("fact"),
        "tags": [graph_manager.embedder.embed("t")],
    }]
    mem.memory_embedding["procedural"] = [{
        "subgoal": graph_manager.embedder.embed("do"),
    }]
    graph.insert(mem)

    graph.load()
    assert len(graph.episodic_nodes) == 1
    assert len(graph.semantic_nodes) == 1
    assert len(graph.procedural_nodes) == 1
    assert len(graph.session_ids["s1"]) == 1
    assert len(graph.session_semantic_ids["s1"]) == 1
    assert len(graph.session_procedural_ids["s1"]) == 1


def test_load_rebuilds_tag_links_from_semantic_metadata(graph_manager):
    graph_id = graph_manager.create_graph("reload_tags")
    graph = graph_manager.get_graph(graph_id)

    mem = Memory.from_structured(embedder=graph_manager.embedder)
    mem.memory["semantic"] = [
        {"semantic_memory": "fact one", "tags": ["shared"]},
        {"semantic_memory": "fact two", "tags": ["shared"]},
    ]
    mem.memory_embedding["semantic"] = [
        {
            "semantic_memory": graph_manager.embedder.embed("fact one"),
            "tags": [graph_manager.embedder.embed("shared")],
        },
        {
            "semantic_memory": graph_manager.embedder.embed("fact two"),
            "tags": [graph_manager.embedder.embed("shared")],
        },
    ]
    graph.insert(mem)

    graph.load()
    assert len(graph.tag_nodes) == 1
    assert len(graph.tag_nodes[0].semantic_nodes) == 2
    assert len(graph.semantic_nodes[0].tag_nodes) == 1
    assert len(graph.semantic_nodes[1].tag_nodes) == 1


def test_load_rebuilds_subgoal_links_from_procedural_metadata(graph_manager):
    graph_id = graph_manager.create_graph("reload_subgoals")
    graph = graph_manager.get_graph(graph_id)

    mem = Memory.from_structured(embedder=graph_manager.embedder)
    mem.memory["procedural"] = [
        {"subgoal": "deploy app", "procedural_memory": "build image", "return": 1.0},
        {"subgoal": "deploy app", "procedural_memory": "roll out release", "return": 1.0},
    ]
    mem.memory_embedding["procedural"] = [
        {"subgoal": graph_manager.embedder.embed("deploy app")},
        {"subgoal": graph_manager.embedder.embed("deploy app")},
    ]
    graph.insert(mem)

    graph.load()
    assert len(graph.subgoal_nodes) == 1
    assert len(graph.subgoal_nodes[0].procedural_nodes) == 2
    assert len(graph.procedural_nodes[0].subgoal_nodes) == 1
    assert len(graph.procedural_nodes[1].subgoal_nodes) == 1
