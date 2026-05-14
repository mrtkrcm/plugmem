"""Smoke tests for the experimental SqliteVecStorage backend.

Goals (kept narrow, since this backend is not yet wired through DI):

1. Round-trip a semantic node with embedding → query_semantic returns it.
2. ``update_semantic`` upserts the vec table when embedding is supplied
   (regression: this used to silently no-op the vector).
3. Method signatures parity with ``ChromaStorage`` for ``update_*`` so a
   future DI swap won't crash on kwarg mismatches.
"""
from __future__ import annotations

import inspect
import sqlite3
from unittest.mock import MagicMock

import pytest

pytest.importorskip("sqlite_vec")

from plugmem.core.memory_graph import MemoryGraph  # noqa: E402
from plugmem.storage.sqlite_vec import _ensure_vec_db  # noqa: E402
from plugmem.storage.chroma import ChromaStorage  # noqa: E402
from plugmem.storage.sqlite_vec import SqliteVecStorage  # noqa: E402


def _emb(seed: float) -> list[float]:
    """Deterministic 768-d unit-ish vector."""
    return [seed + i * 1e-4 for i in range(768)]


def test_semantic_roundtrip(tmp_path):
    db = SqliteVecStorage(str(tmp_path / "p.db"))
    db.create_graph("g")
    db.add_semantic_batch("g", [
        {
            "semantic_id": 0,
            "text": "use uv not pip",
            "embedding": _emb(0.1),
            "tags": ["python"],
            "time": 1,
            "source": "explicit",
            "confidence": 0.9,
            "provenance": {"language": "python"},
        },
        {
            "semantic_id": 1,
            "text": "use swift-format",
            "embedding": _emb(0.5),
            "tags": ["swift"],
            "time": 2,
            "source": "correction",
            "confidence": 0.8,
            "provenance": {"language": "swift"},
        },
    ])

    all_sem = db.get_all_semantic("g")
    assert len(all_sem["ids"]) == 2
    assert "use uv not pip" in all_sem["documents"]

    # Nearest neighbour around _emb(0.1) should pick semantic_id=0 first.
    result = db.query_semantic("g", _emb(0.1), n_results=2)
    assert result["ids"][0] == 0


def test_update_semantic_upserts_vector(tmp_path):
    """Regression: update_semantic(embedding=…) must touch the vec table."""
    db = SqliteVecStorage(str(tmp_path / "p.db"))
    db.create_graph("g")
    db.add_semantic_batch("g", [
        {"semantic_id": 0, "text": "old", "embedding": _emb(0.1), "tags": []},
    ])

    # Move the embedding far away — a query near the old vector should no
    # longer return the row first (it's now near _emb(0.9)).
    db.update_semantic("g", 0, embedding=_emb(0.9))

    near_new = db.query_semantic("g", _emb(0.9), n_results=1)
    assert near_new["ids"] == [0]

    # Metadata-only update still works:
    db.update_semantic("g", 0, text="new", metadata_updates={"is_active": 0})
    all_sem = db.get_all_semantic("g")
    assert "new" in all_sem["documents"]
    metas_by_id = {m["semantic_id"]: m for m in all_sem["metadatas"]}
    assert metas_by_id[0]["is_active"] == 0


def _check_method_parity(method: str) -> None:
    """Assert that *method* has identical parameter signatures on both backends."""
    chroma_sig = inspect.signature(getattr(ChromaStorage, method))
    sqlite_sig = inspect.signature(getattr(SqliteVecStorage, method))
    cp = list(chroma_sig.parameters.items())
    sp = list(sqlite_sig.parameters.items())
    assert len(cp) == len(sp), (
        f"{method}: parameter count differs "
        f"(chroma={len(cp)} sqlite={len(sp)})"
    )
    for (ck, cv), (sk, sv) in zip(cp, sp):
        assert ck == sk, f"{method}: param name {ck!r} != {sk!r}"
        assert cv.kind == sv.kind, f"{method}.{ck}: kind {cv.kind} != {sv.kind}"
        if cv.default is not inspect.Parameter.empty and sv.default is not inspect.Parameter.empty:
            assert type(cv.default) is type(sv.default), (
                f"{method}.{ck}: default type {type(cv.default)} != {type(sv.default)}"
            )


_METHODS_TO_CHECK = (
    # update_* methods
    "update_semantic", "update_procedural", "update_tag",
    "update_subgoal", "update_tag_metadata_batch",
    # query_* methods
    "query_semantic", "query_tag", "query_procedural", "query_subgoal",
)


def test_method_signatures_match_chroma():
    """All public method signatures must match between backends."""
    for method in _METHODS_TO_CHECK:
        _check_method_parity(method)


def test_procedural_and_tag_roundtrip(tmp_path):
    db = SqliteVecStorage(str(tmp_path / "p.db"))
    db.create_graph("g")
    db.add_procedural_batch("g", [
        {"procedural_id": 0, "text": "run uv sync", "embedding": _emb(0.2),
         "source": "correction", "confidence": 0.7, "provenance": {"language": "python"}},
    ])
    db.add_tag_batch("g", [
        {"tag_id": 0, "tag": "python", "embedding": _emb(0.3), "importance": 2,
         "semantic_ids": [0, 1]},
    ])

    proc = db.query_procedural("g", _emb(0.2), n_results=1)
    assert proc["documents"] == ["run uv sync"]

    tags = db.get_all_tags("g")
    assert tags["documents"] == ["python"]
    assert tags["metadatas"][0]["semantic_ids"] == "[0, 1]"


def test_build_storage_selector_dispatches(tmp_path):
    """build_storage(cfg) returns the right backend for each selector."""
    from plugmem.api.dependencies import build_storage
    from plugmem.config import PlugMemConfig

    cfg = PlugMemConfig(
        storage_backend="sqlite_vec",
        sqlite_vec_path=str(tmp_path / "via-di.db"),
    )
    storage = build_storage(cfg)
    assert isinstance(storage, SqliteVecStorage)

    # Unknown backend → clear error.
    cfg_bad = PlugMemConfig(storage_backend="bogus")
    try:
        build_storage(cfg_bad)
    except ValueError as exc:
        assert "Unknown storage_backend" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for bogus backend")


def test_delete_graph_drops_tables(tmp_path):
    db = SqliteVecStorage(str(tmp_path / "p.db"))
    db.create_graph("g")
    db.add_semantic_batch("g", [
        {"semantic_id": 0, "text": "x", "embedding": _emb(0.1), "tags": []},
    ])
    assert db.graph_exists("g")
    db.delete_graph("g")
    assert not db.graph_exists("g")
    assert "g" not in db.list_graphs()


def test_list_graphs_returns_names(tmp_path):
    """Regression: list_graphs must return the correct graph name (not a substring)."""
    db = SqliteVecStorage(str(tmp_path / "p.db"))
    assert db.list_graphs() == []
    db.create_graph("g")
    assert db.list_graphs() == ["g"]
    db.create_graph("my_graph")
    assert db.list_graphs() == ["g", "my_graph"]


def test_update_subgoal_procedural_ids(tmp_path):
    """Regression: update_subgoal must JSON-encode procedural_ids (not write raw repr)."""
    db = SqliteVecStorage(str(tmp_path / "p.db"))
    db.create_graph("g")
    db.add_subgoal_batch("g", [
        {"subgoal_id": 0, "subgoal": "install deps", "time": 1, "procedural_ids": []},
    ])
    db.update_subgoal("g", 0, metadata_updates={"procedural_ids": [10, 20]})
    all_sg = db.get_all_subgoals("g")
    assert all_sg["metadatas"][0]["procedural_ids"] == "[10, 20]"


def test_query_semantic_with_where(tmp_path):
    """query_semantic(where=...) must filter by metadata fields."""
    db = SqliteVecStorage(str(tmp_path / "p.db"))
    db.create_graph("g")
    db.add_semantic_batch("g", [
        {"semantic_id": 0, "text": "active", "embedding": _emb(0.1), "tags": [],
         "is_active": 1},
        {"semantic_id": 1, "text": "inactive", "embedding": _emb(0.11), "tags": [],
         "is_active": 0},
    ])
    # Without filter both match.
    all_res = db.query_semantic("g", _emb(0.1), n_results=5)
    assert len(all_res["ids"]) == 2

    # With where is_active=1, only semantic_id=0 should appear.
    filtered = db.query_semantic("g", _emb(0.1), n_results=5, where={"is_active": 1})
    assert filtered["ids"] == [0]


def test_schema_dim_mismatch_raises(tmp_path):
    """Opening an existing db with a different embedding dim must raise a clear error."""
    path = tmp_path / "dim_mismatch.db"
    db = SqliteVecStorage(str(path), embedding_dim=32)
    db.create_graph("g")
    db.add_semantic_batch("g", [
        {"semantic_id": 0, "text": "x", "embedding": [0.1] * 32, "tags": []},
    ])
    db.close()

    with pytest.raises(ValueError, match="schema dim=32 does not match embedder dim=768"):
        SqliteVecStorage(str(path), embedding_dim=768)

    # Reopening with the same dim works.
    db2 = SqliteVecStorage(str(path), embedding_dim=32)
    assert db2.graph_exists("g")


def test_storage_with_explicit_dim(tmp_path):
    """build_sqlite_vec_storage accepts an explicit dim; round-trip with 32-d works."""
    from plugmem.api.dependencies import build_sqlite_vec_storage
    from plugmem.config import PlugMemConfig

    cfg = PlugMemConfig(
        storage_backend="sqlite_vec",
        sqlite_vec_path=str(tmp_path / "dim32.db"),
    )
    storage = build_sqlite_vec_storage(cfg, embedding_dim=32)
    storage.create_graph("g")
    emb = [0.5] * 32
    storage.add_semantic_batch("g", [
        {"semantic_id": 0, "text": "hello", "embedding": emb, "tags": []},
    ])
    result = storage.query_semantic("g", emb, n_results=1)
    assert result["ids"] == [0]
    assert result["documents"] == ["hello"]


def test_reload_preserves_episodic_content(tmp_path):
    db = SqliteVecStorage(str(tmp_path / "episodic.db"), embedding_dim=32)
    db.create_graph("g")
    db.add_episodic_batch("g", [
        {
            "episodic_id": 0,
            "observation": "compiler error",
            "action": "inspect logs",
            "time": "2026-05-14T10:00:00Z",
            "session_id": "s1",
            "subgoal": "debug build",
            "state": "failing",
            "reward": "-1",
        }
    ])

    graph = MemoryGraph("g", storage=db, llm=MagicMock(), embedder=MagicMock())
    graph.load()

    assert len(graph.episodic_nodes) == 1
    node = graph.episodic_nodes[0]
    assert node.observation == "compiler error"
    assert node.action == "inspect logs"


def test_loaded_embeddings_are_decoded_as_float_vectors(tmp_path):
    db = SqliteVecStorage(str(tmp_path / "embeddings.db"), embedding_dim=32)
    db.create_graph("g")
    emb = _emb32(0.25)
    db.add_semantic_batch("g", [
        {"semantic_id": 0, "text": "decoded", "embedding": emb, "tags": []},
    ])

    all_sem = db.get_all_semantic("g")
    loaded = all_sem["embeddings"][0]

    assert loaded is not None
    assert len(loaded) == 32
    assert loaded == pytest.approx(emb)


def test_existing_db_migrates_son_semantic_ids_column(tmp_path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    _ensure_vec_db(conn)
    conn.execute(
        """
        CREATE TABLE "g_semantic_meta" (
            semantic_id INTEGER PRIMARY KEY,
            text TEXT NOT NULL DEFAULT '',
            tags TEXT NOT NULL DEFAULT '[]',
            time INTEGER NOT NULL DEFAULT 0,
            is_active INTEGER NOT NULL DEFAULT 1,
            credibility INTEGER NOT NULL DEFAULT 10,
            session_id TEXT,
            date TEXT NOT NULL DEFAULT '',
            source TEXT,
            confidence REAL NOT NULL DEFAULT 0.5,
            provenance TEXT NOT NULL DEFAULT '{}',
            tag_ids TEXT NOT NULL DEFAULT '[]',
            episodic_ids TEXT NOT NULL DEFAULT '[]',
            bro_semantic_ids TEXT NOT NULL DEFAULT '[]'
        )
        """
    )
    conn.execute(
        """
        CREATE VIRTUAL TABLE "g_semantic_vec" USING vec0(
            semantic_id INTEGER PRIMARY KEY,
            embedding float[32]
        )
        """
    )
    conn.close()

    db = SqliteVecStorage(str(path), embedding_dim=32)
    db.add_semantic_batch("g", [
        {
            "semantic_id": 0,
            "text": "migrated",
            "embedding": _emb32(0.4),
            "tags": [],
            "son_semantic_ids": [1, 2],
        },
    ])

    all_sem = db.get_all_semantic("g")
    assert all_sem["metadatas"][0]["semantic_id"] == 0


def _emb32(seed: float) -> list[float]:
    """Deterministic 32-d vector for tests."""
    return [seed + i * 1e-4 for i in range(32)]


def test_search_by_text_semantic(tmp_path):
    db = SqliteVecStorage(str(tmp_path / "p.db"), embedding_dim=32)
    db.create_graph("g")
    db.add_semantic_batch("g", [
        {"semantic_id": 0, "text": "use uv not pip", "embedding": _emb32(0.1), "tags": [], "is_active": 1},
        {"semantic_id": 1, "text": "use swift-format", "embedding": _emb32(0.2), "tags": [], "is_active": 1},
        {"semantic_id": 2, "text": "uv is fast", "embedding": _emb32(0.3), "tags": [], "is_active": 0},
    ])

    result = db.search_by_text("g", "semantic", "uv", limit=10)
    assert set(result["ids"]) == {0, 2}
    assert "use uv not pip" in result["documents"]

    # only_active filter
    active = db.search_by_text("g", "semantic", "uv", limit=10, only_active=True)
    assert active["ids"] == [0]


def test_search_by_text_procedural(tmp_path):
    db = SqliteVecStorage(str(tmp_path / "p.db"), embedding_dim=32)
    db.create_graph("g")
    db.add_procedural_batch("g", [
        {"procedural_id": 0, "text": "run uv sync", "embedding": _emb32(0.1), "source": None, "confidence": 0.5},
        {"procedural_id": 1, "text": "run swift build", "embedding": _emb32(0.2), "source": None, "confidence": 0.5},
    ])
    result = db.search_by_text("g", "procedural", "uv", limit=10)
    assert result["ids"] == [0]


def test_search_by_text_tag(tmp_path):
    db = SqliteVecStorage(str(tmp_path / "p.db"), embedding_dim=32)
    db.create_graph("g")
    db.add_tag_batch("g", [
        {"tag_id": 0, "tag": "python", "embedding": _emb32(0.1)},
        {"tag_id": 1, "tag": "swift", "embedding": _emb32(0.2)},
    ])
    result = db.search_by_text("g", "tag", "py", limit=10)
    assert result["ids"] == [0]
    assert result["documents"] == ["python"]


def test_search_by_text_subgoal(tmp_path):
    db = SqliteVecStorage(str(tmp_path / "p.db"), embedding_dim=32)
    db.create_graph("g")
    db.add_subgoal_batch("g", [
        {"subgoal_id": 0, "subgoal": "install deps", "time": 1, "procedural_ids": []},
        {"subgoal_id": 1, "subgoal": "build project", "time": 2, "procedural_ids": []},
    ])
    result = db.search_by_text("g", "subgoal", "deps", limit=10)
    assert result["ids"] == [0]


def test_search_by_text_episodic(tmp_path):
    db = SqliteVecStorage(str(tmp_path / "p.db"), embedding_dim=32)
    db.create_graph("g")
    db.add_episodic_batch("g", [
        {"episodic_id": 0, "observation": "error: file not found", "action": "check path"},
        {"episodic_id": 1, "observation": "build succeeded", "action": "run tests"},
    ])
    # Search in observation
    obs = db.search_by_text("g", "episodic", "error", limit=10)
    assert obs["ids"] == [0]
    # Search in action
    act = db.search_by_text("g", "episodic", "check", limit=10)
    assert act["ids"] == [0]


def test_get_nodes_by_session(tmp_path):
    db = SqliteVecStorage(str(tmp_path / "p.db"), embedding_dim=32)
    db.create_graph("g")
    db.add_episodic_batch("g", [
        {"episodic_id": 0, "observation": "obs-A", "action": "act-A", "session_id": "s1"},
        {"episodic_id": 1, "observation": "obs-B", "action": "act-B", "session_id": "s2"},
    ])
    db.add_semantic_batch("g", [
        {"semantic_id": 0, "text": "fact-A", "embedding": _emb32(0.1), "tags": [], "session_id": "s1"},
        {"semantic_id": 1, "text": "fact-B", "embedding": _emb32(0.2), "tags": [], "session_id": "s2"},
    ])
    db.add_procedural_batch("g", [
        {"procedural_id": 0, "text": "proc-A", "embedding": _emb32(0.1), "source": None, "confidence": 0.5, "session_id": "s1"},
        {"procedural_id": 1, "text": "proc-B", "embedding": _emb32(0.2), "source": None, "confidence": 0.5, "session_id": "s2"},
    ])

    session = db.get_nodes_by_session("g", "s1")
    assert len(session["episodic"]) == 1
    assert session["episodic"][0]["observation"] == "obs-A"
    assert len(session["semantic"]) == 1
    assert session["semantic"][0]["text"] == "fact-A"
    assert len(session["procedural"]) == 1
    assert session["procedural"][0]["text"] == "proc-A"

    # s2 should have different data
    session2 = db.get_nodes_by_session("g", "s2")
    assert len(session2["episodic"]) == 1
    assert session2["episodic"][0]["observation"] == "obs-B"
