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

import pytest

pytest.importorskip("sqlite_vec")

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


def test_update_method_signatures_match_chroma():
    """Future DI selector must not break on kwarg shape mismatches."""
    for method in ("update_semantic", "update_procedural", "update_tag",
                   "update_subgoal", "update_tag_metadata_batch"):
        chroma_sig = inspect.signature(getattr(ChromaStorage, method))
        sqlite_sig = inspect.signature(getattr(SqliteVecStorage, method))
        assert list(chroma_sig.parameters) == list(sqlite_sig.parameters), (
            f"{method} parameter list drifted: "
            f"chroma={list(chroma_sig.parameters)} "
            f"sqlite={list(sqlite_sig.parameters)}"
        )


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
    assert tags["metadatas"][0]["semantic_ids"] == [0, 1]


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
