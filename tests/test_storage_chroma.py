from __future__ import annotations

from plugmem.storage.chroma import ChromaStorage


class _DummyCollection:
    def __init__(self):
        self.calls = []

    def add(self, **kwargs):
        self.calls.append(kwargs)


def test_batch_add_uses_single_call_when_all_rows_have_embeddings():
    storage = object.__new__(ChromaStorage)
    col = _DummyCollection()
    storage._col = lambda graph_id, node_type: col  # type: ignore[method-assign]

    storage._batch_add(
        "g",
        "semantic",
        ids=["0", "1"],
        docs=["a", "b"],
        metas=[{"semantic_id": 0}, {"semantic_id": 1}],
        embs=[[0.1, 0.2], [0.3, 0.4]],
        any_emb=True,
    )

    assert len(col.calls) == 1
    assert col.calls[0]["ids"] == ["0", "1"]
    assert col.calls[0]["embeddings"] == [[0.1, 0.2], [0.3, 0.4]]


def test_batch_add_falls_back_to_row_calls_for_mixed_embeddings():
    storage = object.__new__(ChromaStorage)
    col = _DummyCollection()
    storage._col = lambda graph_id, node_type: col  # type: ignore[method-assign]

    storage._batch_add(
        "g",
        "semantic",
        ids=["0", "1"],
        docs=["a", "b"],
        metas=[{"semantic_id": 0}, {"semantic_id": 1}],
        embs=[[0.1, 0.2], None],
        any_emb=True,
    )

    assert len(col.calls) == 2
    assert "embeddings" in col.calls[0]
    assert "embeddings" not in col.calls[1]
