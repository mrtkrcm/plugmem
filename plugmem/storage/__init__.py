from __future__ import annotations

from typing import TYPE_CHECKING, Union

from plugmem.storage.chroma import ChromaStorage

# SqliteVecStorage is imported lazily (via TYPE_CHECKING for type annotation)
# so the optional sqlite-vec dependency is only needed when the backend is
# actually selected through STORAGE_BACKEND=sqlite_vec.
if TYPE_CHECKING:
    from plugmem.storage.sqlite_vec import SqliteVecStorage

# Storage backends share a duck-typed interface (create_graph, list_graphs,
# add_/query_/get_all_ for each node type, recall audit, text search,
# session-scoped queries). Code paths that accept either backend should
# annotate against this alias.
StorageBackend = Union[ChromaStorage, "SqliteVecStorage"]

__all__ = ["ChromaStorage", "SqliteVecStorage", "StorageBackend"]
