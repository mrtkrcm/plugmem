"""SqliteVec-backed storage for PlugMem memory graphs.

**Experimental.** Alternative to :class:`ChromaStorage` that uses sqlite-vec
for vector similarity search and regular SQLite tables for metadata. Shares
the same public method signatures so :class:`~plugmem.graph_manager.GraphManager`
can be pointed at either backend.

Selectable via ``STORAGE_BACKEND=sqlite_vec`` or through ``plugmem init``.
Coverage is limited to a smoke test (``tests/test_storage_sqlite_vec.py``);
exercise it against your own workload before relying on it in production.

Install with::

    pip install -e ".[sqlite-vec]"

See ``CLAUDE.md → Storage backends`` for the gap list before promoting it
from experimental.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import sqlite_vec
from sqlite_vec import serialize_float32

logger = logging.getLogger(__name__)

NODE_TYPES = ("semantic", "procedural", "tag", "subgoal", "episodic")

_VEC_DIMS: dict = {
    "semantic": 768,
    "procedural": 768,
    "tag": 768,
    "subgoal": 768,
}


def _serialize(v: Any) -> bytes:
    """Convert a list of floats (or numpy array) to sqlite-vec binary format.

    Callers must gate on ``embedding is not None`` — passing ``None`` is a bug.
    """
    if hasattr(v, "tolist"):
        v = v.tolist()
    return serialize_float32(v)


def _ensure_vec_db(conn: sqlite3.Connection) -> None:
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


_SCHEMA_SEMANTIC = """
CREATE TABLE IF NOT EXISTS "{gid}_semantic_meta" (
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
);
"""
_SCHEMA_SEMANTIC_VEC = """
CREATE VIRTUAL TABLE IF NOT EXISTS "{gid}_semantic_vec" USING vec0(
    semantic_id INTEGER PRIMARY KEY,
    embedding float[{dim}]
);
"""

_SCHEMA_PROCEDURAL = """
CREATE TABLE IF NOT EXISTS "{gid}_procedural_meta" (
    procedural_id INTEGER PRIMARY KEY,
    text TEXT NOT NULL DEFAULT '',
    subgoal TEXT NOT NULL DEFAULT '',
    subgoal_id INTEGER,
    time INTEGER NOT NULL DEFAULT 0,
    return_value REAL NOT NULL DEFAULT 0.0,
    source TEXT,
    confidence REAL NOT NULL DEFAULT 0.5,
    session_id TEXT,
    provenance TEXT NOT NULL DEFAULT '{}',
    episodic_ids TEXT NOT NULL DEFAULT '[]'
);
"""
_SCHEMA_PROCEDURAL_VEC = """
CREATE VIRTUAL TABLE IF NOT EXISTS "{gid}_procedural_vec" USING vec0(
    procedural_id INTEGER PRIMARY KEY,
    embedding float[{dim}]
);
"""

_SCHEMA_TAG = """
CREATE TABLE IF NOT EXISTS "{gid}_tag_meta" (
    tag_id INTEGER PRIMARY KEY,
    tag TEXT NOT NULL,
    importance INTEGER NOT NULL DEFAULT 1,
    time INTEGER NOT NULL DEFAULT 0,
    semantic_ids TEXT NOT NULL DEFAULT '[]'
);
"""
_SCHEMA_TAG_VEC = """
CREATE VIRTUAL TABLE IF NOT EXISTS "{gid}_tag_vec" USING vec0(
    tag_id INTEGER PRIMARY KEY,
    embedding float[{dim}]
);
"""

_SCHEMA_SUBGOAL = """
CREATE TABLE IF NOT EXISTS "{gid}_subgoal_meta" (
    subgoal_id INTEGER PRIMARY KEY,
    subgoal TEXT NOT NULL DEFAULT '',
    time INTEGER NOT NULL DEFAULT 0,
    procedural_ids TEXT NOT NULL DEFAULT '[]'
);
"""
_SCHEMA_SUBGOAL_VEC = """
CREATE VIRTUAL TABLE IF NOT EXISTS "{gid}_subgoal_vec" USING vec0(
    subgoal_id INTEGER PRIMARY KEY,
    embedding float[{dim}]
);
"""

_SCHEMA_EPISODIC = """
CREATE TABLE IF NOT EXISTS "{gid}_episodic" (
    episodic_id INTEGER PRIMARY KEY,
    observation TEXT NOT NULL DEFAULT '',
    action TEXT NOT NULL DEFAULT '',
    time TEXT NOT NULL DEFAULT '',
    session_id TEXT,
    subgoal TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL DEFAULT '',
    reward TEXT NOT NULL DEFAULT ''
);
"""

_SCHEMA_RECALL = """
CREATE TABLE IF NOT EXISTS "{gid}_recall_audit" (
    recall_id INTEGER PRIMARY KEY AUTOINCREMENT,
    endpoint TEXT NOT NULL DEFAULT '',
    ts TEXT NOT NULL DEFAULT '',
    graph_time INTEGER NOT NULL DEFAULT 0,
    session_id TEXT,
    observation TEXT NOT NULL DEFAULT '',
    goal TEXT NOT NULL DEFAULT '',
    subgoal TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL DEFAULT '',
    task_type TEXT NOT NULL DEFAULT '',
    mode TEXT NOT NULL DEFAULT '',
    next_subgoal TEXT NOT NULL DEFAULT '',
    query_tags TEXT NOT NULL DEFAULT '[]',
    selected_semantic_ids TEXT NOT NULL DEFAULT '[]',
    selected_procedural_ids TEXT NOT NULL DEFAULT '[]',
    n_messages INTEGER NOT NULL DEFAULT 0
);
"""


def _fmt(sql: str, gid: str, dim: int = 768) -> str:
    return sql.replace("{gid}", gid).replace("{dim}", str(dim))


class SqliteVecStorage:
    """sqlite-vec backed storage for PlugMem memory graphs.

    Each graph gets 10 tables (5 metadata + 5 vec virtual tables)
    plus 1 recall audit table.
    """

    def __init__(self, db_path: str):
        self._lock = threading.Lock()
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        _ensure_vec_db(self._conn)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        # Track which graphs have had their schema created
        self._initialized: set[str] = set()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------ #
    # Schema init
    # ------------------------------------------------------------------ #

    def _ensure_graph_tables(self, graph_id: str) -> None:
        if graph_id in self._initialized:
            return
        with self._lock:
            cur = self._conn.cursor()
            for sql in [_SCHEMA_SEMANTIC, _SCHEMA_SEMANTIC_VEC]:
                cur.execute(_fmt(sql, graph_id))
            for sql in [_SCHEMA_PROCEDURAL, _SCHEMA_PROCEDURAL_VEC]:
                cur.execute(_fmt(sql, graph_id))
            for sql in [_SCHEMA_TAG, _SCHEMA_TAG_VEC]:
                cur.execute(_fmt(sql, graph_id))
            for sql in [_SCHEMA_SUBGOAL, _SCHEMA_SUBGOAL_VEC]:
                cur.execute(_fmt(sql, graph_id))
            cur.execute(_fmt(_SCHEMA_EPISODIC, graph_id))
            cur.execute(_fmt(_SCHEMA_RECALL, graph_id))
            self._conn.commit()
            self._initialized.add(graph_id)

    # ------------------------------------------------------------------ #
    # Graph lifecycle
    # ------------------------------------------------------------------ #

    def create_graph(self, graph_id: str) -> None:
        self._ensure_graph_tables(graph_id)

    def delete_graph(self, graph_id: str) -> None:
        with self._lock:
            cur = self._conn.cursor()
            for tbl in [
                f'"{graph_id}_semantic_meta"', f'"{graph_id}_semantic_vec"',
                f'"{graph_id}_procedural_meta"', f'"{graph_id}_procedural_vec"',
                f'"{graph_id}_tag_meta"', f'"{graph_id}_tag_vec"',
                f'"{graph_id}_subgoal_meta"', f'"{graph_id}_subgoal_vec"',
                f'"{graph_id}_episodic"', f'"{graph_id}_recall_audit"',
            ]:
                cur.execute(f"DROP TABLE IF EXISTS {tbl}")
            self._conn.commit()
            self._initialized.discard(graph_id)

    def list_graphs(self) -> List[str]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT DISTINCT SUBSTR(name, 1, LENGTH(name) - 12) "
            "FROM sqlite_master WHERE type='table' AND name LIKE '%_semantic_meta'"
        )
        return sorted(row[0] for row in cur.fetchall())

    def graph_exists(self, graph_id: str) -> bool:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (f"{graph_id}_semantic_meta",),
        )
        return cur.fetchone() is not None

    def get_graph_stats(self, graph_id: str) -> Dict[str, int]:
        self._ensure_graph_tables(graph_id)
        cur = self._conn.cursor()
        counts: Dict[str, int] = {}
        for nt in NODE_TYPES:
            tbl = nt if nt in ("episodic", "recall_audit") else f"{nt}_meta"
            cur.execute(f'SELECT COUNT(*) FROM "{graph_id}_{tbl}"')
            counts[nt] = cur.fetchone()[0]
        return counts

    # ------------------------------------------------------------------ #
    # Episodic
    # ------------------------------------------------------------------ #

    def add_episodic(self, graph_id: str, **kwargs) -> None:
        self.add_episodic_batch(graph_id, [kwargs])

    def add_episodic_batch(self, graph_id: str, items: List[Dict]) -> None:
        self._ensure_graph_tables(graph_id)
        with self._lock:
            cur = self._conn.cursor()
            for item in items:
                cur.execute(
                    f'INSERT INTO "{graph_id}_episodic"(episodic_id,observation,action,time,session_id,subgoal,state,reward) '
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (
                        item["episodic_id"],
                        item.get("observation", ""),
                        item.get("action", ""),
                        item.get("time", ""),
                        item.get("session_id"),
                        item.get("subgoal", ""),
                        item.get("state", ""),
                        item.get("reward", ""),
                    ),
                )
            self._conn.commit()

    def get_episodic(self, graph_id: str, episodic_id: int) -> Optional[Dict]:
        self._ensure_graph_tables(graph_id)
        cur = self._conn.cursor()
        cur.execute(f'SELECT * FROM "{graph_id}_episodic" WHERE episodic_id=?', (episodic_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def get_all_episodic(self, graph_id: str) -> Dict:
        self._ensure_graph_tables(graph_id)
        cur = self._conn.cursor()
        cur.execute(f'SELECT * FROM "{graph_id}_episodic" ORDER BY episodic_id')
        rows = cur.fetchall()
        return {"documents": [dict(r) for r in rows], "metadatas": [dict(r) for r in rows]}

    # ------------------------------------------------------------------ #
    # Semantic
    # ------------------------------------------------------------------ #

    def add_semantic(self, graph_id: str, **kwargs) -> None:
        self.add_semantic_batch(graph_id, [kwargs])

    def add_semantic_batch(self, graph_id: str, items: List[Dict]) -> None:
        self._ensure_graph_tables(graph_id)
        with self._lock:
            cur = self._conn.cursor()
            for item in items:
                provenance = item.get("provenance") or {}
                cur.execute(
                    f'INSERT INTO "{graph_id}_semantic_meta"('
                    "semantic_id,text,tags,time,is_active,credibility,session_id,date,"
                    "source,confidence,provenance,tag_ids,episodic_ids,bro_semantic_ids) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        item["semantic_id"],
                        item.get("text", ""),
                        json.dumps(item.get("tags", [])),
                        item.get("time", 0),
                        item.get("is_active", 1),
                        item.get("credibility", 10),
                        item.get("session_id"),
                        item.get("date", ""),
                        item.get("source"),
                        float(item.get("confidence", 0.5)),
                        json.dumps(provenance),
                        json.dumps(item.get("tag_ids", [])),
                        json.dumps(item.get("episodic_ids", [])),
                        json.dumps(item.get("bro_semantic_ids", [])),
                    ),
                )
                emb = item.get("embedding")
                if emb is not None:
                    cur.execute(
                        f'INSERT INTO "{graph_id}_semantic_vec"(semantic_id,embedding) VALUES (?,?)',
                        (item["semantic_id"], _serialize(emb)),
                    )
            self._conn.commit()

    def update_semantic(
        self,
        graph_id: str,
        semantic_id: int,
        text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Matches ``ChromaStorage.update_semantic`` exactly."""
        self._ensure_graph_tables(graph_id)
        with self._lock:
            cur = self._conn.cursor()
            updates: Dict[str, Any] = {}
            if text is not None:
                updates["text"] = text
            if metadata_updates:
                for k, v in metadata_updates.items():
                    if k in ("tags", "provenance", "tag_ids", "episodic_ids", "bro_semantic_ids"):
                        updates[k] = json.dumps(v)
                    else:
                        updates[k] = v
            if updates:
                sets = ", ".join(f"{k}=?" for k in updates)
                vals = list(updates.values()) + [semantic_id]
                cur.execute(f'UPDATE "{graph_id}_semantic_meta" SET {sets} WHERE semantic_id=?', vals)
            if embedding is not None:
                # vec0 virtual tables don't accept INSERT OR REPLACE on the
                # rowid PK — delete-then-insert to upsert.
                cur.execute(
                    f'DELETE FROM "{graph_id}_semantic_vec" WHERE semantic_id=?',
                    (semantic_id,),
                )
                cur.execute(
                    f'INSERT INTO "{graph_id}_semantic_vec"(semantic_id,embedding) VALUES (?,?)',
                    (semantic_id, _serialize(embedding)),
                )
            self._conn.commit()

    def query_semantic(self, graph_id: str, query_embedding, n_results: int = 10, where: Optional[Dict] = None) -> Dict:
        self._ensure_graph_tables(graph_id)
        cur = self._conn.cursor()
        emb_bytes = _serialize(query_embedding)

        cur.execute(
            f'SELECT semantic_id, distance FROM "{graph_id}_semantic_vec" '
            "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (emb_bytes, max(1, min(n_results, 100))),
        )
        vec_rows = cur.fetchall()

        ids = [r[0] for r in vec_rows]
        docs, metas = [], []
        for sid in ids:
            cur.execute(f'SELECT * FROM "{graph_id}_semantic_meta" WHERE semantic_id=?', (sid,))
            row = cur.fetchone()
            if row is not None:
                d = dict(row)
                docs.append(d.pop("text", ""))
                metas.append(d)
        return {"documents": docs, "metadatas": metas, "ids": ids}

    def get_all_semantic(self, graph_id: str) -> Dict:
        self._ensure_graph_tables(graph_id)
        cur = self._conn.cursor()
        cur.execute(f'SELECT * FROM "{graph_id}_semantic_meta" ORDER BY semantic_id')
        rows = cur.fetchall()
        docs, metas, ids = [], [], []
        for r in rows:
            d = dict(r)
            ids.append(d["semantic_id"])
            docs.append(d.pop("text", ""))
            d["tags"] = json.loads(d.get("tags", "[]"))
            d["provenance"] = json.loads(d.get("provenance", "{}"))
            d["tag_ids"] = json.loads(d.get("tag_ids", "[]"))
            d["episodic_ids"] = json.loads(d.get("episodic_ids", "[]"))
            d["bro_semantic_ids"] = json.loads(d.get("bro_semantic_ids", "[]"))
            metas.append(d)
        return {"documents": docs, "metadatas": metas, "ids": ids}

    # ------------------------------------------------------------------ #
    # Tag
    # ------------------------------------------------------------------ #

    def add_tag(self, graph_id: str, **kwargs) -> None:
        self.add_tag_batch(graph_id, [kwargs])

    def add_tag_batch(self, graph_id: str, items: List[Dict]) -> None:
        self._ensure_graph_tables(graph_id)
        with self._lock:
            cur = self._conn.cursor()
            for item in items:
                cur.execute(
                    f'INSERT INTO "{graph_id}_tag_meta"(tag_id,tag,importance,time,semantic_ids) '
                    "VALUES (?,?,?,?,?)",
                    (
                        item["tag_id"],
                        item.get("tag", ""),
                        item.get("importance", 1),
                        item.get("time", 0),
                        json.dumps(item.get("semantic_ids", [])),
                    ),
                )
                emb = item.get("embedding")
                if emb is not None:
                    cur.execute(
                        f'INSERT INTO "{graph_id}_tag_vec"(tag_id,embedding) VALUES (?,?)',
                        (item["tag_id"], _serialize(emb)),
                    )
            self._conn.commit()

    def update_tag(
        self,
        graph_id: str,
        tag_id: int,
        tag: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Matches ``ChromaStorage.update_tag`` exactly."""
        self._ensure_graph_tables(graph_id)
        with self._lock:
            cur = self._conn.cursor()
            updates: Dict[str, Any] = {}
            if tag is not None:
                updates["tag"] = tag
            if metadata_updates:
                for k, v in metadata_updates.items():
                    if k == "semantic_ids":
                        updates[k] = json.dumps(v)
                    else:
                        updates[k] = v
            if updates:
                sets = ", ".join(f"{k}=?" for k in updates)
                vals = list(updates.values()) + [tag_id]
                cur.execute(f'UPDATE "{graph_id}_tag_meta" SET {sets} WHERE tag_id=?', vals)
            if embedding is not None:
                cur.execute(
                    f'DELETE FROM "{graph_id}_tag_vec" WHERE tag_id=?',
                    (tag_id,),
                )
                cur.execute(
                    f'INSERT INTO "{graph_id}_tag_vec"(tag_id,embedding) VALUES (?,?)',
                    (tag_id, _serialize(embedding)),
                )
            self._conn.commit()

    def update_tag_metadata_batch(
        self,
        graph_id: str,
        metadata_updates_by_id: Dict[int, Dict[str, Any]],
    ) -> None:
        """Matches ``ChromaStorage.update_tag_metadata_batch`` exactly."""
        if not metadata_updates_by_id:
            return
        self._ensure_graph_tables(graph_id)
        with self._lock:
            cur = self._conn.cursor()
            for tag_id, updates in metadata_updates_by_id.items():
                if not updates:
                    continue
                payload: Dict[str, Any] = {}
                for k, v in updates.items():
                    payload[k] = json.dumps(v) if k == "semantic_ids" else v
                sets = ", ".join(f"{k}=?" for k in payload)
                vals = list(payload.values()) + [int(tag_id)]
                cur.execute(f'UPDATE "{graph_id}_tag_meta" SET {sets} WHERE tag_id=?', vals)
            self._conn.commit()

    def query_tag(self, graph_id: str, query_embedding, n_results: int = 10) -> Dict:
        self._ensure_graph_tables(graph_id)
        cur = self._conn.cursor()
        emb_bytes = _serialize(query_embedding)
        cur.execute(
            f'SELECT tag_id, distance FROM "{graph_id}_tag_vec" '
            "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (emb_bytes, max(1, min(n_results, 100))),
        )
        ids = [r[0] for r in cur.fetchall()]
        docs, metas = [], []
        for tid in ids:
            cur.execute(f'SELECT * FROM "{graph_id}_tag_meta" WHERE tag_id=?', (tid,))
            row = cur.fetchone()
            if row:
                d = dict(row)
                d["semantic_ids"] = json.loads(d.get("semantic_ids", "[]"))
                docs.append(d.pop("tag", ""))
                metas.append(d)
        return {"documents": docs, "metadatas": metas, "ids": ids}

    def get_all_tags(self, graph_id: str) -> Dict:
        self._ensure_graph_tables(graph_id)
        cur = self._conn.cursor()
        cur.execute(f'SELECT * FROM "{graph_id}_tag_meta" ORDER BY tag_id')
        rows = cur.fetchall()
        docs, metas = [], []
        for r in rows:
            d = dict(r)
            d["semantic_ids"] = json.loads(d.get("semantic_ids", "[]"))
            docs.append(d.pop("tag", ""))
            metas.append(d)
        return {"documents": docs, "metadatas": metas}

    # ------------------------------------------------------------------ #
    # Subgoal
    # ------------------------------------------------------------------ #

    def add_subgoal(self, graph_id: str, **kwargs) -> None:
        self.add_subgoal_batch(graph_id, [kwargs])

    def add_subgoal_batch(self, graph_id: str, items: List[Dict]) -> None:
        self._ensure_graph_tables(graph_id)
        with self._lock:
            cur = self._conn.cursor()
            for item in items:
                cur.execute(
                    f'INSERT INTO "{graph_id}_subgoal_meta"(subgoal_id,subgoal,time,procedural_ids) '
                    "VALUES (?,?,?,?)",
                    (
                        item["subgoal_id"],
                        item.get("subgoal", ""),
                        item.get("time", 0),
                        json.dumps(item.get("procedural_ids", [])),
                    ),
                )
                emb = item.get("embedding")
                if emb is not None:
                    cur.execute(
                        f'INSERT INTO "{graph_id}_subgoal_vec"(subgoal_id,embedding) VALUES (?,?)',
                        (item["subgoal_id"], _serialize(emb)),
                    )
            self._conn.commit()

    def update_subgoal(
        self,
        graph_id: str,
        subgoal_id: int,
        subgoal: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._ensure_graph_tables(graph_id)
        with self._lock:
            cur = self._conn.cursor()
            if subgoal is not None:
                cur.execute(f'UPDATE "{graph_id}_subgoal_meta" SET subgoal=? WHERE subgoal_id=?', (subgoal, subgoal_id))
            if metadata_updates:
                sets = ", ".join(f"{k}=?" for k in metadata_updates)
                vals = list(metadata_updates.values()) + [subgoal_id]
                cur.execute(f'UPDATE "{graph_id}_subgoal_meta" SET {sets} WHERE subgoal_id=?', vals)
            if embedding is not None:
                cur.execute(
                    f'DELETE FROM "{graph_id}_subgoal_vec" WHERE subgoal_id=?',
                    (subgoal_id,),
                )
                cur.execute(
                    f'INSERT INTO "{graph_id}_subgoal_vec"(subgoal_id,embedding) VALUES (?,?)',
                    (subgoal_id, _serialize(embedding)),
                )
            self._conn.commit()

    def query_subgoal(self, graph_id: str, query_embedding, n_results: int = 10) -> Dict:
        self._ensure_graph_tables(graph_id)
        cur = self._conn.cursor()
        emb_bytes = _serialize(query_embedding)
        cur.execute(
            f'SELECT subgoal_id, distance FROM "{graph_id}_subgoal_vec" '
            "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (emb_bytes, max(1, min(n_results, 100))),
        )
        ids = [r[0] for r in cur.fetchall()]
        docs, metas = [], []
        for sid in ids:
            cur.execute(f'SELECT * FROM "{graph_id}_subgoal_meta" WHERE subgoal_id=?', (sid,))
            row = cur.fetchone()
            if row:
                d = dict(row)
                d["procedural_ids"] = json.loads(d.get("procedural_ids", "[]"))
                docs.append(d.pop("subgoal", ""))
                metas.append(d)
        return {"documents": docs, "metadatas": metas}

    def get_all_subgoals(self, graph_id: str) -> Dict:
        self._ensure_graph_tables(graph_id)
        cur = self._conn.cursor()
        cur.execute(f'SELECT * FROM "{graph_id}_subgoal_meta" ORDER BY subgoal_id')
        rows = cur.fetchall()
        docs, metas = [], []
        for r in rows:
            d = dict(r)
            d["procedural_ids"] = json.loads(d.get("procedural_ids", "[]"))
            docs.append(d.pop("subgoal", ""))
            metas.append(d)
        return {"documents": docs, "metadatas": metas}

    # ------------------------------------------------------------------ #
    # Procedural
    # ------------------------------------------------------------------ #

    def add_procedural(self, graph_id: str, **kwargs) -> None:
        self.add_procedural_batch(graph_id, [kwargs])

    def add_procedural_batch(self, graph_id: str, items: List[Dict]) -> None:
        self._ensure_graph_tables(graph_id)
        with self._lock:
            cur = self._conn.cursor()
            for item in items:
                provenance = item.get("provenance") or {}
                cur.execute(
                    f'INSERT INTO "{graph_id}_procedural_meta"('
                    "procedural_id,text,subgoal,subgoal_id,time,return_value,"
                    "source,confidence,session_id,provenance,episodic_ids) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        item["procedural_id"],
                        item.get("text", ""),
                        item.get("subgoal", ""),
                        item.get("subgoal_id"),
                        item.get("time", 0),
                        item.get("return_value", 0.0),
                        item.get("source"),
                        float(item.get("confidence", 0.5)),
                        item.get("session_id"),
                        json.dumps(provenance),
                        json.dumps(item.get("episodic_ids", [])),
                    ),
                )
                emb = item.get("embedding")
                if emb is not None:
                    cur.execute(
                        f'INSERT INTO "{graph_id}_procedural_vec"(procedural_id,embedding) VALUES (?,?)',
                        (item["procedural_id"], _serialize(emb)),
                    )
            self._conn.commit()

    def update_procedural(
        self,
        graph_id: str,
        procedural_id: int,
        text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Matches ``ChromaStorage.update_procedural`` exactly."""
        self._ensure_graph_tables(graph_id)
        with self._lock:
            cur = self._conn.cursor()
            updates: Dict[str, Any] = {}
            if text is not None:
                updates["text"] = text
            if metadata_updates:
                for k, v in metadata_updates.items():
                    if k in ("provenance", "episodic_ids"):
                        updates[k] = json.dumps(v)
                    else:
                        updates[k] = v
            if updates:
                sets = ", ".join(f"{k}=?" for k in updates)
                vals = list(updates.values()) + [procedural_id]
                cur.execute(f'UPDATE "{graph_id}_procedural_meta" SET {sets} WHERE procedural_id=?', vals)
            if embedding is not None:
                cur.execute(
                    f'DELETE FROM "{graph_id}_procedural_vec" WHERE procedural_id=?',
                    (procedural_id,),
                )
                cur.execute(
                    f'INSERT INTO "{graph_id}_procedural_vec"(procedural_id,embedding) VALUES (?,?)',
                    (procedural_id, _serialize(embedding)),
                )
            self._conn.commit()

    def query_procedural(self, graph_id: str, query_embedding, n_results: int = 10) -> Dict:
        self._ensure_graph_tables(graph_id)
        cur = self._conn.cursor()
        emb_bytes = _serialize(query_embedding)
        cur.execute(
            f'SELECT procedural_id, distance FROM "{graph_id}_procedural_vec" '
            "WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (emb_bytes, max(1, min(n_results, 100))),
        )
        ids = [r[0] for r in cur.fetchall()]
        docs, metas = [], []
        for pid in ids:
            cur.execute(f'SELECT * FROM "{graph_id}_procedural_meta" WHERE procedural_id=?', (pid,))
            row = cur.fetchone()
            if row:
                d = dict(row)
                d["episodic_ids"] = json.loads(d.get("episodic_ids", "[]"))
                d["provenance"] = json.loads(d.get("provenance", "{}"))
                docs.append(d.pop("text", ""))
                metas.append(d)
        return {"documents": docs, "metadatas": metas}

    def get_all_procedural(self, graph_id: str) -> Dict:
        self._ensure_graph_tables(graph_id)
        cur = self._conn.cursor()
        cur.execute(f'SELECT * FROM "{graph_id}_procedural_meta" ORDER BY procedural_id')
        rows = cur.fetchall()
        docs, metas = [], []
        for r in rows:
            d = dict(r)
            d["episodic_ids"] = json.loads(d.get("episodic_ids", "[]"))
            d["provenance"] = json.loads(d.get("provenance", "{}"))
            docs.append(d.pop("text", ""))
            metas.append(d)
        return {"documents": docs, "metadatas": metas}

    # ------------------------------------------------------------------ #
    # Recall audit
    # ------------------------------------------------------------------ #

    def add_recall(self, graph_id: str, **kwargs) -> int:
        self._ensure_graph_tables(graph_id)
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                f'INSERT INTO "{graph_id}_recall_audit"('
                "endpoint,ts,graph_time,session_id,observation,goal,subgoal,state,"
                "task_type,mode,next_subgoal,query_tags,selected_semantic_ids,"
                "selected_procedural_ids,n_messages) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    kwargs.get("endpoint", ""),
                    kwargs.get("ts", ""),
                    kwargs.get("graph_time", 0),
                    kwargs.get("session_id"),
                    kwargs.get("observation", ""),
                    kwargs.get("goal", ""),
                    kwargs.get("subgoal", ""),
                    kwargs.get("state", ""),
                    kwargs.get("task_type", ""),
                    kwargs.get("mode", ""),
                    kwargs.get("next_subgoal", ""),
                    json.dumps(kwargs.get("query_tags", [])),
                    json.dumps(kwargs.get("selected_semantic_ids", [])),
                    json.dumps(kwargs.get("selected_procedural_ids", [])),
                    kwargs.get("n_messages", 0),
                ),
            )
            self._conn.commit()
            return cur.lastrowid or 0

    def list_recalls(self, graph_id: str, session_id: Optional[str] = None, limit: int = 50) -> List[Dict]:
        self._ensure_graph_tables(graph_id)
        cur = self._conn.cursor()
        if session_id:
            cur.execute(
                f'SELECT * FROM "{graph_id}_recall_audit" WHERE session_id=? ORDER BY recall_id DESC LIMIT ?',
                (session_id, limit),
            )
        else:
            cur.execute(f'SELECT * FROM "{graph_id}_recall_audit" ORDER BY recall_id DESC LIMIT ?', (limit,))
        rows = cur.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["query_tags"] = json.loads(d.get("query_tags", "[]"))
            d["selected_semantic_ids"] = json.loads(d.get("selected_semantic_ids", "[]"))
            d["selected_procedural_ids"] = json.loads(d.get("selected_procedural_ids", "[]"))
            result.append(d)
        return result

    def list_sessions(self, graph_id: str) -> List[str]:
        self._ensure_graph_tables(graph_id)
        cur = self._conn.cursor()
        cur.execute(
            f'SELECT DISTINCT session_id FROM "{graph_id}_episodic" WHERE session_id IS NOT NULL '
            "UNION "
            f'SELECT DISTINCT session_id FROM "{graph_id}_recall_audit" WHERE session_id IS NOT NULL'
        )
        return [row[0] for row in cur.fetchall() if row[0]]

    def list_recall_sessions(self, graph_id: str) -> List[str]:
        self._ensure_graph_tables(graph_id)
        cur = self._conn.cursor()
        cur.execute(f'SELECT DISTINCT session_id FROM "{graph_id}_recall_audit" WHERE session_id IS NOT NULL')
        return [row[0] for row in cur.fetchall() if row[0]]
