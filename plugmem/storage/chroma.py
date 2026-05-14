"""ChromaDB storage wrapper for PlugMem memory graphs.

Replaces all file-based save_*/update_* functions with ChromaDB operations.
Each memory graph gets 5 collections: semantic, procedural, tag, subgoal, episodic.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import chromadb
import numpy as np

logger = logging.getLogger(__name__)

NODE_TYPES = ("semantic", "procedural", "tag", "subgoal", "episodic")


def _collection_name(graph_id: str, node_type: str) -> str:
    return f"{graph_id}_{node_type}"


def _to_list(v: Any) -> Optional[List[float]]:
    """Convert numpy arrays or lists to plain float lists for ChromaDB."""
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        return v.astype(np.float32).tolist()
    if isinstance(v, list):
        return v
    return list(v)


def _serialize_list(v: Any) -> str:
    """Serialize a Python list to JSON string for ChromaDB metadata."""
    if v is None:
        return "[]"
    return json.dumps(v)


def _deserialize_list(s: Any) -> list:
    """Deserialize a JSON string back to a Python list.

    Accepts both raw JSON strings (ChromaStorage returns these) and
    already-deserialized Python lists (SqliteVecStorage returns these),
    so callers work with either backend.
    """
    if isinstance(s, list):
        return s
    if not s:
        return []
    return json.loads(s)


def _serialize_metadata(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize list-valued fields for ChromaDB metadata storage."""
    return {
        k: (_serialize_list(v) if isinstance(v, list) else v)
        for k, v in updates.items()
    }


def _combine_where_clauses(clauses: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


class ChromaStorage:
    """Manages ChromaDB collections for PlugMem memory graphs."""

    def __init__(
        self,
        client: chromadb.ClientAPI,
        embedding_function=None,
        embedding_client=None,
    ):
        self._client = client
        self._col_cache: Dict[str, Any] = {}
        self._recall_col_cache: Dict[str, Any] = {}
        # Monotonic recall_id per graph; seeded from col.count() on first use.
        # NOTE: single-process only. Multi-worker uvicorn deployments will
        # collide on recall_id because each worker holds its own counter and
        # only refreshes from col.count() once. PlugMem ships as a single
        # daemon (see plugmem.cli) so this is acceptable; revisit if the
        # service ever scales horizontally.
        self._recall_counters: Dict[str, int] = {}
        # Build embedding function: prefer explicit, then wrap client, else None
        if embedding_function is not None:
            self._embedding_fn = embedding_function
        elif embedding_client is not None:
            from plugmem.clients.embedding import PlugMemEmbeddingFunction
            self._embedding_fn = PlugMemEmbeddingFunction(embedding_client)
        else:
            self._embedding_fn = None

    # ------------------------------------------------------------------ #
    # Graph lifecycle
    # ------------------------------------------------------------------ #

    def create_graph(self, graph_id: str) -> None:
        """Create 5 collections for a new memory graph."""
        for node_type in NODE_TYPES:
            name = _collection_name(graph_id, node_type)
            self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=self._embedding_fn,
            )
        logger.info("Created graph %s with 5 collections", graph_id)

    def delete_graph(self, graph_id: str) -> None:
        """Delete all collections for a memory graph (incl. recall audit)."""
        for node_type in NODE_TYPES:
            try:
                self._client.delete_collection(_collection_name(graph_id, node_type))
            except Exception:
                logger.debug("delete_graph: no %s collection for %s", node_type, graph_id)
            # Evict cached collection handle for the deleted collection.
            self._col_cache.pop(_collection_name(graph_id, node_type), None)
        try:
            self._client.delete_collection(f"{graph_id}_recall_audit")
        except Exception:
            logger.debug("delete_graph: no recall_audit collection for %s", graph_id)
        self._recall_col_cache.pop(graph_id, None)
        self._recall_counters.pop(graph_id, None)

    def list_graphs(self) -> List[str]:
        """List all graph IDs by inspecting collection names."""
        collections = self._client.list_collections()
        graph_ids: set[str] = set()
        for col in collections:
            # col may be a Collection object or a string depending on chromadb version.
            # In chromadb >= 0.6 list_collections returns names (strings); accessing
            # `.name` on the proxy raises NotImplementedError despite hasattr being True.
            if isinstance(col, str):
                col_name = col
            else:
                try:
                    col_name = col.name
                except (AttributeError, NotImplementedError):
                    col_name = str(col)
            for nt in NODE_TYPES:
                suffix = f"_{nt}"
                if col_name.endswith(suffix):
                    graph_ids.add(col_name[: -len(suffix)])
                    break
        return sorted(graph_ids)

    def graph_exists(self, graph_id: str) -> bool:
        """Check if a graph exists."""
        try:
            self._client.get_collection(_collection_name(graph_id, "semantic"))
            return True
        except Exception:
            return False

    def get_graph_stats(self, graph_id: str) -> Dict[str, int]:
        """Return node counts per type."""
        return {nt: self._col(graph_id, nt).count() for nt in NODE_TYPES}

    # ------------------------------------------------------------------ #
    # Collection accessors
    # ------------------------------------------------------------------ #

    def _col(self, graph_id: str, node_type: str):
        key = _collection_name(graph_id, node_type)
        if key not in self._col_cache:
            self._col_cache[key] = self._client.get_collection(
                key, embedding_function=self._embedding_fn,
            )
        return self._col_cache[key]

    def _batch_add(
        self,
        graph_id: str,
        node_type: str,
        ids: List[str],
        docs: List[str],
        metas: List[Dict[str, Any]],
        embs: List[Optional[List[float]]],
        any_emb: bool,
    ) -> None:
        """Shared batch-add path.

        Chroma requires embedding presence to be consistent within one batched
        ``add`` call. Batch the common all-embedded / no-embedded cases and
        only fall back to per-row inserts for mixed batches.
        """
        if not ids:
            return
        col = self._col(graph_id, node_type)
        if any_emb and all(e is not None for e in embs):
            col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
            return
        if any_emb:
            for i, e in enumerate(embs):
                row_kwargs: Dict[str, Any] = {
                    "ids": [ids[i]],
                    "documents": [docs[i]],
                    "metadatas": [metas[i]],
                }
                if e is not None:
                    row_kwargs["embeddings"] = [e]
                col.add(**row_kwargs)
            return
        col.add(ids=ids, documents=docs, metadatas=metas)

    # ------------------------------------------------------------------ #
    # Episodic nodes
    # ------------------------------------------------------------------ #

    def add_episodic(
        self,
        graph_id: str,
        episodic_id: int,
        observation: str = "",
        action: str = "",
        time: Any = "",
        session_id: Optional[str] = None,
        subgoal: str = "",
        state: str = "",
        reward: str = "",
        embedding: Optional[List[float]] = None,
    ) -> None:
        doc = f"{observation}\n{action}" if observation or action else ""
        metadata: Dict[str, Any] = {
            "episodic_id": episodic_id,
            "observation": observation,
            "action": action,
            "time": str(time),
            "subgoal": subgoal,
            "state": state,
            "reward": reward,
        }
        if session_id is not None:
            metadata["session_id"] = session_id
        col = self._col(graph_id, "episodic")
        kwargs: Dict[str, Any] = {
            "ids": [str(episodic_id)],
            "documents": [doc],
            "metadatas": [metadata],
        }
        if embedding is not None:
            kwargs["embeddings"] = [_to_list(embedding)]
        col.add(**kwargs)

    def add_episodic_batch(
        self,
        graph_id: str,
        steps: List[Dict[str, Any]],
    ) -> None:
        """Batch-insert episodic nodes.

        Each entry in ``steps`` must carry the same keys accepted by
        ``add_episodic`` (``episodic_id``, optional ``embedding``, etc.).
        Saves N-1 ChromaDB round-trips relative to per-node insert.
        """
        if not steps:
            return
        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict[str, Any]] = []
        embs: List[Optional[List[float]]] = []
        any_emb = False
        for s in steps:
            obs = s.get("observation", "")
            act = s.get("action", "")
            ids.append(str(s["episodic_id"]))
            docs.append(f"{obs}\n{act}" if obs or act else "")
            meta: Dict[str, Any] = {
                "episodic_id": s["episodic_id"],
                "observation": obs,
                "action": act,
                "time": str(s.get("time", "")),
                "subgoal": s.get("subgoal", ""),
                "state": s.get("state", ""),
                "reward": s.get("reward", ""),
            }
            sid = s.get("session_id")
            if sid is not None:
                meta["session_id"] = sid
            metas.append(meta)
            emb = s.get("embedding")
            if emb is not None:
                any_emb = True
            embs.append(_to_list(emb) if emb is not None else None)
        self._batch_add(graph_id, "episodic", ids, docs, metas, embs, any_emb)

    def get_episodic(self, graph_id: str, episodic_id: int) -> Optional[Dict]:
        col = self._col(graph_id, "episodic")
        result = col.get(ids=[str(episodic_id)], include=["documents", "metadatas"])
        if not result["ids"]:
            return None
        return result["metadatas"][0]

    def get_all_episodic(self, graph_id: str) -> Dict:
        col = self._col(graph_id, "episodic")
        return col.get(include=["documents", "metadatas"])

    # ------------------------------------------------------------------ #
    # Semantic nodes
    # ------------------------------------------------------------------ #

    def add_semantic(
        self,
        graph_id: str,
        semantic_id: int,
        text: str,
        embedding: Optional[List[float]] = None,
        tags: Optional[List[str]] = None,
        tag_ids: Optional[List[int]] = None,
        time: int = 0,
        is_active: bool = True,
        episodic_ids: Optional[List[int]] = None,
        bro_semantic_ids: Optional[List[int]] = None,
        son_semantic_ids: Optional[List[int]] = None,
        session_id: Optional[str] = None,
        credibility: int = 10,
        date: str = "",
        source: Optional[str] = None,
        confidence: float = 0.5,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> None:
        metadata: Dict[str, Any] = {
            "semantic_id": semantic_id,
            "tags": _serialize_list(tags or []),
            "tag_ids": _serialize_list(tag_ids or []),
            "time": time,
            "is_active": is_active,
            "episodic_ids": _serialize_list(episodic_ids or []),
            "bro_semantic_ids": _serialize_list(bro_semantic_ids or []),
            "son_semantic_ids": _serialize_list(son_semantic_ids or []),
            "credibility": credibility,
            "date": date,
            "confidence": float(confidence),
        }
        if session_id is not None:
            metadata["session_id"] = session_id
        if source is not None:
            metadata["source"] = source
        if provenance:
            for k, v in provenance.items():
                if v is not None:
                    metadata[f"provenance_{k}"] = str(v)

        col = self._col(graph_id, "semantic")
        kwargs: Dict[str, Any] = {
            "ids": [str(semantic_id)],
            "documents": [text],
            "metadatas": [metadata],
        }
        if embedding is not None:
            kwargs["embeddings"] = [_to_list(embedding)]
        col.add(**kwargs)

    def add_semantic_batch(
        self,
        graph_id: str,
        rows: List[Dict[str, Any]],
    ) -> None:
        """Batch-insert semantic nodes. Each row mirrors ``add_semantic`` kwargs."""
        if not rows:
            return
        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict[str, Any]] = []
        embs: List[Optional[List[float]]] = []
        any_emb = False
        for r in rows:
            sid = r["semantic_id"]
            ids.append(str(sid))
            docs.append(r["text"])
            meta: Dict[str, Any] = {
                "semantic_id": sid,
                "tags": _serialize_list(r.get("tags") or []),
                "tag_ids": _serialize_list(r.get("tag_ids") or []),
                "time": r.get("time", 0),
                "is_active": r.get("is_active", True),
                "episodic_ids": _serialize_list(r.get("episodic_ids") or []),
                "bro_semantic_ids": _serialize_list(r.get("bro_semantic_ids") or []),
                "son_semantic_ids": _serialize_list(r.get("son_semantic_ids") or []),
                "credibility": r.get("credibility", 10),
                "date": r.get("date", ""),
                "confidence": float(r.get("confidence", 0.5)),
            }
            if r.get("session_id") is not None:
                meta["session_id"] = r["session_id"]
            if r.get("source") is not None:
                meta["source"] = r["source"]
            provenance = r.get("provenance")
            if provenance:
                for k, v in provenance.items():
                    if v is not None:
                        meta[f"provenance_{k}"] = str(v)
            metas.append(meta)
            emb = r.get("embedding")
            if emb is not None:
                any_emb = True
            embs.append(_to_list(emb) if emb is not None else None)
        self._batch_add(graph_id, "semantic", ids, docs, metas, embs, any_emb)

    def update_semantic(
        self,
        graph_id: str,
        semantic_id: int,
        text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> None:
        col = self._col(graph_id, "semantic")
        kwargs: Dict[str, Any] = {"ids": [str(semantic_id)]}
        if text is not None:
            kwargs["documents"] = [text]
        if embedding is not None:
            kwargs["embeddings"] = [_to_list(embedding)]
        if metadata_updates:
            kwargs["metadatas"] = [_serialize_metadata(metadata_updates)]
        col.update(**kwargs)

    def query_semantic(
        self,
        graph_id: str,
        query_embedding: List[float],
        n_results: int = 20,
        where: Optional[Dict] = None,
    ) -> Dict:
        col = self._col(graph_id, "semantic")
        kwargs: Dict[str, Any] = {
            "query_embeddings": [_to_list(query_embedding)],
            "n_results": min(n_results, max(col.count(), 1)),
            "include": ["documents", "metadatas", "distances", "embeddings"],
        }
        if where:
            kwargs["where"] = where
        return col.query(**kwargs)

    def get_all_semantic(self, graph_id: str) -> Dict:
        col = self._col(graph_id, "semantic")
        return col.get(include=["documents", "metadatas", "embeddings"])

    # ------------------------------------------------------------------ #
    # Tag nodes
    # ------------------------------------------------------------------ #

    def add_tag(
        self,
        graph_id: str,
        tag_id: int,
        tag: str,
        embedding: Optional[List[float]] = None,
        semantic_ids: Optional[List[int]] = None,
        time: int = 0,
        importance: int = 1,
    ) -> None:
        metadata: Dict[str, Any] = {
            "tag_id": tag_id,
            "semantic_ids": _serialize_list(semantic_ids or []),
            "time": time,
            "importance": importance,
        }
        col = self._col(graph_id, "tag")
        kwargs: Dict[str, Any] = {
            "ids": [str(tag_id)],
            "documents": [tag],
            "metadatas": [metadata],
        }
        if embedding is not None:
            kwargs["embeddings"] = [_to_list(embedding)]
        col.add(**kwargs)

    def add_tag_batch(
        self,
        graph_id: str,
        rows: List[Dict[str, Any]],
    ) -> None:
        """Batch-insert tag nodes. Each row mirrors ``add_tag`` kwargs."""
        if not rows:
            return
        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict[str, Any]] = []
        embs: List[Optional[List[float]]] = []
        any_emb = False
        for r in rows:
            tid = r["tag_id"]
            ids.append(str(tid))
            docs.append(r["tag"])
            metas.append({
                "tag_id": tid,
                "semantic_ids": _serialize_list(r.get("semantic_ids") or []),
                "time": r.get("time", 0),
                "importance": r.get("importance", 1),
            })
            emb = r.get("embedding")
            if emb is not None:
                any_emb = True
            embs.append(_to_list(emb) if emb is not None else None)
        self._batch_add(graph_id, "tag", ids, docs, metas, embs, any_emb)

    def update_tag(
        self,
        graph_id: str,
        tag_id: int,
        tag: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> None:
        col = self._col(graph_id, "tag")
        kwargs: Dict[str, Any] = {"ids": [str(tag_id)]}
        if tag is not None:
            kwargs["documents"] = [tag]
        if embedding is not None:
            kwargs["embeddings"] = [_to_list(embedding)]
        if metadata_updates:
            kwargs["metadatas"] = [_serialize_metadata(metadata_updates)]
        col.update(**kwargs)

    def update_tag_metadata_batch(
        self,
        graph_id: str,
        metadata_updates_by_id: Dict[int, Dict[str, Any]],
    ) -> None:
        """Batch-update metadata for multiple tag rows."""
        if not metadata_updates_by_id:
            return
        ids: List[str] = []
        metas: List[Dict[str, Any]] = []
        for tag_id, updates in metadata_updates_by_id.items():
            ids.append(str(tag_id))
            metas.append(_serialize_metadata(updates))
        self._col(graph_id, "tag").update(ids=ids, metadatas=metas)

    def query_tag(
        self,
        graph_id: str,
        query_embedding: List[float],
        n_results: int = 10,
    ) -> Dict:
        col = self._col(graph_id, "tag")
        return col.query(
            query_embeddings=[_to_list(query_embedding)],
            n_results=min(n_results, max(col.count(), 1)),
            include=["documents", "metadatas", "distances", "embeddings"],
        )

    def get_all_tags(self, graph_id: str) -> Dict:
        col = self._col(graph_id, "tag")
        return col.get(include=["documents", "metadatas", "embeddings"])

    # ------------------------------------------------------------------ #
    # Subgoal nodes
    # ------------------------------------------------------------------ #

    def add_subgoal(
        self,
        graph_id: str,
        subgoal_id: int,
        subgoal: str,
        embedding: Optional[List[float]] = None,
        procedural_ids: Optional[List[int]] = None,
        time: int = 0,
    ) -> None:
        metadata: Dict[str, Any] = {
            "subgoal_id": subgoal_id,
            "procedural_ids": _serialize_list(procedural_ids or []),
            "time": time,
        }
        col = self._col(graph_id, "subgoal")
        kwargs: Dict[str, Any] = {
            "ids": [str(subgoal_id)],
            "documents": [subgoal],
            "metadatas": [metadata],
        }
        if embedding is not None:
            kwargs["embeddings"] = [_to_list(embedding)]
        col.add(**kwargs)

    def add_subgoal_batch(
        self,
        graph_id: str,
        rows: List[Dict[str, Any]],
    ) -> None:
        """Batch-insert subgoal nodes. Each row mirrors ``add_subgoal`` kwargs."""
        if not rows:
            return
        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict[str, Any]] = []
        embs: List[Optional[List[float]]] = []
        any_emb = False
        for r in rows:
            sid = r["subgoal_id"]
            ids.append(str(sid))
            docs.append(r["subgoal"])
            metas.append({
                "subgoal_id": sid,
                "procedural_ids": _serialize_list(r.get("procedural_ids") or []),
                "time": r.get("time", 0),
            })
            emb = r.get("embedding")
            if emb is not None:
                any_emb = True
            embs.append(_to_list(emb) if emb is not None else None)
        self._batch_add(graph_id, "subgoal", ids, docs, metas, embs, any_emb)

    def update_subgoal(
        self,
        graph_id: str,
        subgoal_id: int,
        subgoal: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> None:
        col = self._col(graph_id, "subgoal")
        kwargs: Dict[str, Any] = {"ids": [str(subgoal_id)]}
        if subgoal is not None:
            kwargs["documents"] = [subgoal]
        if embedding is not None:
            kwargs["embeddings"] = [_to_list(embedding)]
        if metadata_updates:
            kwargs["metadatas"] = [_serialize_metadata(metadata_updates)]
        col.update(**kwargs)

    def query_subgoal(
        self,
        graph_id: str,
        query_embedding: List[float],
        n_results: int = 5,
    ) -> Dict:
        col = self._col(graph_id, "subgoal")
        return col.query(
            query_embeddings=[_to_list(query_embedding)],
            n_results=min(n_results, max(col.count(), 1)),
            include=["documents", "metadatas", "distances", "embeddings"],
        )

    def get_all_subgoals(self, graph_id: str) -> Dict:
        col = self._col(graph_id, "subgoal")
        return col.get(include=["documents", "metadatas", "embeddings"])

    # ------------------------------------------------------------------ #
    # Procedural nodes
    # ------------------------------------------------------------------ #

    def add_procedural(
        self,
        graph_id: str,
        procedural_id: int,
        text: str,
        embedding: Optional[List[float]] = None,
        subgoal: str = "",
        subgoal_id: Optional[int] = None,
        episodic_ids: Optional[List[int]] = None,
        time: int = 0,
        return_value: float = 0.0,
        source: Optional[str] = None,
        confidence: float = 0.5,
        session_id: Optional[str] = None,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> None:
        metadata: Dict[str, Any] = {
            "procedural_id": procedural_id,
            "subgoal": subgoal,
            "time": time,
            "return": return_value,
            "episodic_ids": _serialize_list(episodic_ids or []),
            "confidence": float(confidence),
        }
        if subgoal_id is not None:
            metadata["subgoal_id"] = subgoal_id
        if source is not None:
            metadata["source"] = source
        if session_id is not None:
            metadata["session_id"] = session_id
        if provenance:
            for k, v in provenance.items():
                if v is not None:
                    metadata[f"provenance_{k}"] = str(v)
        col = self._col(graph_id, "procedural")
        kwargs: Dict[str, Any] = {
            "ids": [str(procedural_id)],
            "documents": [text],
            "metadatas": [metadata],
        }
        if embedding is not None:
            kwargs["embeddings"] = [_to_list(embedding)]
        col.add(**kwargs)

    def add_procedural_batch(
        self,
        graph_id: str,
        rows: List[Dict[str, Any]],
    ) -> None:
        """Batch-insert procedural nodes. Each row mirrors ``add_procedural`` kwargs."""
        if not rows:
            return
        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict[str, Any]] = []
        embs: List[Optional[List[float]]] = []
        any_emb = False
        for r in rows:
            pid = r["procedural_id"]
            ids.append(str(pid))
            docs.append(r["text"])
            meta: Dict[str, Any] = {
                "procedural_id": pid,
                "subgoal": r.get("subgoal", ""),
                "time": r.get("time", 0),
                "return": r.get("return_value", 0.0),
                "episodic_ids": _serialize_list(r.get("episodic_ids") or []),
                "confidence": float(r.get("confidence", 0.5)),
            }
            if r.get("subgoal_id") is not None:
                meta["subgoal_id"] = r["subgoal_id"]
            if r.get("source") is not None:
                meta["source"] = r["source"]
            if r.get("session_id") is not None:
                meta["session_id"] = r["session_id"]
            provenance = r.get("provenance")
            if provenance:
                for k, v in provenance.items():
                    if v is not None:
                        meta[f"provenance_{k}"] = str(v)
            metas.append(meta)
            emb = r.get("embedding")
            if emb is not None:
                any_emb = True
            embs.append(_to_list(emb) if emb is not None else None)
        self._batch_add(graph_id, "procedural", ids, docs, metas, embs, any_emb)

    def update_procedural(
        self,
        graph_id: str,
        procedural_id: int,
        text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> None:
        col = self._col(graph_id, "procedural")
        kwargs: Dict[str, Any] = {"ids": [str(procedural_id)]}
        if text is not None:
            kwargs["documents"] = [text]
        if embedding is not None:
            kwargs["embeddings"] = [_to_list(embedding)]
        if metadata_updates:
            kwargs["metadatas"] = [_serialize_metadata(metadata_updates)]
        col.update(**kwargs)

    def query_procedural(
        self,
        graph_id: str,
        query_embedding: List[float],
        n_results: int = 10,
    ) -> Dict:
        col = self._col(graph_id, "procedural")
        return col.query(
            query_embeddings=[_to_list(query_embedding)],
            n_results=min(n_results, max(col.count(), 1)),
            include=["documents", "metadatas", "distances", "embeddings"],
        )

    def get_all_procedural(self, graph_id: str) -> Dict:
        col = self._col(graph_id, "procedural")
        return col.get(include=["documents", "metadatas", "embeddings"])

    # ------------------------------------------------------------------ #
    # Text search (Inspector UI)
    # ------------------------------------------------------------------ #

    def search_by_text(self, graph_id: str, node_type: str, query: str, limit: int = 50, only_active: bool = False) -> Dict:
        """Substring search on node text fields via ChromaDB ``$contains``.

        Inspector UI only — not for agent use.
        """
        col = self._col(graph_id, node_type)
        kwargs: Dict[str, Any] = {
            "limit": limit,
            "include": ["documents", "metadatas"],
        }
        if query:
            kwargs["where_document"] = {"$contains": query}
        if only_active and node_type == "semantic":
            kwargs["where"] = {"is_active": True}
        return col.get(**kwargs)

    def browse_nodes(
        self,
        graph_id: str,
        node_type: str,
        limit: int = 50,
        offset: int = 0,
        source_in: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        provenance_filters: Optional[Dict[str, List[str]]] = None,
    ) -> Dict:
        """Fetch semantic/procedural nodes via storage-level metadata filters."""
        col = self._col(graph_id, node_type)
        where_clauses: List[Dict[str, Any]] = []
        if source_in:
            where_clauses.append({"source": {"$in": list(source_in)}})
        if min_confidence is not None:
            where_clauses.append({"confidence": {"$gte": float(min_confidence)}})
        if provenance_filters:
            for key, values in provenance_filters.items():
                if values:
                    where_clauses.append({f"provenance_{key}": {"$in": list(values)}})

        where = _combine_where_clauses(where_clauses)
        kwargs: Dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "include": ["documents", "metadatas"],
        }
        if where is not None:
            kwargs["where"] = where
        result = col.get(**kwargs)
        if where is None:
            result["count"] = col.count()
        else:
            result["count"] = len((col.get(where=where) or {}).get("ids", []))
        return result

    # ------------------------------------------------------------------ #
    # Recall audit log
    # ------------------------------------------------------------------ #
    #
    # A separate collection per graph (`{graph_id}_recall_audit`) records
    # every /retrieve, /reason, and /recall_trace call. Lazily created on
    # first append; not enumerated by `list_graphs` because the suffix is
    # outside NODE_TYPES.

    def _recall_col(self, graph_id: str):
        cached = self._recall_col_cache.get(graph_id)
        if cached is not None:
            return cached
        col = self._client.get_or_create_collection(
            name=f"{graph_id}_recall_audit",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embedding_fn,
        )
        self._recall_col_cache[graph_id] = col
        return col

    def add_recall(
        self,
        graph_id: str,
        *,
        endpoint: str,
        observation: str,
        ts: str,
        graph_time: int = 0,
        session_id: Optional[str] = None,
        goal: str = "",
        subgoal: str = "",
        state: str = "",
        task_type: str = "",
        mode: str = "",
        next_subgoal: str = "",
        query_tags: Optional[List[str]] = None,
        selected_semantic_ids: Optional[List[int]] = None,
        selected_procedural_ids: Optional[List[int]] = None,
        n_messages: int = 0,
        embedding: Optional[List[float]] = None,
    ) -> int:
        """Append one recall to the audit log. Returns the assigned recall_id."""
        col = self._recall_col(graph_id)
        if graph_id not in self._recall_counters:
            self._recall_counters[graph_id] = col.count()
        recall_id = self._recall_counters[graph_id]
        self._recall_counters[graph_id] = recall_id + 1
        metadata: Dict[str, Any] = {
            "recall_id": recall_id,
            "endpoint": endpoint,
            "ts": ts,
            "graph_time": graph_time,
            "observation": observation,
            "goal": goal,
            "subgoal": subgoal,
            "state": state,
            "task_type": task_type,
            "mode": mode,
            "next_subgoal": next_subgoal,
            "query_tags": _serialize_list(query_tags or []),
            "selected_semantic_ids": _serialize_list(selected_semantic_ids or []),
            "selected_procedural_ids": _serialize_list(selected_procedural_ids or []),
            "n_messages": n_messages,
        }
        if session_id is not None:
            metadata["session_id"] = session_id
        kwargs: Dict[str, Any] = {
            "ids": [str(recall_id)],
            "documents": [observation or ""],
            "metadatas": [metadata],
        }
        if embedding is not None:
            kwargs["embeddings"] = [_to_list(embedding)]
        col.add(**kwargs)
        return recall_id

    def list_recalls(
        self,
        graph_id: str,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return audit rows newest-first, optionally filtered by session_id."""
        col = self._recall_col(graph_id)
        kwargs: Dict[str, Any] = {"include": ["metadatas"]}
        if session_id is not None:
            kwargs["where"] = {"session_id": session_id}
        data = col.get(**kwargs)
        rows = list(data.get("metadatas") or [])
        for row in rows:
            row["query_tags"] = _deserialize_list(row.get("query_tags", "[]"))
            row["selected_semantic_ids"] = _deserialize_list(row.get("selected_semantic_ids", "[]"))
            row["selected_procedural_ids"] = _deserialize_list(row.get("selected_procedural_ids", "[]"))
        rows.sort(key=lambda r: r.get("recall_id", 0), reverse=True)
        return rows[: max(0, limit)]

    def list_sessions(self, graph_id: str) -> List[str]:
        """Distinct session_ids that appear anywhere in the graph (nodes or recalls)."""
        seen: set = set()
        for node_type in ("episodic", "semantic", "procedural"):
            col = self._col(graph_id, node_type)
            data = col.get(include=["metadatas"])
            for meta in data.get("metadatas", []) or []:
                sid = meta.get("session_id")
                if sid:
                    seen.add(sid)
        try:
            audit_col = self._recall_col(graph_id)
            data = audit_col.get(include=["metadatas"])
            for meta in data.get("metadatas", []) or []:
                sid = meta.get("session_id")
                if sid:
                    seen.add(sid)
        except Exception as exc:
            logger.warning("list_sessions: error reading recall audit for %s: %s", graph_id, exc)
        return sorted(seen)

    def list_recall_sessions(self, graph_id: str) -> List[str]:
        """Distinct session_ids that appear in the recall audit log only."""
        try:
            data = self._recall_col(graph_id).get(include=["metadatas"])
        except Exception as exc:
            logger.warning("list_recall_sessions: error reading recall audit for %s: %s", graph_id, exc)
            return []

        seen: set[str] = set()
        for meta in data.get("metadatas", []) or []:
            sid = meta.get("session_id")
            if sid:
                seen.add(sid)
        return sorted(seen)

    # ------------------------------------------------------------------ #
    # Session-scoped queries (Inspector UI)
    # ------------------------------------------------------------------ #

    def get_nodes_by_session(self, graph_id: str, session_id: str) -> Dict[str, List[Dict]]:
        """Fetch episodic, semantic, and procedural nodes for one session.

        Avoids loading the entire graph — Inspector UI only.
        Metadata is returned in raw Chroma format (callers deserialize as needed).
        """
        result: Dict[str, List[Dict]] = {"episodic": [], "semantic": [], "procedural": []}

        col_e = self._col(graph_id, "episodic")
        data_e = col_e.get(where={"session_id": session_id}, include=["metadatas"])
        result["episodic"] = list(data_e.get("metadatas", []) or [])

        col_s = self._col(graph_id, "semantic")
        data_s = col_s.get(where={"session_id": session_id}, include=["documents", "metadatas"])
        docs_s = list(data_s.get("documents", []) or [])
        metas_s = list(data_s.get("metadatas", []) or [])
        for i, d in enumerate(metas_s):
            d["text"] = docs_s[i] if i < len(docs_s) else ""
            d["tags"] = _deserialize_list(d.get("tags", "[]"))
            d["tag_ids"] = _deserialize_list(d.get("tag_ids", "[]"))
            d["episodic_ids"] = _deserialize_list(d.get("episodic_ids", "[]"))
            d["bro_semantic_ids"] = _deserialize_list(d.get("bro_semantic_ids", "[]"))
        result["semantic"] = metas_s

        col_p = self._col(graph_id, "procedural")
        data_p = col_p.get(where={"session_id": session_id}, include=["documents", "metadatas"])
        docs_p = list(data_p.get("documents", []) or [])
        metas_p = list(data_p.get("metadatas", []) or [])
        for i, d in enumerate(metas_p):
            d["text"] = docs_p[i] if i < len(docs_p) else ""
            d["episodic_ids"] = _deserialize_list(d.get("episodic_ids", "[]"))
        result["procedural"] = metas_p

        return result
