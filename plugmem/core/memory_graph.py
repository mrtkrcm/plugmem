"""Unified MemoryGraph backed by ChromaDB.

Merges the three build_from_disk variants and two insert variants into
single unified methods. All persistence goes through ChromaStorage.
"""
from __future__ import annotations

import heapq
import json
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from plugmem.clients.embedding import EmbeddingClient, get_similarity
from plugmem.clients.llm import LLMClient
from plugmem.clients.llm_router import LLMRouter
from plugmem.core.graph_node import (
    EpisodicNode,
    ProceduralNode,
    SemanticNode,
    SubgoalNode,
    TagNode,
)
from plugmem.core.memory import Memory
from plugmem.core.graph_node import PROVENANCE_FIELDS
from plugmem.core.normalize import normalize_memory
from plugmem.core.value_base import ValueBase
from plugmem.core.value_functions import (
    ProceduralEqual,
    ProceduralRelevant,
    SemanticEqual,
    SemanticRelevant,
    SemanticRelevant4Episodic,
    SubgoalEqual,
    SubgoalRelevant,
    TagEqual,
    TagRelevant,
)
from plugmem.inference.retrieving import get_mode, get_new_semantic, get_new_subgoal, get_plan
from plugmem.prompts.reasoning import (
    DefaultEpisodicPrompt,
    DefaultProceduralPrompt,
    DefaultSemanticPrompt,
)
from plugmem.prompts.registry import PromptRegistry
from plugmem.storage import StorageBackend
from plugmem.storage.chroma import _deserialize_list

logger = logging.getLogger(__name__)

_REASONING_MAP = {
    "episodic_memory": ("reasoning_episodic", DefaultEpisodicPrompt),
    "semantic_memory": ("reasoning_semantic", DefaultSemanticPrompt),
    "procedural_memory": ("reasoning_procedural", DefaultProceduralPrompt),
}


def _passes_metadata_filter(
    node,
    min_confidence: Optional[float],
    source_in: Optional[List[str]],
    provenance_filters: Optional[Dict[str, List[str]]] = None,
) -> bool:
    if min_confidence is not None and getattr(node, "confidence", 0.5) < min_confidence:
        return False
    if source_in is not None and getattr(node, "source", None) not in source_in:
        return False
    if provenance_filters:
        node_prov: dict = getattr(node, "provenance", {}) or {}
        for key, values in provenance_filters.items():
            nv = node_prov.get(key)
            if nv is None or nv not in values:
                return False
    return True


class MemoryGraph:
    """Unified memory graph with ChromaDB-backed persistence."""

    def __init__(
        self,
        graph_id: str,
        storage: StorageBackend,
        llm: LLMClient,
        embedder: EmbeddingClient,
        prompts: Optional[PromptRegistry] = None,
        tag_equal: ValueBase = None,
        tag_relevant: ValueBase = None,
        semantic_equal: ValueBase = None,
        semantic_relevant: ValueBase = None,
        subgoal_equal: ValueBase = None,
        subgoal_relevant: ValueBase = None,
        procedural_equal: ValueBase = None,
        procedural_relevant: ValueBase = None,
    ):
        self.graph_id = graph_id
        self.storage = storage
        self.embedder = embedder
        self.prompts = prompts

        # LLM routing — if a plain LLMClient is passed, wrap it so all
        # roles use the same client.  If an LLMRouter is passed, each
        # operation category gets the role-specific client.
        if isinstance(llm, LLMRouter):
            self._router = llm
        else:
            self._router = LLMRouter.from_single_client(llm)
        # Convenience aliases used throughout this file
        self.llm: LLMClient = self._router.for_role("default")
        self.structuring_llm: LLMClient = self._router.structuring
        self.retrieval_llm: LLMClient = self._router.retrieval
        self.reasoning_llm: LLMClient = self._router.reasoning
        self.consolidation_llm: LLMClient = self._router.consolidation

        # Value functions (defaults if not provided)
        self.tag_equal = tag_equal or TagEqual()
        self.tag_relevant = tag_relevant or TagRelevant()
        self.semantic_equal = semantic_equal or SemanticEqual()
        self.semantic_relevant = semantic_relevant or SemanticRelevant()
        self.semantic_relevant4episodic = SemanticRelevant4Episodic()
        self.subgoal_equal = subgoal_equal or SubgoalEqual()
        self.subgoal_relevant = subgoal_relevant or SubgoalRelevant()
        self.procedural_equal = procedural_equal or ProceduralEqual()
        self.procedural_relevant = procedural_relevant or ProceduralRelevant()

        # In-memory node lists (populated by load)
        self.episodic_nodes: List[EpisodicNode] = []
        self.semantic_nodes: List[SemanticNode] = []
        self.tag_nodes: List[TagNode] = []
        self.subgoal_nodes: List[SubgoalNode] = []
        self.procedural_nodes: List[ProceduralNode] = []

        # Time counters
        self.semantic_time = 0
        self.procedural_time = 0

        # Lookup dicts
        self.tag2node: Dict[str, TagNode] = {}
        self.subgoal2node: Dict[str, SubgoalNode] = {}
        self.episodic_id2node: Dict[int, EpisodicNode] = {}
        self.semantic_id2node: Dict[int, SemanticNode] = {}
        self.procedural_id2node: Dict[int, ProceduralNode] = {}
        self.subgoal_id2node: Dict[int, SubgoalNode] = {}
        self.tag_id2node: Dict[int, TagNode] = {}

        # Session tracking (for LongMemEval episodic retrieval)
        self.session_ids: Dict[str, List[EpisodicNode]] = {}
        self.session_semantic_ids: Dict[str, List[SemanticNode]] = {}
        self.session_procedural_ids: Dict[str, List[ProceduralNode]] = {}
        # Memoized stringified sessions, invalidated when a session gains a node.
        self._session_str_cache: Dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # Unified load from ChromaDB
    # ------------------------------------------------------------------ #

    def load(self) -> Dict[str, int]:
        """Load all nodes from ChromaDB collections into memory."""
        self._reset_loaded_state()
        self._load_episodic_nodes()
        sem_data = self._load_semantic_nodes()
        tag_data = self._load_tag_nodes()
        sg_data = self._load_subgoal_nodes()
        proc_data = self._load_procedural_nodes()
        self._rebuild_lookups()
        self._link_nodes(sem_data=sem_data, tag_data=tag_data, sg_data=sg_data, proc_data=proc_data)

        stats = {
            "episodic": len(self.episodic_nodes),
            "semantic": len(self.semantic_nodes),
            "tag": len(self.tag_nodes),
            "subgoal": len(self.subgoal_nodes),
            "procedural": len(self.procedural_nodes),
        }
        logger.info("Graph %s loaded. Stats: %s", self.graph_id, stats)
        return stats

    def _reset_loaded_state(self) -> None:
        """Clear in-memory state before reloading from storage."""
        self.episodic_nodes = []
        self.semantic_nodes = []
        self.tag_nodes = []
        self.subgoal_nodes = []
        self.procedural_nodes = []

        self.semantic_time = 0
        self.procedural_time = 0

        self.tag2node = {}
        self.subgoal2node = {}
        self.episodic_id2node = {}
        self.semantic_id2node = {}
        self.procedural_id2node = {}
        self.subgoal_id2node = {}
        self.tag_id2node = {}

        self.session_ids = {}
        self.session_semantic_ids = {}
        self.session_procedural_ids = {}
        self._session_str_cache = {}

    def _track_session_node(self, node_type: str, node) -> None:
        sid = getattr(node, "session_id", None)
        if sid is None:
            return
        if node_type == "episodic":
            self.session_ids.setdefault(sid, []).append(node)
            self._session_str_cache.pop(sid, None)
        elif node_type == "semantic":
            self.session_semantic_ids.setdefault(sid, []).append(node)
        elif node_type == "procedural":
            self.session_procedural_ids.setdefault(sid, []).append(node)

    def get_session_nodes(self, session_id: str) -> Dict[str, List[Any]]:
        return {
            "episodic": self.session_ids.get(session_id, []),
            "semantic": self.session_semantic_ids.get(session_id, []),
            "procedural": self.session_procedural_ids.get(session_id, []),
        }

    def list_session_ids(self) -> List[str]:
        seen = set(self.session_ids)
        seen.update(self.session_semantic_ids)
        seen.update(self.session_procedural_ids)
        return sorted(seen)

    def _load_episodic_nodes(self) -> None:
        data = self.storage.get_all_episodic(self.graph_id)
        for i, meta in enumerate(data.get("metadatas", [])):
            node = EpisodicNode(
                episodic_id=meta["episodic_id"],
                observation=meta.get("observation", ""),
                action=meta.get("action", ""),
                time=meta.get("time", ""),
                session_id=meta.get("session_id"),
                subgoal=meta.get("subgoal", ""),
                state=meta.get("state", ""),
                reward=meta.get("reward", ""),
            )
            self.episodic_nodes.append(node)
            self._track_session_node("episodic", node)

    @staticmethod
    def _safe_embeddings(data: Dict, fallback_len: int) -> list:
        """Extract embeddings from ChromaDB result, handling None/numpy arrays."""
        embs = data.get("embeddings")
        if embs is None:
            return [None] * fallback_len
        return list(embs)

    def _load_semantic_nodes(self) -> Dict:
        data = self.storage.get_all_semantic(self.graph_id)
        docs = data.get("documents", [])
        metas = data.get("metadatas", [])
        embs = self._safe_embeddings(data, len(docs))
        for i, (doc, meta, emb) in enumerate(zip(docs, metas, embs)):
            _time = meta.get("time", 0)
            if not isinstance(_time, int):
                _time = 0
            provenance = {k: meta.get(f"provenance_{k}") for k in PROVENANCE_FIELDS if meta.get(f"provenance_{k}") is not None}
            node = SemanticNode(
                semantic_id=meta["semantic_id"],
                semantic_memory_str=doc or "",
                embedding=emb,
                time=_time,
                is_active=meta.get("is_active", True),
                session_id=meta.get("session_id"),
                date=meta.get("date", ""),
                credibility=meta.get("credibility", 10),
                source=meta.get("source"),
                confidence=float(meta.get("confidence", 0.5)),
                provenance=provenance,
            )
            node.tags = _deserialize_list(meta.get("tags", "[]"))
            self.semantic_nodes.append(node)
            self._track_session_node("semantic", node)
            self.semantic_time = max(self.semantic_time, _time + 1)
        return data

    def _load_tag_nodes(self) -> Dict:
        data = self.storage.get_all_tags(self.graph_id)
        docs = data.get("documents", [])
        metas = data.get("metadatas", [])
        embs = self._safe_embeddings(data, len(docs))
        for doc, meta, emb in zip(docs, metas, embs):
            node = TagNode(
                tag=doc or "",
                tag_id=meta["tag_id"],
                embedding=emb,
                time=meta.get("time", 0),
                importance=meta.get("importance", 1),
            )
            self.tag_nodes.append(node)
        return data

    def _load_subgoal_nodes(self) -> Dict:
        data = self.storage.get_all_subgoals(self.graph_id)
        docs = data.get("documents", [])
        metas = data.get("metadatas", [])
        embs = self._safe_embeddings(data, len(docs))
        for doc, meta, emb in zip(docs, metas, embs):
            node = SubgoalNode(
                subgoal=doc or "",
                subgoal_id=meta["subgoal_id"],
                embedding=emb,
                time=meta.get("time", 0),
            )
            self.subgoal_nodes.append(node)
        return data

    def _load_procedural_nodes(self) -> Dict:
        data = self.storage.get_all_procedural(self.graph_id)
        docs = data.get("documents", [])
        metas = data.get("metadatas", [])
        embs = self._safe_embeddings(data, len(docs))
        for doc, meta, emb in zip(docs, metas, embs):
            provenance = {k: meta.get(f"provenance_{k}") for k in PROVENANCE_FIELDS if meta.get(f"provenance_{k}") is not None}
            node = ProceduralNode(
                procedural_id=meta["procedural_id"],
                procedural_memory_str=doc or "",
                embedding=emb,
                time=meta.get("time", 0),
                return_value=meta.get("return", 0.0),
                source=meta.get("source"),
                confidence=float(meta.get("confidence", 0.5)),
                session_id=meta.get("session_id"),
                provenance=provenance,
            )
            self.procedural_nodes.append(node)
            self._track_session_node("procedural", node)
            self.procedural_time = max(self.procedural_time, node.time + 1)
        return data

    def _rebuild_lookups(self) -> None:
        self.episodic_id2node = {n.episodic_id: n for n in self.episodic_nodes}
        self.semantic_id2node = {n.semantic_id: n for n in self.semantic_nodes}
        self.tag_id2node = {n.tag_id: n for n in self.tag_nodes}
        self.tag2node = {n.tag: n for n in self.tag_nodes}
        self.subgoal_id2node = {n.subgoal_id: n for n in self.subgoal_nodes}
        self.subgoal2node = {n.subgoal: n for n in self.subgoal_nodes}
        self.procedural_id2node = {n.procedural_id: n for n in self.procedural_nodes}

    def _find_matching_subgoal(
        self,
        subgoal: str,
        subgoal_embedding,
    ) -> Optional[SubgoalNode]:
        """Return the best existing subgoal candidate without mutating state."""
        if not self.subgoal_nodes:
            return None

        best_value = -1.0
        best_node = None
        procedural_time = self.procedural_time
        for sg_node in self.subgoal_nodes:
            relevance = get_similarity(subgoal_embedding, sg_node.embedding)
            value = self.subgoal_equal.evaluate(
                Relevance=relevance,
                Recency=procedural_time - sg_node.time,
                Importance=sg_node.importance,
            )
            if value > best_value:
                best_node = sg_node
                best_value = value

        if best_value < self.subgoal_equal.value_threshold:
            return None
        return best_node

    def _link_nodes(
        self,
        *,
        sem_data: Dict,
        tag_data: Dict,
        sg_data: Dict,
        proc_data: Dict,
    ) -> None:
        """Re-establish in-memory cross-references between nodes using stored IDs."""
        sem_id2node = self.semantic_id2node
        epis_id2node = self.episodic_id2node

        # Link semantic -> episodic
        for meta, sem_node in zip(sem_data.get("metadatas", []), self.semantic_nodes):
            episodic_ids = _deserialize_list(meta.get("episodic_ids", "[]"))
            for eid in episodic_ids:
                epis_node = epis_id2node.get(eid)
                if epis_node is not None:
                    sem_node.episodic_nodes.append(epis_node)
            bro_ids = _deserialize_list(meta.get("bro_semantic_ids", "[]"))
            for bid in bro_ids:
                bro_node = sem_id2node.get(bid)
                if bro_node is not None:
                    sem_node.bro_semantic_nodes.append(bro_node)

        # Link tags <-> semantics from semantic `tag_ids` metadata. Fall back to
        # tag-side `semantic_ids` only for older stored graphs.
        for meta, sem_node in zip(sem_data.get("metadatas", []), self.semantic_nodes):
            tag_ids = _deserialize_list(meta.get("tag_ids", "[]"))
            if tag_ids:
                for tid in tag_ids:
                    tag_node = self.tag_id2node.get(tid)
                    if tag_node is None:
                        continue
                    if sem_node not in tag_node.semantic_nodes:
                        tag_node.semantic_nodes.append(sem_node)
                    if tag_node not in sem_node.tag_nodes:
                        sem_node.tag_nodes.append(tag_node)
            else:
                for tag_str in getattr(sem_node, "tags", []):
                    tag_node = self.tag2node.get(tag_str)
                    if tag_node is None:
                        continue
                    if sem_node not in tag_node.semantic_nodes:
                        tag_node.semantic_nodes.append(sem_node)
                    if tag_node not in sem_node.tag_nodes:
                        sem_node.tag_nodes.append(tag_node)

        for meta, tag_node in zip(tag_data.get("metadatas", []), self.tag_nodes):
            if tag_node.semantic_nodes:
                continue
            semantic_ids = _deserialize_list(meta.get("semantic_ids", "[]"))
            for sid in semantic_ids:
                sem_node = sem_id2node.get(sid)
                if sem_node is None:
                    continue
                tag_node.semantic_nodes.append(sem_node)
                if tag_node not in sem_node.tag_nodes:
                    sem_node.tag_nodes.append(tag_node)

        # Link procedurals -> episodics and subgoals from procedural metadata.
        for meta, proc_node in zip(proc_data.get("metadatas", []), self.procedural_nodes):
            episodic_ids = _deserialize_list(meta.get("episodic_ids", "[]"))
            for eid in episodic_ids:
                epis_node = epis_id2node.get(eid)
                if epis_node is not None:
                    proc_node.episodic_nodes.append(epis_node)
            subgoal_id = meta.get("subgoal_id")
            if subgoal_id is not None:
                sg_node = self.subgoal_id2node.get(subgoal_id)
                if sg_node is not None:
                    if proc_node not in sg_node.procedural_nodes:
                        sg_node.procedural_nodes.append(proc_node)
                    if sg_node not in proc_node.subgoal_nodes:
                        proc_node.subgoal_nodes.append(sg_node)
                    sg_node.is_active = True

        for meta, sg_node in zip(sg_data.get("metadatas", []), self.subgoal_nodes):
            if sg_node.procedural_nodes:
                continue
            proc_ids = _deserialize_list(meta.get("procedural_ids", "[]"))
            for pid in proc_ids:
                proc_node = self.procedural_id2node.get(pid)
                if proc_node is None:
                    continue
                sg_node.procedural_nodes.append(proc_node)
                if sg_node not in proc_node.subgoal_nodes:
                    proc_node.subgoal_nodes.append(sg_node)
            if sg_node.procedural_nodes:
                sg_node.is_active = True

    # ------------------------------------------------------------------ #
    # Dedupe / upsert for coding memories
    # ------------------------------------------------------------------ #

    def _find_matching_semantic(
        self,
        text: str,
        source: Optional[str],
        threshold: float = 0.85,
    ) -> Optional[SemanticNode]:
        """Find an existing semantic node with similar text and same source."""
        if not source or not text:
            return None
        text_emb = self.embedder.embed(text)
        for node in self.semantic_nodes:
            if not node.is_active:
                continue
            if source != getattr(node, "source", None):
                continue
            node_emb = node.embedding
            if node_emb is None:
                continue
            sim = get_similarity(text_emb, node_emb)
            if sim >= threshold:
                return node
        return None

    def _find_matching_procedural(
        self,
        subgoal: str,
        text: str,
        source: Optional[str],
        threshold: float = 0.85,
    ) -> Optional[ProceduralNode]:
        """Find an existing procedural node with similar text and same source."""
        if not source or not text:
            return None
        text_emb = self.embedder.embed(text)
        for node in self.procedural_nodes:
            if source != getattr(node, "source", None):
                continue
            node_emb = node.embedding
            if node_emb is None:
                continue
            sim = get_similarity(text_emb, node.embedding)
            if sim >= threshold:
                return node
        return None

    # ------------------------------------------------------------------ #
    # Unified insert
    # ------------------------------------------------------------------ #

    def _empty_insert_memory(self) -> Memory:
        """Create an empty Memory suitable for programmatic inserts."""
        return Memory.from_structured(embedder=self.embedder, time=self.semantic_time)

    def insert(self, memory: Memory) -> None:
        """Insert structured memory into the graph and persist to ChromaDB."""
        normalize_memory(memory)

        # session_id stamps every node created by this insert. Used by the
        # Sessions view + recall audit to group nodes by run.
        default_sid: Optional[str] = getattr(memory, "session_id", None)

        # 1. Episodic nodes — collect first, then batch-insert once.
        episodic_nodes: List[List[EpisodicNode]] = []
        epis_batch: List[Dict[str, Any]] = []
        for i, trajectory in enumerate(memory.memory["episodic"]):
            episodic_nodes.append([])
            for step in trajectory:
                epis_id = len(self.episodic_nodes)
                observation = step.get("observation", "") if isinstance(step, dict) else str(step)
                action = step.get("action", "") if isinstance(step, dict) else ""
                time_val = step.get("time", self.semantic_time) if isinstance(step, dict) else self.semantic_time

                epis_node = EpisodicNode(
                    episodic_id=epis_id,
                    observation=observation,
                    action=action,
                    time=time_val,
                    session_id=step.get("session_id", default_sid) if isinstance(step, dict) else default_sid,
                    subgoal=step.get("subgoal", "") if isinstance(step, dict) else "",
                    state=step.get("state", "") if isinstance(step, dict) else "",
                    reward=step.get("reward", "") if isinstance(step, dict) else "",
                )
                self.episodic_nodes.append(epis_node)
                self.episodic_id2node[epis_id] = epis_node
                episodic_nodes[i].append(epis_node)
                self._track_session_node("episodic", epis_node)

                epis_batch.append({
                    "episodic_id": epis_id,
                    "observation": epis_node.observation,
                    "action": epis_node.action,
                    "time": epis_node.time,
                    "session_id": epis_node.session_id,
                    "subgoal": epis_node.subgoal,
                    "state": epis_node.state,
                    "reward": epis_node.reward,
                })
        if epis_batch:
            self.storage.add_episodic_batch(self.graph_id, epis_batch)

        all_episodic_ids = [n.episodic_id for group in episodic_nodes for n in group]

        # 2. Semantic nodes — defer new-tag adds; update_tag stays per-row.
        # Tags created within this insert are buffered and batch-persisted once.
        # We no longer update reverse `semantic_ids` metadata on existing tags
        # for every insert; links are rebuilt from semantic `tag_ids` on load.
        curr_sem_nodes: List[SemanticNode] = []
        new_tag_by_id: Dict[int, Dict[str, Any]] = {}
        for sem_item, sem_emb_item in zip(
            memory.memory["semantic"],
            memory.memory_embedding["semantic"],
        ):
            sem_str = sem_item["semantic_memory"]
            if not sem_str:
                continue

            sem_id = len(self.semantic_nodes)
            sem_node = SemanticNode(
                semantic_id=sem_id,
                semantic_memory_str=sem_str,
                embedding=sem_emb_item["semantic_memory"],
                time=self.semantic_time,
                source=sem_item.get("source"),
                confidence=float(sem_item.get("confidence", 0.5)),
                session_id=sem_item.get("session_id", default_sid),
                provenance=sem_item.get("provenance"),
            )

            # Link episodic nodes
            traj_num = sem_item.get("trajectory_num", 0)
            turn_num = sem_item.get("turn_num", 0)
            if traj_num < len(episodic_nodes) and turn_num < len(episodic_nodes[traj_num]):
                sem_node.episodic_nodes.append(episodic_nodes[traj_num][turn_num])
            else:
                sem_node.episodic_nodes = [n for group in episodic_nodes for n in group]

            # Process tags
            for tag_str, tag_emb in zip(sem_item["tags"], sem_emb_item["tags"]):
                tag_node = self.tag2node.get(tag_str)
                if tag_node is None:
                    tag_id = len(self.tag_nodes)
                    tag_node = TagNode(
                        tag=tag_str, tag_id=tag_id,
                        embedding=tag_emb, time=self.semantic_time,
                    )
                    self.tag_nodes.append(tag_node)
                    self.tag2node[tag_str] = tag_node
                    self.tag_id2node[tag_id] = tag_node
                    new_tag_by_id[tag_id] = {
                        "tag_id": tag_id,
                        "tag": tag_str,
                        "embedding": tag_emb,
                        "semantic_ids": [sem_id],
                        "time": self.semantic_time,
                    }
                else:
                    tag_node.semantic_nodes.append(sem_node)

                sem_node.tag_nodes.append(tag_node)
                sem_node.tags.append(tag_node.tag)
                if sem_node not in tag_node.semantic_nodes:
                    tag_node.semantic_nodes.append(sem_node)

            sem_node.tags = list(set(sem_node.tags))
            self.semantic_nodes.append(sem_node)
            self.semantic_id2node[sem_id] = sem_node
            self._track_session_node("semantic", sem_node)
            curr_sem_nodes.append(sem_node)
            self.semantic_time += 1

        if new_tag_by_id:
            self.storage.add_tag_batch(self.graph_id, list(new_tag_by_id.values()))

        curr_semantic_by_id = {n.semantic_id: n for n in curr_sem_nodes}

        # Persist semantic nodes with bro_semantic_ids — collect then batch.
        sem_batch: List[Dict[str, Any]] = []
        for sem_node in curr_sem_nodes:
            bro_ids = [n.semantic_id for n in curr_sem_nodes if n.semantic_id != sem_node.semantic_id]
            sem_node.bro_semantic_nodes = [
                curr_semantic_by_id[bid] for bid in bro_ids if bid in curr_semantic_by_id
            ]

            embedding_list = sem_node.embedding
            if isinstance(embedding_list, np.ndarray):
                embedding_list = embedding_list.tolist()

            provenance = getattr(sem_node, "provenance", None) or {}
            sem_batch.append({
                "semantic_id": sem_node.semantic_id,
                "text": sem_node.semantic_memory_str,
                "embedding": embedding_list,
                "tags": sem_node.tags,
                "tag_ids": [t.tag_id for t in sem_node.tag_nodes],
                "time": sem_node.time,
                "session_id": sem_node.session_id,
                "episodic_ids": [e.episodic_id for e in sem_node.episodic_nodes],
                "bro_semantic_ids": bro_ids,
                "source": sem_node.source,
                "confidence": sem_node.confidence,
                "provenance": provenance,
            })
        if sem_batch:
            self.storage.add_semantic_batch(self.graph_id, sem_batch)

        # 3. Procedural + subgoal nodes — collect, flush at end.
        proc_batch: List[Dict[str, Any]] = []
        new_subgoal_by_id: Dict[int, Dict[str, Any]] = {}
        for proc_item, proc_emb_item in zip(
            memory.memory["procedural"],
            memory.memory_embedding["procedural"],
        ):
            proc_str = proc_item.get("procedural_memory", "")
            if not proc_str:
                continue

            subgoal_str = proc_item["subgoal"]
            subgoal_embedding = proc_emb_item["subgoal"]
            proc_embedding = self.embedder.embed(proc_str)

            # Find the best existing subgoal by similarity, not just exact text,
            # so procedural inserts can actually consolidate related subgoals.
            subgoal_node = self.subgoal2node.get(subgoal_str)
            if subgoal_node is None:
                subgoal_node = self._find_matching_subgoal(subgoal_str, subgoal_embedding)
            is_new_subgoal = subgoal_node is None
            if subgoal_node is not None:
                if subgoal_str == subgoal_node.get_subgoal():
                    merged_str = subgoal_str
                else:
                    merged_str = get_new_subgoal(
                        self.consolidation_llm, subgoal_node.get_subgoal(), subgoal_str,
                        prompts=self.prompts, graph_id=self.graph_id,
                    )
                old_subgoal = subgoal_node.subgoal
                subgoal_node.subgoal = merged_str
                subgoal_node.embedding = self.embedder.embed(merged_str)
                subgoal_node.time = self.procedural_time
                if old_subgoal != merged_str:
                    if self.subgoal2node.get(old_subgoal) is subgoal_node:
                        self.subgoal2node.pop(old_subgoal, None)
                    existing = self.subgoal2node.get(merged_str)
                    if existing is None or existing is subgoal_node:
                        self.subgoal2node[merged_str] = subgoal_node
                    else:
                        logger.warning(
                            "Subgoal alias collision while canonicalizing '%s' -> '%s' in graph %s",
                            old_subgoal, merged_str, self.graph_id,
                        )
            else:
                sg_id = len(self.subgoal_nodes)
                subgoal_node = SubgoalNode(
                    subgoal=subgoal_str, subgoal_id=sg_id,
                    embedding=subgoal_embedding, time=self.procedural_time,
                )
                self.subgoal_nodes.append(subgoal_node)
                self.subgoal2node[subgoal_str] = subgoal_node
                self.subgoal_id2node[sg_id] = subgoal_node

            proc_id = len(self.procedural_nodes)
            proc_node = ProceduralNode(
                procedural_id=proc_id,
                procedural_memory_str=proc_str,
                embedding=proc_embedding,
                time=self.procedural_time,
                return_value=proc_item.get("return", 0.0),
                source=proc_item.get("source"),
                confidence=float(proc_item.get("confidence", 0.5)),
                session_id=proc_item.get("session_id", default_sid),
                provenance=proc_item.get("provenance"),
            )
            traj_num = proc_item.get("trajectory_num", 0)
            if traj_num < len(episodic_nodes):
                proc_node.episodic_nodes = list(episodic_nodes[traj_num])

            subgoal_node.activate([proc_node])
            proc_node.subgoal_nodes.append(subgoal_node)
            proc_node.subgoals.append(subgoal_node.subgoal)
            self.procedural_nodes.append(proc_node)
            self.procedural_id2node[proc_id] = proc_node
            self._track_session_node("procedural", proc_node)

            # Persist
            sg_emb = subgoal_node.embedding
            if isinstance(sg_emb, np.ndarray):
                sg_emb = sg_emb.tolist()
            proc_emb_list = proc_embedding if isinstance(proc_embedding, list) else proc_embedding.tolist() if isinstance(proc_embedding, np.ndarray) else proc_embedding

            provenance = getattr(proc_node, "provenance", None) or {}
            proc_batch.append({
                "procedural_id": proc_id,
                "text": proc_str,
                "embedding": proc_emb_list,
                "subgoal": subgoal_node.subgoal,
                "subgoal_id": subgoal_node.subgoal_id,
                "episodic_ids": [e.episodic_id for e in proc_node.episodic_nodes],
                "time": self.procedural_time,
                "return_value": proc_node.return_value,
                "source": proc_node.source,
                "confidence": proc_node.confidence,
                "session_id": proc_node.session_id,
                "provenance": provenance,
            })
            sg_proc_ids = [p.procedural_id for p in subgoal_node.procedural_nodes]
            if is_new_subgoal:
                new_subgoal_by_id[subgoal_node.subgoal_id] = {
                    "subgoal_id": subgoal_node.subgoal_id,
                    "subgoal": subgoal_node.subgoal,
                    "embedding": sg_emb,
                    "procedural_ids": sg_proc_ids,
                    "time": subgoal_node.time,
                }
            else:
                pending = new_subgoal_by_id.get(subgoal_node.subgoal_id)
                if pending is not None:
                    # Subgoal created earlier in this insert; mutate the
                    # pending row instead of updating a not-yet-persisted row.
                    pending["subgoal"] = subgoal_node.subgoal
                    pending["embedding"] = sg_emb
                    pending["procedural_ids"] = sg_proc_ids
                    pending["time"] = subgoal_node.time
                else:
                    self.storage.update_subgoal(
                        self.graph_id, subgoal_id=subgoal_node.subgoal_id,
                        subgoal=subgoal_node.subgoal, embedding=sg_emb,
                        metadata_updates={
                            "time": subgoal_node.time,
                        },
                    )

            self.procedural_time += 1

        if proc_batch:
            self.storage.add_procedural_batch(self.graph_id, proc_batch)
        if new_subgoal_by_id:
            self.storage.add_subgoal_batch(self.graph_id, list(new_subgoal_by_id.values()))

        self._rebuild_lookups()
        logger.info("Inserted memory into graph %s", self.graph_id)

    # ------------------------------------------------------------------ #
    # Retrieval methods (unchanged logic, uses injected clients)
    # ------------------------------------------------------------------ #

    def retrieve_tag_nodes(
        self,
        tag: str,
        tag_embedding=None,
        value_func: ValueBase = None,
        make_tag_nodes: bool = False,
        _trace: Optional[List[Dict[str, Any]]] = None,
    ) -> List[TagNode]:
        if tag_embedding is None:
            tag_embedding = self.embedder.embed(tag)

        # Batch-embed any tag nodes missing an embedding to avoid per-node round-trips.
        missing = [tn for tn in self.tag_nodes if tn.embedding is None]
        if missing:
            embeddings = self.embedder.embed_batch([tn.tag for tn in missing])
            for tn, emb in zip(missing, embeddings):
                tn.embedding = emb

        evaluations: List[Dict[str, Any]] = []
        values = []
        semantic_time = self.semantic_time
        for tag_node in self.tag_nodes:
            relevance = get_similarity(tag_embedding, tag_node.embedding)
            recency = semantic_time - tag_node.time
            value = value_func.evaluate(
                Relevance=relevance,
                Recency=recency,
                Importance=tag_node.importance,
            )
            values.append((value, tag_node.tag_id))
            if _trace is not None:
                evaluations.append({
                    "tag_id": tag_node.tag_id,
                    "tag": tag_node.tag,
                    "relevance": float(relevance),
                    "recency": int(recency),
                    "importance": float(tag_node.importance),
                    "value": float(value),
                })

        values.sort(reverse=True, key=lambda x: x[0])
        topk = values[: value_func.k]

        result = []
        selected_ids: set[int] = set()
        for value, tag_id in topk:
            if value < value_func.value_threshold:
                break
            node = self.tag_id2node.get(tag_id)
            if node:
                result.append(node)
                selected_ids.add(tag_id)

        if not result and make_tag_nodes:
            tag_id = len(self.tag_nodes)
            tag_node = TagNode(tag=tag, tag_id=tag_id, embedding=tag_embedding, time=semantic_time)
            self.tag_nodes.append(tag_node)
            self.tag2node[tag] = tag_node
            self.tag_id2node[tag_id] = tag_node
            result.append(tag_node)

        if _trace is not None:
            for ev in evaluations:
                ev["selected"] = ev["tag_id"] in selected_ids
                ev["query_tag"] = tag
            evaluations.sort(key=lambda d: d["value"], reverse=True)
            _trace.extend(evaluations)

        return result

    def retrieve_semantic_nodes(
        self,
        semantic_memory: Dict[str, Any],
        semantic_memory_embedding: Optional[Dict[str, Any]] = None,
        value_func_tag: Optional[ValueBase] = None,
        value_func: Optional[ValueBase] = None,
        min_confidence: Optional[float] = None,
        source_in: Optional[List[str]] = None,
        provenance_filters: Optional[Dict[str, List[str]]] = None,
        _trace: Optional[Dict[str, Any]] = None,
    ) -> List[SemanticNode]:
        if value_func_tag is None or value_func is None:
            raise ValueError("value_func_tag and value_func must not be None.")

        if semantic_memory_embedding is None:
            semantic_memory_embedding = {
                "semantic_memory": self.embedder.embed(semantic_memory["semantic_memory"]),
                "tags": [self.embedder.embed(t) for t in semantic_memory.get("tags", [])],
            }

        query_embedding = semantic_memory_embedding["semantic_memory"]
        query_tags: List[str] = semantic_memory.get("tags", [])

        # Phase 1: direct embedding similarity top-5
        sem_node_topk = 5
        top_sim_nodes, sim_list = self._semantic_similarity_candidates(
            query_embedding=query_embedding,
            n_results=max(sem_node_topk, value_func.k * 2),
            min_confidence=min_confidence,
            source_in=source_in,
            provenance_filters=provenance_filters,
        )
        top_sim_nodes = top_sim_nodes[:sem_node_topk]

        if _trace is not None:
            _trace["semantic_topk_by_similarity"] = [
                {
                    "semantic_id": sid,
                    "similarity": float(sim),
                    "text": (self.semantic_id2node[sid].get_semantic_memory() or "")[:200]
                            if sid in self.semantic_id2node else "",
                }
                for sim, sid in sim_list[:sem_node_topk]
                if sid in self.semantic_id2node
            ]
            _trace["query_tags"] = list(query_tags)

        # Phase 2: tag-based voting
        tag_trace: Optional[List[Dict[str, Any]]] = [] if _trace is not None else None
        tag_nodes = []
        for tag, tag_emb in zip(query_tags, semantic_memory_embedding["tags"]):
            tag_nodes.extend(self.retrieve_tag_nodes(
                tag=tag, tag_embedding=tag_emb,
                value_func=value_func_tag, _trace=tag_trace,
            ))

        tag_vote: Dict[int, Dict[str, float]] = {}
        for tag_node in tag_nodes:
            for sem_node in tag_node.semantic_nodes:
                sid = sem_node.semantic_id
                if sid not in tag_vote:
                    tag_vote[sid] = {"cnt": 0, "importance": 0.0}
                tag_vote[sid]["cnt"] += 1
                if tag_node.tag in query_tags:
                    tag_vote[sid]["importance"] += 5.0 * tag_node.importance
                else:
                    tag_vote[sid]["importance"] += float(tag_node.importance)

        for sem_node in top_sim_nodes:
            sid = sem_node.semantic_id
            if sid not in tag_vote:
                tag_vote[sid] = {"cnt": 0, "importance": 0.0}
            tag_vote[sid]["cnt"] += 1
            tag_vote[sid]["importance"] += 2.0

        if _trace is not None:
            _trace["tag_candidates"] = tag_trace or []

        # Phase 3: score candidates
        candidate_nodes = list(set(
            [self.semantic_id2node[sid] for sid in tag_vote if sid in self.semantic_id2node]
            + top_sim_nodes
        ))
        candidate_nodes = [
            n for n in candidate_nodes
            if _passes_metadata_filter(n, min_confidence, source_in, provenance_filters)
        ]

        candidate_trace: List[Dict[str, Any]] = []
        values = []
        semantic_time = self.semantic_time
        for sem_node in candidate_nodes:
            if sem_node.embedding is None:
                sem_node.embedding = self.embedder.embed(sem_node.get_semantic_memory())
            relevance = get_similarity(query_embedding, sem_node.embedding)
            num_tags = max(1, len(sem_node.tags))
            importance_score = tag_vote.get(sem_node.semantic_id, {}).get("importance", 0.0) / num_tags
            tag_votes_cnt = int(tag_vote.get(sem_node.semantic_id, {}).get("cnt", 0))
            recency = (semantic_time - sem_node.time) if isinstance(sem_node.time, int) else 0
            value = value_func.evaluate(
                Relevance=relevance, Recency=recency,
                Importance=importance_score, Credibility=sem_node.credibility,
                Source=getattr(sem_node, "source", None),
                Confidence=getattr(sem_node, "confidence", 0.5),
            )
            values.append((value, sem_node.semantic_id))
            if _trace is not None:
                candidate_trace.append({
                    "semantic_id": sem_node.semantic_id,
                    "text": (sem_node.get_semantic_memory() or "")[:240],
                    "tags": list(sem_node.tags),
                    "relevance": float(relevance),
                    "recency": int(recency),
                    "importance": float(importance_score),
                    "credibility": int(getattr(sem_node, "credibility", 0)),
                    "tag_votes": tag_votes_cnt,
                    "value": float(value),
                    "is_active": bool(sem_node.is_active),
                })

        values.sort(reverse=True, key=lambda x: x[0])
        kept = values[: value_func.k]

        result = []
        selected_ids: set[int] = set()
        for value, sid in kept:
            if value < value_func.value_threshold:
                break
            node = self.semantic_id2node.get(sid)
            if node:
                result.append(node)
                selected_ids.add(sid)

        if _trace is not None:
            for c in candidate_trace:
                c["selected"] = c["semantic_id"] in selected_ids
            candidate_trace.sort(key=lambda d: d["value"], reverse=True)
            _trace["semantic_candidates"] = candidate_trace
            _trace["k"] = int(value_func.k)
            _trace["value_threshold"] = float(value_func.value_threshold)

        return result

    def retrieve_semantic_nodes_wo_tag(
        self,
        semantic_memory: dict,
        semantic_memory_embedding=None,
        value_func: ValueBase = None,
    ) -> List[SemanticNode]:
        if semantic_memory_embedding is None:
            semantic_memory_embedding = {
                "semantic_memory": self.embedder.embed(semantic_memory["semantic_memory"]),
            }

        embedding = semantic_memory_embedding["semantic_memory"]
        values = []
        semantic_time = self.semantic_time
        candidate_nodes, _ = self._semantic_similarity_candidates(
            query_embedding=embedding,
            n_results=max(value_func.k * 8, 20),
        )
        for sem_node in candidate_nodes:
            relevance = get_similarity(embedding, sem_node.embedding)
            recency = (semantic_time - sem_node.time) if isinstance(sem_node.time, int) else 0
            value = value_func.evaluate(
                Relevance=relevance, Recency=recency,
                Credibility=sem_node.credibility,
            )
            values.append((value, sem_node.semantic_id))

        values.sort(reverse=True, key=lambda x: x[0])
        values = values[: value_func.k]

        result = []
        for value, sid in values:
            if value < value_func.value_threshold:
                break
            node = self.semantic_id2node.get(sid)
            if node:
                result.append(node)
        return result

    def _semantic_similarity_candidates(
        self,
        *,
        query_embedding,
        n_results: int,
        min_confidence: Optional[float] = None,
        source_in: Optional[List[str]] = None,
        provenance_filters: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[List[SemanticNode], List[Tuple[float, int]]]:
        """Get semantic candidates by vector similarity, preferring Chroma query.

        Returns candidate nodes plus `(similarity, semantic_id)` tuples in ranked
        order for trace/debug use. Falls back to an in-memory full scan if the
        vector store query is unavailable or returns too few usable rows.
        """
        target = max(1, min(n_results, len(self.semantic_nodes)))
        if target <= 0:
            return [], []

        ranked_ids: List[int] = []
        seen_ids: set[int] = set()

        try:
            result = self.storage.query_semantic(
                self.graph_id,
                query_embedding=query_embedding,
                n_results=min(max(target * 3, 10), max(len(self.semantic_nodes), 1)),
            )
            metas = (result.get("metadatas") or [[]])[0] or []
            for meta in metas:
                sid = meta.get("semantic_id")
                if sid is None or sid in seen_ids:
                    continue
                seen_ids.add(sid)
                ranked_ids.append(sid)
                if len(ranked_ids) >= target:
                    break
        except Exception:
            logger.debug("semantic candidate query failed; falling back to full scan", exc_info=True)

        candidate_nodes: List[SemanticNode] = []
        for sid in ranked_ids:
            node = self.semantic_id2node.get(sid)
            if node is None or not node.is_active:
                continue
            if not _passes_metadata_filter(node, min_confidence, source_in, provenance_filters):
                continue
            if node.embedding is None:
                node.embedding = self.embedder.embed(node.get_semantic_memory())
            candidate_nodes.append(node)

        if len(candidate_nodes) < target:
            fallback = []
            for node in self.semantic_nodes:
                if node.semantic_id in seen_ids:
                    continue
                if not node.is_active:
                    continue
                if not _passes_metadata_filter(node, min_confidence, source_in, provenance_filters):
                    continue
                if node.embedding is None:
                    node.embedding = self.embedder.embed(node.get_semantic_memory())
                sim = get_similarity(query_embedding, node.embedding)
                fallback.append((sim, node.semantic_id))
            fallback.sort(reverse=True, key=lambda x: x[0])
            for _, sid in fallback:
                node = self.semantic_id2node.get(sid)
                if node is None:
                    continue
                candidate_nodes.append(node)
                if len(candidate_nodes) >= target:
                    break

        sim_list = []
        for node in candidate_nodes:
            sim_list.append((get_similarity(query_embedding, node.embedding), node.semantic_id))
        sim_list.sort(reverse=True, key=lambda x: x[0])
        ordered_nodes = [self.semantic_id2node[sid] for _, sid in sim_list if sid in self.semantic_id2node]
        return ordered_nodes, sim_list

    def retrieve_episodic_nodes(self, observation: str) -> str:
        semantic_nodes = self.retrieve_semantic_nodes_wo_tag(
            semantic_memory={"semantic_memory": observation},
            value_func=self.semantic_relevant4episodic,
        )
        semantic_nodes = semantic_nodes[:30]

        vote_session: Dict[str, int] = {}
        for sn in semantic_nodes:
            sid = sn.session_id
            if sid is not None:
                vote_session[sid] = vote_session.get(sid, 0) + 1

        parts: List[str] = []
        cnt = 0
        for key, value in vote_session.items():
            if value >= 3:
                parts.append(f"Relevant Memory {cnt}:\n{self.get_session_memory(key)}")
                cnt += 1
        for sn in semantic_nodes:
            sid = sn.session_id
            if sid is None or vote_session.get(sid, 0) < 3:
                parts.append(f"Relevant Memory {cnt}:\n{sn.get_semantic_memory()}\n")
                cnt += 1
        return "".join(parts)

    def get_session_memory(self, session_id: str) -> str:
        if session_id not in self.session_ids:
            return "There is no relevant memory"
        cached = self._session_str_cache.get(session_id)
        if cached is not None:
            return cached
        nodes = self.session_ids[session_id]
        parts = []
        if nodes:
            parts.append(nodes[0].get_date())
        for node in nodes:
            parts.append(node.get_episodic_memory(date=False))
        out = "\n".join(parts) + "\n"
        self._session_str_cache[session_id] = out
        return out

    def retrieve_subgoal_nodes(
        self, subgoal: str, subgoal_embedding=None, value_func: ValueBase = None,
    ) -> Optional[SubgoalNode]:
        if subgoal_embedding is None:
            subgoal_embedding = self.embedder.embed(subgoal)

        best_value = -1.0
        best_node = None
        procedural_time = self.procedural_time
        for sg_node in self.subgoal_nodes:
            relevance = get_similarity(subgoal_embedding, sg_node.embedding)
            value = value_func.evaluate(
                Relevance=relevance,
                Recency=procedural_time - sg_node.time,
                Importance=sg_node.importance,
            )
            if value > best_value:
                best_node = sg_node
                best_value = value

        if best_value < value_func.value_threshold:
            return None
        best_node.importance += 1
        return best_node

    def retrieve_procedural_nodes(
        self, subgoal: str, value_func_subgoal: ValueBase, value_func: ValueBase,
        min_confidence: Optional[float] = None,
        source_in: Optional[List[str]] = None,
        provenance_filters: Optional[Dict[str, List[str]]] = None,
        _trace: Optional[Dict[str, Any]] = None,
    ) -> List[ProceduralNode]:
        embedding = self.embedder.embed(subgoal)
        subgoal_node = self.retrieve_subgoal_nodes(
            subgoal=subgoal, value_func=value_func_subgoal,
        )
        if _trace is not None:
            _trace["subgoal_query"] = subgoal
            _trace["subgoal_match"] = (
                None if subgoal_node is None
                else {"subgoal_id": subgoal_node.subgoal_id,
                      "subgoal": subgoal_node.subgoal}
            )
            _trace["procedural_candidates"] = []
        if subgoal_node is None:
            return []

        candidate_trace: List[Dict[str, Any]] = []
        values = []
        procedural_time = self.procedural_time
        for proc_node in subgoal_node.procedural_nodes:
            if not _passes_metadata_filter(proc_node, min_confidence, source_in, provenance_filters):
                continue
            relevance = get_similarity(embedding, proc_node.embedding)
            recency = procedural_time - proc_node.time
            value = value_func.evaluate(
                Relevance=relevance,
                Return=proc_node.return_value,
                Recency=recency,
                Source=getattr(proc_node, "source", None),
                Confidence=getattr(proc_node, "confidence", 0.5),
            )
            values.append((value, proc_node.procedural_id))
            if _trace is not None:
                candidate_trace.append({
                    "procedural_id": proc_node.procedural_id,
                    "text": (proc_node.get_procedural_memory() or "")[:240],
                    "subgoal": subgoal_node.subgoal,
                    "relevance": float(relevance),
                    "recency": int(recency),
                    "return": float(proc_node.return_value),
                    "value": float(value),
                })

        values.sort(reverse=True, key=lambda x: x[0])
        kept = values[: value_func.k]

        result = []
        selected_ids: set[int] = set()
        for value, pid in kept:
            if value < value_func.value_threshold:
                break
            node = self.procedural_id2node.get(pid)
            if node:
                result.append(node)
                selected_ids.add(pid)

        if _trace is not None:
            for c in candidate_trace:
                c["selected"] = c["procedural_id"] in selected_ids
            candidate_trace.sort(key=lambda d: d["value"], reverse=True)
            _trace["procedural_candidates"] = candidate_trace
            _trace["k"] = int(value_func.k)
            _trace["value_threshold"] = float(value_func.value_threshold)

        return result

    # ------------------------------------------------------------------ #
    # Retrieve + Reason pipeline
    # ------------------------------------------------------------------ #

    def retrieve_memory(
        self,
        goal: str = None,
        subgoal: str = None,
        state: str = None,
        observation: str = None,
        time: str = "",
        task_type: str = "",
        mode: str = None,
        min_confidence: Optional[float] = None,
        source_in: Optional[List[str]] = None,
        provenance_filters: Optional[Dict[str, List[str]]] = None,
        _audit: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any], str]:
        if mode is None:
            mode = get_mode(
                self.retrieval_llm, observation=observation, task_type=task_type,
                prompts=self.prompts, graph_id=self.graph_id,
            )
        logger.info("mode: %s", mode)

        query_tags: List[str] = []
        next_subgoal = subgoal or observation or ""
        if mode in ["procedural_memory", "episodic_memory"]:
            next_subgoal, query_tags = get_plan(
                self.retrieval_llm, goal=goal, subgoal=subgoal, state=state, observation=observation,
                prompts=self.prompts, graph_id=self.graph_id,
            )
        logger.info("query_tags: %s", query_tags)

        prompt_name, fallback_cls = _REASONING_MAP.get(mode, ("reasoning_semantic", DefaultSemanticPrompt))
        if self.prompts is not None:
            prompt_template = self.prompts.get(prompt_name, graph_id=self.graph_id)
        else:
            prompt_template = fallback_cls()

        semantic_nodes, procedural_nodes = [], []
        if mode in ["semantic_memory", "episodic_memory"]:
            semantic_nodes = self.retrieve_semantic_nodes(
                semantic_memory={"semantic_memory": observation, "tags": query_tags},
                value_func_tag=self.tag_relevant,
                value_func=self.semantic_relevant,
                min_confidence=min_confidence,
                source_in=source_in,
                provenance_filters=provenance_filters,
            )
        if mode in ["procedural_memory", "episodic_memory"]:
            procedural_nodes = self.retrieve_procedural_nodes(
                subgoal=next_subgoal,
                value_func_subgoal=self.subgoal_relevant,
                value_func=self.procedural_relevant,
                min_confidence=min_confidence,
                source_in=source_in,
                provenance_filters=provenance_filters,
            )

        semantic_memory_str = ""
        procedural_memory_str = ""
        episodic_memory_str = ""

        if mode == "episodic_memory":
            episodic_memory_str = self.retrieve_episodic_nodes(observation=observation)
        elif mode == "semantic_memory":
            if not semantic_nodes:
                semantic_memory_str = "No relevant fact"
            else:
                semantic_memory_str = "\n".join(
                    f"Fact {i}: {sn.get_semantic_memory()}"
                    for i, sn in enumerate(semantic_nodes)
                ) + "\n"
        elif mode == "procedural_memory":
            if not procedural_nodes:
                procedural_memory_str = "No relevant experiences"
            else:
                procedural_memory_str = "\n".join(
                    f"Experience {i}: {pn.get_procedural_memory()}"
                    for i, pn in enumerate(procedural_nodes)
                ) + "\n"
        else:
            raise ValueError(f"Invalid mode: {mode}")

        variables = {
            "goal": goal,
            "subgoal": subgoal,
            "state": state,
            "observation": observation,
            "semantic_memory": semantic_memory_str,
            "procedural_memory": procedural_memory_str,
            "episodic_memory": episodic_memory_str,
            "time": time,
            "information": episodic_memory_str,
            "question": observation,
        }
        messages = prompt_template.build_messages(variables)
        messages = [{"role": m.role, "content": m.content} for m in messages]

        if _audit is not None:
            _audit["next_subgoal"] = next_subgoal or ""
            _audit["query_tags"] = list(query_tags or [])
            _audit["selected_semantic_ids"] = [n.semantic_id for n in semantic_nodes]
            _audit["selected_procedural_ids"] = [n.procedural_id for n in procedural_nodes]

        return messages, variables, mode

    def retrieve_with_trace(
        self,
        observation: str,
        goal: str = None,
        subgoal: str = None,
        state: str = None,
        time: str = "",
        task_type: str = "",
        mode: Optional[str] = None,
        query_tags: Optional[List[str]] = None,
        next_subgoal: Optional[str] = None,
        auto_plan: bool = False,
        provenance_filters: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """Run the retrieval pipeline with full instrumentation.

        Design constraint: this is the *only* trace-producing entrypoint;
        production retrieval (``retrieve_memory``) does not pay the cost.

        ``auto_plan`` controls whether the LLM planner fills in missing
        ``mode`` / ``query_tags`` / ``next_subgoal``. When False (default),
        sensible no-LLM fallbacks are used so the demo works without any
        LLM service configured.
        """
        # 1. Plan / mode resolution
        plan_source: Dict[str, str] = {}
        if mode is None:
            if auto_plan:
                mode = get_mode(
                    self.retrieval_llm, observation=observation,
                    task_type=task_type, prompts=self.prompts, graph_id=self.graph_id,
                )
                plan_source["mode"] = "llm"
            else:
                mode = "semantic_memory"
                plan_source["mode"] = "default"
        else:
            plan_source["mode"] = "override"

        if (query_tags is None or next_subgoal is None) and auto_plan:
            llm_subgoal, llm_tags = get_plan(
                self.retrieval_llm, goal=goal, subgoal=subgoal, state=state,
                observation=observation, prompts=self.prompts, graph_id=self.graph_id,
            )
            if next_subgoal is None:
                next_subgoal = llm_subgoal
                plan_source["next_subgoal"] = "llm"
            if query_tags is None:
                query_tags = llm_tags
                plan_source["query_tags"] = "llm"

        if next_subgoal is None:
            next_subgoal = subgoal or observation or ""
            plan_source.setdefault("next_subgoal", "default")
        else:
            plan_source.setdefault("next_subgoal", "override")

        if query_tags is None:
            query_tags = []
            plan_source.setdefault("query_tags", "default")
        else:
            plan_source.setdefault("query_tags", "override")

        # 2. Retrieve with traces
        sem_trace: Dict[str, Any] = {}
        proc_trace: Dict[str, Any] = {}
        semantic_nodes: List[SemanticNode] = []
        procedural_nodes: List[ProceduralNode] = []

        if mode in ("semantic_memory", "episodic_memory"):
            semantic_nodes = self.retrieve_semantic_nodes(
                semantic_memory={"semantic_memory": observation, "tags": query_tags},
                value_func_tag=self.tag_relevant,
                value_func=self.semantic_relevant,
                provenance_filters=provenance_filters,
                _trace=sem_trace,
            )
        if mode in ("procedural_memory", "episodic_memory"):
            procedural_nodes = self.retrieve_procedural_nodes(
                subgoal=next_subgoal,
                value_func_subgoal=self.subgoal_relevant,
                value_func=self.procedural_relevant,
                provenance_filters=provenance_filters,
                _trace=proc_trace,
            )

        # 3. Build memory text & rendered prompt — same shape as retrieve_memory
        semantic_memory_str = ""
        procedural_memory_str = ""
        episodic_memory_str = ""

        if mode == "episodic_memory":
            episodic_memory_str = self.retrieve_episodic_nodes(observation=observation)
        elif mode == "semantic_memory":
            if not semantic_nodes:
                semantic_memory_str = "No relevant fact"
            else:
                semantic_memory_str = "".join(
                    f"Fact {i}: {n.get_semantic_memory()}\n"
                    for i, n in enumerate(semantic_nodes)
                )
        elif mode == "procedural_memory":
            if not procedural_nodes:
                procedural_memory_str = "No relevant experiences"
            else:
                procedural_memory_str = "".join(
                    f"Experience {i}: {n.get_procedural_memory()}\n"
                    for i, n in enumerate(procedural_nodes)
                )

        prompt_name, fallback_cls = _REASONING_MAP.get(mode, ("reasoning_semantic", DefaultSemanticPrompt))
        if self.prompts is not None:
            prompt_template = self.prompts.get(prompt_name, graph_id=self.graph_id)
        else:
            prompt_template = fallback_cls()

        variables = {
            "goal": goal, "subgoal": subgoal, "state": state, "observation": observation,
            "semantic_memory": semantic_memory_str,
            "procedural_memory": procedural_memory_str,
            "episodic_memory": episodic_memory_str,
            "time": time, "information": episodic_memory_str, "question": observation,
        }
        rendered = prompt_template.build_messages(variables)
        rendered_prompt = [{"role": m.role, "content": m.content} for m in rendered]

        return {
            "mode": mode,
            "plan": {
                "next_subgoal": next_subgoal,
                "query_tags": list(query_tags),
                "source": plan_source,
            },
            "trace": {
                "semantic": sem_trace,
                "procedural": proc_trace,
            },
            "selected": {
                "semantic_ids": [n.semantic_id for n in semantic_nodes],
                "procedural_ids": [n.procedural_id for n in procedural_nodes],
            },
            "rendered_prompt": rendered_prompt,
        }

    def retrieve_and_reason(
        self,
        goal: str = None,
        subgoal: str = None,
        state: str = None,
        observation: str = None,
        time: str = "",
        task_type: str = "",
        mode: str = None,
        min_confidence: Optional[float] = None,
        source_in: Optional[List[str]] = None,
    ) -> str:
        messages, _, _ = self.retrieve_memory(
            goal=goal, subgoal=subgoal, state=state,
            observation=observation, time=time,
            task_type=task_type, mode=mode,
            min_confidence=min_confidence, source_in=source_in,
        )
        return self.reasoning_llm.complete(messages=messages)

    # ------------------------------------------------------------------ #
    # Semantic merging / consolidation
    # ------------------------------------------------------------------ #

    def merge_semantic(self, id1: int, id2: int) -> Tuple[SemanticNode, bool, bool]:
        sem1 = self.semantic_id2node[id1]
        sem2 = self.semantic_id2node[id2]
        merge_decision = get_new_semantic(
            self.consolidation_llm, sem1.get_semantic_memory(), sem2.get_semantic_memory(),
            prompts=self.prompts, graph_id=self.graph_id,
        )

        merged_str = merge_decision["merged_statement"]
        if_del_1 = merge_decision["deactivate_earlier"]
        if_del_2 = merge_decision["deactivate_later"]

        embedding = self.embedder.embed(merged_str)
        merged_node = SemanticNode(
            semantic_id=len(self.semantic_nodes),
            semantic_memory_str=merged_str,
            embedding=embedding,
            time=self.semantic_time,
            son=[sem1, sem2],
        )
        self.semantic_nodes.append(merged_node)

        # Combine episodic and tag links
        epis_ids = set()
        for en in sem1.episodic_nodes + sem2.episodic_nodes:
            epis_ids.add(en.episodic_id)
        merged_node.episodic_nodes = [self.episodic_id2node[eid] for eid in epis_ids if eid in self.episodic_id2node]

        tag_ids = set()
        for tn in sem1.tag_nodes + sem2.tag_nodes:
            tag_ids.add(tn.tag_id)
        merged_node.tag_nodes = [self.tag_id2node[tid] for tid in tag_ids if tid in self.tag_id2node]
        merged_node.tags = list(set(sem1.tags + sem2.tags))

        # Persist merged node
        emb_list = embedding if isinstance(embedding, list) else embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        self.storage.add_semantic(
            self.graph_id,
            semantic_id=merged_node.semantic_id,
            text=merged_str,
            embedding=emb_list,
            tags=merged_node.tags,
            tag_ids=[t.tag_id for t in merged_node.tag_nodes],
            time=merged_node.time,
            episodic_ids=[e.episodic_id for e in merged_node.episodic_nodes],
            son_semantic_ids=[sem1.semantic_id, sem2.semantic_id],
        )

        return merged_node, if_del_1, if_del_2

    def update_semantic_subgraph(
        self,
        *,
        merge_threshold: float = 0.5,
        max_merges_per_node: int = 1,
        max_candidates_per_tag: int = 200,
        max_total_candidates: int = 800,
        min_credibility_to_keep_active: int = -10,
        credibility_decay: int = 0,
        only_update_recent_window: Optional[int] = None,
        allow_merge_with_common_episodic_nodes: bool = False,
    ) -> Dict[str, int]:
        stats = {
            "scanned_semantic": 0,
            "skipped_inactive": 0,
            "merged_pairs": 0,
            "new_semantic_nodes": 0,
            "soft_deactivated": 0,
        }

        time_st = self.semantic_time

        # Credibility decay
        if credibility_decay != 0:
            for sn in self.semantic_nodes:
                if sn.time < time_st and sn.is_active:
                    sn.credibility -= credibility_decay

        # Determine scope
        if only_update_recent_window is None:
            scope = [i for i, s in enumerate(self.semantic_nodes) if s.time < time_st]
        else:
            low = max(0, time_st - only_update_recent_window)
            scope = [i for i, s in enumerate(self.semantic_nodes) if low <= s.time < time_st]

        for idx in scope:
            sem_node = self.semantic_nodes[idx]
            stats["scanned_semantic"] += 1

            if not sem_node.is_active:
                stats["skipped_inactive"] += 1
                continue
            if sem_node.updated:
                continue

            if sem_node.credibility < min_credibility_to_keep_active:
                sem_node.is_active = False
                self.storage.update_semantic(
                    self.graph_id, sem_node.semantic_id,
                    metadata_updates={"is_active": False},
                )
                stats["soft_deactivated"] += 1
                continue

            # Collect candidates via tags
            cand_ids = set()
            for tag_node in sem_node.tag_nodes:
                ids = [s.semantic_id for s in tag_node.semantic_nodes]
                if len(ids) > max_candidates_per_tag:
                    ids = random.sample(ids, k=max_candidates_per_tag)
                cand_ids.update(ids)
                if len(cand_ids) >= max_total_candidates:
                    break
            cand_ids.discard(sem_node.semantic_id)

            filtered = []
            for cid in cand_ids:
                cand = self.semantic_id2node.get(cid)
                if cand is None or not cand.is_active or cand.time >= time_st or cand.updated:
                    continue
                if cand.semantic_id <= sem_node.semantic_id:
                    continue
                filtered.append(cid)

            if not filtered:
                continue

            scored = []
            for cid in filtered:
                cand = self.semantic_id2node[cid]
                rel = get_similarity(sem_node.embedding, cand.embedding)
                val = self.semantic_equal.evaluate(Relevance=rel)
                scored.append((val, cid))

            k = getattr(self.semantic_equal, "k", 10)
            topk = heapq.nlargest(k, scored, key=lambda x: x[0])

            merges_done = 0
            for val, cid in topk:
                if val < merge_threshold:
                    continue

                cand = self.semantic_id2node[cid]
                if not allow_merge_with_common_episodic_nodes:
                    epid_1 = {e.episodic_id for e in sem_node.episodic_nodes}
                    epid_2 = {e.episodic_id for e in cand.episodic_nodes}
                    if epid_1 & epid_2:
                        continue

                new_node, del_1, del_2 = self.merge_semantic(sem_node.semantic_id, cid)
                if del_1:
                    sem_node.is_active = False
                    self.storage.update_semantic(
                        self.graph_id, sem_node.semantic_id,
                        metadata_updates={"is_active": False},
                    )
                if del_2:
                    cand.is_active = False
                    self.storage.update_semantic(
                        self.graph_id, cid,
                        metadata_updates={"is_active": False},
                    )

                stats["merged_pairs"] += 1
                stats["new_semantic_nodes"] += 1
                self.semantic_id2node[new_node.semantic_id] = new_node

                # Repair tag edges
                for t in new_node.tag_nodes:
                    if new_node not in t.semantic_nodes:
                        t.semantic_nodes.append(new_node)

                merges_done += 1
                if merges_done >= max_merges_per_node:
                    break

            if merges_done > 0:
                sem_node.updated = True
                if merges_done >= 3:
                    break

        logger.info("Consolidation stats: %s", stats)
        return stats
