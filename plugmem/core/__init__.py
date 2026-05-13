"""plugmem.core — import-friendly library entrypoint.

Usage (no daemon required)::

    from plugmem.core import (
        extract_coding_memories,
        SemanticRelevant,
        compute_source_boost,
        get_similarity,
        passes_metadata_filter,
        PROVENANCE_FIELDS,
    )

    memories, rejections = extract_coding_memories(llm, candidates)

Library-mode callers supply their own LLM client, embedder, and storage.
The daemon wraps this same layer with FastAPI + ChromaDB + CLI.
"""
from __future__ import annotations

from plugmem.core.graph_node import (
    EpisodicNode,
    ProceduralNode,
    PROVENANCE_FIELDS,
    SemanticNode,
    SubgoalNode,
    TagNode,
)
from plugmem.core.memory_graph import (
    MemoryGraph,
    _passes_metadata_filter as passes_metadata_filter,
)
from plugmem.core.value_base import ValueBase
from plugmem.core.value_functions import (
    ProceduralEqual,
    ProceduralRelevant,
    SemanticEqual,
    SemanticRelevant,
    SubgoalEqual,
    SubgoalRelevant,
    TagEqual,
    TagRelevant,
    compute_source_boost,
)
from plugmem.inference.promotion import extract_coding_memories

from plugmem.clients.embedding import get_similarity

__all__ = [
    # Node types
    "EpisodicNode",
    "SemanticNode",
    "ProceduralNode",
    "TagNode",
    "SubgoalNode",
    # Extraction
    "extract_coding_memories",
    # Value functions for ranking
    "ValueBase",
    "TagEqual",
    "TagRelevant",
    "SemanticEqual",
    "SemanticRelevant",
    "SubgoalEqual",
    "SubgoalRelevant",
    "ProceduralEqual",
    "ProceduralRelevant",
    "compute_source_boost",
    # Filtering
    "passes_metadata_filter",
    # Similarity
    "get_similarity",
    # Provenance
    "PROVENANCE_FIELDS",
    # Full graph (daemon-reliant)
    "MemoryGraph",
]
