"""FastAPI dependency injection — wires up config, clients, and GraphManager."""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

from plugmem.clients.embedding import (
    EmbeddingClient,
    HTTPEmbeddingClient,
    LocalDeterministicEmbeddingClient,
    PlugMemEmbeddingFunction,
)
from plugmem.clients.llm import LLMClient, OpenAICompatibleLLMClient
from plugmem.clients.llm_router import LLMRouter
from plugmem.config import PlugMemConfig
from plugmem.graph_manager import GraphManager
from plugmem.storage.chroma import ChromaStorage
from plugmem.storage import StorageBackend

logger = logging.getLogger(__name__)


@lru_cache
def get_config() -> PlugMemConfig:
    """Build config from environment variables."""
    return PlugMemConfig(
        llm_base_url=os.getenv("LLM_BASE_URL", ""),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", ""),
        embedding_base_url=os.getenv("EMBEDDING_BASE_URL", ""),
        embedding_model=os.getenv("EMBEDDING_MODEL", "nvidia/NV-Embed-v2"),
        embedding_api_key=os.getenv("EMBEDDING_API_KEY", ""),
        storage_backend=os.getenv("STORAGE_BACKEND", "chroma"),
        chroma_mode=os.getenv("CHROMA_MODE", "persistent"),
        chroma_path=os.getenv("CHROMA_PATH", "./data/chroma"),
        chroma_host=os.getenv("CHROMA_HOST", "localhost"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        sqlite_vec_path=os.getenv("SQLITE_VEC_PATH", "./data/plugmem.db"),
        api_key=os.getenv("PLUGMEM_API_KEY"),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "5")),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        llm_top_p=float(os.getenv("LLM_TOP_P", "1.0")),
        llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
        embedding_max_text_len=int(os.getenv("EMBEDDING_MAX_TEXT_LEN", "8192")),
    )


_llm_client: LLMClient | LLMRouter | None = None
_embedding_client: EmbeddingClient | None = None
_graph_manager: GraphManager | None = None


def get_llm(config: PlugMemConfig | None = None) -> LLMClient | LLMRouter:
    """Return the LLM client(s).

    If ``LLM_CONFIG_PATH`` points to a YAML file, an :class:`LLMRouter` with
    per-role clients is returned.  Otherwise a single
    :class:`OpenAICompatibleLLMClient` built from env vars is returned.
    """
    global _llm_client
    if _llm_client is None:
        cfg = config or get_config()

        llm_config_path = os.getenv("LLM_CONFIG_PATH", "")
        if llm_config_path and Path(llm_config_path).is_file():
            _llm_client = LLMRouter.from_yaml(llm_config_path)
            logger.info("LLM routing loaded from %s", llm_config_path)
        else:
            _llm_client = OpenAICompatibleLLMClient(
                base_url=cfg.llm_base_url,
                api_key=cfg.llm_api_key,
                model=cfg.llm_model,
                max_retries=cfg.max_retries,
            )
    return _llm_client


_OPENAI_EMBED_URL = "https://api.openai.com/v1/embeddings"
_OPENAI_DEFAULT_MODEL = "text-embedding-3-small"


def get_embedder(config: PlugMemConfig | None = None) -> EmbeddingClient:
    """Resolve the embedder via a three-tier cascade.

    1. ``EMBEDDING_BASE_URL`` set → ``HTTPEmbeddingClient`` against that URL,
       optionally authed with ``EMBEDDING_API_KEY``. The canonical path for
       self-hosted NV-Embed-v2 or any custom provider.
    2. ``OPENAI_API_KEY`` set → OpenAI ``text-embedding-3-small`` (override
       the model with ``EMBEDDING_MODEL``). Convenient no-GPU fallback.
    3. Otherwise → ``LocalDeterministicEmbeddingClient`` (sha256-based, demo
       only — not semantically meaningful).
    """
    global _embedding_client
    if _embedding_client is not None:
        return _embedding_client

    cfg = config or get_config()
    openai_key = os.getenv("OPENAI_API_KEY", "")

    if cfg.embedding_base_url:
        _embedding_client = HTTPEmbeddingClient(
            base_url=cfg.embedding_base_url,
            model=cfg.embedding_model,
            api_key=cfg.embedding_api_key or None,
            max_text_len=cfg.embedding_max_text_len,
        )
        logger.info(
            "Embedder: HTTP %s (model=%s, auth=%s)",
            cfg.embedding_base_url, cfg.embedding_model,
            "yes" if cfg.embedding_api_key else "no",
        )
        return _embedding_client

    if openai_key:
        # User didn't pin a base URL but has an OpenAI key — use it.
        # EMBEDDING_MODEL still wins so they can pick -3-large or similar.
        model = cfg.embedding_model
        if model == "nvidia/NV-Embed-v2":
            # Default config value — assume they want OpenAI's default since
            # they didn't override either field.
            model = _OPENAI_DEFAULT_MODEL
        _embedding_client = HTTPEmbeddingClient(
            base_url=_OPENAI_EMBED_URL,
            model=model,
            api_key=openai_key,
            max_text_len=cfg.embedding_max_text_len,
        )
        logger.info(
            "Embedder: OpenAI fallback %s (paid via OPENAI_API_KEY). "
            "Set EMBEDDING_BASE_URL to override or unset OPENAI_API_KEY "
            "to fall through to the local deterministic embedder.",
            model,
        )
        return _embedding_client

    logger.warning(
        "Embedder: LocalDeterministicEmbeddingClient (sha256). "
        "Suitable for the demo and tests, NOT for retrieval quality. "
        "Set EMBEDDING_BASE_URL or OPENAI_API_KEY to enable a real embedder."
    )
    _embedding_client = LocalDeterministicEmbeddingClient()
    return _embedding_client


def build_chroma_storage(cfg: PlugMemConfig) -> ChromaStorage:
    """Construct a ChromaStorage from config — the single place that builds Chroma clients."""
    import chromadb

    if cfg.chroma_mode == "http":
        chroma_client = chromadb.HttpClient(host=cfg.chroma_host, port=cfg.chroma_port)
    elif cfg.chroma_mode == "ephemeral":
        chroma_client = chromadb.EphemeralClient()
    else:
        chroma_client = chromadb.PersistentClient(path=cfg.chroma_path)

    embedder = get_embedder(cfg)
    embed_fn = PlugMemEmbeddingFunction(embedder)
    return ChromaStorage(
        client=chroma_client,
        embedding_function=embed_fn,
        embedding_client=embedder,
    )


def build_sqlite_vec_storage(cfg: PlugMemConfig, embedding_dim: int = 768):
    """Construct an experimental SqliteVecStorage from config.

    Import is lazy so the optional ``sqlite-vec`` dependency only matters
    when this backend is actually selected via ``STORAGE_BACKEND=sqlite_vec``.
    ``embedding_dim`` defaults to 768 but callers should probe the active
    embedder and pass the real dimension (e.g. 32 for local deterministic).
    """
    try:
        from plugmem.storage.sqlite_vec import SqliteVecStorage
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "STORAGE_BACKEND=sqlite_vec requires the optional sqlite-vec "
            'dependency. Install with `pip install -e ".[sqlite-vec]"`.'
        ) from exc

    db_path = Path(cfg.sqlite_vec_path).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteVecStorage(str(db_path), embedding_dim=embedding_dim)


def build_storage(cfg: PlugMemConfig, embedding_dim: int = 768) -> StorageBackend:
    """Dispatch on ``cfg.storage_backend`` — single entry point for the API layer."""
    backend = (cfg.storage_backend or "chroma").lower()
    if backend == "chroma":
        return build_chroma_storage(cfg)
    if backend == "sqlite_vec":
        return build_sqlite_vec_storage(cfg, embedding_dim=embedding_dim)
    raise ValueError(
        f"Unknown storage_backend {backend!r}. Expected 'chroma' or 'sqlite_vec'."
    )


def get_graph_manager(config: PlugMemConfig | None = None) -> GraphManager:
    global _graph_manager
    if _graph_manager is None:
        cfg = config or get_config()
        embedder = get_embedder(cfg)
        # Probe the embedder output dimension for sqlite_vec schema creation.
        probe = embedder.embed("x")
        storage = build_storage(cfg, embedding_dim=len(probe))
        llm = get_llm(cfg)

        _graph_manager = GraphManager(
            storage=storage,
            llm=llm,
            embedder=embedder,
        )
    return _graph_manager


def reset_singletons() -> None:
    """Reset cached singletons — useful for testing."""
    global _llm_client, _embedding_client, _graph_manager
    _llm_client = None
    _embedding_client = None
    _graph_manager = None
    get_config.cache_clear()
