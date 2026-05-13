# PlugMem — Agent Context

## Scope (what PlugMem is not)

PlugMem stores **agent experience** (corrections, preferences, procedures), scoped by provenance. It is **not a code index**.

For code-structure questions ("what calls X?", "where is Y defined?", impact radius, call graphs) the companion is **CocoIndex-code** (`~/code/coco/cocoindex-code/`, MCP tools: `codebase_search`, `codebase_symbol`, `codebase_impact`, `codebase_graph_*`, `codebase_context_*`, `codebase_workflow`). The two MCP servers co-deploy — route code-structure to cocoindex-code, experience/correction to PlugMem.

Concrete API contract:
- **Agent list surface** = `GET /graphs/{gid}/nodes` (and `plugmem coding list`). Filters: `language`, `repo`, `source_in`, `min_confidence`. Metadata only, never content.
- **Agent recall surfaces** = `POST /retrieve`, `POST /reason`. Accept `provenance_filters: {key: [values]}` (keys: repo, branch, commit, language, filepath, package_manager, tool_name, tool_version, os, component).
- **Inspector-UI-only** = `GET /graphs/{gid}/search` (content substring). Do not call from agents — the OpenAPI summary already warns.

When extending coding-agent surfaces: provenance/source/confidence are the right filter axes; memory content text is not.

## Architecture

```
plugmem/
  api/            FastAPI routes (graphs, memories, retrieval, extract, promote, inspector, demo)
  cli/            Typer CLI (init, start, stop, restart, status, logs, health)
  clients/        LLM client (OpenAI-compatible), embedding client, LLM router
  core/           Memory, MemoryGraph, graph node types, value functions, normalization
  inference/      LLM-powered structuring, promotion extraction, retrieval logic
  prompts/        Prompt templates for structuring, retrieving, reasoning
  storage/        ChromaDB persistence wrapper
  config.py       Pydantic config model
  graph_manager.py
tests/            pytest suite
```

## Public surfaces

PlugMem exposes three callers' contracts. Don't conflate them.

1. **`plugmem.core` library** — pure-function primitives (`extract_coding_memories`, value functions, `compute_source_boost`, `passes_metadata_filter`, `PROVENANCE_FIELDS`). No daemon needed. Use this when embedding PlugMem inside another runtime. Re-exports live in `plugmem/core/__init__.py`.
2. **HTTP API + daemon** — FastAPI service for cross-session persistence and multi-agent coordination. Routes documented under `plugmem/api/routes/`.
3. **MCP server + CLI** — agent integrations. Thin layers over (2). Do not duplicate logic that belongs in (1) or (2).

When adding a new capability, ask which layer owns it. Most new logic should land in `plugmem.core` and be *exposed* by (2) and (3).

## Storage backends

Two backends behind a shared duck-typed interface (alias `StorageBackend` in `plugmem/storage/__init__.py`):

- **`ChromaStorage`** — production default. Wired through `api/dependencies.py::build_chroma_storage`.
- **`SqliteVecStorage`** — **experimental**. Implementation lives at `plugmem/storage/sqlite_vec.py`; install with `pip install -e ".[sqlite-vec]"`. Selectable via `STORAGE_BACKEND=sqlite_vec` or through the init wizard. Stay on `chroma` for production until the experimental gates close.

SqliteVecStorage status (selecting via `STORAGE_BACKEND=sqlite_vec` / `cfg.storage_backend = "sqlite_vec"`):

- ✅ `build_sqlite_vec_storage` + `build_storage` selector in `api/dependencies.py`.
- ✅ `storage_backend` field on `PlugMemConfig` + CLI config; env var `STORAGE_BACKEND`.
- ✅ Smoke tests in `tests/test_storage_sqlite_vec.py` cover round-trip, vec-table upsert on `update_semantic`, method-signature parity vs `ChromaStorage`, DI dispatch.
- ✅ Init wizard step in `plugmem init` — user can select backend at setup.
- ⚠️ Still missing: full retrieval/promotion integration tests on this backend; large-graph benchmarks. Stay on `chroma` for production until those land.

vec0 quirk: virtual tables do **not** accept `INSERT OR REPLACE` on the rowid PK. Use `DELETE WHERE id=? ; INSERT INTO …` to upsert (see `update_semantic` and friends).

## Key Principles

- **All LLM calls go through `LLMClient`** (or `LLMRouter` for role-specific routing). Never call an API directly.
- **All storage goes through `ChromaStorage`**. Raw `chromadb` client construction is confined to `api/dependencies.py::build_chroma_storage` (the DI factory). Nothing else should import `chromadb` directly.
- **Routes call `get_llm()` / `get_embedder()` directly** (not via Depends) — this is a codebase-wide pattern. Tests manage state via `deps.reset_singletons()`.
- **Value functions** (`ValueBase`) control retrieval scoring. Pluggable per node type (tag, semantic, procedural, subgoal). Source-aware scoring boosts explicit corrections over inferred failure deltas via `compute_source_boost()`.
- **Promote endpoint** (`POST /graphs/{graph_id}/promote`) combines extraction + insertion atomically, including dedupe/upsert for repeated coding signals.

## Memory Graph Pipeline

1. **Structure**: `Memory.append(obs, action)` → Memory.close() — LLM extracts subgoals, states, rewards, semantic facts, procedures
2. **Insert**: `MemoryGraph.insert(memory)` — persists to ChromaDB, builds node links
3. **Promote** (coding-agent path): `POST /graphs/{graph_id}/promote` — accepts candidates, runs LLM extraction, inserts with dedupe/upsert, returns node IDs + rejection reasons
4. **Retrieve**: `MemoryGraph.retrieve_memory(observation)` — mode detection → embedding similarity + tag voting → value function scoring → prompt assembly
5. **Reason**: `MemoryGraph.retrieve_and_reason(observation)` — retrieve + LLM synthesis

## Source / Confidence Metadata

Nodes can carry `source` (where the memory came from: failure_delta, correction, merged, repeated_lookup, explicit) and `confidence` (0.0-1.0). Filtered at retrieval time via `min_confidence` and `source_in` params.

## Provenance Filtering

Retrieval requests (`/retrieve`, `/reason`, `/recall_trace`) accept optional `provenance_filters: Dict[str, List[str]]` to scope recall by provenance metadata (e.g. `{"language": ["python"]}` or `{"repo": ["org/repo"]}`). This is implemented via `_passes_metadata_filter` in `core/memory_graph.py:59` and threaded through `retrieve_semantic_nodes`, `retrieve_procedural_nodes`, and `retrieve_memory`.

## Coding Provenance

Semantic and procedural memories can carry optional provenance metadata for coding-agent use: `repo`, `branch`, `commit`, `language`, `filepath`, `package_manager`, `tool_name`, `tool_version`, `os`, `component`. Stored as `provenance_*` keys in ChromaDB metadata, round-tripped on load. The promote endpoint and extractor prompt both support provenance.

## Source-Aware Scoring

Value functions (`SemanticRelevant`, `ProceduralRelevant`) apply a confidence-scaled boost based on memory source via `compute_source_boost()`:
- `explicit` → +0.3×confidence
- `correction` → +0.25×confidence
- `failure_delta` → +0.1×confidence
- `merged` / `repeated_lookup` → +0.05×confidence
- `None` (legacy) → no boost

This ensures explicit user corrections outrank inferred failure deltas at retrieval time.

## Dedupe / Upsert

The promote endpoint uses `MemoryGraph._find_matching_semantic()` and `_find_matching_procedural()` to detect near-duplicates (text similarity ≥ 0.85 + same source). On match, confidence is bumped (max) and freshness is updated; no new node is created. Repeated corrections like "use uv, not pip" converge to a single node with increasing confidence.

## CLI

```bash
plugmem init      # Interactive setup wizard (LLM, embedding, service, coding profile)
plugmem start     # Start the daemon (uvicorn subprocess)
plugmem stop      # Stop the daemon (SIGTERM → SIGKILL)
plugmem restart   # Stop + start
plugmem status    # PID, URL, health flags
plugmem logs      # Tail or dump the daemon log
plugmem health    # One-shot /health check
plugmem coding scaffold    # Create a coding memory graph with language conventions
plugmem coding promote     # Promote coding signals into the graph from the CLI
plugmem coding recall      # Retrieve relevant memories via /reason or /retrieve
plugmem coding list        # List stored semantic or procedural nodes
```

Config lives at `$XDG_CONFIG_HOME/plugmem/config.toml`. Env vars override at runtime.
A `[coding]` section stores defaults for coding-agent operations (``default_graph``, ``default_repo``, ``source_filter``, ``min_confidence``, etc.).

## Commands

```bash
uv run pytest tests/    # pytest suite, ~13s
uv run mypy plugmem/    # must pass clean
uv run mypy claude-code-plugmem-plugin/   # MCP server mypy check
uv run plugmem --help   # CLI entry point
uv pip install -e ".[dev]"  # install with dev deps (pytest, mypy)
```

## Rules

- Add `max_length` to all new Pydantic string fields
- Never use `except:` — always `except Exception:` and log
- New routes must have at least a smoke test
- mypy must pass before merging
- AI agents: README.md, CLAUDE.md, and ONBOARDING.md are the three docs that must be kept in sync

## File Hotspots

| File | Lines | Why complex |
|------|-------|-------------|
| `core/memory_graph.py` | 1383 | Core business logic: insert, retrieve, reason, consolidate |
| `storage/chroma.py` | 625 | 30 CRUD functions across 5 node types + audit log |
| `api/routes/inspector.py` | 709 | Inspector UI backend + serializers |
| `api/schemas.py` | 382 | All request/response models |
