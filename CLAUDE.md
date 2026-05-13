# PlugMem — Agent Context

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
