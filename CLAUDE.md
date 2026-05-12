# PlugMem — Agent Context

## Architecture

```
plugmem/
  api/            FastAPI routes (graphs, memories, retrieval, extract, inspector, demo)
  cli/            Typer CLI (init, start, stop, restart, status, logs, health)
  clients/        LLM client (OpenAI-compatible), embedding client, LLM router
  core/           Memory, MemoryGraph, graph node types, value functions, normalization
  inference/      LLM-powered structuring and retrieval logic
  prompts/        Prompt templates for structuring, retrieving, reasoning
  storage/        ChromaDB persistence wrapper
  config.py       Pydantic config model
  graph_manager.py
tests/            pytest suite (87 tests)
```

## Key Principles

- **All LLM calls go through `LLMClient`** (or `LLMRouter` for role-specific routing). Never call an API directly.
- **All storage goes through `ChromaStorage`**. No direct ChromaDB client usage outside it.
- **Routes call `get_llm()` / `get_embedder()` directly** (not via Depends) — this is a codebase-wide pattern. Tests manage state via `deps.reset_singletons()`.
- **Value functions** (`ValueBase`) control retrieval scoring. Pluggable per node type (tag, semantic, procedural, subgoal).

## Memory Graph Pipeline

1. **Structure**: `Memory.append(obs, action)` → Memory.close() — LLM extracts subgoals, states, rewards, semantic facts, procedures
2. **Insert**: `MemoryGraph.insert(memory)` — persists to ChromaDB, builds node links
3. **Retrieve**: `MemoryGraph.retrieve_memory(observation)` — mode detection → embedding similarity + tag voting → value function scoring → prompt assembly
4. **Reason**: `MemoryGraph.retrieve_and_reason(observation)` — retrieve + LLM synthesis

## Source / Confidence Metadata

Nodes can carry `source` (where the memory came from: failure_delta, correction, merged, repeated_lookup, explicit) and `confidence` (0.0-1.0). Filtered at retrieval time via `min_confidence` and `source_in` params.

## CLI

```bash
plugmem init      # Interactive setup wizard (LLM, embedding, service config)
plugmem start     # Start the daemon (uvicorn subprocess)
plugmem stop      # Stop the daemon (SIGTERM → SIGKILL)
plugmem restart   # Stop + start
plugmem status    # PID, URL, health flags
plugmem logs      # Tail or dump the daemon log
plugmem health    # One-shot /health check
```

Config lives at `$XDG_CONFIG_HOME/plugmem/config.toml`. Env vars override at runtime.

## Commands

```bash
uv run pytest tests/    # 86 tests, ~13s
uv run mypy plugmem/    # must pass clean
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
| `api/schemas.py` | 381 | All request/response models |
