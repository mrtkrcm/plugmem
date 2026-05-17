# PlugMem

PlugMem is a pluggable long-term memory service for LLM agents. It stores durable experience as structured semantic, procedural, and episodic memories instead of replaying full transcripts on every turn.

Paper: [PlugMem: A Task-Agnostic Plugin Memory Module for LLM Agents](https://arxiv.org/abs/2603.03296)

<p align="center">
  <img src="assets/plugmem_pipeline.png" alt="PlugMem pipeline"/>
</p>

## What ships in this repo

- A FastAPI memory service with graph, insert, retrieve, reason, promote, and inspector routes
- A local CLI for setup, daemon management, and coding-agent workflows
- A Claude Code MCP server in [claude-code-plugmem-plugin/](claude-code-plugmem-plugin/)
- An OpenClaw plugin in [openclaw-plugmem-plugin/](openclaw-plugmem-plugin/)
- An inspector SPA served from `/inspector/`
- Chroma storage by default, plus an experimental `sqlite_vec` backend

## Core capabilities

- Three memory types: semantic facts, procedural workflows, episodic trajectories
- Graph-scoped storage so each agent, user, or project can keep its own namespace
- Coding-memory promotion from corrections, failure deltas, repeated lookups, and explicit guidance
- Provenance-aware recall filters such as `repo`, `branch`, `language`, `filepath`, `tool_name`, and `component`
- Source-aware ranking and confidence thresholds for retrieval
- High-throughput structured ingestion through `/api/v1/graphs/{graph_id}/memories/batch`
- Direct library imports for extraction, ranking, and filtering primitives

## Quick Start

### 1. Install

```bash
uv sync
uv pip install -e ".[dev]"
```

Optional storage backend:

```bash
uv pip install -e ".[sqlite-vec]"
```

### 2. Initialize local config

```bash
plugmem init
```

The wizard writes your local config to `~/.config/plugmem/config.toml` and can configure:

- LLM endpoint
- Embedding endpoint
- Service host, port, and API key
- Storage backend
- Coding-agent defaults

### 3. Start the service

```bash
plugmem start
plugmem health
```

The API base is `http://127.0.0.1:8080/api/v1` by default.

### 4. Create a graph and insert memory

```bash
curl -X POST http://127.0.0.1:8080/api/v1/graphs \
  -H "Content-Type: application/json" \
  -d '{"graph_id":"my-agent"}'

curl -X POST http://127.0.0.1:8080/api/v1/graphs/my-agent/memories \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "structured",
    "semantic": [
      {
        "semantic_memory": "User prefers async standups",
        "tags": ["preference", "communication"],
        "source": "explicit",
        "confidence": 0.95
      }
    ]
  }'
```

### 5. Retrieve with reasoning

```bash
curl -X POST http://127.0.0.1:8080/api/v1/graphs/my-agent/reason \
  -H "Content-Type: application/json" \
  -d '{"observation":"How does the user prefer to communicate?"}'
```

### 6. Open the inspector

Point a browser at:

```text
http://127.0.0.1:8080/inspector/
```

The inspector is useful for browsing graphs, nodes, recall traces, and sessions without writing ad hoc curl commands.

## Coding-Agent Workflow

The current first-class workflow in this repo is `plugmem coding`.

### Recommended setup path

```bash
plugmem init
plugmem start
plugmem coding attach /path/to/repo
```

`attach` does the following:

- detects repo identity and branch from git when available
- detects language, package manager, and project profile from common build files
- updates the local coding profile in `~/.config/plugmem/config.toml`
- health-checks the running PlugMem service
- creates or reuses a graph
- seeds language conventions
- ingests durable repo guidance from files such as `AGENTS.md`, `README.md`, `Justfile`, `Package.swift`, `pyproject.toml`, and `package.json`
- verifies setup with a sample recall

### Main coding commands

```bash
plugmem coding attach /path/to/repo
plugmem coding scaffold --graph my-agent --language python
plugmem coding promote --graph my-agent --kind correction --window "use uv, not pip"
plugmem coding recall --graph my-agent "how do we install deps here?"
plugmem coding list --graph my-agent --type semantic --language python
```

### Promotion examples

```bash
plugmem coding promote --kind correction --window "use httpx, not requests"
plugmem coding promote --kind failure_delta --window "pip install failed, uv sync succeeded"
plugmem coding promote --from-file candidates.txt --json
```

### Recall examples

```bash
plugmem coding recall "how do we run tests here?" --language python
plugmem coding recall "how do we install deps?" --raw
plugmem coding list --type procedural --query "uv" --limit 20
```

### Coding provenance filters

Recall and browse can be scoped by source and provenance metadata. Common fields:

- `repo`
- `branch`
- `commit`
- `language`
- `filepath`
- `package_manager`
- `tool_name`
- `tool_version`
- `os`
- `component`

## Claude Code MCP Server

The MCP server lives in [claude-code-plugmem-plugin/server.py](claude-code-plugmem-plugin/server.py).

Add this to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "plugmem": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/absolute/path/to/PlugMem/claude-code-plugmem-plugin",
        "server.py"
      ],
      "env": {
        "PLUGMEM_BASE_URL": "http://127.0.0.1:8080",
        "PLUGMEM_API_KEY": "dev-key-change-me",
        "PLUGMEM_DEFAULT_GRAPH": "coding-agent"
      }
    }
  }
}
```

Current MCP tools:

- `plugmem_remember`
- `plugmem_recall`
- `plugmem_promote`
- `plugmem_browse`

Behavior notes:

- the server auto-creates the configured graph on first use if it does not exist
- `plugmem_recall` supports source filters, confidence thresholds, and full provenance scoping
- `plugmem_browse` is intended for debugging stored semantic and procedural memories

## OpenClaw Plugin

The OpenClaw plugin lives in [openclaw-plugmem-plugin/](openclaw-plugmem-plugin/). Full setup is documented in [openclaw-plugmem-plugin/ONBOARDING.md](openclaw-plugmem-plugin/ONBOARDING.md).

Highlights:

- registers `plugmem.remember` and `plugmem.recall`
- auto-saves trajectories on session reset and before compaction
- supports shared read-only graphs for cross-agent semantic memory fan-in

## API Surface

Main routes under `/api/v1`:

- `GET /health`
- `POST /graphs`
- `GET /graphs`
- `GET /graphs/{graph_id}`
- `DELETE /graphs/{graph_id}`
- `GET /graphs/{graph_id}/stats`
- `GET /graphs/{graph_id}/nodes`
- `POST /graphs/{graph_id}/memories`
- `POST /graphs/{graph_id}/memories/batch`
- `POST /graphs/{graph_id}/retrieve`
- `POST /graphs/{graph_id}/reason`
- `POST /graphs/{graph_id}/promote`
- `POST /graphs/{graph_id}/consolidate`
- `POST /extract`

### Promotion endpoint

```bash
curl -X POST http://127.0.0.1:8080/api/v1/graphs/my-agent/promote \
  -H "Content-Type: application/json" \
  -d '{
    "candidates": [
      {"kind": "correction", "window": "user said: use uv, not pip"},
      {"kind": "failure_delta", "window": "pip install failed, uv sync succeeded"}
    ]
  }'
```

### Extraction-only endpoint

```bash
curl -X POST http://127.0.0.1:8080/api/v1/extract \
  -H "Content-Type: application/json" \
  -d '{
    "candidates": [
      {"kind": "correction", "window": "use httpx, not requests"}
    ]
  }'
```

### Structured batch ingestion

```bash
curl -X POST http://127.0.0.1:8080/api/v1/graphs/my-agent/memories/batch \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "mode": "structured",
        "session_id": "run-A",
        "semantic": [
          {"semantic_memory": "User prefers async communication", "tags": ["preference"]}
        ]
      }
    ]
  }'
```

## Library Mode

If you do not want to run the daemon, the scoring and extraction primitives are importable directly.

```python
import requests
from plugmem.clients.llm import OpenAICompatibleLLMClient
from plugmem.core import (
    ProceduralRelevant,
    SemanticRelevant,
    extract_coding_memories,
    get_similarity,
    passes_metadata_filter,
)

llm = OpenAICompatibleLLMClient(
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    model="gpt-4o-mini",
)

memories, rejected = extract_coding_memories(llm, candidates=[
    {"kind": "correction", "window": "use httpx, not requests"},
])

res = requests.get("http://127.0.0.1:8080/api/v1/graphs/my-agent/nodes?node_type=semantic")
nodes = res.json().get("nodes", [])

query_emb = [0.1, 0.2, 0.3]
candidate_emb = [0.3, 0.4, 0.5]
similarity = get_similarity(query_emb, candidate_emb)
rank = SemanticRelevant().evaluate(Relevance=similarity, Source="correction", Confidence=0.9)

filtered = [
    n for n in nodes
    if passes_metadata_filter(n, min_confidence=0.7, source_in=["correction"])
]
```

## Configuration

Common environment variables:

```bash
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_API_KEY="sk-..."
export LLM_MODEL="gpt-4o-mini"

export EMBEDDING_BASE_URL="http://<host>:8001/v1/embeddings"
export EMBEDDING_API_KEY="sk-..."
export EMBEDDING_MODEL="text-embedding-3-small"
export OPENAI_API_KEY="sk-..."

export STORAGE_BACKEND="chroma"
export CHROMA_PATH="./data/chroma"
export SQLITE_VEC_PATH="./data/plugmem.db"

export PLUGMEM_API_KEY="change-me"
```

Notes:

- if `EMBEDDING_BASE_URL` is unset, PlugMem can fall back to OpenAI embeddings via `OPENAI_API_KEY`
- if neither embedding source is configured, the service falls through to a deterministic local embedder suitable for demos and smoke tests, not production retrieval quality
- `sqlite_vec` is available but still experimental

## Deployment

For remote deployment on a single Linux VM, use [docs/remote-deployment.md](docs/remote-deployment.md).

Relevant deployment assets:

- [Dockerfile](Dockerfile)
- [docker-compose.yml](docker-compose.yml)
- [deploy/remote/compose.yaml](deploy/remote/compose.yaml)
- [deploy/remote/.env.example](deploy/remote/.env.example)
- [deploy/systemd/plugmem-compose.service](deploy/systemd/plugmem-compose.service)

## Benchmarks and Research Assets

This repo also includes:

- task-adaptation guides in [examples/task-adaptation/](examples/task-adaptation/)
- benchmark tooling in [scripts/bench/](scripts/bench/)
- evaluation code under `src/eval/`

The token-usage benchmark comparing PlugMem against transcript replay is documented in [scripts/bench/README.md](scripts/bench/README.md).

## Development

```bash
uv run pytest tests/
uv run mypy plugmem/
uv run plugmem --help
```

## Citation

```bibtex
@misc{yang2026plugmemtaskagnosticpluginmemory,
  title={PlugMem: A Task-Agnostic Plugin Memory Module for LLM Agents},
  author={Ke Yang and Zixi Chen and Xuan He and Jize Jiang and Michel Galley and Chenglong Wang and Jianfeng Gao and Jiawei Han and ChengXiang Zhai},
  year={2026},
  eprint={2603.03296},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2603.03296}
}
```
