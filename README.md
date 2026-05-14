# PlugMem

**Plug-and-play long-term memory for LLM agents.** PlugMem organizes experience into compact, reusable knowledge units instead of raw interaction histories.

For more details, see the paper: [https://arxiv.org/abs/2603.03296](https://arxiv.org/abs/2603.03296)

<p align="center">
  <img src="assets/plugmem_pipeline.png" alt="PlugMem Pipeline"/>
</p>

## Quick Start

### Using the CLI (recommended)

```bash
# 1. Setup (see Installation below)
plugmem init                    # interactive wizard (LLM, embedding, etc.)

# 2. Start the daemon
plugmem start

# 3. Create a graph, insert a memory, retrieve
curl -X POST http://localhost:8080/api/v1/graphs -H "Content-Type: application/json" \
  -d '{"graph_id":"my-agent"}'

curl -X POST http://localhost:8080/api/v1/graphs/my-agent/memories -H "Content-Type: application/json" \
  -d '{"mode":"structured","semantic":[{"semantic_memory":"User prefers async standups","tags":["preference"]}]}'

curl -X POST http://localhost:8080/api/v1/graphs/my-agent/reason -H "Content-Type: application/json" \
  -d '{"observation":"How does the user prefer to communicate?"}'
```

### Library mode (no daemon)

If you own your own loop and don't need the HTTP service, the scoring + extraction primitives are importable directly. No FastAPI, no daemon, no CLI.

```python
import requests
from plugmem.clients.llm import OpenAICompatibleLLMClient
from plugmem.core import (
    extract_coding_memories, SemanticRelevant, ProceduralRelevant,
    passes_metadata_filter, get_similarity,
)

# 1. Extract durable memories from coding-agent signals
llm = OpenAICompatibleLLMClient(
    base_url="https://api.openai.com/v1", api_key="sk-...", model="gpt-4o-mini",
)

memories, rejected = extract_coding_memories(llm, candidates=[
    {"kind": "correction", "window": "use httpx, not requests"},
])

# 2. Or fetch existing nodes (requires daemon running)
res = requests.get("http://localhost:8080/api/v1/graphs/my-agent/nodes?node_type=semantic")
nodes = res.json().get("nodes", [])

# 3. Score and rank by relevance + source
query_emb = [0.1, 0.2, ...]          # from your embedder
candidate_emb = [0.3, 0.4, ...]      # from the node
similarity = get_similarity(query_emb, candidate_emb)
rank = SemanticRelevant().evaluate(Relevance=similarity, Source="correction", Confidence=0.9)

# 4. Filter by provenance / source / confidence
filtered = [
    n for n in nodes
    if passes_metadata_filter(n, min_confidence=0.7, source_in=["correction"])
]
```

### Claude Code

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "plugmem": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/PlugMem/claude-code-plugmem-plugin", "server.py"],
      "env": {
        "PLUGMEM_BASE_URL": "http://127.0.0.1:8080",
        "PLUGMEM_API_KEY": "dev-key-change-me",
        "PLUGMEM_DEFAULT_GRAPH": "coding-agent"
      }
    }
  }
}
```

Tools: `plugmem_remember`, `plugmem_recall`, `plugmem_promote`. Graph auto-creates on first use.

`plugmem_recall` supports the same provenance-scoped coding recall model as the CLI and HTTP API. In addition to `source_in` and `min_confidence`, you can filter recall by any of:
`repo`, `branch`, `commit`, `language`, `filepath`, `package_manager`, `tool_name`, `tool_version`, `os`, `component`.

Example MCP tool call shape:

```json
{
  "name": "plugmem_recall",
  "arguments": {
    "observation": "how do we run Python tests here?",
    "language": "python",
    "repo": "org/my-lib",
    "tool_name": "pytest",
    "package_manager": "uv",
    "min_confidence": 0.7
  }
}
```

## Features

- **Three memory types**: Semantic (facts), Procedural (workflows), Episodic (sequences)
- **Graph structure**: Hierarchical knowledge units with typed relationships
- **LLM enhancement**: Intelligent extraction, retrieval, and reasoning
- **Source-aware scoring**: Explicit corrections rank higher than inferred deltas
- **Provenance metadata**: Scope recall by repo, language, branch, etc.
- **Agent integrations**: Plugins for [Claude Code](claude-code-plugmem-plugin/) and [OpenClaw](openclaw-plugmem-plugin/)

## Installation

```bash
uv sync
uv pip install -e ".[dev]"                           # core + dev deps
uv pip install -e ".[sqlite-vec]"                    # optional: sqlite-vec backend
```

---

## Reference

### CLI Reference

```
Usage: plugmem [OPTIONS] COMMAND [ARGS]...

Commands:
  init      Interactive setup wizard (LLM, embedding, service, coding profile).
  start     Start the PlugMem daemon.
  stop      Stop the running daemon.
  restart   Restart the daemon.
  status    Show daemon status, PID, port, last health probe.
  logs      Print or tail the daemon log.
  health    One-shot health check.
  coding    Coding-agent memory commands (attach, scaffold, promote, recall, list).
```

Config: `~/.config/plugmem/config.toml` (XDG). All keys overridable via env vars.

### Coding-Agent CLI

Recommended onboarding for a real repo:

```bash
plugmem coding attach /path/to/repo
```

`attach` inspects the repository, updates the local coding profile, health-checks the daemon, scaffolds the project graph, ingests durable repo guidance from files like `AGENTS.md`, `README.md`, `Justfile`, `Package.swift`, `pyproject.toml`, and `package.json`, and runs a sample recall to verify the setup.

```
plugmem coding attach   /path/to/repo                    # inspect + configure + scaffold + ingest guidance
plugmem coding scaffold --graph my-agent --language python  # create graph with conventions
plugmem coding promote  --graph my-agent --kind correction  # promote signals into memory
plugmem coding recall   --graph my-agent "how to install deps"   # retrieve with reasoning
plugmem coding list     --graph my-agent --type semantic         # browse stored nodes
```

All subcommands accept `--graph` (or env `CODING_DEFAULT_GRAPH`).

Example workflow:

```bash
plugmem init                                    # include coding profile
plugmem start

plugmem coding attach ~/code/my-project
# → Detects repo/language/tooling, updates the coding profile, scaffolds a graph,
#   ingests repo guidance, and verifies recall

plugmem coding scaffold --language python
# → Lower-level graph bootstrap; usually not needed if you used `attach`

plugmem coding promote --kind correction --window "use httpx, not requests"
# → LLM extracts semantic memory, inserts with dedupe

plugmem coding promote --kind failure_delta --window "pip install failed → uv sync succeeded"
# → LLM extracts procedural memory

plugmem coding promote --from-file candidates.txt --json   # batch from file, JSON output
# → One candidate per line, each becomes a `--kind correction` candidate

plugmem coding recall "how to install deps" --language python
# → LLM-synthesized reasoning grounded in Python-relevant memories

plugmem coding list --type semantic --language python --source explicit
plugmem coding list --type procedural --query "uv" --limit 20   # substring filter, cap results
```

The `init` wizard prepopulates `[coding]` defaults in `config.toml`: `source_filter`, `min_confidence`, `default_graph`, `default_repo`, `default_language`, `default_package_manager`. Override at runtime: `CODING_DEFAULT_GRAPH=my-graph plugmem coding promote ...`.

`plugmem coding scaffold` is idempotent for convention seeding:
- If the graph does not exist, it is created and seeded.
- If the graph already exists, PlugMem keeps it and inserts only conventions that are not already present.
- Use `plugmem coding scaffold --force` only when you explicitly want to delete and recreate the graph before reseeding.

`plugmem coding attach` is the preferred setup path for introducing a new project:
- Detects repo identity (`repo`, `branch`) from git
- Detects project language and package manager from common build files
- Chooses a graph id automatically unless you pass `--graph`
- Updates the local `[coding]` profile in `~/.config/plugmem/config.toml`
- Ingests durable project guidance from repo-local docs/build files
- Verifies the live daemon with a sample recall

If the embedding backend is unavailable, `attach` still updates the coding profile and scaffolds what it can, but it will skip guidance ingestion until embeddings are working again.

### Coding-Agent Promotion

Atomic extract + insert:

```bash
curl -X POST http://localhost:8080/api/v1/graphs/my-agent/promote \
  -H "Content-Type: application/json" \
  -d '{
    "candidates": [
      {"kind": "correction", "window": "user said: use uv, not pip"},
      {"kind": "failure_delta", "window": "pip install failed → uv sync succeeded"}
    ]
  }'
# → {"inserted": [...], "dropped": [...]}
```

Optional filters: `source_in: ["correction"]`, `min_confidence: 0.7`.

Extraction only (no insert):

```bash
curl -X POST http://localhost:8080/api/v1/extract \
  -H "Content-Type: application/json" \
  -d '{"candidates": [{"kind": "correction", "window": "use httpx, not requests"}]}'
```

Provenance metadata on insert:

```bash
curl -X POST http://localhost:8080/api/v1/graphs/my-agent/memories \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "structured",
    "semantic": [{
      "semantic_memory": "Use uv, not pip",
      "tags": ["python", "tooling"],
      "source": "correction",
      "confidence": 0.9,
      "provenance": {"repo": "org/repo", "language": "python", "tool_name": "uv"}
    }]
  }'
```

Provenance fields (all optional strings): `repo`, `branch`, `commit`, `language`, `filepath`, `package_manager`, `tool_name`, `tool_version`, `os`, `component`.

Source-aware retrieval:

```bash
# Filter by source kind + confidence
curl -X POST http://localhost:8080/api/v1/graphs/my-agent/retrieve \
  -H "Content-Type: application/json" \
  -d '{"observation": "how to install deps", "mode": "procedural_memory", "source_in": ["correction"], "min_confidence": 0.5}'

# Filter by provenance
curl -X POST http://localhost:8080/api/v1/graphs/my-agent/reason \
  -H "Content-Type: application/json" \
  -d '{"observation": "how to format imports", "provenance_filters": {"language": ["python"], "repo": ["org/my-lib"]}}'
```

Deduplication: The promote endpoint detects near-duplicates (text similarity ≥ 0.85 + same source). Repeated corrections bump confidence on the existing node.

### High-Throughput Ingestion

```bash
curl -X POST http://localhost:8080/api/v1/graphs/my-agent/memories/batch \
  -H "Content-Type: application/json" \
  -d '{"items": [{"mode": "structured", "session_id": "run-A", "semantic": [{"semantic_memory": "...", "tags": ["..."]}]}]}'
```

Batch is structured-mode only, significantly faster than one-per-item.

### Remote Deployment

Single-VM Docker Compose stack:

```bash
git clone https://github.com/mrtkrcm/PlugMem.git /opt/plugmem
cp deploy/remote/.env.example deploy/remote/.env
$EDITOR deploy/remote/.env
cd deploy/remote
docker compose --env-file .env up -d --build
```

Production guidance:
- Set `PLUGMEM_API_KEY`, send as `X-API-Key` header
- Put behind a reverse proxy (Caddy, Nginx, Traefik) for TLS
- Persist Chroma data via the `chroma_data` volume
- Health endpoint: `GET /api/v1/health`

### Environment Variables

```bash
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_API_KEY="sk-..."
export LLM_MODEL="gpt-4o-mini"

export OPENAI_API_KEY="sk-..."                                           # embeddings fallback
export EMBEDDING_BASE_URL="http://<host>:8001/v1/embeddings"             # dedicated embedder
export EMBEDDING_MODEL="text-embedding-3-small"                          # if different from LLM

export STORAGE_BACKEND="chroma"                                          # "chroma" | "sqlite_vec"
export SQLITE_VEC_PATH="./data/plugmem.db"                               # for sqlite-vec backend
export CHROMA_PATH="./data/chroma"                                       # for chroma persistent mode

export PLUGMEM_API_KEY="change-me"                                       # API auth
```

Full list: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_MAX_RETRIES`, `LLM_TEMPERATURE`, `LLM_TOP_P`, `LLM_MAX_TOKENS`, `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY`, `EMBEDDING_MODEL`, `EMBEDDING_MAX_TEXT_LEN`, `OPENAI_API_KEY`, `STORAGE_BACKEND`, `SQLITE_VEC_PATH`, `CHROMA_PATH`, `PLUGMEM_API_KEY`, `TOKEN_USAGE_FILE`, `CODING_*` keys for coding defaults.

### Co-deploying with CocoIndex-code

PlugMem (experience) + [CocoIndex-code](https://github.com/cocoindex-io/cocoindex-code) (code structure) are orthogonal and designed to co-deploy.

Register both MCP servers:

```json
{
  "mcpServers": {
    "plugmem": { "command": "uv", "args": ["run", "--directory", "/path/to/PlugMem/claude-code-plugmem-plugin", "server.py"], "env": { "PLUGMEM_BASE_URL": "http://127.0.0.1:8080", "PLUGMEM_API_KEY": "dev-key-change-me", "PLUGMEM_DEFAULT_GRAPH": "coding-agent" } },
    "cocoindex-code": { "command": "ccc", "args": ["mcp"] }
  }
}
```

```bash
# One-time setup
uv pip install -e ".[dev]"
plugmem init && plugmem start
uv tool install 'cocoindex-code[full]' && ccc setup
```

Rule of thumb: *"How did we fix this before?"* → PlugMem. *"What calls X?"* → CocoIndex-code.

### Scope

|                | PlugMem                                          | CocoIndex-code                                   |
| -------------- | ------------------------------------------------ | ------------------------------------------------ |
| Memory type    | Episodic + semantic (experience-based)           | Epistemic (knowledge about code)                 |
| Source         | Agent interactions (corrections, failures)       | The codebase (git, AST)                          |
| Stores         | Corrections, procedures, preferences             | Symbols, call graphs, chunks                     |
| Answers        | *"How did we fix this last time?"*               | *"What calls X? Where is Y defined?"*            |
| Query surface  | `/retrieve`, `/reason`, `plugmem coding recall`  | `codebase_search`, `codebase_symbol`, ...         |

### Benchmark Reproduction

Task adaptation guides: [examples/task-adaptation/](examples/task-adaptation/)

**WebArena:**
```bash
cd src/eval/webarena
python eval_agentoccam.py --config <config.yaml>
```

**LongMemEval:**
```bash
cd src/eval/longmemeval
python eval_longmemeval_all.py
```

**HotpotQA:**
```bash
cd src/eval/hotpotqa
python build_mem.py
python eval_qa_all.py
```

### Updates

- **2026-05** — Plugin release (OpenClaw, Claude Code). Coding-agent CLI. New SOTA on LongMemEval/HotpotQA.
- **2026-04** — Accepted to ICML 2026.

### Deployment Files

- [Dockerfile](Dockerfile): production image
- [docker-compose.yml](docker-compose.yml): local two-container stack
- [deploy/remote/compose.yaml](deploy/remote/compose.yaml): remote deployment stack
- [deploy/remote/.env.example](deploy/remote/.env.example): production env template
- [deploy/systemd/plugmem-compose.service](deploy/systemd/plugmem-compose.service): start-on-boot unit
- [docs/remote-deployment.md](docs/remote-deployment.md): deployment guide

### Development

```bash
uv run pytest tests/
uv run mypy plugmem/
uv run plugmem --help
```

### Reproducibility

Agent trajectories, memory graph artifacts, and human demonstrations for WebArena available at [Google Drive](https://drive.google.com/drive/folders/15feC6xYsONJhJAb2n1kPjGrjSt0weHXi?usp=sharing) (CC BY 4.0).

### Citation

```
@misc{yang2026plugmemtaskagnosticpluginmemory,
      title={PlugMem: A Task-Agnostic Plugin Memory Module for LLM Agents}, 
      author={Ke Yang and Zixi Chen and Xuan He and Jize Jiang and Michel Galley and Chenglong Wang and Jianfeng Gao and Jiawei Han and ChengXiang Zhai},
      year={2026},
      eprint={2603.03296},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.03296}, 
}
```
