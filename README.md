# PlugMem
<p align="center">
  <img src="assets/plugmem_name_card.png" alt="PlugMem"/>
</p>

**PlugMem** is a **plug-and-play long-term memory system for LLM agents**. Instead of storing and retrieving raw interaction histories, PlugMem organizes experience into **compact, reusable knowledge units**, allowing agents to recall what matters to agent decision-making with minimal context overhead.

The module is **task-agnostic by design** and can be integrated into existing agent pipelines with minimal effort, serving as a general memory backbone for diverse environments such as dialogue agents, knowledge-intensive QA, and web automation. 

For more details, please see the full paper: [https://arxiv.org/abs/2603.03296](https://arxiv.org/abs/2603.03296)

<p align="center">
  <img src="assets/plugmem_pipeline.png" alt="PlugMem Pipeline"/>
</p>

## Table of Contents

- [Updates](#updates)
- [Features](#features)
  - [Plug-in](#plug-in)
  - [Memory](#memory)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Using the CLI (recommended)](#using-the-cli-recommended)
  - [Remote Deployment](#remote-deployment)
  - [Using environment variables](#using-environment-variables)
- [CLI Reference](#cli-reference)
- [Deployment Files](#deployment-files)
- [Reproducibility](#reproducibility)
- [Citation](#citation)

## Updates

- **[2026-05]** 🚀 **Plugin release** — PlugMem now ships as installable plugins for AI coding agents.
  Integrations available for **[OpenClaw](openclaw-plugmem-plugin/)** and **Claude Code** (see `plugin` branch).
  Highlights: inspect your memory graph, test retrieval interactively, and replay past agent sessions.

  <p align="center">
    <img src="assets/plugmem_promotion_headline.png" alt="PlugMem Plugin" width="700"/>
  </p>

- **[2026-05]** 🏆 **New SOTA on LongMemEval & HotpotQA** — With light task adaptation, PlugMem reaches
  **90.2 Acc** on LongMemEval and **79.1 F1 / 91.1% LLM-Judge Acc** on HotpotQA (multi-hop), both
  state-of-the-art results. Because the framework is task-agnostic, it can serve as a drop-in backbone for other work on these benchmarks. → [Step-by-step reproduction guide](examples/task-adaptation/)

- **[2026-04]** 🎉 **PlugMem accepted to ICML 2026!**

## Features
### Plug-in
- **Enhance your agent with 6 lines of code**
```python
# init PlugMem memory graph
mg = MemoryGraph()
# init memory sequence
mem = Memory(...)
mem.append(...)
mem.close()
# insert memory sequence into memory graph
mg.insert(mem)
# retrieve memory and perform reasoning on retrieved nodes
mg.retrieve_and_reason(...)
```
- **Easy to modify**: Apply adaptive strategies by defining different value functions and reasoning prompts.
- **Agent integrations**: Native plugins available for **[OpenClaw](openclaw-plugmem-plugin/)** and **Claude Code** (see `plugin` branch), with a built-in **Memory Inspector** UI for visualizing the memory graph, browsing individual memories, testing retrieval, and replaying agent trajectories.

<p align="center">
  <img src="assets/plugmem_memory_inspector.png" alt="Memory Inspector — Graph view" width="800"/>
  <br/><em>Graph view: explore the full memory graph across semantic, procedural, and episodic nodes</em>
</p>

<p align="center">
  <img src="assets/plugmem_memory_inspector_2.png" alt="Memory Inspector — Browse view" width="800"/>
  <br/><em>Browse view: inspect, filter, and manage individual memory entries</em>
</p>

### Memory
- **Three Memory Types**: 
  - **Semantic** (facts, concepts): User preferences, factual information
  - **Procedural** (workflows, procedures): How-to knowledge, step-by-step processes
  - **Episodic** (interaction sequences): Long interaction sessions stored on disk, referenced by ID
- **Graph Structure**: Maintain hierarchical knowledge units to illustrate the relationship between memories.
- **LLM Enhancement**: Use LLMs for intelligent knowledge extraction, memory retrieval, and reasoning
- **Memory Compression and Evolution**: Naively support updating and evolving the memory graph.

<p align="center">
  <img src="assets/plugmem_structuring.png" alt="PlugMem Structuring"/>
</p>

## Installation

### Service

```bash
uv sync
uv pip install -e ".[dev]"   # includes pytest, mypy
```

For a containerized server deployment, see
[docs/remote-deployment.md](docs/remote-deployment.md).

### Benchmarks (WebArena / LongMemEval / HotpotQA)

1. Install benchmarks in `src/` and follow their installation docs to set up the environment.
2. Install/upgrade `openai==2.6.1`.
3. Additional modifications:
- **WebArena**
```bash
# under src/
cd src
# clone modified AgentOccam
git clone https://github.com/jizej/AgentOccam
# clone 
git clone https://github.com/web-arena-x/webarena
# Enable Scriptbrowserenv to run under async loop (if needed)
cp src/webarena_patch/envs.py src/webarena/browser_env/envs.py
# Enable OPENAI_API_KEY + AZURE_ENDPOINT for trajectory evaluation (if needed)
cp src/webarena_patch/openai_utils.py src/webarena/llms/providers/openai_utils.py
```

## Quick Start

### Using the CLI (recommended)

```bash
# 1. Install
uv sync
uv pip install -e ".[dev]"

# 2. Interactive setup (detects Ollama, probes endpoints, writes config)
plugmem init

# 3. Start the service
plugmem start
# → Daemon started (PID 12345) on http://127.0.0.1:8080

# 4. Check health (or `plugmem status`, `plugmem logs`, `plugmem restart`, `plugmem stop`)
plugmem health

# 5. Create a memory graph
curl -X POST http://localhost:8080/api/v1/graphs \
  -H "Content-Type: application/json" \
  -d '{"graph_id":"my-agent"}'

# 6. Insert a memory
curl -X POST http://localhost:8080/api/v1/graphs/my-agent/memories \
  -H "Content-Type: application/json" \
  -d '{"mode":"structured","semantic":[{"semantic_memory":"User prefers async standups","tags":["preference"]}]}'

# 7. Insert many memories efficiently
curl -X POST http://localhost:8080/api/v1/graphs/my-agent/memories/batch \
  -H "Content-Type: application/json" \
  -d '{
        "items": [
          {
            "mode": "structured",
            "session_id": "run-A",
            "semantic": [{"semantic_memory": "User prefers async standups", "tags": ["preference"]}]
          },
          {
            "mode": "structured",
            "session_id": "run-B",
            "procedural": [{"subgoal": "communicate clearly", "procedural_memory": "Send concise async updates"}]
          }
        ]
      }'

# 8. Retrieve
curl -X POST http://localhost:8080/api/v1/graphs/my-agent/reason \
  -H "Content-Type: application/json" \
  -d '{"observation":"How does the user prefer to communicate?"}'
```

Use `/memories/batch` when you are ingesting many structured items at once.
It is structured-mode only and is much more efficient than issuing one HTTP
request per memory item.

### Remote Deployment

The supported remote path is a single-VM Docker Compose stack:

- [deploy/remote/compose.yaml](deploy/remote/compose.yaml)
- [deploy/remote/.env.example](deploy/remote/.env.example)
- [docs/remote-deployment.md](docs/remote-deployment.md)

Quick path:

```bash
git clone https://github.com/mrtkrcm/PlugMem.git /opt/plugmem
cd /opt/plugmem
cp deploy/remote/.env.example deploy/remote/.env
$EDITOR deploy/remote/.env
cd deploy/remote
docker compose --env-file .env up -d --build
curl http://127.0.0.1:${PLUGMEM_PORT:-8080}/api/v1/health
```

Production guidance:

- Set `PLUGMEM_API_KEY` and send it as `X-API-Key`
- Put the API behind a reverse proxy or cloud firewall
- Persist Chroma data using the `chroma_data` volume
- Keep the health endpoint path as `/api/v1/health`
- Prefer `/api/v1/graphs/{graph_id}/memories/batch` for high-volume structured ingestion

### Using environment variables

```bash
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_API_KEY="<your_llm_api_key>"
export LLM_MODEL="gpt-4o-mini"

# Optional: use OpenAI embeddings without a dedicated embedding server
export OPENAI_API_KEY="<your_openai_api_key>"

# Optional: dedicated embedding server
export EMBEDDING_BASE_URL="http://<your_embedding_host>:8001/v1/embeddings"
export EMBEDDING_API_KEY="<your_embedding_api_key>"
export EMBEDDING_MODEL="text-embedding-3-small"

# Optional: secure the API itself
export PLUGMEM_API_KEY="<your_service_api_key>"
```

Host local inference servers if you are not using a hosted LLM API:

```bash
cd host_local_inference
# Qwen (vLLM) server
bash vllm_deploy.sh
# NV-Embed-v2 server
bash nv_embed_v2_deploy.sh
```

Start the API directly:

```bash
uv run uvicorn plugmem.api.app:app --host 127.0.0.1 --port 8080
```

Run examples for different benchmarks:
   ### WebArena
   ```bash
   cd src/eval/webarena
   python eval_agentoccam.py
   ```
   Options for `eval_agentoccam.py`:
   - `--config`: Path to the YAML config file (required).
   - `--replay-trajectory/--no-replay-trajectory`: Replay a saved trajectory before evaluation.
   - `--trajectory-dir`: Directory containing trajectory JSON files for replay.
   - `--load_memory_graph/--no-load_memory_graph`: Load a persisted memory graph from disk.
   - `--refresh-embeddings/--no-refresh-embeddings`: Refresh embeddings when loading the memory graph.
   - `--read-only-memory/--no-read-only-memory`: Use the memory graph without inserting new memories.
   - `--disable-memory-graph/--no-disable-memory-graph`: Turn off all memory-graph operations.
   ### LongMemEval
   ```bash
   cd src/eval/longmemeval
   python eval_longmemeval_all.py
   ```
   ### HotpotQA
   ```bash
   cd src/eval/hotpotqa
   # It may take several hours to structure memory for hotpotqa_corpus.json.
   python build.py
   #Rebuild the memory graph from structuring result and run test
   python eval_hotpotqa_all.py
   ```

## CLI Reference

```text
Usage: plugmem [OPTIONS] COMMAND [ARGS]...

Commands:
  init      Interactive setup wizard for LLM, embedding, and service settings.
  start     Start the PlugMem service (daemonized by default).
  stop      Stop the running PlugMem daemon.
  restart   Restart the PlugMem daemon.
  status    Show daemon status, PID, port, and last health probe.
  logs      Print or tail the daemon log.
  health    One-shot health check against the running service.
```

The CLI uses XDG paths for config (`~/.config/plugmem/config.toml`), state
(PID file at `~/.local/state/plugmem/plugmem.pid`), and data
(`~/.local/share/plugmem/chroma/`). All config keys can be overridden at
runtime via environment variables — `LLM_API_KEY=sk-... plugmem start`.

## Deployment Files

- [Dockerfile](Dockerfile): production image for the API server
- [docker-compose.yml](docker-compose.yml): local two-container stack for quick development
- [deploy/remote/compose.yaml](deploy/remote/compose.yaml): server-oriented remote deployment stack
- [deploy/remote/.env.example](deploy/remote/.env.example): production env template
- [deploy/systemd/plugmem-compose.service](deploy/systemd/plugmem-compose.service): optional start-on-boot unit
- [docs/remote-deployment.md](docs/remote-deployment.md): end-to-end remote deployment guide

## High-Throughput Ingestion

PlugMem now supports batched structured ingestion:

```text
POST /api/v1/graphs/{graph_id}/memories/batch
```

Notes:

- Only `structured` items are supported in batch mode
- Each item may carry its own `session_id`
- Batched ingestion is significantly faster than one-request-per-item inserts
- Trajectory mode should continue to use `POST /api/v1/graphs/{graph_id}/memories`

### Development

```bash
uv run pytest tests/
uv run mypy plugmem/     # must pass clean
uv run plugmem --help    # CLI entry point
```

## Reproducibility
- We release agent trajectories and memory graph artifacts for all three tasks.
- We release human demonstrations used for WebArena (Under License CC BY 4.0).
- Data available in Google Drive: https://drive.google.com/drive/folders/15feC6xYsONJhJAb2n1kPjGrjSt0weHXi?usp=sharing

## Citation
If you use our code or data, or otherwise found our work helpful, please cite our paper:

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
