# Remote Deployment

This document describes the supported remote deployment path for PlugMem on a
single Linux VM using Docker Compose.

## What you get

- One `plugmem` container serving the API on port `8080`
- One `chroma` container for persistent vector storage
- A production env file template at [deploy/remote/.env.example](../deploy/remote/.env.example)
- A server-ready Compose stack at [deploy/remote/compose.yaml](../deploy/remote/compose.yaml)
- An optional systemd unit at [deploy/systemd/plugmem-compose.service](../deploy/systemd/plugmem-compose.service)

## Prerequisites

- Ubuntu 22.04+ or another modern Linux distribution
- Docker Engine with the Compose plugin installed
- A reachable LLM backend:
  - OpenAI / OpenRouter / Azure-compatible API, or
  - a self-hosted OpenAI-compatible server such as vLLM
- Optional: a dedicated embeddings server

If you do **not** configure `EMBEDDING_BASE_URL`, PlugMem uses:

1. `OPENAI_API_KEY` fallback for OpenAI embeddings, or
2. the built-in deterministic embedder if neither is configured

The deterministic embedder is suitable for demos and smoke tests, not for real
retrieval quality.

## Deploy on a VM

1. Clone the repository onto the server.

```bash
git clone https://github.com/mrtkrcm/PlugMem.git /opt/plugmem
cd /opt/plugmem
```

2. Prepare the deployment env file.

```bash
cp deploy/remote/.env.example deploy/remote/.env
$EDITOR deploy/remote/.env
```

Minimum production settings:

- Set `PLUGMEM_API_KEY` to a strong random secret
- Set `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL`
- Optionally set `OPENAI_API_KEY` or `EMBEDDING_BASE_URL`

3. Start the stack.

```bash
cd /opt/plugmem/deploy/remote
docker compose --env-file .env up -d --build
```

4. Verify health.

```bash
curl http://127.0.0.1:${PLUGMEM_PORT:-8080}/api/v1/health
```

5. Verify authenticated API access.

```bash
curl -X POST http://127.0.0.1:${PLUGMEM_PORT:-8080}/api/v1/graphs \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${PLUGMEM_API_KEY}" \
  -d '{"graph_id":"remote-smoke"}'
```

6. Verify high-throughput ingestion.

```bash
curl -X POST http://127.0.0.1:${PLUGMEM_PORT:-8080}/api/v1/graphs/remote-smoke/memories/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${PLUGMEM_API_KEY}" \
  -d '{
        "items": [
          {
            "mode": "structured",
            "session_id": "deploy-run-1",
            "semantic": [{"semantic_memory": "User prefers async communication.", "tags": ["preference", "communication"]}]
          },
          {
            "mode": "structured",
            "session_id": "deploy-run-2",
            "procedural": [{"subgoal": "communicate clearly", "procedural_memory": "Send concise async updates with blockers and next steps."}]
          }
        ]
      }'
```

## Reverse proxy

The Compose stack publishes the PlugMem API directly on `${PLUGMEM_PORT}`.

Recommended production pattern:

- Bind the published port only on a private interface or behind a cloud firewall
- Put Caddy, Nginx, or Traefik in front of PlugMem for TLS termination
- Keep `PLUGMEM_API_KEY` enabled even behind the reverse proxy

PlugMem’s health endpoint is:

```text
/api/v1/health
```

The main API base is:

```text
/api/v1
```

For high-volume structured ingestion, prefer:

```text
/api/v1/graphs/{graph_id}/memories/batch
```

Batch mode is intended for structured payloads only. Trajectory-mode inserts
should continue to use the single-item `/memories` endpoint.

The inspector SPA is served at:

```text
/inspector/
```

## Persistence and upgrades

- Chroma data is stored in the named Docker volume `chroma_data`
- Rebuild/redeploy after pulling updates:

```bash
cd /opt/plugmem
git pull
cd deploy/remote
docker compose --env-file .env up -d --build
```

- View logs:

```bash
docker compose --env-file .env logs -f plugmem
docker compose --env-file .env logs -f chroma
```

## Start on boot with systemd

If you want the stack to start automatically after reboot:

```bash
sudo cp /opt/plugmem/deploy/systemd/plugmem-compose.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now plugmem-compose
```

Check status:

```bash
sudo systemctl status plugmem-compose
```

## Operational notes

- `plugmem` is stateless except for Chroma; back up the `chroma_data` volume
- `plugmem` currently assumes a single-process service for monotonic recall IDs
- If you scale the API horizontally, revisit recall audit ID allocation and
  persistence semantics first
