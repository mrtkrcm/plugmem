#!/usr/bin/env python3
"""MCP server providing PlugMem tools to Claude Code.

Usage (in Claude Code MCP config):
  {
    "mcpServers": {
      "plugmem": {
        "command": "uv",
        "args": ["run", "--directory", "/path/to/claude-code-plugmem-plugin", "server.py"]
      }
    }
  }

Provides tools:
  - plugmem_remember     Store memories (semantic or trajectory) with optional provenance.
  - plugmem_recall       Retrieve relevant memories with source/confidence filtering.
  - plugmem_promote      Promote coding signals (corrections, failure deltas) into memory.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
logger = logging.getLogger("plugmem-mcp")

# ------------------------------------------------------------------ #
# MCP protocol messages — newline-delimited JSON per MCP stdio spec
# ------------------------------------------------------------------ #

JSON_RPC_VERSION = "2.0"


def _rpc_request(method: str, params: Optional[Dict[str, Any]] = None, _id: int = 1) -> Dict[str, Any]:
    return {"jsonrpc": JSON_RPC_VERSION, "id": _id, "method": method, "params": params or {}}


def _rpc_result(result: Any, _id: int) -> Dict[str, Any]:
    return {"jsonrpc": JSON_RPC_VERSION, "id": _id, "result": result}


def _rpc_error(code: int, message: str, _id: int, data: Any = None) -> Dict[str, Any]:
    err: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": JSON_RPC_VERSION, "id": _id, "error": err}


def _read_line() -> Optional[str]:
    try:
        line = sys.stdin.readline()
        if not line:
            return None
        return line.rstrip("\r\n")
    except EOFError:
        return None


def _write_msg(msg: Dict[str, Any]) -> None:
    data = json.dumps(msg, ensure_ascii=False)
    sys.stdout.write(data + "\n")
    sys.stdout.flush()


# ------------------------------------------------------------------ #
# PlugMem API client (direct HTTP, no deps)
# ------------------------------------------------------------------ #

class PlugMemClient:
    def __init__(self, base_url: str, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        return h

    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode()
        req = Request(url, data=data, headers=self._headers(), method="POST")
        try:
            with urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode())
        except HTTPError as e:
            detail = e.read().decode() if e.fp else str(e)
            raise RuntimeError(f"HTTP {e.code} from {path}: {detail}")
        except URLError as e:
            raise RuntimeError(f"Connection failed: {e.reason}")

    def health(self) -> Dict[str, Any]:
        try:
            req = Request(f"{self.base_url}/api/v1/health", headers=self._headers(), method="GET")
            with urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode())
        except HTTPError as e:
            detail = e.read().decode() if e.fp else str(e)
            raise RuntimeError(f"Health check failed: HTTP {e.code}: {detail}")
        except URLError as e:
            raise RuntimeError(f"PlugMem unreachable: {e.reason}")

    def create_graph(self, graph_id: str) -> Dict[str, Any]:
        return self._post("/api/v1/graphs", {"graph_id": graph_id})

    def insert_structured(self, graph_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        return self._post(f"/api/v1/graphs/{graph_id}/memories", body)

    def insert_trajectory(self, graph_id: str, goal: str, steps: List[Dict[str, str]], session_id: Optional[str] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"mode": "trajectory", "goal": goal, "steps": steps}
        if session_id:
            body["session_id"] = session_id
        return self._post(f"/api/v1/graphs/{graph_id}/memories", body)

    def reason(self, graph_id: str, query: Dict[str, Any]) -> Dict[str, Any]:
        return self._post(f"/api/v1/graphs/{graph_id}/reason", query)

    def retrieve(self, graph_id: str, query: Dict[str, Any]) -> Dict[str, Any]:
        return self._post(f"/api/v1/graphs/{graph_id}/retrieve", query)

    def promote(self, graph_id: str, candidates: List[Dict[str, str]], **filters) -> Dict[str, Any]:
        body: Dict[str, Any] = {"candidates": candidates}
        body.update(filters)
        return self._post(f"/api/v1/graphs/{graph_id}/promote", body)

    def stats(self, graph_id: str) -> Dict[str, Any]:
        req = Request(f"{self.base_url}/api/v1/graphs/{graph_id}/stats", headers=self._headers(), method="GET")
        try:
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except HTTPError as e:
            detail = e.read().decode() if e.fp else str(e)
            raise RuntimeError(f"Stats failed: HTTP {e.code}: {detail}")


# ------------------------------------------------------------------ #
# MCP tool handlers
# ------------------------------------------------------------------ #

class PlugMemMCPServer:
    def __init__(self, client: PlugMemClient, default_graph_id: str = ""):
        self.client = client
        self.default_graph_id = default_graph_id or "coding-agent"
        self._request_id = 0

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def handle_message(self, msg: Dict[str, Any]) -> None:
        method = msg.get("method", "")
        params = msg.get("params", {})
        rid = msg.get("id", self._next_id())

        if method == "initialize":
            self._handle_initialize(rid, params)
        elif method == "notifications/initialized":
            pass  # noop
        elif method == "tools/list":
            self._handle_list_tools(rid)
        elif method == "tools/call":
            self._handle_call_tool(rid, params)
        else:
            self._write_msg(_rpc_error(-32601, f"Method not found: {method}", rid))

    def _handle_initialize(self, rid: int, params: Dict[str, Any]) -> None:
        self._write_msg(_rpc_result({
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "plugmem", "version": "0.1.0"},
        }, rid))

    def _handle_list_tools(self, rid: int) -> None:
        tools = [
            {
                "name": "plugmem_remember",
                "description": (
                    "Store information in long-term memory. "
                    "Supports storing a semantic fact (with optional tags and provenance), "
                    "a procedural memory (subgoal + procedural_text), "
                    "or a full trajectory of observation/action steps. "
                    "Use this when you learn something about the user, the project, "
                    "or a process that should be remembered across sessions."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph_id": {
                            "type": "string",
                            "description": "Memory graph ID (defaults to configured graph)",
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Session identifier to group memories by run",
                        },
                        "text": {
                            "type": "string",
                            "description": "Free-text fact to remember (semantic memory)",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Category tags (e.g. python, tooling, user-pref)",
                        },
                        "subgoal": {
                            "type": "string",
                            "description": "Subgoal for procedural memory (required for procedural mode)",
                        },
                        "procedural_text": {
                            "type": "string",
                            "description": "The steps/recipe for procedural memory (required for procedural mode)",
                        },
                        "source": {
                            "type": "string",
                            "description": "Where this memory came from: explicit, correction, failure_delta",
                            "enum": ["explicit", "correction", "failure_delta"],
                            "default": "explicit",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Confidence in this memory (0.0-1.0)",
                            "default": 0.9,
                        },
                        "repo": {
                            "type": "string",
                            "description": "Git repository slug (e.g. org/repo)",
                        },
                        "branch": {"type": "string", "description": "Git branch name"},
                        "language": {
                            "type": "string",
                            "description": "Programming language (e.g. python, typescript)",
                        },
                        "filepath": {
                            "type": "string",
                            "description": "Relevant file path in the project",
                        },
                        "package_manager": {
                            "type": "string",
                            "description": "Package manager (e.g. uv, pnpm, cargo)",
                        },
                        "tool_name": {
                            "type": "string",
                            "description": "Tool this applies to (e.g. ruff, mypy, vitest)",
                        },
                        "goal": {
                            "type": "string",
                            "description": "Task goal (required for trajectory mode)",
                        },
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "observation": {"type": "string"},
                                    "action": {"type": "string"},
                                },
                                "required": ["observation", "action"],
                            },
                            "description": "Observation/action pairs for trajectory mode",
                        },
                    },
                },
            },
            {
                "name": "plugmem_recall",
                "description": (
                    "Retrieve relevant memories from long-term storage. "
                    "Returns LLM-synthesized reasoning over the most relevant memories. "
                    "Use this when you need to remember past experiences, user preferences, "
                    "project conventions, or procedures learned in earlier sessions."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph_id": {
                            "type": "string",
                            "description": "Memory graph ID (defaults to configured graph)",
                        },
                        "observation": {
                            "type": "string",
                            "description": "Current situation or question to find relevant memories for",
                        },
                        "goal": {
                            "type": "string",
                            "description": "Current task goal for better context matching",
                        },
                        "mode": {
                            "type": "string",
                            "description": "Retrieval mode: auto-detect if omitted",
                            "enum": ["semantic_memory", "episodic_memory", "procedural_memory"],
                        },
                        "source_in": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Only return memories with these source types",
                        },
                        "min_confidence": {
                            "type": "number",
                            "description": "Minimum confidence threshold (0.0-1.0)",
                        },
                        "raw": {
                            "type": "boolean",
                            "description": "Return raw retrieval prompt instead of LLM reasoning",
                        },
                    },
                    "required": ["observation"],
                },
            },
            {
                "name": "plugmem_promote",
                "description": (
                    "Extract durable memory nodes from a coding session signal and store them. "
                    "Use this when you notice a pattern worth remembering: a user correction, "
                    "a failure-then-success trace, or a repeated lookup. "
                    "Returns inserted node IDs and any rejected candidates with reasons."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph_id": {
                            "type": "string",
                            "description": "Memory graph ID (defaults to configured graph)",
                        },
                        "kind": {
                            "type": "string",
                            "description": "Type of signal",
                            "enum": ["correction", "failure_delta", "explicit", "repeated_lookup"],
                        },
                        "window": {
                            "type": "string",
                            "description": "Text context describing the signal",
                        },
                        "source_in": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Source types to accept (e.g. correction, explicit)",
                        },
                        "source_filter": {
                            "type": "string",
                            "description": "Deprecated: use source_in instead",
                        },
                        "min_confidence": {
                            "type": "number",
                            "description": "Minimum confidence threshold (0.0-1.0)",
                        },
                    },
                    "required": ["kind", "window"],
                },
            },
        ]
        self._write_msg(_rpc_result({"tools": tools}, rid))

    def _write_msg(self, msg: Dict[str, Any]) -> None:
        _write_msg(msg)

    def _handle_call_tool(self, rid: int, params: Dict[str, Any]) -> None:
        name = params.get("name", "")
        args = params.get("arguments", {})
        try:
            if name == "plugmem_remember":
                result = self._tool_remember(args)
            elif name == "plugmem_recall":
                result = self._tool_recall(args)
            elif name == "plugmem_promote":
                result = self._tool_promote(args)
            else:
                raise ValueError(f"Unknown tool: {name}")
            self._write_msg(_rpc_result({"content": [{"type": "text", "text": result}]}, rid))
        except Exception as e:
            logger.exception("Tool call failed: %s", name)
            self._write_msg(_rpc_error(-32603, str(e), rid, {"traceback": traceback.format_exc()}))

    def _resolve_graph(self, args: Dict[str, Any]) -> str:
        return args.get("graph_id", self.default_graph_id)

    def _ensure_graph(self, graph_id: str) -> None:
        """Auto-create the graph if it doesn't exist."""
        try:
            self.client.stats(graph_id)
        except RuntimeError as e:
            if "404" in str(e):
                self.client.create_graph(graph_id)
                logger.info("Auto-created graph '%s'", graph_id)
            else:
                raise

    def _build_provenance(self, args: Dict[str, Any]) -> Dict[str, str]:
        prov: Dict[str, str] = {}
        for k in ("repo", "branch", "language", "filepath", "package_manager", "tool_name"):
            v = args.get(k)
            if v:
                prov[k] = str(v)
        return prov

    def _tool_remember(self, args: Dict[str, Any]) -> str:
        graph_id = self._resolve_graph(args)
        self._ensure_graph(graph_id)
        session_id = args.get("session_id")

        # Trajectory mode
        if args.get("steps") and args.get("goal"):
            steps = args["steps"]
            goal = args["goal"]
            result = self.client.insert_trajectory(graph_id, goal, steps, session_id)
            stats = result.get("stats", {})
            parts = [f"Stored trajectory ({len(steps)} steps). Graph now has {stats.get('semantic', 0)} semantic, {stats.get('procedural', 0)} procedural, {stats.get('episodic', 0)} episodic."]
            if session_id:
                parts.append(f"Grouped under session: {session_id}")
            return "\n".join(parts)

        # Procedural mode: subgoal + procedural_text
        subgoal = args.get("subgoal", "")
        procedural_text = args.get("procedural_text", "")
        if subgoal and procedural_text:
            source = args.get("source", "explicit")
            confidence = float(args.get("confidence", 0.9))
            provenance = self._build_provenance(args)
            proc: Dict[str, Any] = {
                "subgoal": subgoal,
                "procedural_memory": procedural_text,
                "source": source,
                "confidence": confidence,
            }
            if provenance:
                proc["provenance"] = provenance
            result = self.client.insert_structured(graph_id, {
                "mode": "structured",
                "procedural": [proc],
                **({"session_id": session_id} if session_id else {}),
            })
            stats = result.get("stats", {})
            parts = [f"Remembered procedure: {subgoal[:80]}"]
            parts.append(f"Source: {source}, confidence: {confidence}")
            if provenance:
                parts.append(f"Provenance: {', '.join(f'{k}={v}' for k, v in provenance.items())}")
            parts.append(f"Graph now has {stats.get('procedural', 0)} procedural memories.")
            return "\n".join(parts)

        # Semantic mode
        text = args.get("text", "")
        if not text:
            return "Nothing stored. Provide `text` (semantic), `subgoal`+`procedural_text` (procedural), or `goal`+`steps` (trajectory)."

        source = args.get("source", "explicit")
        confidence = float(args.get("confidence", 0.9))
        tags = args.get("tags", [])
        provenance = self._build_provenance(args)

        sem: Dict[str, Any] = {
            "semantic_memory": text,
            "tags": tags,
            "source": source,
            "confidence": confidence,
        }
        if provenance:
            sem["provenance"] = provenance

        result = self.client.insert_structured(graph_id, {
            "mode": "structured",
            "semantic": [sem],
            **({"session_id": session_id} if session_id else {}),
        })
        stats = result.get("stats", {})
        parts = [f"Remembered: {text[:120]}{'...' if len(text) > 120 else ''}"]
        parts.append(f"Source: {source}, confidence: {confidence}")
        if provenance:
            parts.append(f"Provenance: {', '.join(f'{k}={v}' for k, v in provenance.items())}")
        parts.append(f"Graph now has {stats.get('semantic', 0)} semantic memories.")
        return "\n".join(parts)

    def _tool_recall(self, args: Dict[str, Any]) -> str:
        graph_id = self._resolve_graph(args)
        self._ensure_graph(graph_id)
        observation = args.get("observation", "")
        if not observation:
            return "Provide an `observation` to recall memories for."

        query: Dict[str, Any] = {"observation": observation}
        for k in ("goal", "mode"):
            v = args.get(k)
            if v:
                query[k] = v

        # Apply source/confidence filters
        source_in = args.get("source_in")
        if source_in:
            query["source_in"] = source_in
        min_conf = args.get("min_confidence")
        if min_conf is not None:
            query["min_confidence"] = float(min_conf)

        raw = args.get("raw", False)
        if raw:
            result = self.client.retrieve(graph_id, query)
            prompt = result.get("reasoning_prompt", [])
            lines = [f"[{result.get('mode', '?')}] Retrieved memories:"]
            for m in prompt:
                lines.append(f"\n## {m.get('role', '')}")
                lines.append(m.get("content", ""))
            return "\n".join(lines)

        result = self.client.reason(graph_id, query)
        mode = result.get("mode", "?")
        reasoning = result.get("reasoning", "")
        return f"[{mode}]\n{reasoning}"

    def _tool_promote(self, args: Dict[str, Any]) -> str:
        graph_id = self._resolve_graph(args)
        self._ensure_graph(graph_id)
        kind = args.get("kind", "correction")
        window = args.get("window", "")
        if not window:
            return "Provide a `window` describing the signal to promote."

        candidates = [{"kind": kind, "window": window}]
        filters: Dict[str, Any] = {}
        sf = args.get("source_filter") or args.get("source_in")
        if sf:
            if isinstance(sf, list):
                filters["source_in"] = sf
            elif isinstance(sf, str) and "," in sf:
                logger.warning("source_filter comma string deprecated, use source_in array")
                filters["source_in"] = [s.strip() for s in sf.split(",") if s.strip()]
            elif isinstance(sf, str):
                filters["source_in"] = [sf]
        mc = args.get("min_confidence")
        if mc is not None:
            filters["min_confidence"] = float(mc)

        result = self.client.promote(graph_id, candidates, **filters)
        inserted = result.get("inserted", [])
        dropped = result.get("dropped", [])

        lines = []
        if inserted:
            lines.append(f"Promoted {len(inserted)} memory node(s):")
            for m in inserted:
                mem = m.get("memory", {})
                mtype = mem.get("type", "?")
                src = mem.get("source", "?")
                conf = mem.get("confidence", "?")
                text = mem.get("semantic_memory") or mem.get("procedural_memory", "")
                lines.append(f"  [ID {m['node_id']}] {mtype} ({src}, conf={conf}): {text[:120]}")
        else:
            lines.append("No memories were promoted.")

        if dropped:
            lines.append(f"\n{len(dropped)} candidate(s) rejected:")
            for d in dropped:
                lines.append(f"  #{d.get('index', '?')} ({d.get('kind', '?')}): {d.get('reason', '?')}")

        return "\n".join(lines)


# ------------------------------------------------------------------ #
# Main entry point
# ------------------------------------------------------------------ #

def main() -> None:
    base_url = os.environ.get("PLUGMEM_BASE_URL", "http://127.0.0.1:8080")
    api_key = os.environ.get("PLUGMEM_API_KEY", "")
    default_graph = os.environ.get("PLUGMEM_DEFAULT_GRAPH", "coding-agent")

    # Verify connectivity before accepting requests
    client = PlugMemClient(base_url, api_key)
    try:
        h = client.health()
        if h.get("status") != "ok":
            logger.warning("PlugMem health: status=%s", h.get("status"))
    except Exception as e:
        logger.warning("PlugMem unreachable at %s: %s", base_url, e)
        logger.warning("Tools will return errors until the service is available.")

    server = PlugMemMCPServer(client, default_graph)

    logger.info("PlugMem MCP server starting (base_url=%s, graph=%s)", base_url, default_graph)

    while True:
        line = _read_line()
        if line is None:
            break
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON from stdin: %r", line[:200])
            continue
        server.handle_message(msg)


if __name__ == "__main__":
    main()
