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
  - plugmem_browse       Inspect stored semantic/procedural memories for debugging.
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List, Optional
from urllib.request import Request, build_opener
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

from plugmem.coding_contract import (
    PROVENANCE_FILTER_KEYS,
    build_promote_body,
    build_provenance_filters,
    build_recall_body,
)

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
logger = logging.getLogger("plugmem-mcp")
_shared_opener = build_opener()
# Route all requests through a reusable opener while keeping a patchable
# module-level symbol for the lightweight test harness.
urlopen = _shared_opener.open

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
        return self._perform(req, timeout=60, path=path)

    def _get(self, path: str, query: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        if query:
            url += "?" + urlencode(query, doseq=True)
        req = Request(url, headers=self._headers(), method="GET")
        return self._perform(req, timeout=30, path=path)

    def _perform(self, req: Request, *, timeout: int, path: str) -> Dict[str, Any]:
        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                with urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read().decode())
            except HTTPError as e:
                detail = e.read().decode() if e.fp else str(e)
                if e.code >= 500 and attempt < 2:
                    time.sleep(0.1 * (attempt + 1))
                    last_err = e
                    continue
                raise RuntimeError(f"http_error:{e.code}:{path}:{detail}")
            except URLError as e:
                if attempt < 2:
                    time.sleep(0.1 * (attempt + 1))
                    last_err = e
                    continue
                raise RuntimeError(f"connection_error:{path}:{e.reason}")
        if last_err is not None:
            raise RuntimeError(str(last_err))
        raise RuntimeError(f"request_failed:{path}")

    def health(self) -> Dict[str, Any]:
        try:
            return self._get("/api/v1/health")
        except RuntimeError as e:
            raise RuntimeError(f"health_failed:{e}")

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
        return self._get(f"/api/v1/graphs/{graph_id}/stats")

    def browse(self, graph_id: str, *, node_type: str = "semantic", limit: int = 20, **filters) -> Dict[str, Any]:
        query: Dict[str, Any] = {"node_type": node_type, "limit": limit}
        for key, value in filters.items():
            if value is None or value == []:
                continue
            query[key] = value
        return self._get(f"/api/v1/graphs/{graph_id}/nodes", query=query)


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
                            "description": "Where this memory came from",
                            "enum": ["explicit", "correction", "failure_delta", "merged", "repeated_lookup"],
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
                        "subgoal": {
                            "type": "string",
                            "description": "Current immediate subgoal to bias procedural or episodic recall",
                        },
                        "state": {
                            "type": "string",
                            "description": "Current agent/runtime state for recall context",
                        },
                        "task_type": {
                            "type": "string",
                            "description": "Task class label used by the retrieval pipeline",
                        },
                        "time": {
                            "type": "string",
                            "description": "Timestamp or logical time marker for recall context",
                        },
                        "session_id": {
                            "type": "string",
                            "description": "Session identifier for recall audit logging",
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
                        "language": {
                            "type": "string",
                            "description": "Restrict recall to memories with matching provenance.language",
                        },
                        "repo": {
                            "type": "string",
                            "description": "Restrict recall to memories with matching provenance.repo",
                        },
                        "branch": {
                            "type": "string",
                            "description": "Restrict recall to memories with matching provenance.branch",
                        },
                        "commit": {
                            "type": "string",
                            "description": "Restrict recall to memories with matching provenance.commit",
                        },
                        "filepath": {
                            "type": "string",
                            "description": "Restrict recall to memories with matching provenance.filepath",
                        },
                        "package_manager": {
                            "type": "string",
                            "description": "Restrict recall to memories with matching provenance.package_manager",
                        },
                        "tool_name": {
                            "type": "string",
                            "description": "Restrict recall to memories with matching provenance.tool_name",
                        },
                        "tool_version": {
                            "type": "string",
                            "description": "Restrict recall to memories with matching provenance.tool_version",
                        },
                        "os": {
                            "type": "string",
                            "description": "Restrict recall to memories with matching provenance.os",
                        },
                        "component": {
                            "type": "string",
                            "description": "Restrict recall to memories with matching provenance.component",
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
                    "Extract durable memory nodes from coding signals and store them. "
                    "Accepts one or more candidates (each with kind + window). "
                    "For a single candidate, use the kind+window shorthand. "
                    "For multiple, use the candidates array. "
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
                            "description": "Type of signal (shorthand for single candidate)",
                            "enum": ["correction", "failure_delta", "explicit", "repeated_lookup"],
                        },
                        "window": {
                            "type": "string",
                            "description": "Text context (shorthand for single candidate)",
                        },
                        "candidates": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "kind": {"type": "string"},
                                    "window": {"type": "string"},
                                },
                                "required": ["kind", "window"],
                            },
                            "description": "Multiple candidates (use instead of kind+window)",
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
                },
            },
            {
                "name": "plugmem_browse",
                "description": (
                    "Browse stored semantic or procedural memories with metadata filters. "
                    "Useful for debugging what PlugMem currently knows in Claude Code."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "graph_id": {"type": "string", "description": "Memory graph ID"},
                        "node_type": {"type": "string", "enum": ["semantic", "procedural"], "default": "semantic"},
                        "limit": {"type": "number", "description": "Maximum nodes to return", "default": 20},
                        "source_in": {"type": "array", "items": {"type": "string"}},
                        "min_confidence": {"type": "number"},
                        "language": {"type": "string"},
                        "repo": {"type": "string"},
                        "branch": {"type": "string"},
                        "commit": {"type": "string"},
                        "filepath": {"type": "string"},
                        "package_manager": {"type": "string"},
                        "tool_name": {"type": "string"},
                        "tool_version": {"type": "string"},
                        "os": {"type": "string"},
                        "component": {"type": "string"},
                        "debug": {"type": "boolean", "description": "Return raw JSON payload"},
                    },
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
            elif name == "plugmem_browse":
                result = self._tool_browse(args)
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

    def _detect_context_provenance(self) -> Dict[str, str]:
        prov: Dict[str, str] = {}
        cwd = os.environ.get("PLUGMEM_WORKDIR", os.getcwd())
        p = pathlib.Path(cwd)
        for env_key, prov_key in (
            ("PLUGMEM_DEFAULT_LANGUAGE", "language"),
            ("PLUGMEM_DEFAULT_PACKAGE_MANAGER", "package_manager"),
            ("PLUGMEM_DEFAULT_COMPONENT", "component"),
        ):
            if os.environ.get(env_key):
                prov[prov_key] = os.environ[env_key]
        try:
            prov["repo"] = (
                subprocess.check_output(
                    ["git", "remote", "get-url", "origin"],
                    cwd=cwd, stderr=subprocess.DEVNULL, timeout=3,
                ).decode().strip().removesuffix(".git")
            )
        except Exception:
            pass
        try:
            prov["branch"] = (
                subprocess.check_output(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    cwd=cwd, stderr=subprocess.DEVNULL, timeout=3,
                ).decode().strip()
            )
        except Exception:
            pass
        if "package_manager" not in prov:
            if (p / "uv.lock").exists() or (p / "pyproject.toml").exists():
                prov["package_manager"] = "uv"
            elif (p / "pnpm-lock.yaml").exists():
                prov["package_manager"] = "pnpm"
            elif (p / "package-lock.json").exists():
                prov["package_manager"] = "npm"
            elif (p / "Cargo.toml").exists():
                prov["package_manager"] = "cargo"
        if "language" not in prov:
            if (p / "pyproject.toml").exists():
                prov["language"] = "python"
            elif (p / "package.json").exists():
                prov["language"] = "typescript"
            elif (p / "Cargo.toml").exists():
                prov["language"] = "rust"
        return prov

    def _build_provenance(self, args: Dict[str, Any]) -> Dict[str, str]:
        prov = self._detect_context_provenance()
        for k in PROVENANCE_FILTER_KEYS:
            v = args.get(k)
            if v:
                prov[k] = str(v)
        return prov

    def _build_provenance_filters(self, args: Dict[str, Any]) -> Dict[str, List[str]]:
        merged = dict(self._detect_context_provenance())
        for key in PROVENANCE_FILTER_KEYS:
            value = args.get(key)
            if value:
                merged[key] = str(value)
        return build_provenance_filters(merged)

    def _build_recall_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        prov = self._build_provenance_filters(args)
        return build_recall_body(
            observation=args.get("observation", ""),
            goal=args.get("goal"),
            subgoal=args.get("subgoal"),
            state=args.get("state"),
            task_type=args.get("task_type"),
            time=args.get("time"),
            session_id=args.get("session_id"),
            mode=args.get("mode"),
            source_in=args.get("source_in"),
            min_confidence=args.get("min_confidence"),
            provenance_filters=prov or None,
        )

    def _tool_remember(self, args: Dict[str, Any]) -> str:
        graph_id = self._resolve_graph(args)
        self._ensure_graph(graph_id)
        session_id = args.get("session_id")
        debug = bool(args.get("debug", False))

        # Trajectory mode
        if args.get("steps") and args.get("goal"):
            steps = args["steps"]
            goal = args["goal"]
            result = self.client.insert_trajectory(graph_id, goal, steps, session_id)
            if debug:
                return json.dumps({"graph_id": graph_id, "request": {"goal": goal, "steps": steps, "session_id": session_id}, "response": result}, indent=2)
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
            if debug:
                return json.dumps({"graph_id": graph_id, "request": {"mode": "structured", "procedural": [proc], **({"session_id": session_id} if session_id else {})}, "response": result}, indent=2)
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
        if debug:
            return json.dumps({"graph_id": graph_id, "request": {"mode": "structured", "semantic": [sem], **({"session_id": session_id} if session_id else {})}, "response": result}, indent=2)
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

        query = self._build_recall_query(args)
        debug = bool(args.get("debug", False))

        raw = args.get("raw", False)
        if raw:
            result = self.client.retrieve(graph_id, query)
            if debug:
                return json.dumps({"graph_id": graph_id, "request": query, "response": result}, indent=2)
            prompt = result.get("reasoning_prompt", [])
            lines = [f"[{result.get('mode', '?')}] Retrieved memories:"]
            for m in prompt:
                lines.append(f"\n## {m.get('role', '')}")
                lines.append(m.get("content", ""))
            return "\n".join(lines)

        result = self.client.reason(graph_id, query)
        if debug:
            return json.dumps({"graph_id": graph_id, "request": query, "response": result}, indent=2)
        mode = result.get("mode", "?")
        reasoning = result.get("reasoning", "")
        return f"[{mode}]\n{reasoning}"

    def _tool_promote(self, args: Dict[str, Any]) -> str:
        graph_id = self._resolve_graph(args)
        self._ensure_graph(graph_id)
        debug = bool(args.get("debug", False))

        # Build candidates from either the array form or kind+window shorthand
        candidates = args.get("candidates")
        if candidates is None:
            kind = args.get("kind", "correction")
            window = args.get("window", "")
            if not window:
                return "Provide `window` (single candidate) or `candidates` array."
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

        body = build_promote_body(candidates=candidates, source_in=filters.get("source_in"), min_confidence=filters.get("min_confidence"))
        result = self.client.promote(graph_id, body["candidates"], source_in=body.get("source_in"), min_confidence=body.get("min_confidence"))
        if debug:
            return json.dumps({"graph_id": graph_id, "request": body, "response": result}, indent=2)
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

    def _tool_browse(self, args: Dict[str, Any]) -> str:
        graph_id = self._resolve_graph(args)
        self._ensure_graph(graph_id)
        node_type = args.get("node_type", "semantic")
        limit = int(args.get("limit", 20))
        prov = self._build_provenance_filters(args)
        result = self.client.browse(
            graph_id,
            node_type=node_type,
            limit=limit,
            source_in=args.get("source_in"),
            min_confidence=float(args["min_confidence"]) if args.get("min_confidence") is not None else None,
            **prov,
        )
        if args.get("debug"):
            return json.dumps({"graph_id": graph_id, "node_type": node_type, "response": result}, indent=2)
        nodes = result.get("nodes", [])
        lines = [f"{len(nodes)} {node_type} node(s)"]
        for node in nodes:
            text = node.get("semantic_memory") or node.get("procedural_memory") or ""
            lines.append(f"  [{node_type}] {text[:160]}")
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
