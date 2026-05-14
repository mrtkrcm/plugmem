"""Smoke tests for the PlugMem MCP server using fake urlopen."""
from __future__ import annotations

import json
from io import StringIO
from typing import Any, Callable, Dict, Optional
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from server import PlugMemClient, PlugMemMCPServer


# ------------------------------------------------------------------ #
# Fake urlopen
# ------------------------------------------------------------------ #

_STATS_OK = {"graph_id": "test-graph", "stats": {"semantic": 2, "procedural": 1, "episodic": 3}}


class FakeResponse:
    def __init__(self, data: dict):
        self._data = json.dumps(data).encode()

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# Stores the POST handler per-test
_current_post_handler: Optional[Callable] = None


def _fake_urlopen(req, **kwargs):
    url = req.full_url if hasattr(req, "full_url") else str(req)
    method = getattr(req, "method", "GET") or "GET"

    # Delegate to handler for any method if one is set
    if _current_post_handler:
        try:
            path = url.split("/api/v1")[-1] if "/api/v1" in url else url
            body = json.loads(req.data) if getattr(req, "data", None) else {}
            return FakeResponse(_current_post_handler(path, body))
        except HTTPError:
            raise
        except Exception as e:
            return FakeResponse({"status": "error", "message": str(e)})

    if method == "GET":
        return FakeResponse(_STATS_OK)

    return FakeResponse({"status": "ok"})


def _set_post_handler(handler: Callable):
    global _current_post_handler
    _current_post_handler = handler


@pytest.fixture(autouse=True)
def _reset_handler():
    _set_post_handler(lambda path, body: {"status": "ok", "stats": {"semantic": 1}})
    yield
    _set_post_handler(None)


@pytest.fixture
def patch_urlopen():
    patcher = patch("server.urlopen", _fake_urlopen)
    patcher.start()
    yield
    patcher.stop()


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def client():
    return PlugMemClient("http://localhost:8080", "test-key")


@pytest.fixture
def server(client):
    return PlugMemMCPServer(client, "test-graph")


# ------------------------------------------------------------------ #
# RPC helpers
# ------------------------------------------------------------------ #

def _mcp_msg(method: str, params: Optional[Dict[str, Any]] = None, rid: int = 1) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}


def _collect_output(stdout: StringIO) -> list[dict]:
    lines = stdout.getvalue().strip().split("\n")
    return [json.loads(line) for line in lines if line.strip()]


# ------------------------------------------------------------------ #
# Tool listing
# ------------------------------------------------------------------ #

def test_list_tools(server, patch_urlopen):
    out = StringIO()
    with patch("server.sys.stdout", out):
        server.handle_message(_mcp_msg("tools/list"))
    msgs = _collect_output(out)
    assert len(msgs) == 1
    tools = msgs[0]["result"]["tools"]
    tool_names = [t["name"] for t in tools]
    assert "plugmem_remember" in tool_names
    assert "plugmem_recall" in tool_names
    assert "plugmem_promote" in tool_names
    assert "plugmem_browse" in tool_names


# ------------------------------------------------------------------ #
# _ensure_graph auto-create
# ------------------------------------------------------------------ #

def test_ensure_graph_auto_creates(client, patch_urlopen):
    """When stats returns 404, create_graph should be called."""
    server = PlugMemMCPServer(client, "auto-create-graph")
    calls: list = []

    def handler(path, body):
        calls.append({"path": path, "body": body})
        if "stats" in path:
            raise HTTPError("http://localhost/stats", 404, "Not found", {}, None)
        return {"graph_id": "auto-create-graph", "stats": {}}

    _set_post_handler(handler)

    out = StringIO()
    with patch("server.sys.stdout", out):
        result = server._tool_remember({"text": "test fact"})
    assert "Remembered" in result

    create_calls = [c for c in calls if "graphs" in c["path"] and "stats" not in c["path"] and "memories" not in c["path"] and "promote" not in c["path"]]
    assert len(create_calls) >= 1

    stats_calls = [c for c in calls if "stats" in c["path"]]
    assert len(stats_calls) >= 1


# ------------------------------------------------------------------ #
# plugmem_remember — semantic
# ------------------------------------------------------------------ #

def test_remember_semantic(server, patch_urlopen):
    calls: list = []

    def handler(path, body):
        calls.append({"path": path, "body": body})
        return {"status": "ok", "stats": {"semantic": 1}}

    _set_post_handler(handler)

    out = StringIO()
    with patch("server.sys.stdout", out):
        result = server._tool_remember({"text": "UV is preferred over pip", "tags": ["python"], "source": "explicit", "confidence": 0.9})
    assert "Remembered" in result
    assert "UV is preferred" in result


# ------------------------------------------------------------------ #
# plugmem_remember — procedural
# ------------------------------------------------------------------ #

def test_remember_procedural(server, patch_urlopen):
    def handler(path, body):
        return {"status": "ok", "stats": {"procedural": 1}}

    _set_post_handler(handler)

    out = StringIO()
    with patch("server.sys.stdout", out):
        result = server._tool_remember({"subgoal": "install deps", "procedural_text": "Run uv sync"})
    assert "Remembered procedure" in result
    assert "install deps" in result


# ------------------------------------------------------------------ #
# plugmem_remember — trajectory
# ------------------------------------------------------------------ #

def test_remember_trajectory(server, patch_urlopen):
    _set_post_handler(lambda path, body: {"status": "ok", "stats": {"semantic": 1, "procedural": 0, "episodic": 3}})

    out = StringIO()
    with patch("server.sys.stdout", out):
        result = server._tool_remember({
            "goal": "Fix bug",
            "steps": [{"observation": "error", "action": "fix it"}],
        })
    assert "Stored trajectory" in result


# ------------------------------------------------------------------ #
# plugmem_recall
# ------------------------------------------------------------------ #

def test_recall(server, patch_urlopen):
    calls: list[dict] = []

    def handler(path, body):
        calls.append({"path": path, "body": body})
        return {
        "mode": "semantic_memory",
        "reasoning": "Use UV for Python dependency management",
        "reasoning_prompt": [{"role": "user", "content": "prompt"}],
        }

    _set_post_handler(handler)

    out = StringIO()
    with patch("server.sys.stdout", out):
        result = server._tool_recall({
            "observation": "how to install deps",
            "language": "python",
            "repo": "org/repo",
        })
    assert "semantic_memory" in result
    assert "Use UV" in result
    reason_calls = [c for c in calls if c["path"].endswith("/reason")]
    assert reason_calls
    prov = reason_calls[0]["body"]["provenance_filters"]
    assert prov["language"] == ["python"]
    assert prov["repo"] == ["org/repo"]


def test_recall_debug_returns_structured_payload(server, patch_urlopen):
    _set_post_handler(lambda path, body: {
        "mode": "semantic_memory",
        "reasoning": "debug reasoning",
        "reasoning_prompt": [{"role": "user", "content": "prompt"}],
    })
    payload = json.loads(server._tool_recall({
        "observation": "how to install deps",
        "debug": True,
    }))
    assert payload["request"]["observation"] == "how to install deps"
    assert payload["response"]["reasoning"] == "debug reasoning"


def test_recall_supports_full_provenance_filter_set(server, patch_urlopen):
    calls: list[dict] = []

    def handler(path, body):
        calls.append({"path": path, "body": body})
        return {
            "mode": "semantic_memory",
            "reasoning": "scoped",
            "reasoning_prompt": [{"role": "user", "content": "prompt"}],
        }

    _set_post_handler(handler)

    result = server._tool_recall({
        "observation": "how to run tests",
        "branch": "main",
        "commit": "abc123",
        "filepath": "src/app.py",
        "package_manager": "uv",
        "tool_name": "pytest",
        "tool_version": "8.0",
        "os": "macos",
        "component": "api",
    })
    assert "scoped" in result
    reason_calls = [c for c in calls if c["path"].endswith("/reason")]
    prov = reason_calls[0]["body"]["provenance_filters"]
    assert prov["branch"] == ["main"]
    assert prov["commit"] == ["abc123"]
    assert prov["filepath"] == ["src/app.py"]
    assert prov["package_manager"] == ["uv"]
    assert prov["tool_name"] == ["pytest"]
    assert prov["tool_version"] == ["8.0"]
    assert prov["os"] == ["macos"]
    assert prov["component"] == ["api"]


def test_recall_forwards_extended_retrieval_context(server, patch_urlopen):
    calls: list[dict] = []

    def handler(path, body):
        calls.append({"path": path, "body": body})
        return {
            "mode": "procedural_memory",
            "reasoning": "use the stored procedure",
            "reasoning_prompt": [{"role": "user", "content": "prompt"}],
        }

    _set_post_handler(handler)

    result = server._tool_recall({
        "observation": "tests are failing",
        "goal": "restore green CI",
        "subgoal": "fix flaky pytest",
        "state": "red build",
        "task_type": "debugging",
        "time": "2026-05-14T10:00:00Z",
        "session_id": "sess-42",
        "mode": "procedural_memory",
    })
    assert "procedural_memory" in result
    reason_calls = [c for c in calls if c["path"].endswith("/reason")]
    body = reason_calls[0]["body"]
    assert body["observation"] == "tests are failing"
    assert body["goal"] == "restore green CI"
    assert body["subgoal"] == "fix flaky pytest"
    assert body["state"] == "red build"
    assert body["task_type"] == "debugging"
    assert body["time"] == "2026-05-14T10:00:00Z"
    assert body["session_id"] == "sess-42"
    assert body["mode"] == "procedural_memory"


# ------------------------------------------------------------------ #
# plugmem_promote
# ------------------------------------------------------------------ #

def test_promote(server, patch_urlopen):
    _set_post_handler(lambda path, body: {
        "inserted": [{
            "node_type": "semantic",
            "node_id": 0,
            "memory": {
                "type": "semantic",
                "semantic_memory": "Use uv not pip",
                "source": "correction",
                "confidence": 0.9,
            },
        }],
        "dropped": [],
    })

    out = StringIO()
    with patch("server.sys.stdout", out):
        result = server._tool_promote({"kind": "correction", "window": "use uv"})
    assert "Promoted" in result
    assert "ID 0" in result


def test_browse_tool(server, patch_urlopen):
    _set_post_handler(lambda path, body: {"graph_id": "test-graph", "node_type": "semantic", "count": 1, "nodes": [{"semantic_memory": "Use uv, not pip"}]})
    result = server._tool_browse({"node_type": "semantic", "limit": 5})
    assert "1 semantic node(s)" in result
    assert "Use uv, not pip" in result
