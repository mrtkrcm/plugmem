"""``plugmem coding`` — coding-agent memory management.

Subcommands:

- ``scaffold`` — Create a memory graph with sensible defaults for coding agents,
  optionally seeding it with known tool/language conventions.

- ``promote`` — Promote coding signals (corrections, failure deltas) into
  the graph from the command line.

- ``recall`` — Retrieve relevant memories via /reason or /retrieve.

- ``list`` — List stored semantic or procedural nodes with optional filters.
"""
from __future__ import annotations

import json
import os
import subprocess
import pathlib
from typing import Any, Dict, List, Optional

import typer
import urllib.parse
import urllib.request
import urllib.error

from plugmem.cli.config import default_config_path, load_config
from plugmem.cli.wizard.ui import error, info, success, warn

coding_app = typer.Typer(
    name="coding",
    help="Coding-agent memory commands: scaffold, promote, recall, list.",
    no_args_is_help=True,
    add_completion=False,
)


# ------------------------------------------------------------------ #
# helpers
# ------------------------------------------------------------------ #

def _resolve_url_and_key(cfg) -> tuple[str, str]:
    host = cfg.service.host or "127.0.0.1"
    port = cfg.service.port or 8080
    base = os.environ.get("PLUGMEM_BASE_URL", f"http://{host}:{port}")
    api_key = cfg.service.api_key or os.environ.get("PLUGMEM_API_KEY", "")
    return base.rstrip("/"), api_key


def _headers(api_key: str) -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if api_key:
        h["X-API-Key"] = api_key
    return h


def _api(url: str, headers: Dict[str, str], *, method: str = "POST", body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body_text = e.read().decode() if e.fp else ""
        raise SystemExit(f"HTTP {e.code} from {url}: {body_text}")
    except urllib.error.URLError as e:
        raise SystemExit(f"Connection failed: {e.reason}")


def _api_post(url: str, headers: Dict[str, str], body: Dict[str, Any]) -> Dict[str, Any]:
    return _api(url, headers, method="POST", body=body)


def _api_get(url: str, headers: Dict[str, str]) -> Dict[str, Any]:
    return _api(url, headers, method="GET")


def _detect_git_provenance(path: Optional[str] = None) -> Dict[str, str]:
    """Try to extract repo, branch from the current git context."""
    prov: Dict[str, str] = {}
    cwd = path or os.getcwd()
    try:
        prov["repo"] = (
            subprocess.check_output(
                ["git", "remote", "get-url", "origin"],
                cwd=cwd, stderr=subprocess.DEVNULL, timeout=5,
            )
            .decode()
            .strip()
            .removesuffix(".git")
        )
    except Exception:
        pass
    try:
        prov["branch"] = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=cwd, stderr=subprocess.DEVNULL, timeout=5,
            )
            .decode()
            .strip()
        )
    except Exception:
        pass
    return prov


_KNOWN_CONVENTIONS: Dict[str, List[str]] = {
    "python": [
        "Use uv, not pip, for dependency management",
        "Run `uv sync` instead of `pip install -r requirements.txt`",
        "Format with `ruff format`",
        "Type-check with `mypy .`",
        "Run tests with `pytest -x`",
    ],
    "typescript": [
        "Use pnpm, not npm, for package management",
        "Run `pnpm tsc --noEmit` to type-check",
        "Format with `prettier --write`",
        "Run tests with `vitest run`",
    ],
    "rust": [
        "Use `cargo check` to verify compilation without producing a binary",
        "Run `cargo clippy` for linting",
        "Run `cargo test` for tests",
    ],
    "go": [
        "Use `go mod tidy` to clean up dependencies",
        "Run `go vet ./...` for static analysis",
        "Run `go test ./...` for tests",
    ],
    "swift": [
        "Use SwiftPM (Package.swift) for dependency management, not CocoaPods or Xcode projects",
        "Format with `swift-format`",
        "Run tests with `swift test`",
        "Use `swift build` to verify compilation",
    ],
    "kotlin": [
        "Use Gradle for dependency management",
        "Format with `ktlint`",
        "Run tests with `gradle test`",
        "Run static analysis with `detekt`",
    ],
    "ruby": [
        "Use Bundler for dependency management",
        "Format with `rubocop -A`",
        "Run tests with `rspec`",
        "Run `bundle exec` instead of bare commands",
    ],
}


# ------------------------------------------------------------------ #
# scaffold
# ------------------------------------------------------------------ #

@coding_app.command(
    "scaffold",
    help=(
        "Create a memory graph for coding-agent use with sensible defaults. "
        "Optionally seed it with language-specific tool conventions so the "
        "agent can recall project standards immediately."
    ),
)
def scaffold_cmd(
    graph_id: str = typer.Option(
        "", "--graph", "-g",
        help="Graph ID (default: auto-generated or from config).",
    ),
    language: str = typer.Option(
        "", "--language", "-l",
        help="Seed conventions for this language (python, typescript, rust, go).",
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Overwrite existing graph seeding.",
    ),
) -> None:
    cfg = load_config(default_config_path())
    base, api_key = _resolve_url_and_key(cfg)
    gid = graph_id or cfg.coding.default_graph or "coding-agent"

    # 1. Create the graph
    info(f"Creating graph '{gid}'...")
    _api_post(f"{base}/api/v1/graphs", _headers(api_key), {"graph_id": gid})
    info(f"[bold]Graph '{gid}' ready[/bold]")

    # 2. Auto-detect git provenance
    prov = _detect_git_provenance()
    if not prov.get("repo"):
        repo = cfg.coding.default_repo or ""
        if repo:
            prov["repo"] = repo
    lang = language or cfg.coding.default_language or ""
    if lang:
        prov["language"] = lang

    # 3. Seed language conventions
    conv = _KNOWN_CONVENTIONS.get(lang, [])
    if conv:
        info(f"Seeding {len(conv)} {lang} conventions...")
        for text in conv:
            tags = [lang, "convention", "tooling"]
            body: Dict[str, Any] = {
                "mode": "structured",
                "semantic": [{
                    "semantic_memory": text,
                    "tags": tags,
                    "source": "explicit",
                    "confidence": 0.9,
                    "provenance": dict(prov) if prov else None,
                }],
            }
            _api_post(f"{base}/api/v1/graphs/{gid}/memories", _headers(api_key), body)
        success(f"Seeded {len(conv)} semantic memories into '{gid}'.")
    else:
        info("No conventions seeded (no --language or language not recognized).")

    # 4. Print summary
    if prov:
        info(f"  Provenance: repo={prov.get('repo', '—')}, branch={prov.get('branch', '—')}")
    info(f"  Source filter (default): {cfg.coding.source_filter}")
    info(f"  Min confidence (default): {cfg.coding.min_confidence}")
    stats = _api_get(f"{base}/api/v1/graphs/{gid}/stats", _headers(api_key))
    s = stats.get("stats", {})
    success(f"Graph '{gid}' ready.  semantic={s.get('semantic', 0)}  procedural={s.get('procedural', 0)}  tag={s.get('tag', 0)}")
    info("")
    info("Next: promote signals with:")
    info(f"  plugmem coding promote --graph {gid} --kind correction --window \"use uv, not pip\"")
    info("Or retrieve:")
    info(f'  curl -X POST {base}/api/v1/graphs/{gid}/reason -H "Content-Type: application/json" -d \'{{"observation":"how to install deps"}}\'')


# ------------------------------------------------------------------ #
# promote
# ------------------------------------------------------------------ #

@coding_app.command(
    "promote",
    help=(
        "Promote one or more coding signals into the memory graph. "
        "Runs LLM extraction and inserts accepted memories atomically, "
        "returning node IDs and any rejected candidates with reasons. "
        "Pass --window multiple times for multiple candidates, or "
        "use --from-file to read newline-delimited candidates from a file."
    ),
)
def promote_cmd(
    graph_id: str = typer.Option(
        "", "--graph", "-g",
        help="Target graph ID.",
    ),
    kind: str = typer.Option(
        "correction", "--kind", "-k",
        help="Candidate kind (used for all candidates unless differently sourced).",
    ),
    window: List[str] = typer.Option(
        [], "--window", "-w",
        help="Text context for a candidate (repeatable).",
    ),
    from_file: Optional[str] = typer.Option(
        None, "--from-file",
        help="Read newline-delimited candidate texts from file.",
    ),
    source_in: List[str] = typer.Option(
        [], "--source-in",
        help="Source types to accept (repeatable, e.g. --source-in correction --source-in explicit).",
    ),
    source_filter: Optional[str] = typer.Option(
        None, "--source-filter",
        help="Deprecated: comma-separated source types. Use --source-in instead.",
    ),
    min_confidence: float = typer.Option(
        -1.0, "--min-confidence", "-c",
        help="Minimum confidence threshold (0-1).",
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j",
        help="Output raw JSON instead of human-readable summary.",
    ),
) -> None:
    cfg = load_config(default_config_path())
    base, api_key = _resolve_url_and_key(cfg)
    gid = graph_id or cfg.coding.default_graph or "coding-agent"

    # Build candidates list
    texts: List[str] = list(window)
    if from_file:
        p = pathlib.Path(from_file)
        if not p.exists():
            error(f"File not found: {from_file}")
            raise typer.Exit(1)
        texts.extend(p.read_text().strip().splitlines())

    if not texts:
        error("Provide at least one candidate via --window (repeatable) or --from-file.")
        raise typer.Exit(1)

    body: Dict[str, Any] = {
        "candidates": [{"kind": kind, "window": t} for t in texts],
    }

    # Apply config defaults for filters — prefer --source-in array
    sf: Optional[str] = None
    if source_in:
        body["source_in"] = list(source_in)
    elif source_filter:
        sf = source_filter
    elif cfg.coding.source_filter:
        sf = cfg.coding.source_filter
    if sf:
        body["source_in"] = [s.strip() for s in sf.split(",") if s.strip()]
    mc = min_confidence if min_confidence >= 0 else cfg.coding.min_confidence
    if mc > 0:
        body["min_confidence"] = mc

    resp = _api_post(f"{base}/api/v1/graphs/{gid}/promote", _headers(api_key), body)

    if json_output:
        print(json.dumps(resp, indent=2))
        return

    inserted = resp.get("inserted", [])
    dropped = resp.get("dropped", [])

    if inserted:
        for m in inserted:
            mem = m.get("memory", {})
            conf = mem.get("confidence", "?")
            src = mem.get("source", "?")
            if mem.get("type") == "semantic":
                info(f"  [ID {m['node_id']}] semantic  ({src}, conf={conf})  {mem.get('semantic_memory', '')[:100]}")
            else:
                info(f"  [ID {m['node_id']}] procedural  ({src}, conf={conf})  {mem.get('procedural_memory', '')[:100]}")
        success(f"Promoted {len(inserted)} memory node(s).")
    else:
        warn("No memories were promoted.")

    if dropped:
        warn(f"{len(dropped)} candidate(s) rejected:")
        for d in dropped:
            info(f"  #{d.get('index', '?')} ({d.get('kind', '?')}): {d.get('reason', '—')}")


# ------------------------------------------------------------------ #
# recall
# ------------------------------------------------------------------ #

@coding_app.command(
    "recall",
    help=(
        "Retrieve relevant memories from the coding memory graph. "
        "By default calls /reason for LLM-synthesized reasoning. "
        "Use --raw to get the raw retrieval prompt instead."
    ),
)
def recall_cmd(
    graph_id: str = typer.Option(
        "", "--graph", "-g",
        help="Target graph ID.",
    ),
    observation: str = typer.Argument(
        ..., help="Observation or question to recall memories for.",
    ),
    mode: Optional[str] = typer.Option(
        None, "--mode", "-m",
        help="Retrieval mode: semantic_memory, procedural_memory, episodic_memory.",
    ),
    source_in: List[str] = typer.Option(
        [], "--source-in",
        help="Source types to accept (repeatable).",
    ),
    min_confidence: float = typer.Option(
        -1.0, "--min-confidence", "-c",
        help="Minimum confidence threshold (0-1).",
    ),
    provenance_language: Optional[str] = typer.Option(
        None, "--language", "-l",
        help="Filter by programming language (uses provenance_filters).",
    ),
    provenance_repo: Optional[str] = typer.Option(
        None, "--repo",
        help="Filter by repository (uses provenance_filters).",
    ),
    raw: bool = typer.Option(
        False, "--raw", "-r",
        help="Return raw retrieval prompt instead of LLM reasoning.",
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j",
        help="Output raw JSON instead of human-readable summary.",
    ),
) -> None:
    cfg = load_config(default_config_path())
    base, api_key = _resolve_url_and_key(cfg)
    gid = graph_id or cfg.coding.default_graph or "coding-agent"

    body: Dict[str, Any] = {"observation": observation}
    if mode:
        body["mode"] = mode
    if source_in:
        body["source_in"] = list(source_in)
    else:
        ssf = cfg.coding.source_filter
        if ssf:
            body["source_in"] = [s.strip() for s in ssf.split(",") if s.strip()]
    mc = min_confidence if min_confidence >= 0 else cfg.coding.min_confidence
    if mc > 0:
        body["min_confidence"] = mc

    # Provenance filters as optional CLI args
    prov: Dict[str, List[str]] = {}
    if provenance_language:
        prov["language"] = [provenance_language]
    if provenance_repo:
        prov["repo"] = [provenance_repo]
    if prov:
        body["provenance_filters"] = prov

    endpoint = "/retrieve" if raw else "/reason"
    resp = _api_post(f"{base}/api/v1/graphs/{gid}{endpoint}", _headers(api_key), body)

    if json_output:
        print(json.dumps(resp, indent=2))
        return

    if raw:
        mode_label = resp.get("mode", "?")
        prompt = resp.get("reasoning_prompt", [])
        info(f"[{mode_label}] Retrieved memories:")
        for m in prompt:
            print(f"\n## {m.get('role', '')}")
            print(m.get("content", ""))
    else:
        mode_label = resp.get("mode", "?")
        reasoning = resp.get("reasoning", "")
        info(f"[{mode_label}]")
        print(reasoning)


# ------------------------------------------------------------------ #
# list
# ------------------------------------------------------------------ #

@coding_app.command(
    "list",
    help=(
        "Browse stored coding memories scoped by provenance / source / "
        "confidence. This is an **experience browser**, not a code index — "
        "filters are metadata only. Use `plugmem coding recall` for "
        "content-aware retrieval."
    ),
)
def list_cmd(
    graph_id: str = typer.Option(
        "", "--graph", "-g",
        help="Target graph ID.",
    ),
    node_type: str = typer.Option(
        "semantic", "--type", "-t",
        help="Node type: semantic or procedural.",
    ),
    language: Optional[str] = typer.Option(
        None, "--language", "-l",
        help="Scope by provenance.language (e.g. python, swift).",
    ),
    repo: Optional[str] = typer.Option(
        None, "--repo",
        help="Scope by provenance.repo (e.g. github.com/org/name).",
    ),
    source_in: List[str] = typer.Option(
        [], "--source",
        help=(
            "Filter by source (repeatable): explicit, correction, "
            "failure_delta, merged, repeated_lookup."
        ),
    ),
    min_confidence: float = typer.Option(
        -1.0, "--min-confidence", "-c",
        help="Minimum confidence (0-1).",
    ),
    query: Optional[str] = typer.Option(
        None, "--query", "-q",
        help=(
            "Optional client-side content substring **post-filter** applied "
            "after provenance/source/confidence scoping. For real content "
            "retrieval use `plugmem coding recall`."
        ),
    ),
    limit: int = typer.Option(
        50, "--limit", "-n",
        help="Maximum nodes to return.",
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j",
        help="Output raw JSON instead of table.",
    ),
) -> None:
    cfg = load_config(default_config_path())
    base, api_key = _resolve_url_and_key(cfg)
    gid = graph_id or cfg.coding.default_graph or "coding-agent"

    # Build query string for the metadata-filtered /nodes endpoint
    params: List[str] = [f"node_type={node_type}", f"limit={limit}"]
    if language:
        params.append(f"language={urllib.parse.quote(language)}")
    if repo:
        params.append(f"repo={urllib.parse.quote(repo)}")
    for s in source_in:
        params.append(f"source_in={urllib.parse.quote(s)}")
    mc = min_confidence if min_confidence >= 0 else cfg.coding.min_confidence
    if mc and mc > 0:
        params.append(f"min_confidence={mc}")

    url = f"{base}/api/v1/graphs/{gid}/nodes?" + "&".join(params)
    resp = _api_get(url, _headers(api_key))

    nodes = resp.get("nodes", [])

    def _node_text(n: Dict[str, Any]) -> str:
        return (
            n.get("semantic_memory")
            or n.get("procedural_memory")
            or n.get("text")
            or ""
        )

    # Client-side content post-filter (kept narrow; recall is the real surface)
    if query:
        needle = query.casefold().strip()
        nodes = [n for n in nodes if needle in _node_text(n).casefold()]
        resp = {**resp, "nodes": nodes, "count": len(nodes)}

    if json_output:
        print(json.dumps(resp, indent=2))
        return

    if not nodes:
        info(f"No {node_type} nodes match the given filters.")
        return

    info(f"{len(nodes)} {node_type} node(s):")
    for n in nodes:
        nid = n.get("semantic_id") or n.get("procedural_id", "?")
        text = _node_text(n)
        source = n.get("source") or "—"
        conf = n.get("confidence", "—")
        prov = n.get("provenance") or {}
        lang = prov.get("language", "") if isinstance(prov, dict) else ""
        node_repo = prov.get("repo", "") if isinstance(prov, dict) else ""
        tags = []
        if lang:
            tags.append(lang)
        if node_repo:
            tags.append(node_repo.split("/")[-1])
        tag_suffix = f" [{', '.join(tags)}]" if tags else ""
        info(f"  [ID {nid}] ({source}, conf={conf}){tag_suffix}  {str(text)[:120]}")
