"""``plugmem coding`` — coding-agent memory management.

Subcommands:

- ``attach`` — Inspect a real repo, update the coding profile, scaffold a
  graph, ingest durable guidance, and verify setup.

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
import re
import subprocess
import pathlib
from typing import Any, Dict, List, Optional

import typer
import urllib.parse
import urllib.request
import urllib.error

from plugmem.cli.config import default_config_path, load_config, save_config
from plugmem.coding_contract import build_promote_body, build_recall_body
from plugmem.cli.wizard.ui import error, info, success, warn

coding_app = typer.Typer(
    name="coding",
    help="Coding-agent memory commands: attach, scaffold, promote, recall, list.",
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


def _api_delete(url: str, headers: Dict[str, str]) -> None:
    _api(url, headers, method="DELETE")


def _existing_semantic_texts(base: str, api_key: str, graph_id: str, language: str = "") -> set[str]:
    """Best-effort snapshot of existing semantic memory text for idempotent seeding."""
    params = ["node_type=semantic", "limit=10000", "source_in=explicit"]
    if language:
        params.append(f"language={urllib.parse.quote(language)}")
    params.append(f"component={urllib.parse.quote('seed-convention')}")
    try:
        resp = _api_get(
            f"{base}/api/v1/graphs/{graph_id}/nodes?{'&'.join(params)}",
            _headers(api_key),
        )
    except SystemExit:
        return set()
    texts: set[str] = set()
    for node in resp.get("nodes", []):
        text = _canonicalize_seed_text(node.get("semantic_memory") or "")
        if text:
            texts.add(text)
    return texts


def _canonicalize_seed_text(text: str) -> str:
    """Normalize convention text so trivial formatting drift doesn't reseed it."""
    text = text.casefold().strip()
    text = text.replace("`", "")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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

_PROJECT_PROFILES: Dict[str, List[str]] = {
    "swiftpm-macos-app": [
        "Run `just build` from the repository root for fast debug iteration.",
        "Run `just test` to execute the project's Swift test suite.",
        "Run `just quality` before commit so format, lint, build, and tests all pass.",
    ],
    "swift-package": [
        "Use `swift build` to verify compilation.",
        "Use `swift test` to run the package test suite.",
    ],
    "vite-react": [
        "Use the package manager lockfile in the repo; do not switch package managers casually.",
        "Run the project's test and build scripts from package.json before commit.",
    ],
    "python-project": [
        "Use the repository's configured environment workflow before running tests or scripts.",
        "Run the narrowest relevant test first during iteration, then the full quality gate before commit.",
    ],
}


def _slugify_graph_id(path: pathlib.Path) -> str:
    return re.sub(r"[^a-z0-9]+", "-", path.name.casefold()).strip("-") or "coding-agent"


def _detect_project_language(root: pathlib.Path) -> str:
    if (root / "mdviewer" / "Package.swift").exists() or (root / "Package.swift").exists():
        return "swift"
    if (root / "package.json").exists():
        return "typescript"
    if (root / "pyproject.toml").exists():
        return "python"
    if (root / "Cargo.toml").exists():
        return "rust"
    return ""


def _detect_package_manager(root: pathlib.Path, language: str) -> str:
    if language == "swift":
        return "swiftpm"
    if (root / "pnpm-lock.yaml").exists():
        return "pnpm"
    if (root / "package-lock.json").exists():
        return "npm"
    if (root / "uv.lock").exists():
        return "uv"
    if (root / "Cargo.toml").exists():
        return "cargo"
    return ""


def _detect_primary_tool(root: pathlib.Path, language: str) -> str:
    if (root / "Justfile").exists():
        return "just"
    if language == "swift":
        return "swift-format"
    if language == "python":
        return "pytest"
    return ""


def _detect_project_profile(root: pathlib.Path, language: str) -> str:
    if language == "swift" and (root / "Justfile").exists() and ((root / "mdviewer" / "Package.swift").exists() or (root / "Package.swift").exists()):
        return "swiftpm-macos-app"
    if language == "swift":
        return "swift-package"
    if language == "typescript" and (root / "package.json").exists():
        return "vite-react"
    if language == "python":
        return "python-project"
    return ""


def _guidance_files(root: pathlib.Path) -> List[pathlib.Path]:
    candidates = [
        root / "AGENTS.md",
        root / "README.md",
        root / "Justfile",
        root / "Package.swift",
        root / "pyproject.toml",
        root / "package.json",
    ]
    return [p for p in candidates if p.exists()]


def _extract_guidance_memories(root: pathlib.Path, profile: str) -> List[Dict[str, Any]]:
    memories: List[Dict[str, Any]] = []
    tags = [t for t in [profile, "project-guidance"] if t]
    agents = root / "AGENTS.md"
    if agents.exists():
        text = agents.read_text(encoding="utf-8", errors="ignore")
        patterns = [
            r"Run `just quality` before committing\.[^\n]*",
            r"During dev, run the narrowest failing test only\.[^\n]*",
            r"Never hardcode spacing/colors/durations[^\n]*",
            r"Never use `print\(\)`/`NSLog\(\)`[^\n]*",
            r"Do not change `lineHeightMultiplier`[^\n]*",
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                memories.append({"semantic_memory": m.group(0).strip(), "tags": tags + ["guardrail"]})
    justfile = root / "Justfile"
    if justfile.exists():
        text = justfile.read_text(encoding="utf-8", errors="ignore")
        snippets = [
            ("build workflow", r"just build"),
            ("test workflow", r"just test"),
            ("quality workflow", r"just quality"),
        ]
        for label, pat in snippets:
            if re.search(pat, text):
                memories.append({"semantic_memory": f"Use `{pat}` from the repository root for the project's {label}.", "tags": tags + ["workflow"]})
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for mem in memories:
        key = _canonicalize_seed_text(mem["semantic_memory"])
        if key and key not in seen:
            seen.add(key)
            out.append(mem)
    return out


def _health_snapshot(base: str) -> Dict[str, Any]:
    return _api_get(f"{base}/api/v1/health", {"Accept": "application/json"})


# ------------------------------------------------------------------ #
# attach
# ------------------------------------------------------------------ #

@coding_app.command(
    "attach",
    help=(
        "Inspect a repository, update the local coding profile, health-check "
        "PlugMem, scaffold a project graph, ingest repo guidance, and verify "
        "setup with a sample recall."
    ),
)
def attach_cmd(
    path: str = typer.Argument(..., help="Path to the project repository."),
    graph_id: str = typer.Option("", "--graph", "-g", help="Override graph ID."),
    force: bool = typer.Option(False, "--force", "-f", help="Recreate the graph before reseeding."),
) -> None:
    root = pathlib.Path(path).expanduser().resolve()
    if not root.exists():
        error(f"Project path not found: {root}")
        raise typer.Exit(1)

    cfg_path = default_config_path()
    cfg = load_config(cfg_path)

    git_prov = _detect_git_provenance(str(root))
    language = _detect_project_language(root)
    package_manager = _detect_package_manager(root, language)
    tool_name = _detect_primary_tool(root, language)
    profile = _detect_project_profile(root, language)
    gid = graph_id or cfg.coding.default_graph or _slugify_graph_id(root)

    if git_prov.get("repo"):
        cfg.coding.default_repo = git_prov["repo"]
    if git_prov.get("branch"):
        cfg.coding.default_branch = git_prov["branch"]
    if language:
        cfg.coding.default_language = language
    if package_manager:
        cfg.coding.default_package_manager = package_manager
    if tool_name:
        cfg.coding.default_tool_name = tool_name
    cfg.coding.default_graph = gid
    cfg.coding.default_component = root.name
    save_config(cfg, cfg_path)

    base, _api_key = _resolve_url_and_key(cfg)
    try:
        health = _health_snapshot(base)
    except SystemExit as exc:
        error(f"PlugMem health check failed: {exc}")
        raise typer.Exit(1)

    llm_ok = bool(health.get("llm_available"))
    emb_ok = bool(health.get("embedding_available"))
    storage_ok = bool(health.get("storage_available", health.get("chroma_available")))
    info(f"Detected repo={cfg.coding.default_repo or '—'} branch={cfg.coding.default_branch or '—'} language={language or '—'} package_manager={package_manager or '—'} profile={profile or '—'} graph={gid}")
    info(f"Health: llm={llm_ok} embedding={emb_ok} storage={storage_ok}")
    if not storage_ok:
        error("Storage is unavailable; cannot attach project.")
        raise typer.Exit(1)

    # Reuse scaffold for base conventions.
    os.environ["CODING_DEFAULT_GRAPH"] = gid
    cwd = os.getcwd()
    try:
        os.chdir(root)
        scaffold_cmd(graph_id=gid, language=language, force=force)
    finally:
        os.chdir(cwd)

    if not emb_ok:
        warn("Embedding backend is unavailable. Project-specific guidance ingestion was skipped.")
        raise typer.Exit(0)

    guidance = _extract_guidance_memories(root, profile)
    if guidance:
        provenance = {
            "repo": cfg.coding.default_repo,
            "branch": cfg.coding.default_branch,
            "language": language,
            "package_manager": package_manager,
            "tool_name": tool_name,
            "component": root.name,
        }
        body = {
            "mode": "structured",
            "semantic": [
                {
                    "semantic_memory": item["semantic_memory"],
                    "tags": item["tags"],
                    "source": "explicit",
                    "confidence": 0.95,
                    "provenance": {k: v for k, v in provenance.items() if v},
                }
                for item in guidance
            ],
        }
        _api_post(f"{base}/api/v1/graphs/{gid}/memories", _headers(cfg.service.api_key or os.environ.get("PLUGMEM_API_KEY", "")), body)
        success(f"Ingested {len(guidance)} project guidance memories.")

    sample = build_recall_body(
        observation="how should I build and test this project?",
        mode="semantic_memory",
        source_in=["explicit"],
        provenance_filters={
            "repo": [cfg.coding.default_repo] if cfg.coding.default_repo else [],
            "language": [language] if language else [],
        },
    )
    sample["provenance_filters"] = {k: v for k, v in sample.get("provenance_filters", {}).items() if v}
    resp = _api_post(f"{base}/api/v1/graphs/{gid}/retrieve", _headers(cfg.service.api_key or os.environ.get("PLUGMEM_API_KEY", "")), sample)
    success(f"Project '{root.name}' attached to graph '{gid}'.")
    info(f"Guidance files scanned: {', '.join(p.name for p in _guidance_files(root)) or 'none'}")
    info(f"Sample recall mode: {resp.get('mode', '?')}")
    info("Next: use `plugmem coding recall` and `plugmem coding promote` while working in this repo.")


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
    try:
        _api_post(f"{base}/api/v1/graphs", _headers(api_key), {"graph_id": gid})
    except SystemExit as exc:
        if force and "HTTP 409" in str(exc):
            warn(f"Graph '{gid}' already exists; deleting and recreating because --force was set.")
            _api_delete(f"{base}/api/v1/graphs/{gid}", _headers(api_key))
            _api_post(f"{base}/api/v1/graphs", _headers(api_key), {"graph_id": gid})
        elif "HTTP 409" in str(exc):
            warn(f"Graph '{gid}' already exists; keeping it and seeding only missing conventions.")
        else:
            raise
    info(f"[bold]Graph '{gid}' ready[/bold]")

    # 2. Resolve provenance: config defaults win, git auto-detect as fallback
    prov: Dict[str, str] = {}
    if cfg.coding.default_repo:
        prov["repo"] = cfg.coding.default_repo
    if cfg.coding.default_branch:
        prov["branch"] = cfg.coding.default_branch
    git_prov = _detect_git_provenance()
    prov.setdefault("repo", git_prov.get("repo", ""))
    prov.setdefault("branch", git_prov.get("branch", ""))
    lang = language or cfg.coding.default_language or ""
    if lang:
        prov["language"] = lang

    # 3. Seed language conventions
    conv = _KNOWN_CONVENTIONS.get(lang, [])
    if conv:
        existing = _existing_semantic_texts(base, api_key, gid, language=lang)
        missing = [text for text in conv if _canonicalize_seed_text(text) not in existing]
        info(f"Seeding {len(conv)} {lang} conventions...")
        if missing:
            tags = [lang, "convention", "tooling", f"seed:{lang}"]
            prov_for_seed = dict(prov) if prov else {}
            prov_for_seed["component"] = "seed-convention"
            body: Dict[str, Any] = {
                "mode": "structured",
                "semantic": [
                    {
                        "semantic_memory": text,
                        "tags": tags,
                        "source": "explicit",
                        "confidence": 0.9,
                        "provenance": prov_for_seed,
                    }
                    for text in missing
                ],
            }
            _api_post(f"{base}/api/v1/graphs/{gid}/memories", _headers(api_key), body)
        skipped = len(conv) - len(missing)
        success(f"Seeded {len(missing)} semantic memories into '{gid}'.")
        if skipped:
            info(f"Skipped {skipped} existing convention(s).")
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

    candidates = [{"kind": kind, "window": t} for t in texts]

    # Apply config defaults for filters — prefer --source-in array
    sf: Optional[str] = None
    resolved_source_in: Optional[List[str]] = None
    if source_in:
        resolved_source_in = list(source_in)
    elif source_filter:
        sf = source_filter
    elif cfg.coding.source_filter:
        sf = cfg.coding.source_filter
    if sf:
        resolved_source_in = [s.strip() for s in sf.split(",") if s.strip()]
    mc = min_confidence if min_confidence >= 0 else cfg.coding.min_confidence
    body = build_promote_body(
        candidates=candidates,
        source_in=resolved_source_in,
        min_confidence=mc if mc > 0 else None,
    )

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

    resolved_source_in: Optional[List[str]] = None
    if source_in:
        resolved_source_in = list(source_in)
    else:
        ssf = cfg.coding.source_filter
        if ssf:
            resolved_source_in = [s.strip() for s in ssf.split(",") if s.strip()]
    mc = min_confidence if min_confidence >= 0 else cfg.coding.min_confidence

    # Provenance filters as optional CLI args
    prov: Dict[str, List[str]] = {}
    if provenance_language:
        prov["language"] = [provenance_language]
    if provenance_repo:
        prov["repo"] = [provenance_repo]
    body = build_recall_body(
        observation=observation,
        mode=mode,
        source_in=resolved_source_in,
        min_confidence=mc if mc > 0 else None,
        provenance_filters=prov or None,
    )

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
