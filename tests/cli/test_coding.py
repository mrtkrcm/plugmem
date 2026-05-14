"""Tests for `plugmem coding` commands."""
from __future__ import annotations

from typer.testing import CliRunner

from plugmem.cli.main import app


runner = CliRunner()


def test_scaffold_force_recreates_existing_graph(monkeypatch, tmp_path):
    calls: list[tuple[str, str, dict | None]] = []

    monkeypatch.setattr(
        "plugmem.cli.commands.coding.default_config_path",
        lambda: tmp_path / "config.toml",
    )

    def fake_post(url, headers, body):
        calls.append(("POST", url, body))
        if url.endswith("/api/v1/graphs") and len([c for c in calls if c[0] == "POST" and c[1].endswith("/api/v1/graphs")]) == 1:
            raise SystemExit("HTTP 409 from http://127.0.0.1:8080/api/v1/graphs: already exists")
        if url.endswith("/stats"):
            return {"stats": {"semantic": 0, "procedural": 0, "tag": 0}}
        return {"graph_id": "coding-agent", "stats": {"semantic": 0, "procedural": 0, "tag": 0}}

    def fake_delete(url, headers):
        calls.append(("DELETE", url, None))

    monkeypatch.setattr("plugmem.cli.commands.coding._api_post", fake_post)
    monkeypatch.setattr("plugmem.cli.commands.coding._api_get", lambda url, headers: {"stats": {"semantic": 0, "procedural": 0, "tag": 0}})
    monkeypatch.setattr("plugmem.cli.commands.coding._api_delete", fake_delete)

    result = runner.invoke(app, ["coding", "scaffold", "--force"])
    assert result.exit_code == 0, result.stdout

    create_calls = [c for c in calls if c[0] == "POST" and c[1].endswith("/api/v1/graphs")]
    delete_calls = [c for c in calls if c[0] == "DELETE" and c[1].endswith("/api/v1/graphs/coding-agent")]
    assert len(create_calls) == 2
    assert len(delete_calls) == 1


def test_scaffold_existing_graph_skips_duplicate_conventions(monkeypatch, tmp_path):
    calls: list[tuple[str, str, dict | None]] = []
    seen_get_urls: list[str] = []

    monkeypatch.setattr(
        "plugmem.cli.commands.coding.default_config_path",
        lambda: tmp_path / "config.toml",
    )

    def fake_post(url, headers, body):
        calls.append(("POST", url, body))
        if url.endswith("/api/v1/graphs"):
            raise SystemExit("HTTP 409 from http://127.0.0.1:8080/api/v1/graphs: already exists")
        return {"status": "ok", "stats": {"semantic": 5, "procedural": 0, "tag": 0}}

    def fake_get(url, headers):
        seen_get_urls.append(url)
        if "node_type=semantic" in url:
            return {
                "nodes": [
                    {"semantic_memory": "use uv not pip for dependency management"},
                ]
            }
        return {"stats": {"semantic": 5, "procedural": 0, "tag": 0}}

    monkeypatch.setattr("plugmem.cli.commands.coding._api_post", fake_post)
    monkeypatch.setattr("plugmem.cli.commands.coding._api_get", fake_get)
    monkeypatch.setattr("plugmem.cli.commands.coding._api_delete", lambda url, headers: None)

    result = runner.invoke(app, ["coding", "scaffold", "--language", "python"])
    assert result.exit_code == 0, result.stdout
    assert any("component=seed-convention" in url for url in seen_get_urls)

    seeded = [
        c for c in calls
        if c[0] == "POST" and "/memories" in c[1]
    ]
    assert len(seeded) == 1
    seeded_texts = [item["semantic_memory"] for item in seeded[0][2]["semantic"]]
    assert len(seeded_texts) == 4
    assert "Use uv, not pip, for dependency management" not in seeded_texts
    assert all(item["provenance"]["component"] == "seed-convention" for item in seeded[0][2]["semantic"])


def test_attach_happy_path(monkeypatch, tmp_path):
    repo = tmp_path / "demo"
    repo.mkdir()
    (repo / "AGENTS.md").write_text("Run `just quality` before committing.\n")
    (repo / "Justfile").write_text("build:\n\techo build\n")
    (repo / "package.json").write_text('{"name":"demo"}')

    monkeypatch.setattr(
        "plugmem.cli.commands.coding.default_config_path",
        lambda: tmp_path / "config.toml",
    )
    monkeypatch.setattr(
        "plugmem.cli.commands.coding._detect_git_provenance",
        lambda path=None: {"repo": "git@github.com:org/demo.git", "branch": "main"},
    )
    monkeypatch.setattr(
        "plugmem.cli.commands.coding._detect_project_language",
        lambda root: "typescript",
    )
    monkeypatch.setattr(
        "plugmem.cli.commands.coding._detect_package_manager",
        lambda root, language: "pnpm",
    )
    monkeypatch.setattr(
        "plugmem.cli.commands.coding._detect_primary_tool",
        lambda root, language: "just",
    )
    monkeypatch.setattr(
        "plugmem.cli.commands.coding._detect_project_profile",
        lambda root, language: "vite-react",
    )

    posts: list[tuple[str, dict]] = []

    def fake_post(url, headers, body):
        posts.append((url, body))
        if url.endswith("/graphs"):
            return {"graph_id": "demo", "stats": {}}
        if url.endswith("/retrieve"):
            return {"mode": "semantic_memory", "reasoning_prompt": []}
        return {"status": "ok", "stats": {"semantic": 1, "procedural": 0, "tag": 0}}

    monkeypatch.setattr("plugmem.cli.commands.coding._api_post", fake_post)
    monkeypatch.setattr("plugmem.cli.commands.coding._api_get", lambda url, headers: {"status": "ok", "llm_available": True, "embedding_available": True, "storage_available": True} if url.endswith("/health") else {"stats": {"semantic": 0, "procedural": 0, "tag": 0}, "nodes": []})
    monkeypatch.setattr("plugmem.cli.commands.coding._api_delete", lambda url, headers: None)

    result = runner.invoke(app, ["coding", "attach", str(repo), "--graph", "demo"])
    assert result.exit_code == 0, result.stdout
    assert any(url.endswith("/graphs/demo/memories") for url, _ in posts)
    assert any(url.endswith("/graphs/demo/retrieve") for url, _ in posts)


def test_attach_skips_guidance_when_embedding_unavailable(monkeypatch, tmp_path):
    repo = tmp_path / "demo"
    repo.mkdir()
    (repo / "AGENTS.md").write_text("Run `just quality` before committing.\n")

    monkeypatch.setattr(
        "plugmem.cli.commands.coding.default_config_path",
        lambda: tmp_path / "config.toml",
    )
    monkeypatch.setattr(
        "plugmem.cli.commands.coding._detect_git_provenance",
        lambda path=None: {"repo": "git@github.com:org/demo.git", "branch": "main"},
    )
    monkeypatch.setattr(
        "plugmem.cli.commands.coding._detect_project_language",
        lambda root: "swift",
    )
    monkeypatch.setattr(
        "plugmem.cli.commands.coding._detect_package_manager",
        lambda root, language: "swiftpm",
    )
    monkeypatch.setattr(
        "plugmem.cli.commands.coding._detect_primary_tool",
        lambda root, language: "just",
    )
    monkeypatch.setattr(
        "plugmem.cli.commands.coding._detect_project_profile",
        lambda root, language: "swiftpm-macos-app",
    )
    monkeypatch.setattr("plugmem.cli.commands.coding._api_post", lambda url, headers, body: {"graph_id": "demo", "stats": {}})
    monkeypatch.setattr("plugmem.cli.commands.coding._api_get", lambda url, headers: {"status": "ok", "llm_available": True, "embedding_available": False, "storage_available": True} if url.endswith("/health") else {"stats": {"semantic": 0, "procedural": 0, "tag": 0}, "nodes": []})
    monkeypatch.setattr("plugmem.cli.commands.coding._api_delete", lambda url, headers: None)

    result = runner.invoke(app, ["coding", "attach", str(repo), "--graph", "demo"])
    assert result.exit_code == 0, result.stdout
    assert "Embedding backend is unavailable" in result.stdout
