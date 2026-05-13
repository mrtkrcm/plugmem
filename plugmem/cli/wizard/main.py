"""Wizard orchestrator: walks the three sections + final probe + write."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from plugmem.cli.config import (
    PlugmemConfig,
    default_config_path,
    load_config,
    save_config,
)
from plugmem.cli.wizard.final_probe import run_final_probe
from plugmem.cli.wizard.probes import detect_ollama
from plugmem.cli.wizard.sections import (
    run_embedding_section,
    run_llm_section,
    run_service_section,
)
from plugmem.cli.wizard.ui import error, header, info, prompt_choice, prompt_text, success, warn


def run_wizard(config_path: Optional[Path] = None, *, force: bool = False) -> int:
    path = config_path or default_config_path()
    if path.exists() and not force:
        warn("Config already exists at {}. Pass --force to overwrite.".format(path))
        return 1

    cfg = load_config(path) if path.exists() else PlugmemConfig()

    detected = detect_ollama()

    if not run_llm_section(cfg, ollama=detected):
        error("LLM section was skipped -- config not written.")
        return 1

    if not run_embedding_section(cfg, ollama=detected):
        error("Embedding section was skipped -- config not written.")
        return 1

    run_service_section(cfg)

    header("Coding-agent profile (optional)")
    if prompt_choice("Configure coding-agent defaults?", choices=["yes", "skip"], default="skip") == "yes":
        _run_coding_section(cfg)

    header("Final probe")
    info("Launching the service briefly to verify everything wires up\u2026")
    ok, msg = run_final_probe(cfg)
    if not ok:
        error("Probe failed: {}".format(msg))
        retry = prompt_choice("What now?", choices=["save anyway", "abort"], default="abort")
        if retry == "abort":
            return 1
        warn("Saving config despite probe failure -- fix the issue and re-run `plugmem doctor`.")
    else:
        success(msg)

    written = save_config(cfg, path)
    success("Wrote config to {}".format(written))

    _print_post_setup_summary(cfg)
    return 0


def _run_coding_section(cfg: PlugmemConfig) -> None:
    """Optional coding-agent profile section."""
    info("These defaults are used by `plugmem coding` commands.")
    info("All fields are optional; leave blank to skip.")

    cfg.coding.default_graph = prompt_text(
        "Default graph ID for coding memories",
        default=cfg.coding.default_graph or "coding-agent",
    )
    cfg.coding.default_repo = prompt_text(
        "Default git repo slug (e.g. org/repo)",
        default=cfg.coding.default_repo or "",
        allow_empty=True,
    )
    cfg.coding.default_language = prompt_text(
        "Default programming language",
        default=cfg.coding.default_language or "",
        allow_empty=True,
    )
    cfg.coding.default_package_manager = prompt_text(
        "Default package manager (e.g. uv, pnpm, cargo)",
        default=cfg.coding.default_package_manager or "",
        allow_empty=True,
    )
    cfg.coding.default_tool_name = prompt_text(
        "Default tool name (e.g. ruff, mypy, vitest)",
        default=cfg.coding.default_tool_name or "",
        allow_empty=True,
    )
    src = prompt_text(
        "Source filter (comma-separated, empty = all)",
        default=cfg.coding.source_filter,
        allow_empty=True,
    )
    if src:
        cfg.coding.source_filter = src
    conf = prompt_text(
        "Min confidence threshold (0-1)",
        default=str(cfg.coding.min_confidence),
        allow_empty=True,
    )
    if conf:
        try:
            cfg.coding.min_confidence = float(conf)
        except ValueError:
            warn(f"Invalid confidence '{conf}', keeping {cfg.coding.min_confidence}")

    info("")
    info("After starting the daemon, run:")
    info("  plugmem coding scaffold --language <your_language>")
    info("to create a seeded memory graph.")


def _print_post_setup_summary(cfg: PlugmemConfig) -> None:
    header("Next steps")
    info("Start the daemon:")
    info("    plugmem start")
    info("")
    info("Scaffold a coding-agent memory graph (optional):")
    info("    plugmem coding scaffold --language python")
    info("")
    info("Promote coding signals into memory:")
    info('    plugmem coding promote --kind correction --window "use uv, not pip"')
    info("")
    info("To wire the Claude Code plugin against this instance, export:")
    info("    export PLUGMEM_BASE_URL=http://{}:{}".format(cfg.service.host, cfg.service.port))
    info("    export PLUGMEM_API_KEY={}".format(cfg.service.api_key))
    info("")
    info("Then in your project repo:")
    info("    claude --plugin-dir /absolute/path/to/plugmem-coding-claude-code")
