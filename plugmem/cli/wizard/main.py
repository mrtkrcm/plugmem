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
from plugmem.cli.wizard.ui import error, header, info, prompt_choice, success, warn


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


def _print_post_setup_summary(cfg: PlugmemConfig) -> None:
    header("Next steps")
    info("Start the daemon:")
    info("    plugmem start")
    info("")
    info("To wire the Claude Code plugin against this instance, export:")
    info("    export PLUGMEM_BASE_URL=http://{}:{}".format(cfg.service.host, cfg.service.port))
    info("    export PLUGMEM_API_KEY={}".format(cfg.service.api_key))
    info("")
    info("Then in your project repo:")
    info("    claude --plugin-dir /absolute/path/to/plugmem-coding-claude-code")
