"""``plugmem health`` -- one-shot health check."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import requests
import typer

from plugmem.cli.config import default_config_path, load_config
from plugmem.cli.wizard.ui import console, error


HEALTH_FLAGS = ("llm_available", "embedding_available", "chroma_available")


def health_cmd(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file."
    ),
) -> None:
    path = config_path or default_config_path()
    cfg = load_config(path)
    url = "http://{}:{}/health".format(cfg.service.host, cfg.service.port)

    try:
        resp = requests.get(url, timeout=5.0)
    except requests.RequestException as e:
        error("{} returned error: {}".format(url, e))
        raise typer.Exit(code=2)

    if resp.status_code != 200:
        error("{} returned HTTP {}".format(url, resp.status_code))
        raise typer.Exit(code=2)

    try:
        data = resp.json()
    except ValueError:
        error("{} returned non-JSON".format(url))
        raise typer.Exit(code=2)

    overall_ok = True
    for flag in HEALTH_FLAGS:
        ok = data.get(flag, False)
        mark = "[green]✓[/green]" if ok else "[red]✗[/red]"
        console.print("  {} {}".format(mark, flag))
        if not ok:
            overall_ok = False

    version = data.get("version", "?")
    status = data.get("status", "?")
    console.print("\nstatus: {}, version: {}".format(status, version))

    if not overall_ok:
        raise typer.Exit(code=1)
