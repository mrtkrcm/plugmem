"""``plugmem health`` -- one-shot health check."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import requests
import typer

from plugmem.cli.config import default_config_path, load_config
from plugmem.cli.daemon import HEALTH_PATH
from plugmem.cli.wizard.ui import console, error


def health_cmd(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file."
    ),
) -> None:
    path = config_path or default_config_path()
    cfg = load_config(path)
    url = "http://{}:{}{}".format(cfg.service.host, cfg.service.port, HEALTH_PATH)

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

    backend = data.get("storage_backend", "chroma")
    # Prefer storage_available; fall back to chroma_available for older daemons.
    storage_ok = data.get("storage_available", data.get("chroma_available", False))
    data["storage_available"] = storage_ok
    flags = ["llm_available", "embedding_available", "storage_available"]

    overall_ok = True
    for flag in flags:
        ok = data.get(flag, False)
        label = flag if flag != "storage_available" else f"storage_available ({backend})"
        mark = "[green]✓[/green]" if ok else "[red]✗[/red]"
        console.print("  {} {}".format(mark, label))
        if not ok:
            overall_ok = False

    version = data.get("version", "?")
    status_val = data.get("status", "?")
    console.print("\nstatus: {}, version: {}".format(status_val, version))

    if not overall_ok:
        raise typer.Exit(code=1)
