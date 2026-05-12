"""``plugmem init`` -- interactive setup wizard."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from plugmem.cli.wizard import run_wizard


def init_cmd(
    force: bool = typer.Option(
        False, "--force", help="Overwrite existing config without prompting."
    ),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file.",
    ),
) -> None:
    code = run_wizard(config_path, force=force)
    raise typer.Exit(code=code)
