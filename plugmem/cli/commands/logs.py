"""``plugmem logs`` -- print or tail the daemon log."""
from __future__ import annotations

import time

import typer

from plugmem.cli.config import default_log_file
from plugmem.cli.wizard.ui import error, info


def logs_cmd(
    follow: bool = typer.Option(
        False, "--follow", "-f", help="Tail the log instead of printing it once."
    ),
    lines: int = typer.Option(
        100, "--lines", "-n", help="Number of trailing lines to show before tailing."
    ),
) -> None:
    log_path = default_log_file()
    if not log_path.exists():
        error("No log file at {}.".format(log_path))
        info("The daemon hasn't been started yet, or PLUGMEM_STATE_DIR is overridden.")
        raise typer.Exit(code=1)

    with open(log_path, "rb") as f:
        data = f.read()
    text = data.decode("utf-8", errors="replace")
    last = text.splitlines()[-lines:]
    for line in last:
        typer.echo(line)

    if not follow:
        return

    try:
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            while True:
                chunk = f.read()
                if chunk:
                    typer.echo(chunk.decode("utf-8", errors="replace"), nl=False)
                else:
                    time.sleep(0.3)
    except KeyboardInterrupt:
        pass
