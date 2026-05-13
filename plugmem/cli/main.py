"""Top-level Typer app and subcommand registration."""
from __future__ import annotations

import typer

from plugmem.cli.commands import (
    coding,
    health,
    init,
    logs,
    restart,
    start,
    status,
    stop,
)

app = typer.Typer(
    name="plugmem",
    help=(
        "PlugMem -- pluggable long-term memory for LLM agents.\n\n"
        "Run `plugmem init` to set up your local instance, then "
        "`plugmem start` to launch the service."
    ),
    no_args_is_help=True,
    add_completion=False,
)

app.command("init", help="Interactive setup wizard for LLM, embedding, service, and coding profile.")(init.init_cmd)
app.command("start", help="Start the PlugMem service (daemonized by default).")(start.start_cmd)
app.command("stop", help="Stop the running PlugMem daemon.")(stop.stop_cmd)
app.command("restart", help="Restart the PlugMem daemon.")(restart.restart_cmd)
app.command("status", help="Show daemon status, PID, port, and last health probe.")(status.status_cmd)
app.command("logs", help="Print or tail the daemon log.")(logs.logs_cmd)
app.command("health", help="One-shot health check against the running service.")(health.health_cmd)

# Coding-agent command group
app.add_typer(coding.coding_app)
