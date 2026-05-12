"""Shared Rich console + prompt helpers for the wizard."""
from __future__ import annotations

from typing import List, Optional

from rich.console import Console
from rich.prompt import Prompt

console = Console()


def header(text: str) -> None:
    console.rule("[bold]{}[/bold]".format(text))


def info(text: str) -> None:
    console.print(text)


def success(text: str) -> None:
    console.print("[green]✓[/green] {}".format(text))


def warn(text: str) -> None:
    console.print("[yellow]![/yellow] {}".format(text))


def error(text: str) -> None:
    console.print("[red]✗[/red] {}".format(text))


def prompt_text(
    label: str,
    *,
    default: Optional[str] = None,
    password: bool = False,
    allow_empty: bool = False,
) -> str:
    while True:
        value = Prompt.ask(label, default=default, password=password, console=console)
        value = (value or "").strip()
        if value or allow_empty:
            return value
        warn("Value cannot be empty.")


def prompt_choice(
    label: str,
    choices: List[str],
    *,
    default: Optional[str] = None,
) -> str:
    return Prompt.ask(
        label,
        choices=choices,
        default=default if default in choices else choices[0],
        console=console,
    )


def prompt_action(label: str = "What now?") -> str:
    return prompt_choice(label, ["retry", "edit", "skip"], default="retry")
