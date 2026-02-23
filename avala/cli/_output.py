"""Output formatting utilities for the CLI."""

from __future__ import annotations

import json
from typing import Any, List, Sequence, Tuple

from rich.console import Console
from rich.table import Table

console = Console()


def print_table(title: str, columns: List[str], rows: Sequence[Tuple[str, ...]]) -> None:
    """Print a formatted table to the console."""
    table = Table(title=title)
    for col in columns:
        table.add_column(col)
    for row in rows:
        table.add_row(*row)
    console.print(table)


def print_json(data: Any) -> None:
    """Print formatted JSON to the console."""
    console.print_json(json.dumps(data, indent=2, default=str))


def print_detail(title: str, fields: List[Tuple[str, str]]) -> None:
    """Print a detail view with key-value pairs."""
    table = Table(title=title, show_header=False)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    for key, value in fields:
        table.add_row(key, value)
    console.print(table)
