"""Output formatting utilities for the CLI."""

from __future__ import annotations

import json
from typing import Any, List, Sequence, Tuple

import click
from rich.console import Console
from rich.table import Table

console = Console()


def _get_output_format() -> str:
    """Return the current output format from the Click context, defaulting to 'table'."""
    ctx = click.get_current_context(silent=True)
    if ctx and ctx.obj:
        return ctx.obj.get("output_format", "table")
    return "table"


def print_table(
    title: str,
    columns: List[str],
    rows: Sequence[Tuple[str, ...]],
    *,
    json_keys: List[str] | None = None,
) -> None:
    """Print a formatted table to the console, or JSON when ``--output json`` is active.

    Parameters
    ----------
    json_keys:
        Optional list of snake_case API field names to use as JSON object keys
        instead of the display *columns*.  When provided and output is JSON,
        these keys are used so that programmatic consumers get stable,
        API-aligned field names.
    """
    if _get_output_format() == "json":
        keys = json_keys if json_keys is not None else columns
        data = [dict(zip(keys, row)) for row in rows]
        click.echo(json.dumps(data, indent=2, default=str))
        return

    table = Table(title=title)
    for col in columns:
        table.add_column(col)
    for row in rows:
        table.add_row(*row)
    console.print(table)


def print_json(data: Any) -> None:
    """Print formatted JSON to the console."""
    console.print_json(json.dumps(data, indent=2, default=str))


def print_detail(
    title: str,
    fields: List[Tuple[str, str]],
    *,
    json_keys: List[str] | None = None,
) -> None:
    """Print a detail view with key-value pairs, or JSON when ``--output json`` is active.

    Parameters
    ----------
    json_keys:
        Optional list of snake_case API field names to use as JSON object keys
        instead of the display labels from *fields*.  Must have the same length
        as *fields* when provided.
    """
    if _get_output_format() == "json":
        if json_keys is not None:
            data = dict(zip(json_keys, (v for _, v in fields)))
        else:
            data = dict(fields)
        click.echo(json.dumps(data, indent=2, default=str))
        return

    table = Table(title=title, show_header=False)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    for key, value in fields:
        table.add_row(key, value)
    console.print(table)
