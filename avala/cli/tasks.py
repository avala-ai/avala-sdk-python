"""CLI commands for tasks."""

from __future__ import annotations

import click

from avala.cli._output import print_detail, print_table


@click.group()
def tasks() -> None:
    """Manage tasks."""


@tasks.command("list")
@click.option("--limit", type=int, default=None, help="Maximum number of results")
@click.pass_context
def list_tasks(ctx: click.Context, limit: int | None) -> None:
    """List tasks."""
    client = ctx.obj["client"]
    page = client.tasks.list(limit=limit)
    rows = [(t.uid, t.name or "—", t.type or "—", t.status or "—", str(t.created_at or "—")) for t in page.items]
    print_table(
        "Tasks",
        ["UID", "Name", "Type", "Status", "Created"],
        rows,
        json_keys=["uid", "name", "type", "status", "created_at"],
    )


@tasks.command("get")
@click.argument("uid")
@click.pass_context
def get_task(ctx: click.Context, uid: str) -> None:
    """Get a task by UID."""
    client = ctx.obj["client"]
    t = client.tasks.get(uid)
    print_detail(
        f"Task: {t.uid}",
        [
            ("UID", t.uid),
            ("Name", t.name or "—"),
            ("Type", t.type or "—"),
            ("Status", t.status or "—"),
            ("Project", t.project or "—"),
            ("Created", str(t.created_at or "—")),
            ("Updated", str(t.updated_at or "—")),
        ],
        json_keys=["uid", "name", "type", "status", "project", "created_at", "updated_at"],
    )
