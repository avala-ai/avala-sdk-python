"""CLI commands for projects."""

from __future__ import annotations

import click

from avala.cli._output import print_detail, print_table


@click.group()
def projects() -> None:
    """Manage projects."""


@projects.command("list")
@click.option("--limit", type=int, default=None, help="Maximum number of results")
@click.pass_context
def list_projects(ctx: click.Context, limit: int | None) -> None:
    """List projects."""
    client = ctx.obj["client"]
    page = client.projects.list(limit=limit)
    rows = [(p.uid, p.name, p.status or "—", str(p.created_at or "—")) for p in page.items]
    print_table(
        "Projects", ["UID", "Name", "Status", "Created"], rows, json_keys=["uid", "name", "status", "created_at"]
    )


@projects.command("get")
@click.argument("uid")
@click.pass_context
def get_project(ctx: click.Context, uid: str) -> None:
    """Get a project by UID."""
    client = ctx.obj["client"]
    p = client.projects.get(uid)
    print_detail(
        f"Project: {p.name}",
        [
            ("UID", p.uid),
            ("Name", p.name),
            ("Status", p.status or "—"),
            ("Created", str(p.created_at or "—")),
            ("Updated", str(p.updated_at or "—")),
        ],
        json_keys=["uid", "name", "status", "created_at", "updated_at"],
    )
