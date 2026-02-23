"""CLI commands for exports."""

from __future__ import annotations

import click

from avala.cli._output import print_detail, print_table


@click.group()
def exports() -> None:
    """Manage exports."""


@exports.command("list")
@click.option("--limit", type=int, default=None, help="Maximum number of results")
@click.pass_context
def list_exports(ctx: click.Context, limit: int | None) -> None:
    """List exports."""
    client = ctx.obj["client"]
    page = client.exports.list(limit=limit)
    rows = [(e.uid, e.status or "—", e.download_url or "—", str(e.created_at or "—")) for e in page.items]
    print_table("Exports", ["UID", "Status", "Download URL", "Created"], rows)


@exports.command("get")
@click.argument("uid")
@click.pass_context
def get_export(ctx: click.Context, uid: str) -> None:
    """Get an export by UID."""
    client = ctx.obj["client"]
    e = client.exports.get(uid)
    print_detail(
        f"Export: {e.uid}",
        [
            ("UID", e.uid),
            ("Status", e.status or "—"),
            ("Download URL", e.download_url or "—"),
            ("Created", str(e.created_at or "—")),
            ("Updated", str(e.updated_at or "—")),
        ],
    )


@exports.command("create")
@click.option("--dataset", default=None, help="Dataset UID")
@click.option("--project", default=None, help="Project UID")
@click.pass_context
def create_export(ctx: click.Context, dataset: str | None, project: str | None) -> None:
    """Create a new export."""
    client = ctx.obj["client"]
    e = client.exports.create(dataset=dataset, project=project)
    click.echo(f"Export created: {e.uid} (status: {e.status})")
