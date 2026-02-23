"""CLI commands for datasets."""

from __future__ import annotations

import click

from avala.cli._output import print_detail, print_table


@click.group()
def datasets() -> None:
    """Manage datasets."""


@datasets.command("list")
@click.option("--limit", type=int, default=None, help="Maximum number of results")
@click.pass_context
def list_datasets(ctx: click.Context, limit: int | None) -> None:
    """List datasets."""
    client = ctx.obj["client"]
    page = client.datasets.list(limit=limit)
    rows = [(d.uid, d.name, d.slug, str(d.item_count), d.data_type or "—") for d in page.items]
    print_table("Datasets", ["UID", "Name", "Slug", "Items", "Type"], rows)


@datasets.command("get")
@click.argument("uid")
@click.pass_context
def get_dataset(ctx: click.Context, uid: str) -> None:
    """Get a dataset by UID."""
    client = ctx.obj["client"]
    d = client.datasets.get(uid)
    print_detail(
        f"Dataset: {d.name}",
        [
            ("UID", d.uid),
            ("Name", d.name),
            ("Slug", d.slug),
            ("Items", str(d.item_count)),
            ("Type", d.data_type or "—"),
            ("Created", str(d.created_at or "—")),
            ("Updated", str(d.updated_at or "—")),
        ],
    )
