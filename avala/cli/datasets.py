"""CLI commands for datasets."""

from __future__ import annotations

import json

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


@datasets.command("create")
@click.option("--name", required=True, help="Display name for the dataset")
@click.option("--slug", required=True, help="URL-friendly identifier")
@click.option(
    "--data-type",
    required=True,
    type=click.Choice(["image", "video", "lidar", "mcap"]),
    help="Type of data in the dataset",
)
@click.option("--is-sequence", is_flag=True, default=False, help="Dataset contains sequences")
@click.option(
    "--visibility",
    default="private",
    type=click.Choice(["private", "public"]),
    help="Dataset visibility (default: private)",
)
@click.option("--create-metadata/--no-create-metadata", default=True, help="Create dataset metadata")
@click.option("--provider-config", default=None, help="Provider config as JSON string")
@click.option("--owner", default=None, help="Dataset owner username or email")
@click.pass_context
def create_dataset(
    ctx: click.Context,
    name: str,
    slug: str,
    data_type: str,
    is_sequence: bool,
    visibility: str,
    create_metadata: bool,
    provider_config: str | None,
    owner: str | None,
) -> None:
    """Create a new dataset."""
    client = ctx.obj["client"]
    parsed_config = json.loads(provider_config) if provider_config else None
    d = client.datasets.create(
        name=name,
        slug=slug,
        data_type=data_type,
        is_sequence=is_sequence,
        visibility=visibility,
        create_metadata=create_metadata,
        provider_config=parsed_config,
        owner_name=owner,
    )
    click.echo(f"Dataset created: {d.uid} ({d.name})")
