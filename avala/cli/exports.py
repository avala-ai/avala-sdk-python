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
    print_table(
        "Exports",
        ["UID", "Status", "Download URL", "Created"],
        rows,
        json_keys=["uid", "status", "download_url", "created_at"],
    )


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
        json_keys=["uid", "status", "download_url", "created_at", "updated_at"],
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


@exports.command("wait")
@click.argument("uid")
@click.option("--interval", type=float, default=2.0, help="Seconds between polls (default: 2.0)")
@click.option("--timeout", type=float, default=300.0, help="Maximum seconds to wait (default: 300)")
@click.pass_context
def wait_export(ctx: click.Context, uid: str, interval: float, timeout: float) -> None:
    """Wait for an export to complete."""
    client = ctx.obj["client"]
    click.echo(f"Waiting for export {uid}", nl=False, err=True)
    try:
        e = client.exports.wait(
            uid, interval=interval, timeout=timeout, _on_poll=lambda: click.echo(".", nl=False, err=True)
        )
    except TimeoutError as exc:
        click.echo(err=True)
        raise click.ClickException(str(exc))
    click.echo(err=True)
    print_detail(
        f"Export: {e.uid}",
        [
            ("UID", e.uid),
            ("Status", e.status or "—"),
            ("Download URL", e.download_url or "—"),
            ("Created", str(e.created_at or "—")),
            ("Updated", str(e.updated_at or "—")),
        ],
        json_keys=["uid", "status", "download_url", "created_at", "updated_at"],
    )
