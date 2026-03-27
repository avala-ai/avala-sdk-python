"""CLI commands for inference providers."""

from __future__ import annotations

import json

import click

from avala.cli._output import print_detail, print_table


@click.group("inference-providers")
def inference_providers() -> None:
    """Manage inference providers."""


@inference_providers.command("list")
@click.option("--limit", type=int, default=None, help="Maximum number of results")
@click.pass_context
def list_inference_providers(ctx: click.Context, limit: int | None) -> None:
    """List inference providers."""
    client = ctx.obj["client"]
    page = client.inference_providers.list(limit=limit)
    rows = [
        (
            p.uid,
            p.name,
            p.provider_type or "—",
            "Yes" if p.is_active else "No",
            str(p.created_at or "—"),
        )
        for p in page.items
    ]
    print_table(
        "Inference Providers",
        ["UID", "Name", "Type", "Active", "Created"],
        rows,
        json_keys=["uid", "name", "provider_type", "is_active", "created_at"],
    )


@inference_providers.command("get")
@click.argument("uid")
@click.pass_context
def get_inference_provider(ctx: click.Context, uid: str) -> None:
    """Get an inference provider by UID."""
    client = ctx.obj["client"]
    p = client.inference_providers.get(uid)
    print_detail(
        f"Inference Provider: {p.name}",
        [
            ("UID", p.uid),
            ("Name", p.name),
            ("Description", p.description or "—"),
            ("Type", p.provider_type or "—"),
            ("Config", json.dumps(p.config) if p.config else "—"),
            ("Active", "Yes" if p.is_active else "No"),
            ("Project", p.project or "—"),
            ("Last Test", str(p.last_test_at or "—")),
            ("Last Test OK", str(p.last_test_success) if p.last_test_success is not None else "—"),
            ("Created", str(p.created_at or "—")),
            ("Updated", str(p.updated_at or "—")),
        ],
        json_keys=[
            "uid",
            "name",
            "description",
            "provider_type",
            "config",
            "is_active",
            "project",
            "last_test_at",
            "last_test_success",
            "created_at",
            "updated_at",
        ],
    )


@inference_providers.command("create")
@click.option("--name", required=True, help="Provider name")
@click.option("--provider-type", required=True, type=click.Choice(["http", "sagemaker"]), help="Provider type")
@click.option("--config", "config_json", required=True, help="Provider config as JSON string")
@click.option("--description", default=None, help="Provider description")
@click.option("--project", default=None, help="Project UID to scope the provider to")
@click.pass_context
def create_inference_provider(
    ctx: click.Context,
    name: str,
    provider_type: str,
    config_json: str,
    description: str | None,
    project: str | None,
) -> None:
    """Create a new inference provider."""
    client = ctx.obj["client"]
    try:
        config = json.loads(config_json)
    except json.JSONDecodeError as e:
        raise click.BadParameter(f"Invalid JSON for --config: {e}") from e
    kwargs: dict = {"name": name, "provider_type": provider_type, "config": config}
    if description is not None:
        kwargs["description"] = description
    if project is not None:
        kwargs["project"] = project
    p = client.inference_providers.create(**kwargs)
    click.echo(f"Inference provider created: {p.uid} ({p.name})")


@inference_providers.command("delete")
@click.argument("uid")
@click.confirmation_option(prompt="Are you sure you want to delete this inference provider?")
@click.pass_context
def delete_inference_provider(ctx: click.Context, uid: str) -> None:
    """Delete an inference provider."""
    client = ctx.obj["client"]
    client.inference_providers.delete(uid)
    click.echo(f"Inference provider {uid} deleted.")


@inference_providers.command("test")
@click.argument("uid")
@click.pass_context
def test_inference_provider(ctx: click.Context, uid: str) -> None:
    """Test connectivity for an inference provider."""
    client = ctx.obj["client"]
    result = client.inference_providers.test(uid)
    if result.get("success"):
        click.echo("Connection test passed.")
    else:
        click.echo(f"Connection test failed: {result.get('message', 'unknown error')}")
