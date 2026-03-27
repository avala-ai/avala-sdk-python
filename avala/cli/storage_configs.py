"""CLI commands for storage configs."""

from __future__ import annotations

import click

from avala.cli._output import print_table


@click.group("storage-configs")
def storage_configs() -> None:
    """Manage storage configurations."""


@storage_configs.command("list")
@click.pass_context
def list_storage_configs(ctx: click.Context) -> None:
    """List storage configurations."""
    client = ctx.obj["client"]
    page = client.storage_configs.list()
    rows = [
        (
            sc.uid,
            sc.name,
            sc.provider,
            "Yes" if sc.is_verified else "No",
            str(sc.created_at or "—"),
        )
        for sc in page.items
    ]
    print_table(
        "Storage Configs",
        ["UID", "Name", "Provider", "Verified", "Created"],
        rows,
        json_keys=["uid", "name", "provider", "is_verified", "created_at"],
    )


@storage_configs.command("create")
@click.option("--name", required=True, help="Name for the storage config")
@click.option("--provider", required=True, type=click.Choice(["aws_s3", "gc_storage"]), help="Cloud provider")
@click.option("--s3-bucket-name", default=None, help="S3 bucket name")
@click.option("--s3-bucket-region", default=None, help="S3 bucket region")
@click.option("--s3-bucket-prefix", default=None, help="S3 bucket prefix")
@click.option("--s3-access-key-id", default=None, help="S3 access key ID")
@click.option("--s3-secret-access-key", default=None, help="S3 secret access key")
@click.option("--gc-bucket-name", default=None, help="GCS bucket name")
@click.option("--gc-prefix", default=None, help="GCS prefix")
@click.option("--gc-auth-json", default=None, help="GCS auth JSON content")
@click.pass_context
def create_storage_config(
    ctx: click.Context,
    name: str,
    provider: str,
    s3_bucket_name: str | None,
    s3_bucket_region: str | None,
    s3_bucket_prefix: str | None,
    s3_access_key_id: str | None,
    s3_secret_access_key: str | None,
    gc_bucket_name: str | None,
    gc_prefix: str | None,
    gc_auth_json: str | None,
) -> None:
    """Create a new storage configuration."""
    client = ctx.obj["client"]
    sc = client.storage_configs.create(
        name=name,
        provider=provider,
        s3_bucket_name=s3_bucket_name,
        s3_bucket_region=s3_bucket_region,
        s3_bucket_prefix=s3_bucket_prefix,
        s3_access_key_id=s3_access_key_id,
        s3_secret_access_key=s3_secret_access_key,
        gc_storage_bucket_name=gc_bucket_name,
        gc_storage_prefix=gc_prefix,
        gc_storage_auth_json_content=gc_auth_json,
    )
    click.echo(f"Storage config created: {sc.uid} ({sc.name})")


@storage_configs.command("test")
@click.argument("uid")
@click.pass_context
def test_storage_config(ctx: click.Context, uid: str) -> None:
    """Test connectivity for a storage configuration."""
    client = ctx.obj["client"]
    result = client.storage_configs.test(uid)
    if result.get("verified"):
        click.echo("Connection test passed.")
    else:
        click.echo("Connection test failed.")
        if errors := result.get("errors"):
            for field, msgs in errors.items():
                for msg in msgs:
                    click.echo(f"  {field}: {msg}")


@storage_configs.command("delete")
@click.argument("uid")
@click.confirmation_option(prompt="Are you sure you want to delete this storage config?")
@click.pass_context
def delete_storage_config(ctx: click.Context, uid: str) -> None:
    """Delete a storage configuration."""
    client = ctx.obj["client"]
    client.storage_configs.delete(uid)
    click.echo(f"Storage config {uid} deleted.")
