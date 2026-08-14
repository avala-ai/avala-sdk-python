"""CLI commands for storage configs."""

from __future__ import annotations

import click

from avala.cli._output import print_detail, print_table


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


@storage_configs.command("get")
@click.argument("uid")
@click.pass_context
def get_storage_config(ctx: click.Context, uid: str) -> None:
    """Get a storage configuration by UID."""
    client = ctx.obj["client"]
    sc = client.storage_configs.get(uid)
    print_detail(
        f"Storage Config: {sc.name}",
        [
            ("UID", sc.uid),
            ("Name", sc.name),
            ("Provider", sc.provider),
            ("S3 Bucket", sc.s3_bucket_name or "—"),
            ("S3 Region", sc.s3_bucket_region or "—"),
            ("S3 Prefix", sc.s3_bucket_prefix or "—"),
            ("S3 Accelerated", "Yes" if sc.s3_is_accelerated else "No"),
            ("Auth Method", sc.s3_auth_method or "—"),
            ("GCS Bucket", sc.gc_storage_bucket_name or "—"),
            ("GCS Prefix", sc.gc_storage_prefix or "—"),
            ("Verified", "Yes" if sc.is_verified else "No"),
            ("Created", str(sc.created_at or "—")),
        ],
        json_keys=[
            "uid",
            "name",
            "provider",
            "s3_bucket_name",
            "s3_bucket_region",
            "s3_bucket_prefix",
            "s3_is_accelerated",
            "s3_auth_method",
            "gc_storage_bucket_name",
            "gc_storage_prefix",
            "is_verified",
            "created_at",
        ],
    )


@storage_configs.command("create")
@click.option("--name", required=True, help="Name for the storage config")
@click.option("--provider", required=True, type=click.Choice(["aws_s3", "gc_storage"]), help="Cloud provider")
@click.option("--s3-bucket-name", default=None, help="S3 bucket name")
@click.option("--s3-bucket-region", default=None, help="S3 bucket region")
@click.option("--s3-bucket-prefix", default=None, help="S3 bucket prefix")
@click.option(
    "--s3-auth-method",
    default=None,
    type=click.Choice(["access_key", "iam_role"]),
    help="How Avala authenticates to the bucket. 'iam_role' is the keyless zero-copy setup.",
)
@click.option(
    "--s3-role-arn",
    default=None,
    help="IAM role Avala assumes for --s3-auth-method=iam_role (arn:aws:iam::<acct>:role/<name>).",
)
@click.option(
    "--organization",
    default=None,
    help=(
        "Organization slug that will own this config. Required if you belong to more than one; "
        "omitting it uses your sole organization. There are no personal storage configs."
    ),
)
@click.option(
    "--s3-access-key-id",
    default=None,
    help="S3 access key ID (prefer AVALA_S3_ACCESS_KEY_ID env var)",
)
@click.option(
    "--s3-secret-access-key",
    default=None,
    help="S3 secret access key — NOT RECOMMENDED; use AVALA_S3_SECRET_ACCESS_KEY env var or interactive prompt.",
)
@click.option("--gc-bucket-name", default=None, help="GCS bucket name")
@click.option("--gc-prefix", default=None, help="GCS prefix")
@click.option(
    "--gc-auth-json",
    default=None,
    help="GCS auth JSON content — NOT RECOMMENDED; use AVALA_GC_AUTH_JSON env var or interactive prompt.",
)
@click.pass_context
def create_storage_config(
    ctx: click.Context,
    name: str,
    provider: str,
    s3_bucket_name: str | None,
    s3_bucket_region: str | None,
    s3_bucket_prefix: str | None,
    s3_auth_method: str | None,
    s3_role_arn: str | None,
    organization: str | None,
    s3_access_key_id: str | None,
    s3_secret_access_key: str | None,
    gc_bucket_name: str | None,
    gc_prefix: str | None,
    gc_auth_json: str | None,
) -> None:
    """Create a new storage configuration.

    Secrets (``--s3-secret-access-key``, ``--gc-auth-json``) passed on the
    command line are visible in shell history and ``ps`` output. Prefer
    environment variables (``AVALA_S3_SECRET_ACCESS_KEY``,
    ``AVALA_GC_AUTH_JSON``) or the interactive prompt triggered when the
    flags are omitted.
    """
    import os

    client = ctx.obj["client"]

    # Warn loudly if a secret was passed on the command line.
    if s3_secret_access_key is not None:
        click.echo(
            click.style(
                "WARNING: --s3-secret-access-key on the command line is visible in shell "
                "history and 'ps' output. Use AVALA_S3_SECRET_ACCESS_KEY env var instead.",
                fg="yellow",
            ),
            err=True,
        )
    if gc_auth_json is not None:
        click.echo(
            click.style(
                "WARNING: --gc-auth-json on the command line is visible in shell history "
                "and 'ps' output. Use AVALA_GC_AUTH_JSON env var instead.",
                fg="yellow",
            ),
            err=True,
        )

    # Env var fallback — values only read when the flag is omitted, so explicit
    # --flag still wins (with the warning above). No interactive prompt: it
    # would change the behavior of non-credential invocations.
    if s3_access_key_id is None:
        s3_access_key_id = os.environ.get("AVALA_S3_ACCESS_KEY_ID") or None
    if s3_secret_access_key is None:
        s3_secret_access_key = os.environ.get("AVALA_S3_SECRET_ACCESS_KEY") or None
    if gc_auth_json is None:
        gc_auth_json = os.environ.get("AVALA_GC_AUTH_JSON") or None

    # Never submit a credential the selected config cannot use. The env
    # fallbacks above are unconditional, so an operator with AVALA_S3_* or
    # AVALA_GC_AUTH_JSON exported — likely, if they have ever configured either
    # — would send them regardless of what this config is for. The server
    # validates only the selected provider's branch, but
    # `StorageConfigCreateSerializer` persists every submitted model field, so
    # the config quietly stores secrets it never uses and nobody rotates.
    #
    # Two independent cases, and fixing only the first left a private key
    # behind: a GCS service-account JSON riding along on a keyless S3 config.
    dropped: list[str] = []
    if provider != "gc_storage" and gc_auth_json:
        dropped.append("GCS service-account JSON")
        gc_auth_json = None
    # A role ARN is only meaningful for an AWS iam_role config. The serializer
    # validates just the selected branch but persists every submitted field, so
    # an ARN sent alongside a GCS or access-key config is stored indefinitely —
    # an unrelated IAM identity nobody will audit. Same rule as the credentials
    # below; fixing only those left this newly exposed field behind.
    if s3_role_arn and not (provider == "aws_s3" and s3_auth_method == "iam_role"):
        dropped.append("S3 role ARN")
        s3_role_arn = None
    s3_keys_unusable = provider != "aws_s3" or s3_auth_method == "iam_role"
    if s3_keys_unusable and (s3_access_key_id or s3_secret_access_key):
        dropped.append("S3 access-key credentials")
        s3_access_key_id = None
        s3_secret_access_key = None
    if dropped:
        click.echo(
            click.style(
                f"Ignoring {' and '.join(dropped)}: this config does not use them, "
                "and submitting them would persist secrets it never reads.",
                fg="yellow",
            ),
            err=True,
        )

    # The resource has supported these since it was written; only the CLI never
    # surfaced them, which made the keyless (iam_role) setup unreachable from
    # the command line — including from the onboarding runbook, whose Step 2
    # documented flags that Click rejected before any request was made.
    sc = client.storage_configs.create(
        name=name,
        provider=provider,
        organization=organization,
        s3_bucket_name=s3_bucket_name,
        s3_bucket_region=s3_bucket_region,
        s3_bucket_prefix=s3_bucket_prefix,
        s3_auth_method=s3_auth_method,
        s3_role_arn=s3_role_arn,
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
