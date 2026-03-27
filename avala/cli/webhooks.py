"""CLI commands for webhooks."""

from __future__ import annotations

import click

from avala.cli._output import print_detail, print_table


@click.group("webhooks")
def webhooks() -> None:
    """Manage webhooks."""


@webhooks.command("list")
@click.option("--limit", type=int, default=None, help="Maximum number of results")
@click.pass_context
def list_webhooks(ctx: click.Context, limit: int | None) -> None:
    """List webhooks."""
    client = ctx.obj["client"]
    page = client.webhooks.list(limit=limit)
    rows = [
        (
            w.uid,
            w.target_url,
            ", ".join(w.events),
            "Yes" if w.is_active else "No",
            str(w.created_at or "—"),
        )
        for w in page.items
    ]
    print_table(
        "Webhooks",
        ["UID", "Target URL", "Events", "Active", "Created"],
        rows,
        json_keys=["uid", "target_url", "events", "is_active", "created_at"],
    )


@webhooks.command("get")
@click.argument("uid")
@click.pass_context
def get_webhook(ctx: click.Context, uid: str) -> None:
    """Get a webhook by UID."""
    client = ctx.obj["client"]
    w = client.webhooks.get(uid)
    print_detail(
        f"Webhook: {w.uid}",
        [
            ("UID", w.uid),
            ("Target URL", w.target_url),
            ("Events", ", ".join(w.events)),
            ("Active", "Yes" if w.is_active else "No"),
            ("Created", str(w.created_at or "—")),
            ("Updated", str(w.updated_at or "—")),
        ],
        json_keys=["uid", "target_url", "events", "is_active", "created_at", "updated_at"],
    )


@webhooks.command("create")
@click.option("--target-url", required=True, help="Webhook target URL (HTTPS)")
@click.option("--events", required=True, help="Comma-separated list of event types")
@click.option("--secret", default=None, help="HMAC signing secret (auto-generated if omitted)")
@click.pass_context
def create_webhook(
    ctx: click.Context,
    target_url: str,
    events: str,
    secret: str | None,
) -> None:
    """Create a new webhook."""
    client = ctx.obj["client"]
    event_list = [e.strip() for e in events.split(",")]
    kwargs: dict = {"target_url": target_url, "events": event_list}
    if secret is not None:
        kwargs["secret"] = secret
    w = client.webhooks.create(**kwargs)
    click.echo(f"Webhook created: {w.uid}")
    if w.secret:
        click.echo(f"Secret: {w.secret}")
        click.echo("Save this secret — it will not be shown again.")


@webhooks.command("delete")
@click.argument("uid")
@click.confirmation_option(prompt="Are you sure you want to delete this webhook?")
@click.pass_context
def delete_webhook(ctx: click.Context, uid: str) -> None:
    """Delete a webhook."""
    client = ctx.obj["client"]
    client.webhooks.delete(uid)
    click.echo(f"Webhook {uid} deleted.")


@webhooks.command("test")
@click.argument("uid")
@click.pass_context
def test_webhook(ctx: click.Context, uid: str) -> None:
    """Test a webhook."""
    client = ctx.obj["client"]
    result = client.webhooks.test(uid)
    if result.get("success"):
        click.echo("Webhook test passed.")
    else:
        click.echo(f"Webhook test failed: {result.get('message', 'unknown error')}")


@webhooks.command("deliveries")
@click.option("--limit", type=int, default=None, help="Maximum number of results")
@click.pass_context
def list_deliveries(ctx: click.Context, limit: int | None) -> None:
    """List webhook deliveries."""
    client = ctx.obj["client"]
    page = client.webhook_deliveries.list(limit=limit)
    rows = [
        (
            d.uid,
            d.event_type,
            d.status or "—",
            str(d.response_status) if d.response_status is not None else "—",
            str(d.attempts),
            str(d.created_at or "—"),
        )
        for d in page.items
    ]
    print_table(
        "Webhook Deliveries",
        ["UID", "Event", "Status", "HTTP Status", "Attempts", "Created"],
        rows,
        json_keys=["uid", "event_type", "status", "response_status", "attempts", "created_at"],
    )
