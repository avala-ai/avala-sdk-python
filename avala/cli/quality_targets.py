"""CLI commands for quality targets."""

from __future__ import annotations

import click

from avala.cli._output import print_detail, print_table


@click.group("quality-targets")
def quality_targets() -> None:
    """Manage quality targets."""


@quality_targets.command("list")
@click.option("--project", required=True, help="Project UID")
@click.option("--limit", type=int, default=None, help="Maximum number of results")
@click.pass_context
def list_quality_targets(ctx: click.Context, project: str, limit: int | None) -> None:
    """List quality targets for a project."""
    client = ctx.obj["client"]
    page = client.quality_targets.list(project, limit=limit)
    rows = [
        (
            t.uid,
            t.name,
            t.metric,
            f"{t.operator} {t.threshold}",
            "Yes" if t.is_breached else "No",
            str(t.created_at or "—"),
        )
        for t in page.items
    ]
    print_table(
        "Quality Targets",
        ["UID", "Name", "Metric", "Threshold", "Breached", "Created"],
        rows,
        json_keys=["uid", "name", "metric", "threshold", "is_breached", "created_at"],
    )


@quality_targets.command("get")
@click.option("--project", required=True, help="Project UID")
@click.argument("uid")
@click.pass_context
def get_quality_target(ctx: click.Context, project: str, uid: str) -> None:
    """Get a quality target by UID."""
    client = ctx.obj["client"]
    t = client.quality_targets.get(project, uid)
    print_detail(
        f"Quality Target: {t.name}",
        [
            ("UID", t.uid),
            ("Name", t.name),
            ("Metric", t.metric),
            ("Operator", t.operator),
            ("Threshold", str(t.threshold)),
            ("Severity", t.severity or "—"),
            ("Active", "Yes" if t.is_active else "No"),
            ("Notify Webhook", "Yes" if t.notify_webhook else "No"),
            ("Notify Emails", ", ".join(t.notify_emails) if t.notify_emails else "—"),
            ("Last Evaluated", str(t.last_evaluated_at or "—")),
            ("Last Value", str(t.last_value) if t.last_value is not None else "—"),
            ("Breached", "Yes" if t.is_breached else "No"),
            ("Breach Count", str(t.breach_count)),
            ("Last Breached", str(t.last_breached_at or "—")),
            ("Created", str(t.created_at or "—")),
            ("Updated", str(t.updated_at or "—")),
        ],
        json_keys=[
            "uid",
            "name",
            "metric",
            "operator",
            "threshold",
            "severity",
            "is_active",
            "notify_webhook",
            "notify_emails",
            "last_evaluated_at",
            "last_value",
            "is_breached",
            "breach_count",
            "last_breached_at",
            "created_at",
            "updated_at",
        ],
    )


@quality_targets.command("create")
@click.option("--project", required=True, help="Project UID")
@click.option("--name", required=True, help="Target name")
@click.option("--metric", required=True, help="Metric to monitor")
@click.option("--threshold", required=True, type=float, help="Threshold value")
@click.option("--operator", default=None, help="Comparison operator (gt, lt, gte, lte)")
@click.option("--severity", default=None, type=click.Choice(["warning", "critical"]), help="Alert severity")
@click.pass_context
def create_quality_target(
    ctx: click.Context,
    project: str,
    name: str,
    metric: str,
    threshold: float,
    operator: str | None,
    severity: str | None,
) -> None:
    """Create a new quality target."""
    client = ctx.obj["client"]
    kwargs: dict = {"name": name, "metric": metric, "threshold": threshold}
    if operator is not None:
        kwargs["operator"] = operator
    if severity is not None:
        kwargs["severity"] = severity
    t = client.quality_targets.create(project, **kwargs)
    click.echo(f"Quality target created: {t.uid} ({t.name})")


@quality_targets.command("delete")
@click.option("--project", required=True, help="Project UID")
@click.argument("uid")
@click.confirmation_option(prompt="Are you sure you want to delete this quality target?")
@click.pass_context
def delete_quality_target(ctx: click.Context, project: str, uid: str) -> None:
    """Delete a quality target."""
    client = ctx.obj["client"]
    client.quality_targets.delete(project, uid)
    click.echo(f"Quality target {uid} deleted.")


@quality_targets.command("evaluate")
@click.option("--project", required=True, help="Project UID")
@click.pass_context
def evaluate_quality_targets(ctx: click.Context, project: str) -> None:
    """Evaluate all quality targets for a project."""
    client = ctx.obj["client"]
    results = client.quality_targets.evaluate(project)
    rows = [
        (
            r.uid,
            r.name,
            r.metric,
            f"{r.operator} {r.threshold}",
            str(r.current_value),
            "Yes" if r.is_breached else "No",
            r.severity or "—",
        )
        for r in results
    ]
    print_table(
        "Quality Target Evaluations",
        ["UID", "Name", "Metric", "Threshold", "Current", "Breached", "Severity"],
        rows,
        json_keys=["uid", "name", "metric", "threshold", "current_value", "is_breached", "severity"],
    )
