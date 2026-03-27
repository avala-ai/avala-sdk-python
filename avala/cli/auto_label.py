"""CLI commands for auto-label jobs."""

from __future__ import annotations

import click

from avala.cli._output import print_detail, print_table


@click.group("auto-label")
def auto_label() -> None:
    """Manage auto-label jobs."""


@auto_label.command("list")
@click.option("--project", default=None, help="Filter by project UID")
@click.option("--limit", type=int, default=None, help="Maximum number of results")
@click.pass_context
def list_auto_label_jobs(ctx: click.Context, project: str | None, limit: int | None) -> None:
    """List auto-label jobs."""
    client = ctx.obj["client"]
    page = client.auto_label_jobs.list(project=project, limit=limit)
    rows = [
        (
            j.uid,
            j.status or "—",
            j.model_type or "—",
            f"{j.progress_pct:.1f}%" if j.progress_pct is not None else "—",
            str(j.created_at or "—"),
        )
        for j in page.items
    ]
    print_table(
        "Auto-Label Jobs",
        ["UID", "Status", "Model", "Progress", "Created"],
        rows,
        json_keys=["uid", "status", "model_type", "progress_pct", "created_at"],
    )


@auto_label.command("get")
@click.argument("uid")
@click.pass_context
def get_auto_label_job(ctx: click.Context, uid: str) -> None:
    """Get an auto-label job by UID."""
    client = ctx.obj["client"]
    j = client.auto_label_jobs.get(uid)
    print_detail(
        f"Auto-Label Job: {j.uid}",
        [
            ("UID", j.uid),
            ("Status", j.status or "—"),
            ("Model Type", j.model_type or "—"),
            ("Confidence", str(j.confidence_threshold) if j.confidence_threshold is not None else "—"),
            ("Labels", ", ".join(j.labels) if j.labels else "all"),
            ("Dry Run", "Yes" if j.dry_run else "No"),
            ("Total Items", str(j.total_items)),
            ("Processed", str(j.processed_items)),
            ("Successful", str(j.successful_items)),
            ("Failed", str(j.failed_items)),
            ("Skipped", str(j.skipped_items)),
            ("Progress", f"{j.progress_pct:.1f}%" if j.progress_pct is not None else "—"),
            ("Error", j.error_message or "—"),
            ("Started", str(j.started_at or "—")),
            ("Completed", str(j.completed_at or "—")),
            ("Created", str(j.created_at or "—")),
        ],
        json_keys=[
            "uid",
            "status",
            "model_type",
            "confidence_threshold",
            "labels",
            "dry_run",
            "total_items",
            "processed_items",
            "successful_items",
            "failed_items",
            "skipped_items",
            "progress_pct",
            "error_message",
            "started_at",
            "completed_at",
            "created_at",
        ],
    )


@auto_label.command("create")
@click.option("--project", required=True, help="Project UID")
@click.option("--model-type", default=None, type=click.Choice(["sam3", "yolo"]), help="Inference model to use")
@click.option("--confidence-threshold", default=None, type=float, help="Minimum confidence (0.0-1.0)")
@click.option("--labels", default=None, help="Comma-separated list of labels to filter")
@click.option("--dry-run", is_flag=True, default=False, help="Run inference without saving results")
@click.pass_context
def create_auto_label_job(
    ctx: click.Context,
    project: str,
    model_type: str | None,
    confidence_threshold: float | None,
    labels: str | None,
    dry_run: bool,
) -> None:
    """Create a new auto-label job."""
    client = ctx.obj["client"]
    kwargs: dict = {}
    if model_type is not None:
        kwargs["model_type"] = model_type
    if confidence_threshold is not None:
        kwargs["confidence_threshold"] = confidence_threshold
    if labels is not None:
        kwargs["labels"] = [label.strip() for label in labels.split(",")]
    if dry_run:
        kwargs["dry_run"] = True
    j = client.auto_label_jobs.create(project, **kwargs)
    click.echo(f"Auto-label job created: {j.uid} (status: {j.status})")


@auto_label.command("cancel")
@click.argument("uid")
@click.confirmation_option(prompt="Are you sure you want to cancel this auto-label job?")
@click.pass_context
def cancel_auto_label_job(ctx: click.Context, uid: str) -> None:
    """Cancel an auto-label job."""
    client = ctx.obj["client"]
    client.auto_label_jobs.cancel(uid)
    click.echo(f"Auto-label job {uid} cancelled.")
