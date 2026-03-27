"""CLI commands for consensus."""

from __future__ import annotations

import click

from avala.cli._output import print_detail, print_table


@click.group("consensus")
def consensus() -> None:
    """Manage consensus scoring."""


@consensus.command("summary")
@click.option("--project", required=True, help="Project UID")
@click.pass_context
def consensus_summary(ctx: click.Context, project: str) -> None:
    """Get consensus summary for a project."""
    client = ctx.obj["client"]
    s = client.consensus.get_summary(project)
    print_detail(
        "Consensus Summary",
        [
            ("Mean Score", f"{s.mean_score:.4f}"),
            ("Median Score", f"{s.median_score:.4f}"),
            ("Min Score", f"{s.min_score:.4f}"),
            ("Max Score", f"{s.max_score:.4f}"),
            ("Total Items", str(s.total_items)),
            ("Items with Consensus", str(s.items_with_consensus)),
        ],
        json_keys=["mean_score", "median_score", "min_score", "max_score", "total_items", "items_with_consensus"],
    )


@consensus.command("scores")
@click.option("--project", required=True, help="Project UID")
@click.option("--limit", type=int, default=None, help="Maximum number of results")
@click.pass_context
def consensus_scores(ctx: click.Context, project: str, limit: int | None) -> None:
    """List consensus scores for a project."""
    client = ctx.obj["client"]
    page = client.consensus.list_scores(project, limit=limit)
    rows = [
        (
            s.uid,
            s.dataset_item_uid,
            s.task_name,
            f"{s.score:.4f}",
            str(s.annotator_count),
            str(s.created_at or "—"),
        )
        for s in page.items
    ]
    print_table(
        "Consensus Scores",
        ["UID", "Item", "Task", "Score", "Annotators", "Created"],
        rows,
        json_keys=["uid", "dataset_item_uid", "task_name", "score", "annotator_count", "created_at"],
    )


@consensus.command("compute")
@click.option("--project", required=True, help="Project UID")
@click.pass_context
def consensus_compute(ctx: click.Context, project: str) -> None:
    """Compute consensus scores for a project."""
    client = ctx.obj["client"]
    result = client.consensus.compute(project)
    click.echo(f"Status: {result.status} — {result.message}")


@consensus.command("config")
@click.option("--project", required=True, help="Project UID")
@click.option("--iou-threshold", type=float, default=None, help="Update IoU threshold (0.0-1.0)")
@click.option("--min-agreement-ratio", type=float, default=None, help="Update min agreement ratio (0.0-1.0)")
@click.option("--min-annotations", type=int, default=None, help="Update min annotations required")
@click.pass_context
def consensus_config(
    ctx: click.Context,
    project: str,
    iou_threshold: float | None,
    min_agreement_ratio: float | None,
    min_annotations: int | None,
) -> None:
    """Get or update consensus config for a project."""
    client = ctx.obj["client"]
    if iou_threshold is not None or min_agreement_ratio is not None or min_annotations is not None:
        kwargs: dict = {}
        if iou_threshold is not None:
            kwargs["iou_threshold"] = iou_threshold
        if min_agreement_ratio is not None:
            kwargs["min_agreement_ratio"] = min_agreement_ratio
        if min_annotations is not None:
            kwargs["min_annotations"] = min_annotations
        c = client.consensus.update_config(project, **kwargs)
        click.echo("Consensus config updated.")
    else:
        c = client.consensus.get_config(project)
    print_detail(
        "Consensus Config",
        [
            ("UID", c.uid),
            ("IoU Threshold", str(c.iou_threshold)),
            ("Min Agreement Ratio", str(c.min_agreement_ratio)),
            ("Min Annotations", str(c.min_annotations)),
            ("Created", str(c.created_at or "—")),
            ("Updated", str(c.updated_at or "—")),
        ],
        json_keys=["uid", "iou_threshold", "min_agreement_ratio", "min_annotations", "created_at", "updated_at"],
    )
