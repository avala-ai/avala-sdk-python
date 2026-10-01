"""CLI commands for sequence outcome labels."""

from __future__ import annotations

import json
from typing import Any

import click

from avala.cli._output import print_detail, print_table
from avala.types.sequence_outcome import (
    AUTONOMY_LEVELS,
    EVALUATION_MEMBERSHIPS,
    LEAKAGE_GROUP_KEYS,
    OUTCOME_SOURCES,
    OUTCOMES,
    SequenceOutcome,
)

_DETAIL_KEYS = [
    "uid",
    "sequence_uid",
    "version",
    "outcome",
    "progress",
    "quality",
    "speed",
    "subtasks",
    "mistake_type",
    "recovery_type",
    "failure_stage",
    "autonomy_level",
    "model_version",
    "evaluation_membership",
    "leakage_groups",
    "source",
    "confidence",
    "created_at",
]


def _dash(value: Any) -> str:
    return "—" if value in (None, "", [], {}) else str(value)


def _print_outcome(o: SequenceOutcome) -> None:
    print_detail(
        f"Sequence outcome: {o.sequence_uid} (v{o.version})",
        [
            ("UID", o.uid),
            ("Sequence UID", o.sequence_uid),
            ("Version", str(o.version)),
            ("Outcome", o.outcome),
            ("Progress", _dash(o.progress)),
            ("Quality", _dash(o.quality)),
            ("Speed", _dash(o.speed)),
            ("Subtasks", _dash(json.dumps([s.model_dump() for s in o.subtasks]) if o.subtasks else None)),
            ("Mistake type", _dash(o.mistake_type)),
            ("Recovery type", _dash(o.recovery_type)),
            ("Failure stage", _dash(o.failure_stage)),
            ("Autonomy level", _dash(o.autonomy_level)),
            ("Model version", _dash(o.model_version)),
            ("Evaluation", _dash(o.evaluation_membership)),
            ("Leakage groups", _dash(json.dumps(o.leakage_groups) if o.leakage_groups else None)),
            ("Source", o.source),
            ("Confidence", _dash(o.confidence)),
            ("Created", _dash(o.created_at)),
        ],
        json_keys=_DETAIL_KEYS,
    )


def _split(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


@click.group("sequence-outcomes")
def sequence_outcomes() -> None:
    """Behavioral outcome labels of dataset sequences."""


@sequence_outcomes.command("get")
@click.argument("owner")
@click.argument("slug")
@click.argument("sequence_uid")
@click.pass_context
def get_outcome(ctx: click.Context, owner: str, slug: str, sequence_uid: str) -> None:
    """Show the current outcome label of a sequence."""
    _print_outcome(ctx.obj["client"].sequence_outcomes.get(owner, slug, sequence_uid))


@sequence_outcomes.command("history")
@click.argument("owner")
@click.argument("slug")
@click.argument("sequence_uid")
@click.pass_context
def outcome_history(ctx: click.Context, owner: str, slug: str, sequence_uid: str) -> None:
    """List every version of a sequence's outcome label, newest first."""
    versions = ctx.obj["client"].sequence_outcomes.history(owner, slug, sequence_uid)
    print_table(
        "Outcome history",
        ["Version", "Current", "Outcome", "Source", "Evaluation", "Created"],
        [
            (
                str(o.version),
                "Yes" if o.is_current else "No",
                o.outcome,
                o.source,
                _dash(o.evaluation_membership),
                _dash(o.created_at),
            )
            for o in versions
        ],
        json_keys=["version", "is_current", "outcome", "source", "evaluation_membership", "created_at"],
    )


@sequence_outcomes.command("list")
@click.argument("owner")
@click.argument("slug")
@click.option("--outcome", default=None, help=f"Comma-separated outcomes: {', '.join(OUTCOMES)}")
@click.option("--evaluation-membership", default=None, help=f"Comma-separated: {', '.join(EVALUATION_MEMBERSHIPS)}")
@click.option("--source", default=None, help=f"Comma-separated: {', '.join(OUTCOME_SOURCES)}")
@click.option("--limit", type=int, default=None, help="Maximum number of results")
@click.pass_context
def list_outcomes(
    ctx: click.Context,
    owner: str,
    slug: str,
    outcome: str | None,
    evaluation_membership: str | None,
    source: str | None,
    limit: int | None,
) -> None:
    """List current outcome labels in a dataset."""
    page = ctx.obj["client"].sequence_outcomes.list(
        owner,
        slug,
        outcome=_split(outcome),
        evaluation_membership=_split(evaluation_membership),
        source=_split(source),
        limit=limit,
    )
    print_table(
        "Sequence outcomes",
        ["Sequence UID", "Outcome", "Progress", "Evaluation", "Source", "Version"],
        [
            (
                o.sequence_uid,
                o.outcome,
                _dash(o.progress),
                _dash(o.evaluation_membership),
                o.source,
                str(o.version),
            )
            for o in page.items
        ],
        json_keys=["sequence_uid", "outcome", "progress", "evaluation_membership", "source", "version"],
    )


@sequence_outcomes.command("set")
@click.argument("owner")
@click.argument("slug")
@click.argument("sequence_uid")
@click.option("--outcome", required=True, type=click.Choice(OUTCOMES), help="Behavioral outcome")
@click.option("--progress", type=click.FloatRange(0, 1), default=None, help="Task progress in [0, 1]")
@click.option("--quality", type=click.IntRange(1, 5), default=None, help="Execution quality, 1-5")
@click.option("--speed", type=click.IntRange(1, 5), default=None, help="Execution speed, 1-5")
@click.option(
    "--subtasks-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="JSON file: ordered list of {label, start_ts, end_ts, outcome}",
)
@click.option("--mistake-type", default=None)
@click.option("--recovery-type", default=None)
@click.option("--failure-stage", default=None)
@click.option("--autonomy-level", type=click.Choice(AUTONOMY_LEVELS), default=None)
@click.option("--model-version", default=None, help="Policy/model version for robot rollouts")
@click.option("--evaluation-membership", type=click.Choice(EVALUATION_MEMBERSHIPS), default=None)
@click.option(
    "--leakage-group",
    "leakage_group",
    multiple=True,
    help=f"KEY=ID, repeatable; keys: {', '.join(LEAKAGE_GROUP_KEYS)}",
)
@click.option("--source", type=click.Choice(OUTCOME_SOURCES), default=None, help="Label source (default human)")
@click.option("--confidence", type=click.FloatRange(0, 1), default=None, help="Model confidence (source=model only)")
@click.pass_context
def set_outcome(
    ctx: click.Context,
    owner: str,
    slug: str,
    sequence_uid: str,
    outcome: str,
    progress: float | None,
    quality: int | None,
    speed: int | None,
    subtasks_file: str | None,
    mistake_type: str | None,
    recovery_type: str | None,
    failure_stage: str | None,
    autonomy_level: str | None,
    model_version: str | None,
    evaluation_membership: str | None,
    leakage_group: tuple[str, ...],
    source: str | None,
    confidence: float | None,
) -> None:
    """Record a new current outcome label (replaces the previous version wholesale)."""
    subtasks = None
    if subtasks_file:
        with open(subtasks_file, encoding="utf-8") as fh:
            subtasks = json.load(fh)
        if not isinstance(subtasks, list):
            raise click.BadParameter("must contain a JSON list", param_hint="--subtasks-file")
    leakage_groups = None
    if leakage_group:
        leakage_groups = {}
        for item in leakage_group:
            key, sep, value = item.partition("=")
            if not sep or not key or not value:
                raise click.BadParameter(f"expected KEY=ID, got {item!r}", param_hint="--leakage-group")
            leakage_groups[key] = value
    result = ctx.obj["client"].sequence_outcomes.set(
        owner,
        slug,
        sequence_uid,
        outcome=outcome,
        progress=progress,
        quality=quality,
        speed=speed,
        subtasks=subtasks,
        mistake_type=mistake_type,
        recovery_type=recovery_type,
        failure_stage=failure_stage,
        autonomy_level=autonomy_level,
        model_version=model_version,
        evaluation_membership=evaluation_membership,
        leakage_groups=leakage_groups,
        source=source,
        confidence=confidence,
    )
    _print_outcome(result)
