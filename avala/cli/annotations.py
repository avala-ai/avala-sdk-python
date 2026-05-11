"""CLI commands for annotations bulk-edit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import click

from avala.types.dataset import Dataset


@click.group(name="annotations")
def annotations_group() -> None:
    """Manage annotations (bulk-edit cuboids and other 3D annotations)."""


def _load_edits(input_path: Path) -> list[dict[str, Any]]:
    """Load edits from a JSON array file or a JSONL file (one object per line).

    Detection: ``.jsonl`` extension → line-by-line. Anything else → JSON array.
    """
    if input_path.suffix.lower() == ".jsonl":
        edits: list[dict[str, Any]] = []
        with input_path.open("r") as fh:
            for lineno, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise click.ClickException(f"{input_path}:{lineno}: invalid JSON — {exc}")
                if not isinstance(obj, dict):
                    raise click.ClickException(f"{input_path}:{lineno}: expected JSON object, got {type(obj).__name__}")
                edits.append(obj)
        return edits

    with input_path.open("r") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise click.ClickException(f"{input_path}: invalid JSON — {exc}")
    if not isinstance(data, list):
        raise click.ClickException(f"{input_path}: expected JSON array at top level, got {type(data).__name__}")
    for i, obj in enumerate(data):
        if not isinstance(obj, dict):
            raise click.ClickException(f"{input_path}: item {i} is not a JSON object")
    return data


_VALID_OBJECT_TYPES = {"cuboid", "cuboid 3d", "point cloud polyline"}
_VALID_ACTIONS = {"upsert", "delete"}


def _validate_edit_locally(edit: dict[str, Any], index: int) -> None:
    """Client-side validation that mirrors the server's serializer rules.

    Catches obvious typos before any HTTP call.
    """
    action = edit.get("action")
    if action not in _VALID_ACTIONS:
        raise click.ClickException(f"edit {index}: action must be one of {sorted(_VALID_ACTIONS)}, got {action!r}")

    if "object_uuid" not in edit:
        raise click.ClickException(f"edit {index}: object_uuid is required")

    object_type = edit.get("object_type")
    if action == "upsert":
        if object_type not in _VALID_OBJECT_TYPES:
            raise click.ClickException(
                f"edit {index}: object_type must be one of {sorted(_VALID_OBJECT_TYPES)}, got {object_type!r}"
            )
        if edit.get("object_data") is None:
            raise click.ClickException(f"edit {index}: object_data is required for upsert")

    item_uid = edit.get("dataset_item_uid")
    seq_uid = edit.get("sequence_uid")
    if item_uid and seq_uid:
        raise click.ClickException(f"edit {index}: only one of dataset_item_uid / sequence_uid is allowed")
    if not item_uid and not seq_uid:
        raise click.ClickException(f"edit {index}: one of dataset_item_uid / sequence_uid is required")

    if item_uid and object_type and object_type not in {"cuboid"}:
        raise click.ClickException(
            f"edit {index}: dataset_item_uid does not support object_type={object_type!r} (use 'cuboid')"
        )
    if seq_uid and object_type and object_type not in {"cuboid 3d", "point cloud polyline"}:
        raise click.ClickException(
            f"edit {index}: sequence_uid does not support object_type={object_type!r} "
            f"(use 'cuboid 3d' or 'point cloud polyline')"
        )


def _check_dataset(client: Any, owner: str, slug: str) -> Dataset:
    """Pre-flight: dataset exists, is LIDAR. Returns the :class:`Dataset` model."""
    dataset = client.datasets.get_by_slug(owner, slug)
    if dataset.data_type != "lidar":
        raise click.ClickException(
            f"Dataset {owner}/{slug} has data_type={dataset.data_type!r}, but bulk-edit requires 'lidar'"
        )
    return dataset


def _resolve_required_uids(
    edits: list[dict[str, Any]],
) -> tuple[set[str], set[str]]:
    """Return (set of required dataset_item_uids, set of required sequence_uids)."""
    items: set[str] = set()
    sequences: set[str] = set()
    for edit in edits:
        item_uid = edit.get("dataset_item_uid")
        seq_uid = edit.get("sequence_uid")
        if item_uid:
            items.add(str(item_uid))
        if seq_uid:
            sequences.add(str(seq_uid))
    return items, sequences


def _fetch_known_dataset_item_uids(client: Any, owner: str, slug: str) -> set[str]:
    """Page through all dataset items and return their uids."""
    uids: set[str] = set()
    cursor: str | None = None
    while True:
        page = client.datasets.list_items(owner, slug, cursor=cursor)
        for item in page.items:
            uids.add(str(item.uid))
        if not page.next_cursor:
            break
        cursor = page.next_cursor
    return uids


def _fetch_known_sequence_uids(client: Any, owner: str, slug: str) -> set[str]:
    uids: set[str] = set()
    cursor: str | None = None
    while True:
        page = client.datasets.list_sequences(owner, slug, cursor=cursor)
        for seq in page.items:
            uids.add(str(seq.uid))
        if not page.next_cursor:
            break
        cursor = page.next_cursor
    return uids


def _summarize_object_names(edits: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for edit in edits:
        name = edit.get("object_name") or "Object"
        counts[name] = counts.get(name, 0) + 1
    return counts


def _iter_edits(edits: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    yield from edits


@annotations_group.command("bulk-edit")
@click.option("--owner", required=True, help="Dataset owner (username or org email)")
@click.option("--slug", required=True, help="Dataset slug")
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a JSON array or JSONL file of edits",
)
@click.option("--chunk-size", type=int, default=500, show_default=True, help="Edits per HTTP request")
@click.option("--dry-run", is_flag=True, default=False, help="Validate edits client-side, do NOT post")
@click.option(
    "--check-only",
    is_flag=True,
    default=False,
    help="Pre-flight: dataset exists, data_type=lidar, items/sequences resolve. No POST.",
)
@click.pass_context
def bulk_edit_command(
    ctx: click.Context,
    owner: str,
    slug: str,
    input_path: Path,
    chunk_size: int,
    dry_run: bool,
    check_only: bool,
) -> None:
    """Bulk-create / update / delete cuboid annotations on a LiDAR dataset.

    Posts edits to ``/api/v1/datasets/{owner}/{slug}/bulk-edition/`` in chunks
    of --chunk-size. Idempotent on (object_uuid, dataset_item_uid) — re-running
    is safe and a failed chunk can be re-posted without duplicates.

    The input file is either a JSON array (``.json``) or JSONL (``.jsonl``).
    Each entry follows the server's ``DatasetObjectEditionSerializer``:

    \b
      - action: "upsert" or "delete"
      - object_uuid: stable per-track UUID
      - object_type: "cuboid" (per-frame) or "cuboid 3d" (sequence-level)
      - dataset_item_uid OR sequence_uid (mutually exclusive)
      - object_name: free-form label (e.g. "vehicle"); auto-creates a
        bulk-edition Project per (object_type, object_name) on the dataset
      - object_data: full geometry payload (position/dimensions/rotation)
    """
    edits = _load_edits(input_path)
    if not edits:
        raise click.ClickException(f"{input_path}: no edits found")

    click.echo(f"Loaded {len(edits)} edits from {input_path}", err=True)

    for i, edit in enumerate(edits):
        _validate_edit_locally(edit, i)

    name_counts = _summarize_object_names(edits)
    click.echo("Distinct object_name values:", err=True)
    for name, count in sorted(name_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        click.echo(f"  {name}: {count}", err=True)

    if dry_run:
        click.echo(f"\n[DRY RUN] {len(edits)} edits validated. No HTTP call made.", err=True)
        return

    client = ctx.obj["client"]

    dataset = _check_dataset(client, owner, slug)
    click.echo(
        f"Dataset {owner}/{slug}: data_type={dataset.data_type}, is_sequence={dataset.is_sequence}",
        err=True,
    )

    required_items, required_sequences = _resolve_required_uids(edits)
    if required_items:
        click.echo(f"Resolving {len(required_items)} dataset items...", err=True)
        known_items = _fetch_known_dataset_item_uids(client, owner, slug)
        missing = required_items - known_items
        if missing:
            sample = sorted(missing)[:10]
            raise click.ClickException(
                f"{len(missing)} of {len(required_items)} dataset_item_uid(s) "
                f"do not exist on {owner}/{slug}. Sample: {sample}"
            )
        click.echo(f"  All {len(required_items)} dataset items exist.", err=True)

    if required_sequences:
        click.echo(f"Resolving {len(required_sequences)} sequences...", err=True)
        known_sequences = _fetch_known_sequence_uids(client, owner, slug)
        missing_seq = required_sequences - known_sequences
        if missing_seq:
            sample = sorted(missing_seq)[:10]
            raise click.ClickException(
                f"{len(missing_seq)} of {len(required_sequences)} sequence_uid(s) "
                f"do not exist on {owner}/{slug}. Sample: {sample}"
            )
        click.echo(f"  All {len(required_sequences)} sequences exist.", err=True)

    if check_only:
        click.echo("\n[CHECK ONLY] All pre-flight checks passed. No POST made.", err=True)
        return

    _tqdm: Any
    try:
        from tqdm import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    total_chunks = (len(edits) + chunk_size - 1) // chunk_size
    click.echo(
        f"\nPosting {len(edits)} edits in {total_chunks} chunk(s) of {chunk_size}...",
        err=True,
    )

    progress = _tqdm(total=total_chunks, unit="chunk", desc="Posting") if _tqdm is not None else None
    try:
        for index, _response in client.annotations.bulk_edit(owner, slug, _iter_edits(edits), chunk_size=chunk_size):
            if progress is not None:
                progress.update(1)
            else:
                click.echo(f"  chunk {index + 1}/{total_chunks} ok", err=True)
    finally:
        if progress is not None:
            progress.close()

    click.echo(f"\nDone. {len(edits)} edits posted across {total_chunks} chunk(s).", err=True)
