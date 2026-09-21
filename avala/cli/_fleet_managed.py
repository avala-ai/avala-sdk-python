"""Explicit Fleet managed uploads without legacy readiness or destination inference."""

from __future__ import annotations

import json

import click

from avala._fleet_uploads import collect_source
from avala.cli._output import human_bytes, print_detail
from avala.errors import UploadStateError
from avala.resources.fleet import uploads
from avala.resources.fleet._managed_consumer import validate_managed_inventory, validate_managed_options
from avala.types.fleet_managed_upload import ManagedFleetUploadStatus, canonical_uid

_RETAIN = "Retain the receipt and original files. Resume with the same recording, source, API and credentials."


def upload_managed(
    ctx: click.Context,
    source: str,
    recording_uid: str | None,
    storage_config_uid: str | None,
    workers: int,
    dry_run: bool,
    wait_after: bool,
    wait_timeout: float,
) -> None:
    """Preview locally or explicitly submit and observe exact managed publication."""
    if storage_config_uid is not None:
        raise click.ClickException("--managed cannot be combined with --storage-config.")
    try:
        requested_uid = canonical_uid(recording_uid or "")
    except ValueError:
        raise click.ClickException("Managed uploads require --recording with an existing canonical UUID.") from None
    timeout = wait_timeout if wait_after else 0
    try:
        validate_managed_options(workers, timeout)
        inventory = collect_source(source, state_dir=uploads._STATE_DIR)
        validate_managed_inventory(inventory)
    except UploadStateError as error:
        raise click.ClickException(str(error)) from None
    except Exception:
        raise click.ClickException("Managed source inventory could not be read safely.") from None

    click.echo(f"Source: {json.dumps(str(inventory.source), ensure_ascii=True)}", err=True)
    click.echo(f"Files: {len(inventory.files)} ({human_bytes(inventory.total_bytes)})", err=True)
    if dry_run:
        click.echo("[DRY RUN] Managed MCAP originals:", err=True)
        for file in inventory.files[:20]:
            click.echo(f"  {json.dumps(file.path, ensure_ascii=True)} ({human_bytes(file.size_bytes)})", err=True)
        if len(inventory.files) > 20:
            click.echo(f"  ... and {len(inventory.files) - 20} more files", err=True)
        click.echo("Local validation passed. Server enrollment and available quota have not been checked.", err=True)
        return

    client = ctx.obj.get("client") if ctx.obj else None
    if client is None:
        raise click.ClickException("No API key provided. Set AVALA_API_KEY or run avala configure.")
    try:
        status = client.fleet.uploads.upload_managed_recording(
            requested_uid, source, max_workers=workers, wait_timeout=timeout
        )
    except Exception:
        raise click.ClickException(f"Managed upload could not establish exact evidence. {_RETAIN}") from None
    _show_result(status)
    finalization = status.finalization
    if finalization is None:
        raise click.ClickException(f"Managed publication has not been established. {_RETAIN}")
    if finalization.state == "failed":
        raise click.ClickException(f"Managed publication failed ({finalization.code}). {_RETAIN}")
    if finalization.state == "succeeded":
        click.echo(f"Published dataset {finalization.dataset_uid}. {_RETAIN}", err=True)
    elif wait_after:
        raise click.ClickException(f"Timed out waiting for managed publication; work remains retained. {_RETAIN}")
    else:
        click.echo(f"Managed publication submitted ({finalization.state}). {_RETAIN}", err=True)


def _show_result(status: ManagedFleetUploadStatus) -> None:
    """Emit only bounded typed identity, counters and public state codes."""
    finalization = status.finalization
    fields = [
        ("Protocol", status.protocol),
        ("Recording", status.recording_uid),
        ("Session", status.session_uid),
        ("Upload status", status.status),
        ("Files verified", str(status.confirmed_files)),
        ("Total files", str(status.total_files)),
        ("Bytes verified", str(status.confirmed_bytes)),
        ("Total bytes", str(status.total_bytes)),
        ("Finalization", finalization.state if finalization else "unsubmitted"),
        ("Code", finalization.code if finalization else "unsubmitted"),
        ("Publication", finalization.publication_uid or "" if finalization else ""),
        ("Dataset", finalization.dataset_uid or "" if finalization else ""),
    ]
    print_detail(
        "Managed Fleet upload",
        fields,
        json_keys=[
            "protocol",
            "recording_uid",
            "session_uid",
            "upload_status",
            "confirmed_files",
            "total_files",
            "confirmed_bytes",
            "total_bytes",
            "finalization_state",
            "code",
            "publication_uid",
            "dataset_uid",
        ],
    )
