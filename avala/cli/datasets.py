"""CLI commands for datasets."""

from __future__ import annotations

import json
import mimetypes
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

import click
from avala.cli._output import human_bytes, print_detail, print_table

if TYPE_CHECKING:
    from avala.types.dataset import Dataset


def _make_poll_callback(start_time: float) -> Callable[..., None]:
    """Create a poll callback that prints dataset status with elapsed time."""

    def _on_poll(d: Dataset) -> None:
        elapsed = int(time.monotonic() - start_time)
        click.echo(f"  status={d.status}, items={d.item_count} (elapsed: {elapsed}s)", err=True)

    return _on_poll


@click.group()
def datasets() -> None:
    """Manage datasets."""


@datasets.command("list")
@click.option(
    "--data-type",
    type=str,
    default=None,
    help="Filter by data type (image, video, lidar, mcap, splat)",
)
@click.option(
    "--name",
    type=str,
    default=None,
    help="Filter by name (case-insensitive substring match)",
)
@click.option("--status", type=str, default=None, help="Filter by status (creating, created)")
@click.option(
    "--visibility",
    type=str,
    default=None,
    help="Filter by visibility (private, public)",
)
@click.option("--limit", type=int, default=None, help="Maximum number of results")
@click.pass_context
def list_datasets(
    ctx: click.Context,
    data_type: str | None,
    name: str | None,
    status: str | None,
    visibility: str | None,
    limit: int | None,
) -> None:
    """List datasets."""
    client = ctx.obj["client"]
    page = client.datasets.list(
        data_type=data_type,
        name=name,
        status=status,
        visibility=visibility,
        limit=limit,
    )
    rows = [(d.uid, d.name, d.slug, str(d.item_count), d.data_type or "—") for d in page.items]
    print_table(
        "Datasets",
        ["UID", "Name", "Slug", "Items", "Type"],
        rows,
        json_keys=["uid", "name", "slug", "item_count", "data_type"],
    )


@datasets.command("get")
@click.argument("uid")
@click.pass_context
def get_dataset(ctx: click.Context, uid: str) -> None:
    """Get a dataset by UID."""
    client = ctx.obj["client"]
    d = client.datasets.get(uid)
    print_detail(
        f"Dataset: {d.name}",
        [
            ("UID", d.uid),
            ("Name", d.name),
            ("Slug", d.slug),
            ("Items", str(d.item_count)),
            ("Type", d.data_type or "—"),
            ("Created", str(d.created_at or "—")),
            ("Updated", str(d.updated_at or "—")),
        ],
        json_keys=[
            "uid",
            "name",
            "slug",
            "item_count",
            "data_type",
            "created_at",
            "updated_at",
        ],
    )


@datasets.command("get-sequence")
@click.argument("owner")
@click.argument("slug")
@click.argument("sequence_uid")
@click.pass_context
def get_sequence_cmd(ctx: click.Context, owner: str, slug: str, sequence_uid: str) -> None:
    """Get a sequence's detail (frame count, status, frames array)."""
    client = ctx.obj["client"]
    seq = client.datasets.get_sequence(owner, slug, sequence_uid)
    frames = seq.frames or []
    print_detail(
        f"Sequence: {seq.key or seq.uid}",
        [
            ("UID", seq.uid),
            ("Key", seq.key or "—"),
            ("Status", seq.status or "—"),
            ("Frames", str(len(frames))),
            ("Dataset UID", seq.dataset_uid or "—"),
            ("Lidar calibration", "on" if seq.lidar_calibration_enabled else "off"),
            ("Camera calibration", "on" if seq.camera_calibration_enabled else "off"),
        ],
        json_keys=[
            "uid",
            "key",
            "status",
            "dataset_uid",
            "number_of_frames",
            "lidar_calibration_enabled",
            "camera_calibration_enabled",
        ],
    )


@datasets.command("get-frame")
@click.argument("owner")
@click.argument("slug")
@click.argument("sequence_uid")
@click.argument("frame_idx", type=int)
@click.pass_context
def get_frame_cmd(ctx: click.Context, owner: str, slug: str, sequence_uid: str, frame_idx: int) -> None:
    """Get a single frame's LiDAR JSON metadata (model, xi, alpha, device pose, cameras)."""
    client = ctx.obj["client"]
    frame = client.datasets.get_frame(owner, slug, sequence_uid, frame_idx)
    print_detail(
        f"Frame {frame_idx}",
        [
            ("Frame index", str(frame.frame_index)),
            ("Key", frame.key or "—"),
            ("Model", frame.model or frame.camera_model or "—"),
            ("xi", f"{frame.xi}" if frame.xi is not None else "—"),
            ("alpha", f"{frame.alpha}" if frame.alpha is not None else "—"),
            ("Cameras", str(len(frame.images or []))),
            (
                "Device position",
                (
                    f"x={frame.device_position.x} y={frame.device_position.y} z={frame.device_position.z}"
                    if frame.device_position
                    else "—"
                ),
            ),
        ],
        json_keys=[
            "frame_index",
            "key",
            "model",
            "xi",
            "alpha",
            "device_position",
            "device_heading",
        ],
    )


@datasets.command("get-calibration")
@click.argument("owner")
@click.argument("slug")
@click.argument("sequence_uid")
@click.pass_context
def get_calibration_cmd(ctx: click.Context, owner: str, slug: str, sequence_uid: str) -> None:
    """Get the canonicalized rig calibration for a sequence (derived from frame[0])."""
    client = ctx.obj["client"]
    calib = client.datasets.get_calibration(owner, slug, sequence_uid)
    rows = [
        (
            c.camera_id or "—",
            c.model or "—",
            f"{c.fx}" if c.fx is not None else "—",
            f"{c.fy}" if c.fy is not None else "—",
            f"{c.cx}" if c.cx is not None else "—",
            f"{c.cy}" if c.cy is not None else "—",
            f"{c.xi}" if c.xi is not None else "—",
            f"{c.alpha}" if c.alpha is not None else "—",
        )
        for c in calib.cameras
    ]
    print_table(
        f"Calibration — sequence {sequence_uid}",
        ["Camera", "Model", "fx", "fy", "cx", "cy", "xi", "alpha"],
        rows,
        json_keys=["camera_id", "model", "fx", "fy", "cx", "cy", "xi", "alpha"],
    )


@datasets.command("health")
@click.argument("owner")
@click.argument("slug")
@click.pass_context
def health_cmd(ctx: click.Context, owner: str, slug: str) -> None:
    """Get an ingest/health snapshot for a dataset."""
    client = ctx.obj["client"]
    h = client.datasets.get_health(owner, slug)
    print_detail(
        f"Health: {h.dataset_slug}",
        [
            ("Dataset UID", h.dataset_uid),
            ("Status", h.dataset_status or "—"),
            ("Items", str(h.item_count)),
            ("Sequences", str(h.sequence_count)),
            ("Frames", str(h.total_frames)),
            ("S3 prefix", h.s3_prefix or "—"),
            ("GC storage prefix", h.gc_storage_prefix or "—"),
            ("Last updated", str(h.last_updated_at or "—")),
            ("Ingest OK", "yes" if h.ingest_ok else "no"),
            ("Issues", "; ".join(h.issues) or "—"),
        ],
        json_keys=[
            "dataset_uid",
            "dataset_slug",
            "dataset_status",
            "item_count",
            "sequence_count",
            "total_frames",
            "s3_prefix",
            "gc_storage_prefix",
            "last_updated_at",
            "ingest_ok",
            "issues",
        ],
    )
    for seq in h.sequences:
        click.echo(
            f"  - {seq.key or seq.uid}: frames={seq.frame_count} status={seq.status} "
            f"lidar_calib={'y' if seq.has_lidar_calibration else 'n'} "
            f"cam_calib={'y' if seq.has_camera_calibration else 'n'}",
            err=True,
        )


@datasets.command("create")
@click.option("--name", required=True, help="Display name for the dataset")
@click.option("--slug", required=True, help="URL-friendly identifier")
@click.option(
    "--data-type",
    required=True,
    type=click.Choice(["image", "video", "lidar", "mcap", "splat"]),
    help="Type of data in the dataset",
)
@click.option(
    "--visibility",
    default="private",
    type=click.Choice(["private", "public"]),
    help="Dataset visibility (default: private)",
)
@click.option(
    "--create-metadata/--no-create-metadata",
    default=True,
    help="Create dataset metadata",
)
@click.option("--provider-config", default=None, help="Provider config as JSON string")
@click.option("--owner", default=None, help="Dataset owner username or email")
@click.option(
    "--organization-uid",
    default=None,
    help="Organization public UID to own the dataset (preferred over numeric organization_id for API users)",
)
@click.pass_context
def create_dataset(
    ctx: click.Context,
    name: str,
    slug: str,
    data_type: str,
    visibility: str,
    create_metadata: bool,
    provider_config: str | None,
    owner: str | None,
    organization_uid: str | None,
) -> None:
    """Create a new dataset."""
    client = ctx.obj["client"]
    parsed_config = json.loads(provider_config) if provider_config else None
    d = client.datasets.create(
        name=name,
        slug=slug,
        data_type=data_type,
        visibility=visibility,
        create_metadata=create_metadata,
        provider_config=parsed_config,
        owner_name=owner,
        organization_uid=organization_uid,
    )
    click.echo(f"Dataset created: {d.uid} ({d.name})")


@datasets.command("wait")
@click.argument("uid")
@click.option("--status", default="created", help="Target status to wait for (default: created)")
@click.option(
    "--timeout",
    type=float,
    default=3600.0,
    help="Maximum seconds to wait (default: 3600)",
)
@click.option("--interval", type=float, default=10.0, help="Seconds between polls (default: 10)")
@click.option("--quiet", is_flag=True, default=False, help="Suppress progress output")
@click.pass_context
def wait_dataset(
    ctx: click.Context,
    uid: str,
    status: str,
    timeout: float,
    interval: float,
    quiet: bool,
) -> None:
    """Wait for a dataset to reach a target status."""
    client = ctx.obj["client"]
    start = time.monotonic()

    if not quiet:
        click.echo(f"Waiting for dataset {uid} to reach status '{status}'...", err=True)

    callback = None if quiet else _make_poll_callback(start)
    try:
        d = client.datasets.wait(uid, status=status, interval=interval, timeout=timeout, _on_poll=callback)
    except TimeoutError as exc:
        raise click.ClickException(str(exc))

    if not quiet:
        elapsed = int(time.monotonic() - start)
        click.echo(f"Dataset {uid} reached status '{status}' in {elapsed}s.", err=True)

    print_detail(
        f"Dataset: {d.name}",
        [
            ("UID", d.uid),
            ("Name", d.name),
            ("Slug", d.slug),
            ("Status", d.status or "—"),
            ("Items", str(d.item_count)),
            ("Type", d.data_type or "—"),
            ("Created", str(d.created_at or "—")),
            ("Updated", str(d.updated_at or "—")),
        ],
        json_keys=[
            "uid",
            "name",
            "slug",
            "status",
            "item_count",
            "data_type",
            "created_at",
            "updated_at",
        ],
    )


@datasets.command("upload")
@click.option(
    "--source",
    required=True,
    type=click.Path(exists=True),
    help="Local file or directory containing files to upload",
)
@click.option(
    "--dataset",
    "dataset_uid",
    default=None,
    help="Deprecated; local upload creates a new dataset",
)
@click.option(
    "--storage-config",
    "storage_config_uid",
    default=None,
    help="Deprecated; Avala-managed local upload does not use storage configs",
)
@click.option("--name", required=True, help="Dataset name")
@click.option("--slug", required=True, help="Dataset slug")
@click.option(
    "--data-type",
    required=True,
    type=click.Choice(["image", "video", "lidar", "mcap", "splat"]),
    help="Data type",
)
@click.option("--owner", default=None, help="Dataset owner username or email")
@click.option(
    "--visibility",
    default="private",
    type=click.Choice(["private", "public"]),
    help="Dataset visibility (default: private)",
)
@click.option("--industry", type=int, default=None, help="Industry ID for the dataset")
@click.option("--license", "license_id", type=int, default=None, help="License ID for the dataset")
@click.option(
    "--create-metadata/--no-create-metadata",
    default=True,
    help="Create dataset metadata",
)
@click.option("--aws-profile", default=None, help="Deprecated; ignored for Avala-managed uploads")
@click.option(
    "--workers",
    type=int,
    default=8,
    help="Number of parallel upload threads (default: 8)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Preview what would be uploaded without uploading",
)
@click.option(
    "--wait",
    "wait_after",
    is_flag=True,
    default=False,
    help="Wait for dataset indexing after upload (only meaningful for newly created datasets)",
)
@click.option(
    "--wait-timeout",
    type=float,
    default=3600.0,
    help="Timeout in seconds for --wait (default: 3600)",
)
@click.pass_context
def upload_dataset(
    ctx: click.Context,
    source: str,
    dataset_uid: str | None,
    storage_config_uid: str | None,
    name: str,
    slug: str,
    data_type: str,
    owner: str | None,
    visibility: str,
    industry: int | None,
    license_id: int | None,
    create_metadata: bool,
    aws_profile: str | None,
    workers: int,
    dry_run: bool,
    wait_after: bool,
    wait_timeout: float,
) -> None:
    """Upload local files to Avala-managed dataset storage and create a dataset."""
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    import httpx

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    client = ctx.obj["client"]

    if dataset_uid:
        raise click.ClickException("Local upload creates a new dataset; --dataset is not supported.")
    if storage_config_uid:
        raise click.ClickException("Local upload uses Avala-managed storage; remove --storage-config.")
    if aws_profile:
        raise click.ClickException("Local upload uses Avala-managed storage; remove --aws-profile.")
    if visibility != "private":
        raise click.ClickException("Local upload currently supports private datasets only.")

    source_path = Path(source).resolve()
    local_files: list[tuple[Path, str]] = []
    if source_path.is_file():
        local_files.append((source_path, source_path.name))
    else:
        for root, _, files in os.walk(source_path):
            for fname in sorted(files):
                local_path = Path(root) / fname
                relative = local_path.relative_to(source_path).as_posix()
                local_files.append((local_path, relative))

    if not local_files:
        raise click.ClickException(f"No files found in {source}")

    total_files = len(local_files)
    total_bytes = sum(path.stat().st_size for path, _ in local_files)
    # No client-side quota precheck. The server is the source of truth for
    # the per-user cap (LOCAL_UPLOAD_PER_USER_BYTES, configurable per
    # deployment) and returns 413 when a presign would exceed it. A hard
    # client cap created drift for users with raised limits and shadowed
    # the authoritative server response. Codex review of PR #11356.

    click.echo("Target: Avala-managed dataset upload storage", err=True)
    click.echo(f"Found {total_files} files ({human_bytes(total_bytes)})", err=True)

    if dry_run:
        click.echo("\n[DRY RUN] Would upload:", err=True)
        for local_path, relative in local_files[:20]:
            click.echo(f"  {relative} ({human_bytes(local_path.stat().st_size)})", err=True)
        if total_files > 20:
            click.echo(f"  ... and {total_files - 20} more files", err=True)
        click.echo(f"\nTotal: {total_files} files ({human_bytes(total_bytes)})", err=True)
        click.echo(
            f"Would create dataset: name={name!r}, slug={slug!r}, data_type={data_type!r}",
            err=True,
        )
        return

    failed_count = 0
    skipped_count = 0
    uploaded_bytes = 0
    start_time = time.monotonic()

    # Shared stop flag — workers check this BEFORE issuing a presign so an
    # in-flight worker that hasn't started its presign yet exits cleanly
    # instead of burning a quota reservation. Codex review of PR #11356
    # round 4 flagged that submitting every file up front and cancelling
    # only not-yet-started futures still let already-running workers issue
    # presigns and S3 PUTs after the first failure.
    import threading
    from concurrent.futures import CancelledError

    stop_event = threading.Event()

    class _Skipped(Exception):
        """Sentinel: worker exited because stop_event was set, not a failure."""

    def _upload_one(item: tuple[Path, str]) -> tuple[str, int]:
        if stop_event.is_set():
            raise _Skipped()
        local_path, relative = item
        file_size = local_path.stat().st_size
        if stop_event.is_set():
            raise _Skipped()
        upload_info = client.datasets.create_manual_upload_url(
            dataset_name=name,
            file_path_in_dataset=relative,
            content_length=file_size,
        )
        if stop_event.is_set():
            raise _Skipped()
        fields = upload_info["fields"]
        content_type = (
            fields.get("Content-Type") or mimetypes.guess_type(local_path.name)[0] or "application/octet-stream"
        )
        with local_path.open("rb") as fh:
            response = httpx.post(
                upload_info["url"],
                data=fields,
                files={"file": (local_path.name, fh, content_type)},
                timeout=None,
            )
            response.raise_for_status()
        return (relative, file_size)

    progress = tqdm(total=total_files, unit="file", desc="Uploading", disable=tqdm is None) if tqdm else None

    # Bounded active set — submit only ``workers`` files at a time, and only
    # submit the next file after a prior one completes. Combined with the
    # ``stop_event`` flag, this guarantees no new presigns are issued after
    # the first failure (the previous all-up-front submission left
    # already-running workers issuing presigns and PUTs even after we
    # cancelled the not-started futures).
    first_error: Optional[Exception] = None
    iterator = iter(local_files)
    active: dict = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        # Prime the executor with up to ``workers`` futures.
        for _ in range(max(1, workers)):
            try:
                item = next(iterator)
            except StopIteration:
                break
            active[pool.submit(_upload_one, item)] = item

        while active:
            done_future = next(as_completed(active))
            done_item = active.pop(done_future)
            try:
                _, nbytes = done_future.result()
                uploaded_bytes += nbytes
                # Advance: submit the next pending file iff we're not stopping.
                if first_error is None and not stop_event.is_set():
                    try:
                        nxt = next(iterator)
                        active[pool.submit(_upload_one, nxt)] = nxt
                    except StopIteration:
                        pass
            except CancelledError:
                skipped_count += 1
            except _Skipped:
                skipped_count += 1
            except Exception as exc:
                failed_count += 1
                click.echo(f"  FAILED: {done_item[1]} - {exc}", err=True)
                if first_error is None:
                    first_error = exc
                    stop_event.set()
                    # Cancel anything not started; running workers exit on
                    # the next ``stop_event.is_set()`` check.
                    for pending in list(active):
                        pending.cancel()
            if progress:
                progress.update(1)

    if progress:
        progress.close()

    elapsed = time.monotonic() - start_time
    rate = uploaded_bytes / elapsed if elapsed > 0 else 0
    uploaded_count = total_files - failed_count - skipped_count
    click.echo(
        f"\nDone in {elapsed:.1f}s — uploaded {uploaded_count} files "
        f"({human_bytes(uploaded_bytes)}, {human_bytes(rate)}/s), {failed_count} failed, "
        f"{skipped_count} skipped after first error.",
        err=True,
    )

    if first_error is not None:
        raise click.ClickException(f"Upload failed: {first_error}")
    if failed_count > 0:
        raise click.ClickException(f"{failed_count} file(s) failed to upload")

    dataset = client.datasets.create_from_manual_upload(
        name=name,
        slug=slug,
        data_type=data_type,
        visibility=visibility,
        create_metadata=create_metadata,
        owner_name=owner,
        industry=industry,
        license=license_id,
    )
    click.echo(f"Dataset created: {dataset.uid} ({dataset.name})", err=True)

    if wait_after:
        click.echo(f"\nWaiting for dataset {dataset.uid} to finish indexing...", err=True)
        poll_start = time.monotonic()
        try:
            dataset = client.datasets.wait(
                dataset.uid,
                status="created",
                interval=10.0,
                timeout=wait_timeout,
                _on_poll=_make_poll_callback(poll_start),
            )
        except TimeoutError as exc:
            raise click.ClickException(str(exc))
        click.echo(
            f"Dataset {dataset.uid} is ready (status={dataset.status}, items={dataset.item_count}).",
            err=True,
        )

    # --- Final output ---
    print_detail(
        f"Dataset: {dataset.name}",
        [
            ("UID", dataset.uid),
            ("Name", dataset.name),
            ("Slug", dataset.slug),
            ("Status", dataset.status or "—"),
            ("Items", str(dataset.item_count)),
            ("Type", dataset.data_type or "—"),
        ],
        json_keys=["uid", "name", "slug", "status", "item_count", "data_type"],
    )
