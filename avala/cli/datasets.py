"""CLI commands for datasets."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

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


@datasets.command("transfer")
@click.argument("owner")
@click.argument("slug")
@click.option("--organization-uid", default=None, help="UID of the destination organization")
@click.option("--owner-username", default=None, help="Username of the destination user (yourself only)")
@click.pass_context
def transfer_dataset_cmd(
    ctx: click.Context,
    owner: str,
    slug: str,
    organization_uid: str | None,
    owner_username: str | None,
) -> None:
    """Transfer a dataset to another organization, or to your own account.

    Pass exactly one of --organization-uid or --owner-username. A dataset is
    owned by a user XOR an organization, so the transfer sets one and clears the
    other. Requires the OWNER role on the dataset's current organization, and
    authority at the destination.

    OWNER and SLUG identify the dataset at its CURRENT path; after a successful
    transfer that path no longer resolves, so use the printed owner next time.
    """
    client = ctx.obj["client"]
    if (organization_uid is None) == (owner_username is None):
        raise click.UsageError("Pass exactly one of --organization-uid or --owner-username.")
    d = client.datasets.transfer(
        owner,
        slug,
        organization_uid=organization_uid,
        owner_username=owner_username,
    )
    print_detail(
        f"Transferred: {d.name}",
        [
            ("UID", d.uid),
            ("Name", d.name),
            ("Slug", d.slug),
            ("Owner", d.owner_name or "—"),
            ("New path", f"/@{d.owner_name}/datasets/{d.slug}" if d.owner_name else "—"),
        ],
        json_keys=["uid", "name", "slug", "owner_name"],
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
    "--organization-uid",
    default=None,
    help="Create the dataset under this organization instead of the calling user",
)
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
    "--resume/--no-resume",
    default=True,
    help="Skip files a previous interrupted run already uploaded (default: resume)",
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
    organization_uid: str | None,
    visibility: str,
    industry: int | None,
    license_id: int | None,
    create_metadata: bool,
    aws_profile: str | None,
    workers: int,
    resume: bool,
    dry_run: bool,
    wait_after: bool,
    wait_timeout: float,
) -> None:
    """Upload local files to Avala-managed dataset storage and create a dataset."""
    import os

    from avala.errors import QuotaExceededError
    from avala.resources.datasets import _STATE_DIR, clear_completed, gather_local_files

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

    # Same walk the resource layer uses, so --dry-run lists exactly what the
    # upload will send.
    local_files = gather_local_files(source)
    if not local_files:
        raise click.ClickException(f"No files found in {source}")

    total_files = len(local_files)
    total_bytes = sum(os.path.getsize(path) for path, _ in local_files)

    click.echo("Target: Avala-managed dataset upload storage", err=True)
    click.echo(f"Found {total_files} files ({human_bytes(total_bytes)})", err=True)

    if dry_run:
        click.echo("\n[DRY RUN] Would upload:", err=True)
        for local_path, relative in local_files[:20]:
            click.echo(f"  {relative} ({human_bytes(os.path.getsize(local_path))})", err=True)
        if total_files > 20:
            click.echo(f"  ... and {total_files - 20} more files", err=True)
        click.echo(f"\nTotal: {total_files} files ({human_bytes(total_bytes)})", err=True)
        click.echo(
            f"Would create dataset: name={name!r}, slug={slug!r}, data_type={data_type!r}",
            err=True,
        )
        return

    # The upload loop lives in ``client.datasets.upload_files`` — retries,
    # resume state, and the presigned-host allow-list included. This command
    # used to reimplement it inline, which meant every fix had to be made
    # twice and the CLI silently lacked the resilience the resource grew.
    #
    # Still no client-side quota precheck here: the server is the source of
    # truth for the cap and returns 413 when a presign would exceed it, which
    # surfaces as QuotaExceededError below. A hard client cap created drift for
    # users with raised limits. Codex review of PR #11356.
    progress = tqdm(total=total_files, unit="file", desc="Uploading", disable=tqdm is None) if tqdm else None
    resumed_files = 0

    def _on_progress(_relative: str, _size: int) -> None:
        if progress:
            progress.update(1)

    def _on_skipped(_relative: str) -> None:
        # Files a previous run already uploaded still count toward `total_files`,
        # so without this the bar closes at (say) 1/100 on a successful resume
        # and reads as a near-total failure of an upload that is in fact done.
        nonlocal resumed_files
        resumed_files += 1
        if progress:
            progress.update(1)

    # Captured before the first byte moves; verified again before the create
    # call below. `create_from_local` does this internally, but this command
    # finalizes for itself — and a guard on only one of the two paths is how the
    # rename-mid-upload hole survived its first fix.
    storage_root_before = client.datasets.resolve_storage_root(organization_uid)

    start_time = time.monotonic()
    try:
        uploaded_bytes = client.datasets.upload_files(
            dataset_name=name,
            files=local_files,
            workers=workers,
            on_progress=_on_progress,
            on_skipped=_on_skipped,
            organization_uid=organization_uid,
            resume=resume,
            # The dataset NAME decides the remote prefix; the slug never
            # reaches S3. Keying on the slug meant the standard
            # slug-collision retry re-sent the whole payload against an
            # unchanged prefix. Matches `create_from_local`.
            state_key=name,
            # Keep the checkpoint until the dataset is created below, so a
            # failure at that last step resumes instead of re-sending everything.
            clear_state_on_success=False,
        )
    except QuotaExceededError as exc:
        detail = ""
        if exc.limit is not None and exc.used is not None:
            detail = f" ({human_bytes(exc.used)} of {human_bytes(exc.limit)} already used)"
        raise click.ClickException(f"Storage quota exceeded{detail}. Free space or request a higher cap.")
    except Exception as exc:
        # Resume state is intact — say so, because the natural next move is to
        # re-run the identical command rather than start over.
        click.echo(f"\nUpload failed: {exc}", err=True)
        raise click.ClickException("Upload failed. Re-run the same command to resume from where it stopped.")
    finally:
        # Runs before the exception propagates, so the bar is closed exactly
        # once on every path.
        if progress:
            progress.close()

    elapsed = time.monotonic() - start_time
    rate = uploaded_bytes / elapsed if elapsed > 0 else 0
    # `uploaded_bytes` covers only what moved in THIS run, so attributing it to
    # the whole manifest overstates the transfer on every resume.
    sent_files = total_files - resumed_files
    resumed_note = f" ({resumed_files} already uploaded)" if resumed_files else ""
    click.echo(
        f"\nDone in {elapsed:.1f}s — uploaded {human_bytes(uploaded_bytes)} "
        f"across {sent_files} file(s) at {human_bytes(rate)}/s{resumed_note}.",
        err=True,
    )

    try:
        client.datasets.assert_storage_root_unchanged(storage_root_before, organization_uid=organization_uid)
    except RuntimeError as exc:
        raise click.ClickException(str(exc))

    dataset = client.datasets.create_from_manual_upload(
        name=name,
        slug=slug,
        data_type=data_type,
        visibility=visibility,
        create_metadata=create_metadata,
        owner_name=owner,
        industry=industry,
        license=license_id,
        organization_uid=organization_uid,
    )
    # Dataset exists — the resume checkpoint has nothing left to protect.
    # Must match the fingerprint `upload_files` keyed this run's state by, or
    # the checkpoint for this destination is left behind. Call the resource's
    # helper rather than rebuilding it: a personal upload folds in the resolved
    # user uid, which this module has no way to know.
    clear_completed(
        _STATE_DIR,
        name,
        fingerprint=client.datasets._upload_fingerprint(organization_uid, name),
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
