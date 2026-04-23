"""CLI commands for datasets."""

from __future__ import annotations

import json
import mimetypes
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
@click.option("--data-type", type=str, default=None, help="Filter by data type (image, video, lidar, mcap, splat)")
@click.option("--name", type=str, default=None, help="Filter by name (case-insensitive substring match)")
@click.option("--status", type=str, default=None, help="Filter by status (creating, created)")
@click.option("--visibility", type=str, default=None, help="Filter by visibility (private, public)")
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
    page = client.datasets.list(data_type=data_type, name=name, status=status, visibility=visibility, limit=limit)
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
        json_keys=["uid", "name", "slug", "item_count", "data_type", "created_at", "updated_at"],
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
                f"x={frame.device_position.x} y={frame.device_position.y} z={frame.device_position.z}"
                if frame.device_position
                else "—",
            ),
        ],
        json_keys=["frame_index", "key", "model", "xi", "alpha", "device_position", "device_heading"],
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
            ("Last item updated", str(h.last_item_updated_at or "—")),
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
@click.option("--is-sequence", is_flag=True, default=False, help="Dataset contains sequences")
@click.option(
    "--visibility",
    default="private",
    type=click.Choice(["private", "public"]),
    help="Dataset visibility (default: private)",
)
@click.option("--create-metadata/--no-create-metadata", default=True, help="Create dataset metadata")
@click.option("--provider-config", default=None, help="Provider config as JSON string")
@click.option("--owner", default=None, help="Dataset owner username or email")
@click.pass_context
def create_dataset(
    ctx: click.Context,
    name: str,
    slug: str,
    data_type: str,
    is_sequence: bool,
    visibility: str,
    create_metadata: bool,
    provider_config: str | None,
    owner: str | None,
) -> None:
    """Create a new dataset."""
    client = ctx.obj["client"]
    parsed_config = json.loads(provider_config) if provider_config else None
    d = client.datasets.create(
        name=name,
        slug=slug,
        data_type=data_type,
        is_sequence=is_sequence,
        visibility=visibility,
        create_metadata=create_metadata,
        provider_config=parsed_config,
        owner_name=owner,
    )
    click.echo(f"Dataset created: {d.uid} ({d.name})")


@datasets.command("wait")
@click.argument("uid")
@click.option("--status", default="created", help="Target status to wait for (default: created)")
@click.option("--timeout", type=float, default=3600.0, help="Maximum seconds to wait (default: 3600)")
@click.option("--interval", type=float, default=10.0, help="Seconds between polls (default: 10)")
@click.option("--quiet", is_flag=True, default=False, help="Suppress progress output")
@click.pass_context
def wait_dataset(ctx: click.Context, uid: str, status: str, timeout: float, interval: float, quiet: bool) -> None:
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
        json_keys=["uid", "name", "slug", "status", "item_count", "data_type", "created_at", "updated_at"],
    )


@datasets.command("upload")
@click.option(
    "--source",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Local directory containing files to upload",
)
@click.option("--dataset", "dataset_uid", default=None, help="Existing dataset UID to upload into")
@click.option(
    "--storage-config",
    "storage_config_uid",
    required=True,
    help="Storage config UID (provides S3 bucket/region/prefix)",
)
@click.option("--name", default=None, help="Dataset name (required when creating a new dataset)")
@click.option("--slug", default=None, help="Dataset slug (required when creating a new dataset)")
@click.option(
    "--data-type",
    default=None,
    type=click.Choice(["image", "video", "lidar", "mcap", "splat"]),
    help="Data type (required when creating a new dataset)",
)
@click.option("--is-sequence", is_flag=True, default=False, help="Dataset contains sequences")
@click.option("--owner", default=None, help="Dataset owner username or email")
@click.option(
    "--visibility",
    default="private",
    type=click.Choice(["private", "public"]),
    help="Dataset visibility (default: private)",
)
@click.option("--aws-profile", default=None, help="AWS profile name for credentials (default: boto3 default chain)")
@click.option("--workers", type=int, default=8, help="Number of parallel upload threads (default: 8)")
@click.option("--dry-run", is_flag=True, default=False, help="Preview what would be uploaded without uploading")
@click.option(
    "--wait",
    "wait_after",
    is_flag=True,
    default=False,
    help="Wait for dataset indexing after upload (only meaningful for newly created datasets)",
)
@click.option("--wait-timeout", type=float, default=3600.0, help="Timeout in seconds for --wait (default: 3600)")
@click.pass_context
def upload_dataset(
    ctx: click.Context,
    source: str,
    dataset_uid: str | None,
    storage_config_uid: str,
    name: str | None,
    slug: str | None,
    data_type: str | None,
    is_sequence: bool,
    owner: str | None,
    visibility: str,
    aws_profile: str | None,
    workers: int,
    dry_run: bool,
    wait_after: bool,
    wait_timeout: float,
) -> None:
    """Upload a batch of local files to an S3-backed dataset.

    Uploads ALL files from --source to the S3 location defined by --storage-config.
    This is a batch upload, not an incremental sync. For incremental uploads, use
    ``aws s3 sync`` followed by ``avala datasets create``.

    Either upload to an existing dataset (--dataset) or create a new one
    (--name, --slug, --data-type required). S3 credentials come from the
    standard boto3 chain (env vars, ~/.aws/credentials, IAM role) or --aws-profile.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
    except ImportError:
        raise click.ClickException("boto3 is required for upload. Install with: pip install avala[cli]")

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    client = ctx.obj["client"]

    # --- Phase 1: Validate storage config ---
    sc = client.storage_configs.get(storage_config_uid)
    if sc.provider != "aws_s3":
        raise click.ClickException(f"Upload only supports aws_s3 storage configs, got: {sc.provider}")
    if not sc.s3_bucket_name:
        raise click.ClickException(f"Storage config {sc.uid} has no S3 bucket name")

    bucket = sc.s3_bucket_name
    region = sc.s3_bucket_region or "us-east-1"
    prefix = (sc.s3_bucket_prefix or "").rstrip("/")

    # --- Phase 2: Collect local files ---
    source_path = Path(source).resolve()
    local_files: list[tuple[Path, str]] = []
    for root, _, files in os.walk(source_path):
        for fname in files:
            local_path = Path(root) / fname
            relative = local_path.relative_to(source_path).as_posix()
            s3_key = f"{prefix}/{relative}" if prefix else relative
            local_files.append((local_path, s3_key))

    if not local_files:
        raise click.ClickException(f"No files found in {source}")

    total_files = len(local_files)
    total_bytes = sum(f.stat().st_size for f, _ in local_files)

    click.echo(f"S3 target: s3://{bucket}/{prefix}/", err=True)
    click.echo(f"Found {total_files} files ({human_bytes(total_bytes)})", err=True)

    # --- Phase 3: Dry run ---
    if dry_run:
        click.echo("\n[DRY RUN] Would upload:", err=True)
        for local_path, s3_key in local_files[:20]:
            click.echo(f"  {s3_key} ({human_bytes(local_path.stat().st_size)})", err=True)
        if total_files > 20:
            click.echo(f"  ... and {total_files - 20} more files", err=True)
        click.echo(f"\nTotal: {total_files} files ({human_bytes(total_bytes)})", err=True)
        if not dataset_uid:
            click.echo(f"Would create dataset: name={name!r}, slug={slug!r}, data_type={data_type!r}", err=True)
        return

    # --- Phase 4: Build S3 client (credentials validated here) ---
    try:
        session = boto3.Session(profile_name=aws_profile, region_name=region)
        s3 = session.client("s3")
        # Force credential resolution with a lightweight call
        s3.head_bucket(Bucket=bucket)
    except NoCredentialsError:
        raise click.ClickException(
            "No AWS credentials found.\n\n"
            "Configure credentials via one of:\n"
            "  export AWS_ACCESS_KEY_ID=... && export AWS_SECRET_ACCESS_KEY=...\n"
            "  aws configure\n"
            "  --aws-profile <profile_name>"
        )
    except PartialCredentialsError:
        raise click.ClickException("Incomplete AWS credentials. Both access key and secret key are required.")
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code == "403":
            raise click.ClickException(
                f"Access denied to bucket '{bucket}'. Check your AWS credentials and bucket policy."
            )
        raise click.ClickException(f"S3 error: {exc}")

    # --- Phase 5: Create or resolve dataset ---
    if dataset_uid:
        dataset = client.datasets.get(dataset_uid)
        click.echo(f"Uploading to existing dataset: {dataset.uid} ({dataset.name})", err=True)
    else:
        if not all([name, slug, data_type]):
            raise click.ClickException("--name, --slug, and --data-type are required when not using --dataset")
        provider_config = {
            "provider": "aws_s3",
            "s3_bucket_name": bucket,
            "s3_bucket_region": region,
            "s3_bucket_prefix": prefix,
        }
        dataset = client.datasets.create(
            name=name,  # type: ignore[arg-type]
            slug=slug,  # type: ignore[arg-type]
            data_type=data_type,  # type: ignore[arg-type]
            is_sequence=is_sequence,
            visibility=visibility,
            owner_name=owner,
            provider_config=provider_config,
        )
        click.echo(f"Dataset created: {dataset.uid} ({dataset.name})", err=True)

    # --- Phase 6: Upload ---
    failed_count = 0
    uploaded_bytes = 0
    start_time = time.monotonic()

    def _upload_one(item: tuple[Path, str]) -> tuple[str, int]:
        local_path, s3_key = item
        extra = _extra_args(local_path.name)
        s3.upload_file(str(local_path), bucket, s3_key, ExtraArgs=extra or None)
        return (s3_key, local_path.stat().st_size)

    progress = tqdm(total=total_files, unit="file", desc="Uploading", disable=tqdm is None) if tqdm else None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_upload_one, item): item for item in local_files}
        for future in as_completed(futures):
            try:
                _, nbytes = future.result()
                uploaded_bytes += nbytes
            except Exception as exc:
                failed_count += 1
                item = futures[future]
                click.echo(f"  FAILED: {item[1]} — {exc}", err=True)
            if progress:
                progress.update(1)

    if progress:
        progress.close()

    elapsed = time.monotonic() - start_time
    rate = uploaded_bytes / elapsed if elapsed > 0 else 0
    uploaded_count = total_files - failed_count
    click.echo(
        f"\nDone in {elapsed:.1f}s — uploaded {uploaded_count} files "
        f"({human_bytes(uploaded_bytes)}, {human_bytes(rate)}/s), {failed_count} failed.",
        err=True,
    )

    if failed_count > 0:
        raise click.ClickException(f"{failed_count} file(s) failed to upload")

    # --- Phase 7: Wait for indexing ---
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
        click.echo(f"Dataset {dataset.uid} is ready (status={dataset.status}, items={dataset.item_count}).", err=True)

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


def _extra_args(filename: str) -> dict[str, str]:
    """Return S3 ExtraArgs (headers) based on file extension."""
    lower = filename.lower()
    args: dict[str, str] = {}

    # Cache-Control and Content-Encoding
    if lower.endswith((".alp.gz", ".ali.gz")):
        args["CacheControl"] = "private, immutable, max-age=31536000"
        args["ContentEncoding"] = "gzip"
    elif lower.endswith((".alp", ".ali")):
        args["CacheControl"] = "private, immutable, max-age=31536000"
    elif lower.endswith((".webp", ".jpg", ".jpeg", ".png")):
        args["CacheControl"] = "private, immutable, max-age=31536000"
    elif lower.endswith(".json"):
        args["CacheControl"] = "no-cache"

    # Content-Type
    content_type, _ = mimetypes.guess_type(filename)
    if content_type:
        args["ContentType"] = content_type

    return args
