"""CLI commands for importing datasets into Mission Control from external sources."""

from __future__ import annotations

import click


@click.group(name="import")
def import_group() -> None:
    """Import datasets into Mission Control from external sources."""


@import_group.command("list")
def list_importers() -> None:
    """List the available import sources."""
    from avala.importers import available_importers

    for name in available_importers():
        click.echo(name)


@import_group.command("folder")
@click.option("--source", required=True, type=click.Path(exists=True), help="Local file or directory to import")
@click.option("--name", required=True, help="Dataset name")
@click.option("--slug", required=True, help="Dataset slug")
@click.option(
    "--data-type",
    default=None,
    type=click.Choice(["image", "video", "lidar", "mcap", "splat"]),
    help="Override the auto-detected data type",
)
@click.option("--owner", default=None, help="Dataset owner username or email")
@click.option(
    "--organization-uid",
    default=None,
    help="Create the dataset under this organization instead of the calling user",
)
@click.option("--workers", type=int, default=8, help="Parallel upload threads (default: 8)")
@click.option(
    "--resume/--no-resume",
    default=True,
    help="Skip files a previous interrupted run already uploaded (default: resume)",
)
@click.option("--wait/--no-wait", "wait_after", default=False, help="Wait for indexing to finish")
@click.pass_context
def import_folder_cmd(
    ctx: click.Context,
    source: str,
    name: str,
    slug: str,
    data_type: str | None,
    owner: str | None,
    organization_uid: str | None,
    workers: int,
    resume: bool,
    wait_after: bool,
) -> None:
    """Create a dataset from a local file or directory (auto-detects data type)."""
    from avala.importers import import_folder

    client = ctx.obj["client"]
    dataset = import_folder(
        client,
        source=source,
        name=name,
        slug=slug,
        data_type=data_type,
        owner_name=owner,
        organization_uid=organization_uid,
        workers=workers,
        resume=resume,
        wait=wait_after,
    )
    click.echo(
        f"Dataset created: {dataset.uid} ({dataset.name}) — type={dataset.data_type}, items={dataset.item_count}"
    )


@import_group.command("lerobot")
@click.option("--repo-id", default=None, help="Hugging Face Hub dataset id (e.g. lerobot/svla_so101_pickplace)")
@click.option(
    "--root",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="Local LeRobot dataset directory (instead of, or alongside, --repo-id)",
)
@click.option("--name", required=True, help="Dataset name")
@click.option("--slug", required=True, help="Dataset slug")
@click.option("--episodes", default=None, help="Comma-separated episode indices to import (default: all)")
@click.option("--camera-keys", default=None, help="Comma-separated camera feature keys (default: all cameras)")
@click.option("--fps", type=float, default=None, help="Override the dataset frame rate")
@click.option("--owner", default=None, help="Dataset owner username or email")
@click.option("--workers", type=int, default=8, help="Parallel upload threads (default: 8)")
@click.option("--wait/--no-wait", "wait_after", default=False, help="Wait for indexing to finish")
@click.pass_context
def import_lerobot_cmd(
    ctx: click.Context,
    repo_id: str | None,
    root: str | None,
    name: str,
    slug: str,
    episodes: str | None,
    camera_keys: str | None,
    fps: float | None,
    owner: str | None,
    workers: int,
    wait_after: bool,
) -> None:
    """Import a LeRobot dataset (Hugging Face Hub or local) as an Avala MCAP dataset.

    Each episode becomes one .mcap file: camera streams as foxglove.CompressedImage,
    proprioception (state/action) as protobuf Struct messages. Requires the 'lerobot'
    extra: pip install 'avala[lerobot]'.
    """
    from avala.importers import import_lerobot

    episode_list = [int(e) for e in episodes.split(",") if e.strip() != ""] if episodes else None
    camera_list = [c.strip() for c in camera_keys.split(",") if c.strip() != ""] if camera_keys else None

    client = ctx.obj["client"]
    dataset = import_lerobot(
        client,
        repo_id=repo_id,
        root=root,
        name=name,
        slug=slug,
        episodes=episode_list,
        camera_keys=camera_list,
        fps=fps,
        owner_name=owner,
        workers=workers,
        wait=wait_after,
    )
    click.echo(
        f"Dataset created: {dataset.uid} ({dataset.name}) — type={dataset.data_type}, items={dataset.item_count}"
    )


@import_group.command("rosbag")
@click.argument("bag", type=click.Path(exists=True))
@click.option("--name", required=True, help="Dataset name")
@click.option("--slug", required=True, help="Dataset slug")
@click.option("--image-topics", default=None, help="Comma-separated image topics to convert (default: all)")
@click.option("--owner", default=None, help="Dataset owner username or email")
@click.option("--workers", type=int, default=8, help="Parallel upload threads (default: 8)")
@click.option("--wait/--no-wait", "wait_after", default=False, help="Wait for server-side indexing to finish")
@click.pass_context
def import_rosbag_cmd(
    ctx: click.Context,
    bag: str,
    name: str,
    slug: str,
    image_topics: str | None,
    owner: str | None,
    workers: int,
    wait_after: bool,
) -> None:
    """Import a ROS bag (.bag / .db3) as an Avala MCAP dataset.

    Camera topics (sensor_msgs/Image, sensor_msgs/CompressedImage) are re-encoded as
    foxglove.CompressedImage so they render in Mission Control. Non-image topics are
    skipped this increment. Requires the 'rosbag' extra: pip install 'avala[rosbag]'.
    """
    from avala.importers import import_ros_bag

    topics = [t.strip() for t in image_topics.split(",") if t.strip()] if image_topics else None
    client = ctx.obj["client"]
    dataset = import_ros_bag(
        client,
        bag=bag,
        name=name,
        slug=slug,
        image_topics=topics,
        owner_name=owner,
        workers=workers,
        wait=wait_after,
    )
    click.echo(
        f"Dataset created: {dataset.uid} ({dataset.name}) — type={dataset.data_type}, items={dataset.item_count}"
    )


@import_group.command("cloud")
@click.argument("uri", required=False)
@click.option("--name", required=True, help="Dataset name")
@click.option("--slug", required=True, help="Dataset slug")
@click.option(
    "--data-type",
    required=True,
    type=click.Choice(["image", "video", "lidar", "mcap", "splat"]),
    help="Data type the server should index from the bucket",
)
@click.option(
    "--storage-config",
    "storage_config_uid",
    default=None,
    help=(
        "Reuse a saved storage config for bucket/region/prefix "
        "(credentials still required — the server never returns them)"
    ),
)
@click.option("--region", default=None, envvar="AWS_REGION", help="S3 bucket region (or set AWS_REGION)")
@click.option(
    "--access-key-id", default=None, envvar="AWS_ACCESS_KEY_ID", help="S3 access key id (or set AWS_ACCESS_KEY_ID)"
)
@click.option(
    "--secret-access-key",
    default=None,
    envvar="AWS_SECRET_ACCESS_KEY",
    help="S3 secret access key (or set AWS_SECRET_ACCESS_KEY)",
)
@click.option("--role-arn", default=None, help="S3 IAM role ARN for keyless cross-account access")
@click.option("--accelerated", is_flag=True, default=False, help="Use S3 Transfer Acceleration")
@click.option(
    "--gcs-auth-json",
    default=None,
    help="GCS service-account JSON, or a path to a .json key file",
)
@click.option("--include-extensions", default=None, help="Comma-separated extensions to index (e.g. webp,png)")
@click.option("--ignore-paths", default=None, help="Comma-separated glob paths to skip (matched against object keys)")
@click.option("--owner", default=None, help="Dataset owner username or email")
@click.option("--organization-uid", default=None, help="Owning organization uid (required for keyless --role-arn)")
@click.option(
    "--organization-id", type=int, default=None, help="Owning organization id (alternative to --organization-uid)"
)
@click.option("--wait/--no-wait", "wait_after", default=False, help="Wait for server-side indexing to finish")
@click.pass_context
def import_cloud_cmd(
    ctx: click.Context,
    uri: str | None,
    name: str,
    slug: str,
    data_type: str,
    storage_config_uid: str | None,
    region: str | None,
    access_key_id: str | None,
    secret_access_key: str | None,
    role_arn: str | None,
    accelerated: bool,
    gcs_auth_json: str | None,
    include_extensions: str | None,
    ignore_paths: str | None,
    owner: str | None,
    organization_uid: str | None,
    organization_id: int | None,
    wait_after: bool,
) -> None:
    """Import an existing S3/GCS bucket as a zero-copy dataset (no re-upload).

    URI is s3://bucket/prefix or gs://bucket/prefix. Provide S3 credentials
    (--access-key-id/--secret-access-key or env) or a keyless --role-arn, or a
    --gcs-auth-json for GCS. Keyless --role-arn requires --organization-uid.

    With --storage-config, the bucket, region and prefix come from a saved,
    verified config and URI becomes optional (pass one to select a narrower
    prefix inside the same bucket). Credentials are still needed separately.
    """
    from avala.importers import import_cloud

    client = ctx.obj["client"]
    dataset = import_cloud(
        client,
        uri=uri,
        name=name,
        slug=slug,
        data_type=data_type,
        storage_config_uid=storage_config_uid,
        region=region,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        role_arn=role_arn,
        accelerated=accelerated,
        gcs_auth_json=gcs_auth_json,
        included_extensions=include_extensions,
        ignored_paths=ignore_paths,
        owner_name=owner,
        organization_uid=organization_uid,
        organization_id=organization_id,
        wait=wait_after,
    )
    click.echo(
        f"Dataset created: {dataset.uid} ({dataset.name}) — type={dataset.data_type}, items={dataset.item_count}"
    )


@import_group.command("inspect-robots")
@click.argument("log", type=click.Path(exists=True, dir_okay=False))
@click.option("--dataset", required=True, help="Target dataset as owner/slug")
@click.option("--dry-run", is_flag=True, default=False, help="Print the trial -> outcome mapping; no network calls")
@click.option(
    "--create",
    is_flag=True,
    default=False,
    help="Convert each trial's recorded actions/frames to MCAP and create the dataset (default: attach to existing)",
)
@click.option("--name", default=None, help="Dataset name with --create (default: the slug)")
@click.option("--organization-uid", default=None, help="With --create, create the dataset under this organization")
@click.option(
    "--success-key",
    default=None,
    help="Scorer that decides success (default: success_at_end, reached_goal_state or operator, if present)",
)
@click.option(
    "--success-threshold", type=float, default=0.5, show_default=True, help="Success score at or above this succeeds"
)
@click.option("--score-key", default=None, help="Scorer that grades a success (default: the success scorer)")
@click.option(
    "--expert-threshold",
    type=float,
    default=1.0,
    show_default=True,
    help="A success scoring at or above this is expert_success, below it partial_success",
)
@click.option("--progress-key", default=None, help="Copy this scorer's [0, 1] value into the outcome's progress")
@click.option("--overwrite", is_flag=True, default=False, help="Replace existing human/model labels")
@click.option(
    "--receipt",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Write the full mapping (incl. run id, task and scores per trial) to this JSON file",
)
@click.pass_context
def import_inspect_robots_cmd(
    ctx: click.Context,
    log: str,
    dataset: str,
    dry_run: bool,
    create: bool,
    name: str | None,
    organization_uid: str | None,
    success_key: str | None,
    success_threshold: float,
    score_key: str | None,
    expert_threshold: float,
    progress_key: str | None,
    overwrite: bool,
    receipt: str | None,
) -> None:
    """Import Inspect Robots evaluation logs as outcome-labelled sequences.

    LOG is the JSON eval log an Inspect Robots run writes (<task>_<id>.json). Each trial
    (scene x epoch) labels one sequence: success -> expert_success / partial_success,
    failure -> failure, errored or cancelled -> aborted; source=imported,
    evaluation_membership=held_out_eval, model_version from the run's policy.
    Requires the 'inspect' extra: pip install 'avala[inspect]'.
    """
    import json

    from avala.cli._output import _get_output_format, print_table
    from avala.importers.inspect_robots import import_inspect_robots

    try:
        result = import_inspect_robots(
            None if dry_run else ctx.obj["client"],
            log=log,
            dataset=dataset,
            dry_run=dry_run,
            create=create,
            name=name,
            organization_uid=organization_uid,
            success_key=success_key,
            success_threshold=success_threshold,
            score_key=score_key,
            expert_threshold=expert_threshold,
            progress_key=progress_key,
            overwrite=overwrite,
        )
    except ModuleNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    if receipt:
        with open(receipt, "w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, indent=2, sort_keys=True, default=str)

    def _fmt(value: float | None) -> str:
        return "—" if value is None else f"{value:g}"

    if _get_output_format() == "json":
        # Machine-readable output keeps every field, including reason and metadata.
        click.echo(json.dumps(result.to_dict(), indent=2, sort_keys=True, default=str))
        return

    model = result.rows[0].model_version if result.rows else "—"
    columns = ["Trial", "Outcome", "Success", "Score", "Progress", "Status"]
    if not result.dry_run:
        columns.append("Sequence")
    print_table(
        f"Inspect Robots {'dry run' if result.dry_run else 'import'}: {result.task} "
        f"(run {result.run_id}, model {model}) -> {result.owner}/{result.slug}",
        columns,
        [
            (
                row.trial_id,
                row.outcome or "—",
                _fmt(row.success_value),
                _fmt(row.score_value),
                _fmt(row.progress),
                row.status,
            )
            + (() if result.dry_run else (row.sequence_uid or "—",))
            for row in result.rows
        ],
    )
    for row in result.rows:
        if row.status not in ("planned", "labelled", "unchanged"):
            click.echo(f"{row.trial_id}: {row.status}: {row.reason}", err=True)
    if not result.dry_run:
        counts = {status: result.count(status) for status in sorted({row.status for row in result.rows})}
        click.echo(", ".join(f"{status}={count}" for status, count in counts.items()), err=True)
