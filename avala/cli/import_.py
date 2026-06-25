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
@click.option("--workers", type=int, default=8, help="Parallel upload threads (default: 8)")
@click.option("--wait/--no-wait", "wait_after", default=False, help="Wait for indexing to finish")
@click.pass_context
def import_folder_cmd(
    ctx: click.Context,
    source: str,
    name: str,
    slug: str,
    data_type: str | None,
    owner: str | None,
    workers: int,
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
        workers=workers,
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
@click.argument("uri")
@click.option("--name", required=True, help="Dataset name")
@click.option("--slug", required=True, help="Dataset slug")
@click.option(
    "--data-type",
    required=True,
    type=click.Choice(["image", "video", "lidar", "mcap", "splat"]),
    help="Data type the server should index from the bucket",
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
    uri: str,
    name: str,
    slug: str,
    data_type: str,
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
    """
    from avala.importers import import_cloud

    client = ctx.obj["client"]
    dataset = import_cloud(
        client,
        uri=uri,
        name=name,
        slug=slug,
        data_type=data_type,
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
