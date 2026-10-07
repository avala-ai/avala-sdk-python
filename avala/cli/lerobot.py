"""CLI commands for LeRobot dataset export."""

from __future__ import annotations

import click


@click.group()
def lerobot() -> None:
    """Export Avala datasets to the LeRobot v3 format."""


@lerobot.command("export")
@click.argument("dataset")
@click.option("--repo-id", required=True, help="Target LeRobot/HF dataset id, '<user>/<name>'")
@click.option("--output", "output_dir", required=True, help="Output directory for the LeRobot dataset")
@click.option(
    "--fps",
    type=int,
    default=30,
    show_default=True,
    help="Frames per second; timestamps are synthesized as frame_index/fps",
)
@click.option(
    "--task", default="avala sequence", show_default=True, help="Task/instruction string attached to every frame"
)
@click.option(
    "--camera",
    "cameras",
    multiple=True,
    help="Restrict to these camera(s); repeatable. Default: all cameras on frame 0",
)
@click.option("--state-key", default=None, help="Dotted path into the raw frame dict for an observation.state vector")
@click.option("--action-key", default=None, help="Dotted path into the raw frame dict for an action vector")
@click.option(
    "--ego-pose-state",
    is_flag=True,
    default=False,
    help="Use the 7-dim camera-rig ego pose as observation.state (rig pose, not proprioception)",
)
@click.option("--robot-type", default=None, help="robot_type metadata for the LeRobot dataset")
@click.option(
    "--no-video",
    is_flag=True,
    default=False,
    help="Store frames as PNG image features instead of encoded video (avoids the av/torchcodec stack)",
)
@click.option("--limit", type=int, default=None, help="Convert at most this many sequences")
@click.option("--push", is_flag=True, default=False, help="Push the dataset to the Hugging Face Hub (requires HF auth)")
@click.option(
    "--tag",
    "tags",
    multiple=True,
    help="Extra dataset-card tag(s) for the Hub; repeatable. 'avala' and 'LeRobot' are always added.",
)
@click.option(
    "--license",
    "repo_license",
    default=None,
    help="License for the pushed dataset card (default: lerobot's apache-2.0)",
)
@click.option(
    "--include-outcomes",
    is_flag=True,
    default=False,
    help="Write each episode's Avala sequence outcome label to meta/avala_sequence_outcomes.jsonl",
)
@click.option(
    "--control-source-key",
    default=None,
    help="Dotted path into the raw frame dict for who was driving (policy/teleop/intervention/hold)",
)
@click.option(
    "--control-source-column",
    default="annotation.avala.control_source",
    show_default=True,
    help="LeRobot column name for --control-source-key",
)
@click.option(
    "--backend",
    type=click.Choice(["auto", "lerobot", "core"]),
    default="auto",
    show_default=True,
    help="Writer: the lerobot library, the torch-free core writer, or auto (lerobot if installed)",
)
@click.pass_context
def export_cmd(
    ctx: click.Context,
    dataset: str,
    repo_id: str,
    output_dir: str,
    fps: int,
    task: str,
    cameras: tuple[str, ...],
    state_key: str | None,
    action_key: str | None,
    ego_pose_state: bool,
    robot_type: str | None,
    no_video: bool,
    limit: int | None,
    push: bool,
    tags: tuple[str, ...],
    repo_license: str | None,
    include_outcomes: bool,
    control_source_key: str | None,
    control_source_column: str,
    backend: str,
) -> None:
    """Convert an Avala sequence dataset (DATASET = OWNER/SLUG) to a LeRobot v3 dataset.

    By default this produces a perception/vision-language dataset (cameras +
    timestamps). Pass --state-key/--action-key (or --ego-pose-state) when the source
    frames carry proprioception.
    """
    # Imported lazily so `--help` works without the lerobot extra installed.
    from avala.lerobot import export_dataset

    if "/" not in dataset:
        raise click.BadArgumentUsage("DATASET must be in the form OWNER/SLUG")
    owner, slug = dataset.split("/", 1)

    client = ctx.obj["client"]
    out = export_dataset(
        client,
        owner,
        slug,
        repo_id=repo_id,
        output_dir=output_dir,
        fps=fps,
        task=task,
        camera_keys=list(cameras) or None,
        state_key=state_key,
        action_key=action_key,
        include_ego_pose=ego_pose_state,
        robot_type=robot_type,
        use_videos=not no_video,
        limit=limit,
        push=push,
        tags=list(tags) or None,
        repo_license=repo_license,
        include_outcomes=include_outcomes,
        backend=backend,
        control_source_key=control_source_key,
        control_source_column=control_source_column,
    )
    click.echo(f"LeRobot dataset written to {out}")
