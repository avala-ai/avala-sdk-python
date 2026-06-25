"""Import a LeRobot dataset (local or Hugging Face Hub) into Avala as an MCAP dataset.

Each LeRobot *episode* becomes one ``.mcap`` file, which Avala ingests as one MCAP
episode (one ``.mcap`` = one ``DatasetItem`` = one ``McapEpisode``). Inside each file:

* **Camera streams** are written as ``foxglove.CompressedImage`` (protobuf) — the format
  the Avala server indexer and the Mission Control MCAP viewer both understand, so the
  frames render as image panels.
* **Proprioception** (``observation.state``, ``action``, and any other 1-D numeric
  features) is written as ``google.protobuf.Struct`` messages so the values are preserved
  in the file and inspectable as raw messages.

  NOTE: Mission Control's embedded MCAP viewer renders images, point clouds and logs, but
  does not yet *chart* scalar time-series. State/action are therefore preserved and
  raw-viewable, but not plotted. That's a viewer feature, not an import limitation.

The reader uses the ``lerobot`` library (install via ``pip install 'avala[lerobot]'``);
the converter targets the LeRobot v2.1/v3 dataset API (verified against ``lerobot`` 0.5.x).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence

from avala.importers import register_importer

if TYPE_CHECKING:
    from avala._client import Client
    from avala.types.dataset import Dataset

__all__ = ["import_lerobot", "write_episode_mcap"]


# ──────────────────────────────────────────────────────────────────────────────
# MCAP writing — pure, dependency-light (no lerobot/torch). Testable in isolation.
# ──────────────────────────────────────────────────────────────────────────────
def write_episode_mcap(out_path: str, frames: Iterable[Dict[str, Any]]) -> int:
    """Write ``frames`` to an MCAP file at ``out_path``; return the frame count.

    Each frame is a self-describing dict::

        {
            "timestamp_ns": int,                       # nanoseconds, monotonic
            "images":  {topic: np.ndarray HWC uint8},  # one entry per camera
            "structs": {topic: dict},                  # e.g. {"data": [floats]}
        }

    Camera arrays are JPEG-encoded into ``foxglove.CompressedImage``; struct payloads
    become ``google.protobuf.Struct`` messages. ``mcap_protobuf`` auto-registers the
    protobuf schemas (with the FileDescriptorSet the viewer needs) and writes the
    summary section the server parser requires.
    """
    from io import BytesIO

    from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
    from google.protobuf.struct_pb2 import Struct
    from mcap_protobuf.writer import Writer
    from PIL import Image

    count = 0
    with open(out_path, "wb") as fh, Writer(fh) as writer:
        for frame in frames:
            t_ns = int(frame["timestamp_ns"])
            sec, nsec = divmod(t_ns, 1_000_000_000)

            for topic, arr in (frame.get("images") or {}).items():
                buf = BytesIO()
                Image.fromarray(arr).save(buf, format="JPEG", quality=95)
                msg = CompressedImage()
                msg.timestamp.seconds = sec
                msg.timestamp.nanos = nsec
                msg.frame_id = topic.strip("/").replace("/", ".")
                msg.format = "jpeg"
                msg.data = buf.getvalue()
                writer.write_message(topic=topic, message=msg, log_time=t_ns, publish_time=t_ns)

            for topic, payload in (frame.get("structs") or {}).items():
                struct = Struct()
                struct.update(payload)
                writer.write_message(topic=topic, message=struct, log_time=t_ns, publish_time=t_ns)

            count += 1
    return count


# ──────────────────────────────────────────────────────────────────────────────
# LeRobot → frame-dict adaptation
# ──────────────────────────────────────────────────────────────────────────────
def _to_hwc_uint8(value: Any) -> Any:
    """Normalize a LeRobot image (CHW or HWC, float[0,1] or uint8) to HWC uint8."""
    import numpy as np

    arr = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
    # CHW -> HWC: a leading axis of 1/3/4 that is smaller than the trailing axis is a
    # channel dim (LeRobot returns torch CHW image tensors).
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[2]:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        peak = float(arr.max()) if arr.size else 0.0
        # float [0,1] -> scale to [0,255]; otherwise assume [0,255] space. Clip both
        # branches so out-of-range / negative values can't wrap during the uint8 cast.
        arr = np.clip(arr, 0.0, 1.0) * 255.0 if peak <= 1.0 else np.clip(arr, 0.0, 255.0)
        arr = arr.round().astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return arr


def _to_list(value: Any) -> List[float]:
    """Flatten a tensor / array / scalar of numbers to a flat list of floats."""
    import numpy as np

    arr = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
    return [float(x) for x in np.asarray(arr).reshape(-1)]


def _topic_for_camera(key: str) -> str:
    """``observation.images.laptop`` -> ``/observation/images/laptop``."""
    return "/" + key.replace(".", "/")


def _build_frame(
    sample: Dict[str, Any],
    camera_keys: Sequence[str],
    state_keys: Sequence[str],
    fps: float,
) -> Dict[str, Any]:
    """Convert a single LeRobot sample into a writer frame dict."""
    images = {_topic_for_camera(key): _to_hwc_uint8(sample[key]) for key in camera_keys if key in sample}
    structs: Dict[str, Any] = {}
    for key in state_keys:
        if key in sample:
            structs["/" + key.replace(".", "/")] = {"data": _to_list(sample[key])}

    ts = sample.get("timestamp")
    if ts is not None:
        timestamp_ns = int(float(ts.item() if hasattr(ts, "item") else ts) * 1_000_000_000)
    else:
        frame_index = sample.get("frame_index", 0)
        frame_index = float(frame_index.item() if hasattr(frame_index, "item") else frame_index)
        timestamp_ns = int(frame_index / fps * 1_000_000_000)

    return {"timestamp_ns": timestamp_ns, "images": images, "structs": structs}


def _episode_streams(
    dataset: Any,
    camera_keys: Sequence[str],
    state_keys: Sequence[str],
    fps: float,
) -> Iterator[tuple]:
    """Yield ``(episode_index, frames_iter)`` groups over all loaded frames.

    Frames within a LeRobot episode are contiguous and ordered, so grouping the linear
    ``dataset[i]`` stream by each sample's ``episode_index`` recovers per-episode bounds
    without relying on ``episode_data_index`` (removed in lerobot 0.5) or the
    absolute/relative index remapping that subset loading applies.
    """
    import itertools

    def _frames() -> Iterator[tuple]:
        for i in range(len(dataset)):
            sample = dataset[i]
            ep = sample.get("episode_index", 0)
            ep = int(ep.item() if hasattr(ep, "item") else ep)
            yield ep, _build_frame(sample, camera_keys, state_keys, fps)

    for ep, group in itertools.groupby(_frames(), key=lambda pair: pair[0]):
        yield ep, (frame for _ep, frame in group)


def import_lerobot(
    client: "Client",
    *,
    name: str,
    slug: str,
    repo_id: Optional[str] = None,
    root: Optional[str] = None,
    episodes: Optional[Sequence[int]] = None,
    camera_keys: Optional[Sequence[str]] = None,
    state_keys: Optional[Sequence[str]] = None,
    fps: Optional[float] = None,
    visibility: str = "private",
    owner_name: Optional[str] = None,
    industry: Optional[int] = None,
    license: Optional[int] = None,
    workers: int = 8,
    on_progress: "Optional[Callable[[str, int], None]]" = None,
    wait: bool = False,
    wait_timeout: float = 3600.0,
) -> "Dataset":
    """Import a LeRobot dataset into Avala as an MCAP dataset.

    Provide ``repo_id`` (downloaded from the Hugging Face Hub) and/or ``root`` (a local
    dataset directory). Each episode is converted to one ``.mcap`` file and uploaded; the
    resulting Avala dataset has ``data_type="mcap"``.

    ``camera_keys`` / ``state_keys`` default to the dataset's camera features and
    ``observation.state`` + ``action``. ``episodes`` limits the export to specific episode
    indices (default: all).
    """
    import os
    import tempfile

    if not repo_id and not root:
        raise ValueError("provide repo_id (Hugging Face Hub) and/or root (local dataset path)")

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without the extra
        raise ModuleNotFoundError(
            "LeRobot import requires the 'lerobot' extra. Install it with: pip install 'avala[lerobot]'"
        ) from exc

    rid = repo_id or slug

    # Read metadata first (cheap, no video download) to resolve features and validate the
    # episode selection before pulling any media.
    meta = LeRobotDatasetMetadata(rid, root=root)
    available_cameras = list(meta.camera_keys)
    if camera_keys is not None:
        unknown = [k for k in camera_keys if k not in available_cameras]
        if unknown:
            raise ValueError(f"unknown camera keys {unknown}; available cameras: {sorted(available_cameras)}")
    resolved_cameras: Sequence[str] = list(camera_keys) if camera_keys is not None else available_cameras
    if not resolved_cameras:
        raise ValueError("no camera features found in the LeRobot dataset; pass camera_keys explicitly")
    if state_keys is not None:
        unknown_states = [k for k in state_keys if k not in meta.features]
        if unknown_states:
            raise ValueError(f"unknown state keys {unknown_states}; available features: {sorted(meta.features)}")
    resolved_states: Sequence[str] = (
        list(state_keys)
        if state_keys is not None
        else [k for k in ("observation.state", "action") if k in meta.features]
    )
    resolved_fps: float = float(fps if fps is not None else meta.fps)

    total_episodes = int(meta.total_episodes)
    selected = list(episodes) if episodes is not None else list(range(total_episodes))
    invalid = [ep for ep in selected if not 0 <= ep < total_episodes]
    if invalid:
        raise ValueError(f"episode indices {invalid} out of range [0, {total_episodes})")
    if not selected:
        raise ValueError("no episodes selected to import")

    # Pass ``episodes=`` so LeRobot only downloads/loads the requested episodes.
    dataset = LeRobotDataset(rid, root=root, episodes=selected)

    with tempfile.TemporaryDirectory(prefix="avala-lerobot-") as tmp:
        written = 0
        for ep, frames in _episode_streams(dataset, resolved_cameras, resolved_states, resolved_fps):
            out_path = os.path.join(tmp, f"episode_{ep:06d}.mcap")
            if write_episode_mcap(out_path, frames) > 0:
                written += 1
            else:
                os.remove(out_path)

        if written == 0:
            raise ValueError("no non-empty episodes to import")

        return client.datasets.create_from_local(
            source=tmp,
            name=name,
            slug=slug,
            data_type="mcap",
            visibility=visibility,
            owner_name=owner_name,
            industry=industry,
            license=license,
            workers=workers,
            on_progress=on_progress,
            wait=wait,
            wait_timeout=wait_timeout,
        )


register_importer("lerobot", import_lerobot)
