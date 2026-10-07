"""Bridge between LeRobot frames and Avala MCAP episodes (one LeRobot episode = one ``.mcap``).

Topic layout written for every frame, all with the frame's timestamp as ``log_time``:

==========================================  ==========================  ==========================
LeRobot feature                             MCAP topic                  message
==========================================  ==========================  ==========================
camera (``image``/``video``) ``a.b.c``      ``/a/b/c``                  ``foxglove.CompressedImage``
numeric ``a.b`` (vector or scalar)          ``/a/b``                    ``Struct {"data": [x, ...]}``
string ``a.b``                              ``/a/b``                    ``Struct {"data": "text"}``
``control_source`` / ``*.control_source``   ``/lerobot/control_source``  ``Struct {"data": "policy"}``
per-frame task text                         ``/lerobot/task``           ``Struct {"data": "text"}``
==========================================  ==========================  ==========================

``observation.state`` and ``action`` therefore keep the ``/observation/state`` and
``/action`` topics and the ``{"data": [floats]}`` Struct layout the importer has always
written. Every other non-image per-frame column — numeric or categorical, e.g.
``annotation.vendor.control_source``, ``episode_uuid``, ``failure_type`` — gets its
own topic, so nothing in the source is dropped.

``control_source`` (who was driving each frame: ``policy`` / ``teleop`` /
``intervention`` / ``hold``) lands on the stable topic :data:`CONTROL_SOURCE_TOPIC`
whatever the source column was called (``annotation.<vendor>.control_source`` is common),
so consumers need not know the vendor prefix. Only the first such column moves; any
further one keeps its own path-derived topic. Values are copied verbatim, never mapped
onto the vocabulary above.

Each file also carries one MCAP metadata record, :data:`METADATA_NAME`, holding the
source ``fps``, ``robot_type``, ``episode_index``, ``codebase_version`` and the exact
feature specs plus feature→topic map (JSON strings), so an export back to LeRobot
restores the original column names, dtypes, shapes and names.

Precision: ``Struct`` numbers are IEEE doubles. float16/32/64 and integers up to 2**53
round-trip exactly; larger 64-bit integers do not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from avala.converters.lerobot_v3 import _layout as L
from avala.converters.lerobot_v3._values import as_python_scalar, flat_numbers, to_hwc_uint8

__all__ = [
    "CONTROL_SOURCE_TOPIC",
    "CONTROL_SOURCE_VALUES",
    "DEFAULT_CONTROL_SOURCE_COLUMN",
    "METADATA_NAME",
    "TASK_TOPIC",
    "TopicPlan",
    "build_frame",
    "feature_topic",
    "lerobot_to_mcap",
    "mcap_to_lerobot",
    "plan_topics",
    "write_episode_mcap",
]

CONTROL_SOURCE_TOPIC = "/lerobot/control_source"
TASK_TOPIC = "/lerobot/task"
METADATA_NAME = "avala.lerobot"
METADATA_SCHEMA = "avala.lerobot.mcap/1"
# The vocabulary seen in third-party v3 recordings; documentation only — values are
# carried verbatim and never validated against it.
CONTROL_SOURCE_VALUES = ("policy", "teleop", "intervention", "hold")
# Column name used when exporting a control source that has no recorded original name.
DEFAULT_CONTROL_SOURCE_COLUMN = "annotation.avala.control_source"


def feature_topic(key: str) -> str:
    """``observation.images.laptop`` -> ``/observation/images/laptop``."""
    return "/" + key.replace(".", "/")


def is_control_source(key: str) -> bool:
    return key == "control_source" or key.endswith(".control_source")


def _spec(features: Mapping[str, Any], key: str) -> Dict[str, Any]:
    spec = features.get(key)
    if not isinstance(spec, Mapping) or "dtype" not in spec:
        raise ValueError(f"feature {key!r} has no dtype in the dataset's features: {spec!r}")
    return dict(spec)


@dataclass
class TopicPlan:
    """Which feature goes to which topic, and how it is encoded."""

    cameras: Dict[str, str] = field(default_factory=dict)  # feature -> topic (CompressedImage)
    numeric: Dict[str, str] = field(default_factory=dict)  # feature -> topic (Struct list)
    strings: Dict[str, str] = field(default_factory=dict)  # feature -> topic (Struct str)
    specs: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # carried feature specs

    @property
    def topics(self) -> Dict[str, str]:
        return {**self.cameras, **self.numeric, **self.strings}

    @property
    def keys(self) -> List[str]:
        return list(self.topics)


def plan_topics(
    features: Mapping[str, Any],
    camera_keys: Sequence[str],
    state_keys: Sequence[str] = (),
) -> TopicPlan:
    """Map every carried feature to a topic.

    ``camera_keys`` are the selected cameras (unselected visual features are not
    carried). ``state_keys`` must exist and be numeric; every other non-visual,
    non-bookkeeping feature is carried as well.
    """
    plan = TopicPlan()
    for key in camera_keys:
        plan.cameras[key] = feature_topic(key)
        plan.specs[key] = _spec(features, key)
    control_assigned = False
    ordered = list(dict.fromkeys([*state_keys, *features]))
    for key in ordered:
        if key in L.BOOKKEEPING_KEYS or key == L.TASK_KEY or key in plan.cameras:
            continue
        spec = _spec(features, key)
        dtype = str(spec["dtype"])
        if dtype in L.VISUAL_DTYPES:
            if key in state_keys:
                raise ValueError(f"state key {key!r} is a {dtype} feature, not a numeric vector")
            continue
        if dtype == "string":
            if key in state_keys:
                raise ValueError(f"state key {key!r} is a string feature, not a numeric vector")
            if is_control_source(key) and not control_assigned:
                plan.strings[key] = CONTROL_SOURCE_TOPIC
                control_assigned = True
            else:
                plan.strings[key] = feature_topic(key)
        else:
            plan.numeric[key] = feature_topic(key)
        plan.specs[key] = spec
    # Keep the source column order (an export restores it).
    plan.specs = {k: plan.specs[k] for k in features if k in plan.specs}
    seen: Dict[str, str] = {TASK_TOPIC: "<task>"}
    for key, topic in plan.topics.items():
        if topic in seen:
            raise ValueError(f"features {seen[topic]!r} and {key!r} would both be written to topic {topic!r}")
        seen[topic] = key
    return plan


def build_frame(sample: Mapping[str, Any], plan: TopicPlan, fps: float) -> Dict[str, Any]:
    """Convert one LeRobot sample (library or core reader) into a writer frame dict."""
    images = {topic: to_hwc_uint8(sample[key]) for key, topic in plan.cameras.items() if key in sample}
    structs: Dict[str, Any] = {}
    for key, topic in plan.numeric.items():
        if key in sample:
            structs[topic] = {"data": [float(x) if not isinstance(x, bool) else x for x in flat_numbers(sample[key])]}
    for key, topic in plan.strings.items():
        if key in sample:
            value = as_python_scalar(sample[key])
            if not isinstance(value, str):
                raise ValueError(f"string feature {key!r} has a non-string value {value!r}")
            structs[topic] = {"data": value}
    task = sample.get(L.TASK_KEY)
    if isinstance(task, str) and task:
        structs[TASK_TOPIC] = {"data": task}

    ts = sample.get("timestamp")
    if ts is not None:
        timestamp_ns = int(float(as_python_scalar(ts)) * 1_000_000_000)
    else:
        frame_index = float(as_python_scalar(sample.get("frame_index", 0)))
        timestamp_ns = int(frame_index / fps * 1_000_000_000)
    return {"timestamp_ns": timestamp_ns, "images": images, "structs": structs}


def _jsonable_spec(spec: Mapping[str, Any]) -> Dict[str, Any]:
    out = json.loads(json.dumps(dict(spec), default=lambda o: list(o) if isinstance(o, tuple) else str(o)))
    out["shape"] = [int(x) for x in spec.get("shape", [])]
    return dict(out)


def episode_metadata(
    plan: TopicPlan,
    *,
    fps: float,
    episode_index: int,
    robot_type: Optional[str],
    codebase_version: Optional[str],
) -> Dict[str, str]:
    """The :data:`METADATA_NAME` record for one episode (all values are strings, per MCAP)."""
    return {
        "schema": METADATA_SCHEMA,
        "codebase_version": str(codebase_version or ""),
        "fps": repr(float(fps)),
        "robot_type": robot_type or "",
        "episode_index": str(int(episode_index)),
        "features": json.dumps({k: _jsonable_spec(v) for k, v in plan.specs.items()}),
        "topics": json.dumps({k: plan.topics[k] for k in plan.specs}),
        "task_topic": TASK_TOPIC,
    }


def write_episode_mcap(
    out_path: Union[str, Path],
    frames: Iterable[Dict[str, Any]],
    *,
    metadata: Optional[Dict[str, str]] = None,
) -> int:
    """Write ``frames`` to an MCAP file at ``out_path``; return the frame count.

    Each frame is a self-describing dict::

        {
            "timestamp_ns": int,                       # nanoseconds, monotonic
            "images":  {topic: np.ndarray HWC uint8},  # one entry per camera
            "structs": {topic: dict},                  # e.g. {"data": [floats]} / {"data": "teleop"}
        }

    Camera arrays are JPEG-encoded into ``foxglove.CompressedImage``; struct payloads
    become ``google.protobuf.Struct`` messages. ``mcap_protobuf`` auto-registers the
    protobuf schemas (with the FileDescriptorSet the viewer needs) and writes the summary
    section the server parser requires. ``metadata`` is stored as one
    :data:`METADATA_NAME` metadata record.
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
        if metadata is not None and count:
            # mcap_protobuf's Writer wraps mcap's Writer; metadata records go on the inner one.
            writer._writer.add_metadata(METADATA_NAME, metadata)
    return count


def lerobot_to_mcap(
    root: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    episodes: Optional[Sequence[int]] = None,
    camera_keys: Optional[Sequence[str]] = None,
    state_keys: Optional[Sequence[str]] = None,
    fps: Optional[float] = None,
) -> List[Path]:
    """Convert a local LeRobot v3 dataset to one ``episode_XXXXXX.mcap`` per episode.

    Torch-free (uses :class:`~avala.converters.lerobot_v3.reader.LeRobotV3Dataset`).
    Returns the written paths, in episode order. Empty episodes are skipped.
    """
    from avala.converters.lerobot_v3.reader import LeRobotV3Dataset

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with LeRobotV3Dataset(root) as ds:
        cameras = list(camera_keys) if camera_keys is not None else ds.camera_keys
        unknown = [k for k in cameras if k not in ds.camera_keys]
        if unknown:
            raise ValueError(f"unknown camera keys {unknown}; available cameras: {sorted(ds.camera_keys)}")
        states = list(state_keys) if state_keys is not None else []
        missing = [k for k in states if k not in ds.features]
        if missing:
            raise ValueError(f"unknown state keys {missing}; available features: {sorted(ds.features)}")
        plan = plan_topics(ds.features, cameras, states)
        rate = float(fps if fps is not None else ds.fps)
        selected = list(episodes) if episodes is not None else [int(e["episode_index"]) for e in ds.episodes]
        paths: List[Path] = []
        for ep in selected:
            path = out / f"episode_{ep:06d}.mcap"
            meta = episode_metadata(
                plan,
                fps=rate,
                episode_index=ep,
                robot_type=ds.robot_type,
                codebase_version=str(ds.info.get("codebase_version")),
            )
            frames = (build_frame(s, plan, rate) for s in ds.iter_frames(ep, keys=plan.keys))
            if write_episode_mcap(path, frames, metadata=meta) > 0:
                paths.append(path)
            else:
                path.unlink()
        return paths


# MCAP -> LeRobot v3 lives in ``_mcap_export`` (it needs the constants above).
from avala.converters.lerobot_v3._mcap_export import mcap_to_lerobot  # noqa: E402
