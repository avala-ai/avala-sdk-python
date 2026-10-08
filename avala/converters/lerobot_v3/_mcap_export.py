"""MCAP episodes -> one LeRobot v3 dataset (implementation of ``mcap.mcap_to_lerobot``).

Two input shapes:

* **Importer MCAPs** carry the ``avala.lerobot`` metadata record. The original feature
  specs and feature->topic map are restored and every topic must have a value at every
  frame time.
* **Any other MCAP** is inferred:

  - ``foxglove.CompressedVideo`` (H.264 Annex-B, as Avala recordings and the teleop
    converter write) and ``foxglove.CompressedImage`` topics become
    ``observation.images.*`` video features. H.264 is **copied into the mp4 without
    re-encoding** whenever the frames line up one packet per LeRobot frame, start on a
    keyframe and decode one-to-one (no B-frames); otherwise the camera is decoded and
    re-encoded and a warning says why.
  - ``sensor_msgs/JointState`` (ROS 2 ``ros2msg``, ROS 1 ``ros1msg`` with the optional
    ``mcap-ros2-support`` / ``mcap-ros1-support`` decoders, or the ``Struct`` mirror the
    Avala ROS importer writes): the state topic (``/joint_states``, else the first joint
    topic that is not a command) gives ``observation.state`` = positions, plus
    ``observation.velocity`` / ``observation.effort`` when those arrays are populated.
    A command topic (name containing ``command``, ``cmd``, ``action``, ``target``,
    ``goal``, ``desired`` or ``setpoint``) gives ``action`` = its positions. Without a
    command topic there is **no** ``action``; it is never derived from the state.
  - ``Struct {"data": [numbers]}`` -> float32 vector, ``Struct {"data": "text"}`` ->
    string column, ``/lerobot/control_source`` -> ``annotation.avala.control_source``.
  - A ``foxglove.Log`` named ``task`` (the teleop converter's task text) becomes the
    episode's LeRobot ``task`` unless ``/lerobot/task`` gives one per frame.

  Frames follow the first camera's recorded message times. Other topics contribute their
  latest value at or before each frame (sample-and-hold); frames before every topic has
  published are dropped, never filled.

LeRobot timestamps are ``frame_index / fps`` by definition. The recorded times are kept as
the frame clock (they decide which value lands in which frame), the largest deviation from
the grid is warned about past half a frame, and every episode's maximum skew, dropped
leading frames and per-camera video mode are written to ``meta/avala_export_report.json``.
"""

from __future__ import annotations

import bisect
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from avala.converters.lerobot_v3 import _layout as L
from avala.converters.lerobot_v3.mcap import (
    CONTROL_SOURCE_TOPIC,
    DEFAULT_CONTROL_SOURCE_COLUMN,
    METADATA_NAME,
    TASK_TOPIC,
)

__all__ = ["mcap_to_lerobot", "EXPORT_REPORT_PATH"]

DEFAULT_TASK = "avala episode"
EXPORT_REPORT_PATH = "meta/avala_export_report.json"
_IMAGE_SCHEMA = "foxglove.CompressedImage"
_VIDEO_SCHEMA = "foxglove.CompressedVideo"
_LOG_SCHEMA = "foxglove.Log"
_STRUCT_SCHEMA = "google.protobuf.Struct"
_JOINT_SCHEMAS = frozenset({"sensor_msgs/msg/JointState", "sensor_msgs/JointState"})
_COMMAND_TOKENS = ("command", "cmd", "action", "target", "goal", "desired", "setpoint")
# foxglove.CompressedVideo ``format`` -> libavcodec decoder
_VIDEO_DECODERS = {"h264": "h264", "h265": "hevc", "hevc": "hevc", "vp9": "vp9", "av1": "av1"}
_JOINT_FIELDS = ("position", "velocity", "effort")

Source = Tuple[str, Optional[str]]  # (topic, joint field or None)


@dataclass
class _VideoPacket:
    data: bytes
    fmt: str
    keyframe: bool


@dataclass
class _Joint:
    name: List[str]
    position: List[float]
    velocity: List[float]
    effort: List[float]


@dataclass
class _Episode:
    path: Path
    metadata: Optional[Dict[str, str]]
    series: Dict[str, List[Any]]  # topic -> [(log_time_ns, payload)]
    skipped_topics: List[str]
    task_text: Optional[str] = None


def _annexb_has_idr(data: bytes) -> bool:
    """True when an Annex-B H.264 access unit contains an IDR slice (NAL type 5)."""
    i, n = 0, len(data)
    while i < n - 3:
        if data[i] == 0 and data[i + 1] == 0 and (data[i + 2] == 1 or (data[i + 2] == 0 and data[i + 3] == 1)):
            i += 3 if data[i + 2] == 1 else 4
            if i < n and data[i] & 0x1F == 5:
                return True
        else:
            i += 1
    return False


def _struct_value(msg: Any) -> Any:
    """``Struct {"data": ...}`` -> Python value; a JointState-shaped Struct -> ``_Joint``."""
    if "data" not in msg:
        if "position" in msg and "name" in msg:
            return _Joint(**{f: list(msg[f]) if f in msg else [] for f in ("name", *_JOINT_FIELDS)})
        raise ValueError("Struct message has no 'data' field")
    value = msg["data"]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "keys"):  # nested Struct: no LeRobot equivalent
        raise ValueError("nested Struct 'data' is not supported")
    # ListValue. Not an isinstance check: mcap_protobuf decodes with classes built from the
    # file's own descriptors, which are not google.protobuf.struct_pb2.ListValue.
    items = list(value)
    if any(not isinstance(x, (str, bool, int, float)) for x in items):
        raise ValueError("nested values inside Struct 'data' are not supported")
    return items


def _ros_joint(msg: Any) -> _Joint:
    return _Joint(
        name=[str(n) for n in msg.name],
        position=[float(x) for x in msg.position],
        velocity=[float(x) for x in msg.velocity],
        effort=[float(x) for x in msg.effort],
    )


def _decoder_factories() -> List[Any]:
    from mcap_protobuf.decoder import DecoderFactory

    factories: List[Any] = [DecoderFactory()]
    for module in ("mcap_ros2.decoder", "mcap_ros1.decoder"):
        try:
            import importlib

            factories.append(importlib.import_module(module).DecoderFactory())
        except ModuleNotFoundError:
            pass  # ROS-encoded topics are then reported as skipped
    return factories


def _read_episode(path: Path) -> _Episode:
    from mcap.reader import make_reader

    factories = _decoder_factories()
    series: Dict[str, List[Any]] = {}
    skipped: Dict[str, None] = {}
    decoders: Dict[int, Any] = {}
    task_text: Optional[str] = None
    with open(path, "rb") as fh:
        reader = make_reader(fh)
        metadata: Optional[Dict[str, str]] = None
        for record in reader.iter_metadata():
            if record.name == METADATA_NAME:
                metadata = dict(record.metadata)
        fh.seek(0)
        reader = make_reader(fh)
        for schema, channel, message in reader.iter_messages():
            topic = channel.topic
            if topic in skipped:
                continue
            name = schema.name if schema is not None else ""
            if channel.id not in decoders:
                decoders[channel.id] = next(
                    (d for f in factories if (d := f.decoder_for(channel.message_encoding, schema)) is not None), None
                )
            decode = decoders[channel.id]
            if decode is None:
                skipped[topic] = None
                continue
            decoded = decode(message.data)
            payload: Any
            try:
                if name == _IMAGE_SCHEMA:
                    payload = (bytes(decoded.data), str(decoded.format))
                elif name == _VIDEO_SCHEMA:
                    fmt = str(decoded.format).lower()
                    data = bytes(decoded.data)
                    payload = _VideoPacket(data, fmt, _annexb_has_idr(data) if fmt == "h264" else False)
                elif name == _STRUCT_SCHEMA:
                    payload = _struct_value(decoded)
                elif name in _JOINT_SCHEMAS:
                    payload = _ros_joint(decoded)
                elif name == _LOG_SCHEMA and str(decoded.name) == "task":
                    task_text = task_text or str(decoded.message) or None
                    continue
                else:
                    raise ValueError(name)
            except ValueError:
                skipped[topic] = None
                series.pop(topic, None)
                continue
            series.setdefault(topic, []).append((int(message.log_time), payload))
    for values in series.values():
        values.sort(key=lambda item: item[0])
    return _Episode(path=path, metadata=metadata, series=series, skipped_topics=list(skipped), task_text=task_text)


# ── video decoding ──
class _PacketDecoder:
    """Decode a topic's packets in order; ``frame(i)`` is the picture of packet ``i``.

    Assumes one picture per packet in order (verified by :func:`_one_to_one` before a
    pass-through, and by the count check here otherwise). Requests must not go backwards.
    """

    def __init__(self, packets: List[Any], fmt: str) -> None:
        import av

        codec = _VIDEO_DECODERS.get(fmt)
        if codec is None:
            raise ValueError(f"unsupported CompressedVideo format {fmt!r}")
        self._ctx: Any = av.CodecContext.create(codec, "r")
        self._packets = packets
        self._fed = 0
        self._ready: Dict[int, Any] = {}
        self._produced = 0

    def _feed(self) -> None:
        import av

        if self._fed < len(self._packets):
            frames = self._ctx.decode(av.Packet(self._packets[self._fed].data))
            self._fed += 1
        else:
            frames = self._ctx.decode(None)
            if not frames:
                raise ValueError("decoder produced fewer pictures than packets")
        for frame in frames:
            self._ready[self._produced] = frame.to_ndarray(format="rgb24")
            self._produced += 1

    def frame(self, i: int) -> Any:
        while i not in self._ready:
            if i < self._produced:
                raise ValueError("video frames requested out of order")
            self._feed()
        for stale in [k for k in self._ready if k < i]:
            del self._ready[stale]
        return self._ready[i]


def _one_to_one(packets: List[_VideoPacket]) -> bool:
    """True when every packet decodes to exactly one picture immediately (no reordering delay)."""
    import av

    ctx: Any = av.CodecContext.create(_VIDEO_DECODERS[packets[0].fmt], "r")
    for packet in packets:
        if len(ctx.decode(av.Packet(packet.data))) != 1:
            return False
    return len(ctx.decode(None)) == 0


def _image_shape(payload: Any) -> Tuple[int, int]:
    img = _decode_image(payload)
    return int(img.shape[0]), int(img.shape[1])


def _decode_image(payload: Any) -> Any:
    from io import BytesIO

    import numpy as np
    from PIL import Image

    data, _fmt = payload
    return np.asarray(Image.open(BytesIO(data)).convert("RGB"))


# ── schema inference ──
# Path segments that say "this is a camera stream" rather than which camera it is.
_GENERIC_CAMERA_SEGMENTS = frozenset(
    {"camera", "cameras", "cam", "video", "image", "images", "image_raw", "compressed", "rgb", "color"}
)


def _infer_name(topic: str, *, image: bool) -> str:
    """Feature name for a topic: ``/camera/high/video`` -> ``observation.images.high``."""
    if topic == CONTROL_SOURCE_TOPIC:
        return DEFAULT_CONTROL_SOURCE_COLUMN
    name = topic.strip("/").replace("/", ".")
    if image and not name.startswith("observation.images."):
        parts = [p for p in topic.strip("/").split("/") if p]
        specific = [p for p in parts if p.lower() not in _GENERIC_CAMERA_SEGMENTS] or parts
        name = "observation.images." + "_".join(specific)
    return name


def _is_command(topic: str) -> bool:
    lowered = topic.lower()
    return any(token in lowered for token in _COMMAND_TOKENS)


def _inferred_image_spec(height: int, width: int) -> Dict[str, Any]:
    """A camera feature the way lerobot itself declares one: channels-last, axes named."""
    return {"dtype": "video", "shape": [height, width, 3], "names": list(L.INFERRED_IMAGE_NAMES)}


def _infer_schema(ep: _Episode) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Source]]:
    """Features + feature->(topic, joint field) for an MCAP without our metadata record."""
    features: Dict[str, Dict[str, Any]] = {}
    sources: Dict[str, Source] = {}

    def add(key: str, spec: Dict[str, Any], source: Source) -> None:
        if key in sources:
            raise ValueError(f"topics {sources[key][0]!r} and {source[0]!r} both map to feature {key!r}")
        features[key] = spec
        sources[key] = source

    joint_topics: List[str] = []
    for topic, values in ep.series.items():
        if topic == TASK_TOPIC:
            continue
        first = values[0][1]
        if isinstance(first, _VideoPacket):
            decoder = _PacketDecoder([p for _, p in values], first.fmt)
            img = decoder.frame(0)
            spec = _inferred_image_spec(int(img.shape[0]), int(img.shape[1]))
            add(_infer_name(topic, image=True), spec, (topic, None))
        elif isinstance(first, tuple):
            height, width = _image_shape(first)
            spec = _inferred_image_spec(height, width)
            add(_infer_name(topic, image=True), spec, (topic, None))
        elif isinstance(first, _Joint):
            joint_topics.append(topic)
        elif isinstance(first, str):
            add(_infer_name(topic, image=False), {"dtype": "string", "shape": [1], "names": None}, (topic, None))
        elif isinstance(first, list) and first and all(isinstance(x, (int, float)) for x in first):
            dtype = "bool" if all(isinstance(x, bool) for x in first) else "float32"
            add(_infer_name(topic, image=False), {"dtype": dtype, "shape": [len(first)], "names": None}, (topic, None))

    states = [t for t in joint_topics if not _is_command(t)]
    commands = [t for t in joint_topics if _is_command(t)]
    state_topic = "/joint_states" if "/joint_states" in states else (states[0] if states else None)
    action_topic = commands[0] if commands else None
    for topic in joint_topics:
        first = ep.series[topic][0][1]
        names = list(first.name) or None
        if topic == state_topic:
            keys = {"position": "observation.state", "velocity": "observation.velocity", "effort": "observation.effort"}
        elif topic == action_topic:
            keys = {"position": "action"}
        else:
            base = topic.strip("/").replace("/", ".")
            keys = {f: f"{base}.{f}" for f in _JOINT_FIELDS}
        for joint_field, key in keys.items():
            values = getattr(first, joint_field)
            if not values:
                continue  # an unpopulated array is not a feature
            spec = {"dtype": "float32", "shape": [len(values)], "names": names}
            add(key, spec, (topic, joint_field))
    return features, sources


def _joint_vector(joint: _Joint, joint_field: str, names: Optional[List[str]], where: str) -> List[float]:
    values = list(getattr(joint, joint_field))
    if names and joint.name != names:
        if sorted(joint.name) != sorted(names) or len(joint.name) != len(values):
            raise ValueError(f"{where}: joint names changed from {names} to {joint.name}")
        order = {n: i for i, n in enumerate(joint.name)}
        values = [values[order[n]] for n in names]
    if names and len(values) != len(names):
        raise ValueError(f"{where}: {joint_field} has {len(values)} values for {len(names)} joints (never filled in)")
    return values


# ── frame assembly ──
@dataclass
class _EpisodeResult:
    frames: int = 0
    dropped_leading: int = 0
    max_skew_s: float = 0.0
    video_modes: Dict[str, str] = field(default_factory=dict)


def _lookup_rows(
    ep: _Episode, sources: Dict[str, Source], clock: List[int], *, exact: bool
) -> Tuple[List[Tuple[int, Dict[str, int]]], int]:
    """For each clock time, the index of the value each topic contributes; drop incomplete frames."""
    topics = sorted({topic for topic, _ in sources.values()})
    missing = [t for t in topics if t not in ep.series]
    if missing:
        raise ValueError(f"{ep.path.name}: topics {missing} are missing from this episode")
    times = {t: [ts for ts, _ in ep.series[t]] for t in topics}
    rows: List[Tuple[int, Dict[str, int]]] = []
    dropped = 0
    for t in clock:
        idx: Dict[str, int] = {}
        for topic in topics:
            i = bisect.bisect_right(times[topic], t) - 1
            if i < 0 or (exact and times[topic][i] != t):
                if exact:
                    keys = [k for k, (src, _) in sources.items() if src == topic]
                    raise ValueError(f"{ep.path.name}: frame at t={t}ns has no value for {keys}")
                break
            idx[topic] = i
        else:
            rows.append((t, idx))
            continue
        if rows:
            raise ValueError(f"{ep.path.name}: a topic stopped publishing mid-episode at t={t}ns")
        dropped += 1  # before every topic has published; never fabricated
    return rows, dropped


def _write_episode(
    writer: Any,
    ep: _Episode,
    features: Dict[str, Dict[str, Any]],
    sources: Dict[str, Source],
    *,
    exact: bool,
    rate: int,
    task: str,
    passthrough: bool,
) -> _EpisodeResult:
    import numpy as np

    from avala.converters.lerobot_v3.writer import EncodedVideoFrame

    result = _EpisodeResult()
    if exact:
        clock = sorted({t for topic, _ in sources.values() if topic in ep.series for t, _ in ep.series[topic]})
    else:
        visual = [src for k, (src, _) in sources.items() if features[k]["dtype"] in L.VISUAL_DTYPES]
        primary = visual[0] if visual else next(iter(sources.values()))[0]
        clock = [t for t, _ in ep.series.get(primary, [])]
    rows, result.dropped_leading = _lookup_rows(ep, sources, clock, exact=exact)
    if not rows:
        return result

    # Decide, per video topic, whether its packets can be copied as they are.
    decoders: Dict[str, _PacketDecoder] = {}
    copy_topics: Dict[str, None] = {}
    for key, (topic, _) in sources.items():
        first = ep.series[topic][0][1]
        if not isinstance(first, _VideoPacket):
            continue
        packets = [p for _, p in ep.series[topic]]
        decoders[topic] = _PacketDecoder(packets, first.fmt)
        picked = [idx[topic] for _, idx in rows]
        start = picked[0]
        run = packets[start : start + len(picked)]
        reason = None
        if features[key]["dtype"] != "video":
            reason = "image features requested"
        elif not passthrough:
            reason = "pass-through disabled"
        elif first.fmt != "h264":
            reason = f"format {first.fmt!r} is not h264"
        elif picked != list(range(start, start + len(picked))):
            reason = "its frames do not map one packet per LeRobot frame on the shared clock"
        elif not run[0].keyframe:
            reason = "the first kept packet is not a keyframe"
        elif not _one_to_one(run):
            reason = "packets do not decode one picture each in order (B-frames or delay)"
        if reason is None:
            copy_topics[topic] = None
            result.video_modes[key] = "passthrough"
        else:
            result.video_modes[key] = "reencoded"
            if reason != "image features requested":
                warnings.warn(f"{ep.path.name}: re-encoding {topic}: {reason}", stacklevel=3)

    episode_task = ep.task_text or task
    t0 = rows[0][0]
    for count, (t, idx) in enumerate(rows):
        result.max_skew_s = max(result.max_skew_s, abs((t - t0) / 1e9 - count / rate))
        frame: Dict[str, Any] = {}
        for key, (topic, joint_field) in sources.items():
            value = ep.series[topic][idx[topic]][1]
            spec = features[key]
            if isinstance(value, _VideoPacket):
                image = decoders[topic].frame(idx[topic])
                frame[key] = (
                    EncodedVideoFrame(value.data, value.keyframe, image, value.fmt) if topic in copy_topics else image
                )
            elif spec["dtype"] in L.VISUAL_DTYPES:
                frame[key] = _decode_image(value)
            elif joint_field is not None:
                where = f"{ep.path.name} {topic} t={t}ns"
                frame[key] = np.asarray(_joint_vector(value, joint_field, spec.get("names"), where), dtype="float32")
            elif spec["dtype"] == "string":
                frame[key] = value
            else:
                frame[key] = np.asarray(value, dtype=np.dtype(spec["dtype"]))
        task_value: Any = None
        if TASK_TOPIC in ep.series:
            ti = bisect.bisect_right([ts for ts, _ in ep.series[TASK_TOPIC]], t) - 1
            task_value = ep.series[TASK_TOPIC][ti][1] if ti >= 0 else None
        frame[L.TASK_KEY] = task_value if isinstance(task_value, str) and task_value else episode_task
        writer.add_frame(frame)
        result.frames += 1
    return result


def _signature(features: Dict[str, Dict[str, Any]]) -> Dict[str, Tuple[str, List[int]]]:
    return {k: (str(v["dtype"]), [int(x) for x in v["shape"]]) for k, v in features.items()}


def mcap_to_lerobot(
    mcap_paths: Sequence[Union[str, Path]],
    output_dir: Union[str, Path],
    *,
    repo_id: str,
    fps: Optional[int] = None,
    robot_type: Optional[str] = None,
    use_videos: bool = True,
    task: Optional[str] = None,
    vcodec: str = "libx264",
    passthrough_video: bool = True,
) -> Path:
    """Write Avala MCAP episodes as one LeRobot v3 dataset (one ``.mcap`` = one episode).

    The torch-free export entry point intended for server-side use; see the module
    docstring of :mod:`avala.converters.lerobot_v3._mcap_export` for the mapping.

    ``fps`` defaults to the recorded fps (importer MCAPs), else the rounded median
    interval of the first camera. The per-frame ``task`` is ``/lerobot/task`` when
    present, else the episode's ``foxglove.Log`` ``task`` text, else ``task`` (default
    ``"avala episode"``). ``passthrough_video=False`` forces re-encoding of H.264. The
    first episode defines the schema; later episodes must match it. Writes
    ``meta/avala_export_report.json`` beside lerobot's files.

    Returns the dataset root (``output_dir``), which must not exist or be empty.
    """
    from avala.converters.lerobot_v3.writer import LeRobotV3Writer

    paths = [Path(p) for p in mcap_paths]
    if not paths:
        raise ValueError("no MCAP files given")
    first = _read_episode(paths[0])
    if first.metadata is not None:
        recorded = json.loads(first.metadata["features"])
        features = {k: dict(v) for k, v in recorded.items()}
        sources: Dict[str, Source] = {k: (t, None) for k, t in json.loads(first.metadata["topics"]).items()}
        exact = True
    else:
        recorded = None
        features, sources = _infer_schema(first)
        exact = False
    if not features:
        raise ValueError(f"{first.path.name}: no image, video, joint or Struct topics to export")
    signature = _signature(features)
    for spec in features.values():
        spec.pop("info", None)  # recomputed by the writer from the written video
        if spec["dtype"] in L.VISUAL_DTYPES:
            spec["dtype"] = "video" if use_videos and spec["dtype"] == "video" else "image"

    rate = fps
    if rate is None and first.metadata is not None and first.metadata.get("fps"):
        rate = int(round(float(first.metadata["fps"])))
    if rate is None:
        visual = [src for k, (src, _) in sources.items() if features[k]["dtype"] in L.VISUAL_DTYPES]
        primary = visual[0] if visual else next(iter(sources.values()))[0]
        clock = [t for t, _ in first.series[primary]]
        deltas = sorted(b - a for a, b in zip(clock, clock[1:]))
        if not deltas:
            raise ValueError("cannot infer fps from a single-frame episode; pass fps=")
        rate = int(round(1e9 / deltas[len(deltas) // 2]))

    writer = LeRobotV3Writer.create(
        repo_id=repo_id,
        fps=rate,
        features=features,
        root=output_dir,
        robot_type=robot_type if robot_type is not None else ((first.metadata or {}).get("robot_type") or None),
        use_videos=any(s["dtype"] == "video" for s in features.values()),
        vcodec=vcodec,
    )
    skipped: Dict[str, None] = {}
    report: List[Dict[str, Any]] = []
    try:
        for i, path in enumerate(paths):
            ep = first if i == 0 else _read_episode(path)  # one episode in memory at a time
            skipped.update(dict.fromkeys(ep.skipped_topics))
            if exact:
                if ep.metadata is None:
                    raise ValueError(f"{path.name} has no Avala LeRobot metadata but {first.path.name} does")
                same = {k: t for k, t in json.loads(ep.metadata["topics"]).items()} == {
                    k: t for k, (t, _) in sources.items()
                } and _signature(json.loads(ep.metadata["features"])) == _signature(recorded or {})
                if not same:
                    raise ValueError(f"{path.name} has a different feature schema than {first.path.name}")
            elif i and _signature(_infer_schema(ep)[0]) != signature:
                raise ValueError(f"{path.name} has a different feature schema than {first.path.name}")
            result = _write_episode(
                writer,
                ep,
                features,
                sources,
                exact=exact,
                rate=rate,
                task=task or DEFAULT_TASK,
                passthrough=passthrough_video,
            )
            ep.series = {}  # release this episode's payloads before reading the next
            if not result.frames:
                warnings.warn(f"{path.name}: no complete frames; skipped", stacklevel=2)
                continue
            writer.save_episode()
            if result.max_skew_s > 0.5 / rate:
                warnings.warn(
                    f"{path.name}: recorded timing deviates from frame_index/fps by up to {result.max_skew_s:.3f}s; "
                    "LeRobot timestamps are frame_index/fps",
                    stacklevel=2,
                )
            report.append(
                {
                    "episode_index": writer.num_episodes - 1,
                    "source": path.name,
                    "frames": result.frames,
                    "dropped_leading_frames": result.dropped_leading,
                    "max_timing_skew_s": result.max_skew_s,
                    "video": result.video_modes,
                }
            )
    finally:
        writer.finalize()
    out = Path(output_dir)
    (out / EXPORT_REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    (out / EXPORT_REPORT_PATH).write_text(json.dumps({"fps": rate, "episodes": report}, indent=2) + "\n")
    if skipped:
        warnings.warn(f"skipped topics with no LeRobot mapping: {sorted(skipped)}", stacklevel=2)
    return out
