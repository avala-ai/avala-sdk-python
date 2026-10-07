"""Write a LeRobot v3.0 dataset with pyarrow (+ PyAV for video), no torch.

The public surface deliberately mirrors the subset of ``lerobot.datasets.LeRobotDataset``
that recording code uses — ``create(...)`` / ``add_frame(frame)`` / ``save_episode()`` /
``finalize()`` — so callers can swap one for the other. See ``_layout`` for where each
path and column comes from.

What it writes (same as lerobot 0.5.1, checked against a dataset that library wrote):

* ``meta/info.json`` — ``codebase_version``, totals, ``chunks_size``, file size limits,
  ``fps``, ``splits``, ``data_path``, ``video_path``, ``features`` (user features then the
  five bookkeeping columns; video features gain an ``info`` block).
* ``meta/tasks.parquet`` — ``task_index`` + ``task``, ``task`` marked as the pandas
  index so ``pandas.read_parquet`` returns it as lerobot expects.
* ``meta/episodes/chunk-000/file-000.parquet`` — one row per episode: ``episode_index``,
  ``tasks``, ``length``, ``data/*``, ``dataset_from_index``/``dataset_to_index``,
  ``videos/<key>/*`` (video mode), ``stats/<feature>/<stat>``,
  ``meta/episodes/*``.
* ``data/chunk-XXX/file-XXX.parquet`` — one row per frame. Vectors are
  ``fixed_size_list``, shape-``(1,)`` features are plain scalars, ``string`` features
  are strings, ``image`` features are ``struct<bytes, path>`` PNGs.
* ``videos/<key>/chunk-XXX/file-XXX.mp4`` — episodes concatenated in order; each episode
  row records its ``[from_timestamp, to_timestamp)`` window in that file.
* ``meta/stats.json`` — aggregate stats (see ``_stats``).

Differences from lerobot, all within the format: quantiles are exact rather than
histogram estimates, image stats use a <=150 px subsample per frame (lerobot also
subsamples), and the default video codec is ``libx264`` (lerobot defaults to
``libsvtav1``; both are valid ``video.codec`` values for a v3 reader).

Requires ``pip install 'avala[lerobot-core]'``; video features also need PyAV
(``pip install 'avala[lerobot-core-video]'``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from avala.converters.lerobot_v3 import _layout as L
from avala.converters.lerobot_v3 import _stats
from avala.converters.lerobot_v3._values import NUMERIC_DTYPES, to_hwc_uint8, to_rgb

__all__ = ["EncodedVideoFrame", "LeRobotV3Writer"]

_INSTALL_HINT = "LeRobot v3 writing requires the 'lerobot-core' extra: pip install 'avala[lerobot-core]'"
_VIDEO_HINT = "LeRobot v3 video features require PyAV: pip install 'avala[lerobot-core-video]'"

_PANDAS_TASKS_METADATA = {
    "index_columns": ["task"],
    "column_indexes": [
        {"name": None, "field_name": None, "pandas_type": "unicode", "numpy_type": "object", "metadata": None}
    ],
    "columns": [
        {
            "name": "task_index",
            "field_name": "task_index",
            "pandas_type": "int64",
            "numpy_type": "int64",
            "metadata": None,
        },
        {"name": "task", "field_name": "task", "pandas_type": "unicode", "numpy_type": "object", "metadata": None},
    ],
    "attributes": {},
    "creator": {"library": "avala.converters.lerobot_v3", "version": "1"},
    "pandas_version": "2.0.0",
}


def _pa() -> Any:
    try:
        import pyarrow as pa
        import pyarrow.parquet  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without the extra
        raise ModuleNotFoundError(_INSTALL_HINT) from exc
    return pa


def _np() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without the extra
        raise ModuleNotFoundError(_INSTALL_HINT) from exc
    return np


def _av() -> Any:
    try:
        import av
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(_VIDEO_HINT) from exc
    return av


def _normalize_features(features: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for key, spec in features.items():
        if "/" in key:
            raise ValueError(f"feature names must not contain '/': {key!r}")
        if key in L.DEFAULT_FEATURES or key == L.TASK_KEY:
            raise ValueError(f"{key!r} is reserved by the LeRobot format and is filled in automatically")
        dtype = str(spec["dtype"])
        shape = [int(x) for x in spec["shape"]]
        if dtype in L.VISUAL_DTYPES:
            if len(shape) != 3:
                raise ValueError(f"{key!r}: {dtype} features need a (C, H, W) shape, got {shape}")
        elif dtype == "string":
            if shape != [1]:
                raise ValueError(f"{key!r}: string features must have shape (1,), got {shape}")
        elif dtype in NUMERIC_DTYPES:
            if len(shape) != 1:
                raise NotImplementedError(f"{key!r}: only 1-D numeric features are supported, got shape {shape}")
        else:
            raise ValueError(f"{key!r}: unsupported dtype {dtype!r}")
        normalized: Dict[str, Any] = {"dtype": dtype, "shape": shape, "names": spec.get("names")}
        if "info" in spec:
            normalized["info"] = spec["info"]
        out[key] = normalized
    return out


def _arrow_type(pa: Any, spec: Dict[str, Any]) -> Any:
    dtype = spec["dtype"]
    if dtype == "image":
        return pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    if dtype == "string":
        return pa.string()
    value_type = pa.from_numpy_dtype(_np().dtype(dtype))
    if spec["shape"] == [1]:
        return value_type
    return pa.list_(value_type, spec["shape"][0])


def _hf_feature(spec: Dict[str, Any]) -> Dict[str, Any]:
    """The ``datasets`` feature JSON lerobot embeds in the data parquet (informational)."""
    if spec["dtype"] == "image":
        return {"_type": "Image"}
    if spec["shape"] == [1]:
        return {"dtype": spec["dtype"], "_type": "Value"}
    return {"feature": {"dtype": spec["dtype"], "_type": "Value"}, "length": spec["shape"][0], "_type": "List"}


def _stat_value(value: Any) -> Any:
    return _np().asarray(value).tolist()


@dataclass(frozen=True)
class EncodedVideoFrame:
    """An already-encoded frame to copy into the mp4 without re-encoding.

    ``data`` is one H.264 access unit in Annex-B form (start codes, SPS/PPS in-band on
    keyframes), as carried by ``foxglove.CompressedVideo``; the stream must have no
    B-frames (presentation order == decode order). ``image`` is the same frame decoded
    (HWC uint8): it is used for the shape check and the feature statistics only.

    A pass-through episode is written to its own mp4 file (a valid v3 layout: each episode
    row names its file and ``[from, to)`` window), and its first frame must be a keyframe.
    """

    data: bytes
    keyframe: bool
    image: Any
    codec: str = "h264"


def _probe_video_info(path: Path, fps: int) -> Dict[str, Any]:
    """The ``info`` block lerobot's ``get_video_info`` records, read from the written file."""
    av = _av()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return {
            "video.height": int(stream.codec_context.height),
            "video.width": int(stream.codec_context.width),
            "video.codec": str(stream.codec_context.codec.canonical_name),
            "video.pix_fmt": str(stream.codec_context.pix_fmt or "yuv420p"),
            "video.is_depth_map": False,
            "video.fps": int(fps),
            "video.channels": 3,
            "has_audio": False,
        }


class _PassthroughStream:
    """One mp4 holding exactly one episode of pre-encoded packets (remux, no re-encode)."""

    passthrough = True

    def __init__(self, path: Path, fps: int, width: int, height: int, codec: str) -> None:
        av = _av()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.fps = fps
        self.codec = codec
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream(codec, rate=fps)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = "yuv420p"
        self.frames_written = 0

    def write(self, frame: EncodedVideoFrame) -> None:
        from fractions import Fraction

        av = _av()
        if frame.codec != self.codec:
            raise ValueError(f"pass-through stream is {self.codec}, got a {frame.codec} frame")
        if self.frames_written == 0 and not frame.keyframe:
            raise ValueError("a pass-through episode must start on a keyframe")
        packet = av.Packet(frame.data)
        packet.pts = packet.dts = self.frames_written  # re-timed onto the frame_index / fps grid
        packet.time_base = Fraction(1, self.fps)
        packet.stream = self.stream
        if frame.keyframe:
            packet.is_keyframe = True
        self.container.mux(packet)
        self.frames_written += 1

    def close(self) -> Dict[str, Any]:
        self.container.close()
        return _probe_video_info(self.path, self.fps)

    def size_mb(self) -> float:
        return os.path.getsize(self.path) / (1024**2) if self.path.exists() else 0.0


class _VideoStream:
    """One open ``videos/<key>/chunk-XXX/file-XXX.mp4``; episodes are appended in order."""

    passthrough = False

    def __init__(
        self, path: Path, fps: int, width: int, height: int, vcodec: str, pix_fmt: str, options: Dict[str, str]
    ) -> None:
        av = _av()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.fps = fps
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream(vcodec, rate=fps, options=options)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = pix_fmt
        self.frames_written = 0

    def encode(self, rgb: Any) -> None:
        av = _av()
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts = self.frames_written
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.frames_written += 1

    def close(self) -> Dict[str, Any]:
        for packet in self.stream.encode():
            self.container.mux(packet)
        info = {
            "video.height": int(self.stream.height),
            "video.width": int(self.stream.width),
            "video.codec": str(self.stream.codec_context.codec.canonical_name),
            "video.pix_fmt": str(self.stream.pix_fmt),
            "video.is_depth_map": False,
            "video.fps": int(self.fps),
            "video.channels": 3,
            "has_audio": False,
        }
        self.container.close()
        return info

    def size_mb(self) -> float:
        return os.path.getsize(self.path) / (1024**2) if self.path.exists() else 0.0


class LeRobotV3Writer:
    """Torch-free writer for the LeRobot v3.0 on-disk format.

    Use :meth:`create` (same keywords as ``LeRobotDataset.create``), then per frame
    :meth:`add_frame` (every user feature plus ``"task"``), :meth:`save_episode` after each
    episode, and :meth:`finalize` once at the end — without it ``meta/stats.json`` and
    the episode table are never written and the dataset is not readable.
    """

    def __init__(
        self,
        *,
        repo_id: str,
        fps: int,
        features: Dict[str, Dict[str, Any]],
        root: Path,
        robot_type: Optional[str],
        use_videos: bool,
        vcodec: str,
        pix_fmt: str,
        video_options: Dict[str, str],
        chunks_size: int,
        data_files_size_in_mb: float,
        video_files_size_in_mb: float,
    ) -> None:
        self.repo_id = repo_id
        self.fps = int(fps)
        self.root = root
        self.features = _normalize_features(features)
        self.use_videos = use_videos
        self.vcodec = vcodec
        self.pix_fmt = pix_fmt
        self.video_options = dict(video_options)
        self.video_keys = [k for k, s in self.features.items() if s["dtype"] == "video"]
        self.image_keys = [k for k, s in self.features.items() if s["dtype"] == "image"]
        if self.video_keys and not use_videos:
            raise ValueError(
                f"features contain video keys {self.video_keys} but use_videos is False; "
                "use dtype 'image' or set use_videos=True"
            )
        if self.video_keys:
            _av()  # fail before writing anything
        self.info: Dict[str, Any] = {
            "codebase_version": L.CODEBASE_VERSION,
            "robot_type": robot_type,
            "total_episodes": 0,
            "total_frames": 0,
            "total_tasks": 0,
            "chunks_size": int(chunks_size),
            "data_files_size_in_mb": data_files_size_in_mb,
            "video_files_size_in_mb": video_files_size_in_mb,
            "fps": self.fps,
            "splits": {},
            "data_path": L.DATA_PATH,
            "video_path": L.VIDEO_PATH if use_videos else None,
            "features": {**self.features, **json.loads(json.dumps(L.DEFAULT_FEATURES))},
        }
        self._tasks: Dict[str, int] = {}
        self._episode_rows: List[Dict[str, Any]] = []
        self._episode_stats: List[Dict[str, Any]] = []
        self._buffer: Optional[Dict[str, List[Any]]] = None
        self._stat_frames: Dict[str, List[Any]] = {}
        self._data_chunk = 0
        self._data_file = 0
        self._data_writer: Any = None
        self._data_path: Optional[Path] = None
        self._videos: Dict[str, Any] = {}  # key -> _VideoStream | _PassthroughStream
        self._video_pos: Dict[str, List[int]] = {k: [0, 0] for k in self.video_keys}
        self._video_info: Dict[str, Dict[str, Any]] = {}
        self._finalized = False
        self._write_json(L.INFO_PATH, self.info)

    # ── construction ──
    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: Dict[str, Dict[str, Any]],
        root: Union[str, Path],
        robot_type: Optional[str] = None,
        use_videos: bool = True,
        vcodec: str = "libx264",
        pix_fmt: str = "yuv420p",
        video_options: Optional[Dict[str, str]] = None,
        chunks_size: int = L.DEFAULT_CHUNK_SIZE,
        data_files_size_in_mb: float = L.DEFAULT_DATA_FILE_SIZE_IN_MB,
        video_files_size_in_mb: float = L.DEFAULT_VIDEO_FILE_SIZE_IN_MB,
        **_ignored: Any,
    ) -> "LeRobotV3Writer":
        """Start a new dataset at ``root`` (which must not exist, or be an empty directory).

        Extra keyword arguments that only make sense for the lerobot library (image-writer
        threads, batch encoding) are accepted and ignored so call sites stay portable.
        ``video_options`` defaults to lerobot's random-access settings (``g=2``, ``crf=30``).
        """
        _pa()
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        path = Path(root)
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"{path} already exists and is not empty")
        path.mkdir(parents=True, exist_ok=True)
        return cls(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=path,
            robot_type=robot_type,
            use_videos=use_videos,
            vcodec=vcodec,
            pix_fmt=pix_fmt,
            video_options=video_options if video_options is not None else {"g": "2", "crf": "30"},
            chunks_size=chunks_size,
            data_files_size_in_mb=data_files_size_in_mb,
            video_files_size_in_mb=video_files_size_in_mb,
        )

    # ── helpers ──
    def _write_json(self, rel: str, payload: Any) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=4, ensure_ascii=False)

    @property
    def num_episodes(self) -> int:
        return int(self.info["total_episodes"])

    @property
    def num_frames(self) -> int:
        return int(self.info["total_frames"])

    def _new_buffer(self) -> Dict[str, List[Any]]:
        buf: Dict[str, List[Any]] = {key: [] for key in self.features}
        buf[L.TASK_KEY] = []
        return buf

    def _coerce(self, key: str, value: Any) -> Any:
        np = _np()
        spec = self.features[key]
        dtype = spec["dtype"]
        if dtype in L.VISUAL_DTYPES:
            arr = to_rgb(to_hwc_uint8(value))
            _, height, width = spec["shape"]
            if arr.shape[:2] != (height, width):
                raise ValueError(f"{key!r}: frame is {arr.shape[1]}x{arr.shape[0]}, feature declares {width}x{height}")
            return arr
        if dtype == "string":
            if not isinstance(value, str):
                raise ValueError(f"{key!r}: expected a str, got {type(value).__name__}")
            return value
        arr = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
        arr = np.asarray(arr).reshape(-1)
        if list(arr.shape) != spec["shape"]:
            raise ValueError(f"{key!r}: expected shape {tuple(spec['shape'])}, got {tuple(arr.shape)}")
        return arr.astype(np.dtype(dtype))

    def _stats_subsample(self, arr: Any) -> Any:
        step = max(1, max(arr.shape[0], arr.shape[1]) // 150)
        return arr[::step, ::step]

    # ── recording API ──
    def add_frame(self, frame: Dict[str, Any]) -> None:
        """Buffer one frame. ``frame`` must hold every user feature plus ``"task"``.

        ``timestamp`` / ``frame_index`` must not be passed: like lerobot, they are derived
        as ``frame_index / fps``.
        """
        if self._finalized:
            raise RuntimeError("dataset is finalized; no more frames can be added")
        for forbidden in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            if forbidden in frame:
                raise ValueError(f"{forbidden!r} is computed automatically; do not pass it in add_frame")
        unknown = sorted(set(frame) - set(self.features) - {L.TASK_KEY})
        if unknown:
            raise ValueError(f"frame has keys that are not features: {unknown}")
        missing = sorted(set(self.features) - set(frame))
        if missing:
            raise ValueError(f"frame is missing features: {missing}")
        task = frame.get(L.TASK_KEY)
        if not isinstance(task, str) or not task:
            raise ValueError("frame needs a non-empty 'task' string")

        if self._buffer is None:
            self._buffer = self._new_buffer()
            self._stat_frames = {key: [] for key in (*self.video_keys, *self.image_keys)}
        buf = self._buffer
        for key in self.features:
            raw = frame[key]
            encoded = raw if isinstance(raw, EncodedVideoFrame) else None
            if encoded is not None and self.features[key]["dtype"] != "video":
                raise ValueError(f"{key!r}: an EncodedVideoFrame can only fill a video feature")
            value = self._coerce(key, encoded.image if encoded is not None else raw)
            if self.features[key]["dtype"] == "video":
                if encoded is not None:
                    self._passthrough_video(key, encoded, value)
                else:
                    self._encode_video(key, value)
                buf[key].append(None)
                self._stat_frames[key].append(self._stats_subsample(value))
            elif self.features[key]["dtype"] == "image":
                buf[key].append(value)
                self._stat_frames[key].append(self._stats_subsample(value))
            else:
                buf[key].append(value)
        buf[L.TASK_KEY].append(task)

    def _roll_video(self, key: str) -> None:
        """Close the key's open mp4 (keeping the first file's info) and move to the next file."""
        stream = self._videos.pop(key)
        self._video_info.setdefault(key, stream.close())
        self._video_pos[key] = self._next_file(*self._video_pos[key])

    def _passthrough_video(self, key: str, encoded: EncodedVideoFrame, rgb: Any) -> None:
        stream = self._videos.get(key)
        first_of_episode = len(self._buffer[key]) == 0 if self._buffer is not None else True
        if stream is not None and not stream.passthrough:
            if not first_of_episode:
                raise ValueError(f"{key!r}: cannot mix encoded and pass-through frames in one episode")
            self._roll_video(key)  # the previous file holds earlier, re-encoded episodes
            stream = None
        if stream is None:
            chunk, file = self._video_pos[key]
            path = self.root / L.VIDEO_PATH.format(video_key=key, chunk_index=chunk, file_index=file)
            stream = _PassthroughStream(path, self.fps, int(rgb.shape[1]), int(rgb.shape[0]), encoded.codec)
            self._videos[key] = stream
        assert isinstance(stream, _PassthroughStream)
        stream.write(encoded)

    def _video_stream(self, key: str, rgb: Any) -> Any:
        stream = self._videos.get(key)
        if stream is not None and stream.passthrough:
            raise ValueError(f"{key!r}: cannot mix pass-through and encoded frames in one episode")
        if stream is None:
            chunk, file = self._video_pos[key]
            path = self.root / L.VIDEO_PATH.format(video_key=key, chunk_index=chunk, file_index=file)
            stream = _VideoStream(
                path, self.fps, int(rgb.shape[1]), int(rgb.shape[0]), self.vcodec, self.pix_fmt, self.video_options
            )
            self._videos[key] = stream
        return stream

    def _encode_video(self, key: str, rgb: Any) -> None:
        self._video_stream(key, rgb).encode(rgb)

    def _png(self, rgb: Any) -> bytes:
        from io import BytesIO

        from PIL import Image

        buf = BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        return buf.getvalue()

    def _data_table(self, buf: Dict[str, List[Any]], episode_index: int, length: int, start_index: int) -> Any:
        pa = _pa()
        np = _np()
        task_indices = [self._tasks[t] for t in buf[L.TASK_KEY]]
        fields = []
        arrays = []
        hf: Dict[str, Any] = {}
        for key, spec in self.features.items():
            if spec["dtype"] == "video":
                continue
            arrow_type = _arrow_type(pa, spec)
            values = buf[key]
            if spec["dtype"] == "image":
                column = [{"bytes": self._png(rgb), "path": f"frame-{i:06d}.png"} for i, rgb in enumerate(values)]
                arrays.append(pa.array(column, type=arrow_type))
            elif spec["dtype"] == "string":
                arrays.append(pa.array(values, type=arrow_type))
            elif spec["shape"] == [1]:
                arrays.append(pa.array(np.stack(values).reshape(-1), type=arrow_type))
            else:
                flat = pa.array(np.stack(values).reshape(-1), type=arrow_type.value_type)
                arrays.append(pa.FixedSizeListArray.from_arrays(flat, spec["shape"][0]))
            fields.append(pa.field(key, arrow_type))
            hf[key] = _hf_feature(spec)
        frame_index = np.arange(length, dtype=np.int64)
        bookkeeping = {
            "timestamp": (frame_index / self.fps).astype(np.float32),
            "frame_index": frame_index,
            "episode_index": np.full(length, episode_index, dtype=np.int64),
            "index": np.arange(start_index, start_index + length, dtype=np.int64),
            "task_index": np.asarray(task_indices, dtype=np.int64),
        }
        for key, values in bookkeeping.items():
            spec = L.DEFAULT_FEATURES[key]
            arrays.append(pa.array(values, type=pa.from_numpy_dtype(np.dtype(str(spec["dtype"])))))
            fields.append(pa.field(key, arrays[-1].type))
            hf[key] = {"dtype": spec["dtype"], "_type": "Value"}
        schema = pa.schema(fields, metadata={"huggingface": json.dumps({"info": {"features": hf}})})
        return pa.Table.from_arrays(arrays, schema=schema), bookkeeping

    def _write_data(self, table: Any) -> Dict[str, int]:
        import pyarrow.parquet as pq

        if self._data_writer is not None and self._data_path is not None:
            if os.path.getsize(self._data_path) / (1024**2) >= float(self.info["data_files_size_in_mb"]):
                self._data_writer.close()
                self._data_writer = None
                self._data_chunk, self._data_file = self._next_file(self._data_chunk, self._data_file)
        if self._data_writer is None:
            path = self.root / L.DATA_PATH.format(chunk_index=self._data_chunk, file_index=self._data_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._data_path = path
            self._data_writer = pq.ParquetWriter(str(path), table.schema)
        self._data_writer.write_table(table)
        return {"data/chunk_index": self._data_chunk, "data/file_index": self._data_file}

    def _next_file(self, chunk: int, file: int) -> List[int]:
        if file == int(self.info["chunks_size"]) - 1:
            return [chunk + 1, 0]
        return [chunk, file + 1]

    def save_episode(self) -> None:
        """Write the buffered frames as the next episode."""
        if self._buffer is None or not self._buffer[L.TASK_KEY]:
            raise ValueError("no frames were added to the current episode")
        buf = self._buffer
        length = len(buf[L.TASK_KEY])
        episode_index = self.num_episodes
        start_index = self.num_frames

        episode_tasks: List[str] = []
        for task in buf[L.TASK_KEY]:
            if task not in self._tasks:
                self._tasks[task] = len(self._tasks)
            if task not in episode_tasks:
                episode_tasks.append(task)

        table, bookkeeping = self._data_table(buf, episode_index, length, start_index)
        row: Dict[str, Any] = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": length,
            **self._write_data(table),
            "dataset_from_index": start_index,
            "dataset_to_index": start_index + length,
        }

        for key in self.video_keys:
            stream = self._videos[key]
            start_frame = stream.frames_written - length
            row[f"videos/{key}/chunk_index"] = self._video_pos[key][0]
            row[f"videos/{key}/file_index"] = self._video_pos[key][1]
            row[f"videos/{key}/from_timestamp"] = start_frame / self.fps
            row[f"videos/{key}/to_timestamp"] = stream.frames_written / self.fps

        stats: Dict[str, Any] = {}
        for key, spec in self.features.items():
            if spec["dtype"] == "string":
                continue
            if spec["dtype"] in L.VISUAL_DTYPES:
                stats[key] = _stats.image_stats(self._stat_frames[key])
            else:
                stats[key] = _stats.numeric_stats(_np().stack(buf[key]))
        for key, values in bookkeeping.items():
            stats[key] = _stats.numeric_stats(values)
        for feature, feature_stats in stats.items():
            for name, value in feature_stats.items():
                row[f"stats/{feature}/{name}"] = _stat_value(value)
        row["meta/episodes/chunk_index"] = 0
        row["meta/episodes/file_index"] = 0

        self._episode_rows.append(row)
        self._episode_stats.append(stats)
        self.info["total_episodes"] += 1
        self.info["total_frames"] += length
        self.info["total_tasks"] = len(self._tasks)
        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        self._write_tasks()
        self._write_json(L.INFO_PATH, self.info)
        self._buffer = None
        self._stat_frames = {}

        # Roll video files at episode boundaries once they pass the size limit; a
        # pass-through file always holds exactly one episode.
        for key in self.video_keys:
            stream = self._videos[key]
            if stream.passthrough or stream.size_mb() >= float(self.info["video_files_size_in_mb"]):
                self._roll_video(key)

    def clear_episode_buffer(self) -> None:
        """Drop the frames of the current, unsaved episode (image features only — video
        frames are already encoded and cannot be withdrawn)."""
        if self.video_keys and self._buffer is not None:
            raise RuntimeError("cannot discard a partially encoded video episode")
        self._buffer = None
        self._stat_frames = {}

    def _write_tasks(self) -> None:
        pa = _pa()
        import pyarrow.parquet as pq

        ordered = sorted(self._tasks.items(), key=lambda item: item[1])
        schema = pa.schema(
            [pa.field("task_index", pa.int64()), pa.field("task", pa.large_string())],
            metadata={"pandas": json.dumps(_PANDAS_TASKS_METADATA)},
        )
        table = pa.Table.from_arrays(
            [
                pa.array([i for _, i in ordered], type=pa.int64()),
                pa.array([t for t, _ in ordered], type=pa.large_string()),
            ],
            schema=schema,
        )
        path = self.root / L.TASKS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, str(path))

    def _write_episodes(self) -> None:
        pa = _pa()
        import pyarrow.parquet as pq

        columns: Dict[str, List[Any]] = {}
        for row in self._episode_rows:
            for key in row:
                columns.setdefault(key, [])
        for key in columns:
            columns[key] = [row.get(key) for row in self._episode_rows]
        arrays = []
        for key, values in columns.items():
            if key == "tasks":
                arrays.append(pa.array(values, type=pa.list_(pa.string())))
            elif key.endswith("/from_timestamp") or key.endswith("/to_timestamp"):
                arrays.append(pa.array(values, type=pa.float64()))
            elif not key.startswith("stats/"):
                arrays.append(pa.array(values, type=pa.int64()))
            else:
                arrays.append(pa.array(values))
        table = pa.Table.from_arrays(arrays, names=list(columns))
        path = self.root / L.EPISODES_PATH.format(chunk_index=0, file_index=0)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, str(path))

    def finalize(self) -> None:
        """Close files and write the episode table, stats and final ``info.json``. Idempotent."""
        if self._finalized:
            return
        if self._buffer is not None and self._buffer[L.TASK_KEY]:
            # Same as lerobot: an unsaved episode is not part of the dataset. (Its video frames,
            # if any, stay as an unreferenced tail of the mp4 and are never read.)
            warnings.warn(
                f"discarding {len(self._buffer[L.TASK_KEY])} unsaved frame(s); call save_episode() first to keep them",
                stacklevel=2,
            )
            self._buffer = None
            self._stat_frames = {}
        for key in list(self._videos):
            self._video_info.setdefault(key, self._videos.pop(key).close())
        if self._data_writer is not None:
            self._data_writer.close()
            self._data_writer = None
        for key, video_info in self._video_info.items():
            self.info["features"][key]["info"] = video_info
        if self._episode_rows:
            self._write_episodes()
            self._write_json(L.STATS_PATH, _stats.to_jsonable(_stats.aggregate(self._episode_stats)))
        self._write_json(L.INFO_PATH, self.info)
        self._finalized = True

    def abort(self) -> None:
        """Best-effort close without writing metadata; removes the root if nothing was saved."""
        for stream in self._videos.values():
            try:
                stream.container.close()
            except Exception:  # pragma: no cover - best effort
                pass
        self._videos.clear()
        if self._data_writer is not None:
            self._data_writer.close()
            self._data_writer = None
        if not self._episode_rows:
            shutil.rmtree(self.root, ignore_errors=True)

    def push_to_hub(self, *args: Any, **kwargs: Any) -> None:
        """Not implemented by the torch-free writer.

        Upload the finalized directory with ``huggingface-cli upload <repo_id> <root>
        --repo-type dataset`` and tag it ``v3.0`` (lerobot resolves datasets by that tag),
        or install the ``lerobot`` extra to use its ``push_to_hub``.
        """
        raise NotImplementedError(
            "push_to_hub needs the lerobot library (pip install 'avala[lerobot]'). With the torch-free "
            f"writer, upload {self.root} with `huggingface-cli upload {self.repo_id} {self.root} "
            "--repo-type dataset` and tag the revision 'v3.0'."
        )
