"""Read a LeRobot v3.0 dataset from disk with pyarrow (+ PyAV for video), no torch.

Only ``codebase_version`` ``v3.x`` is accepted (v2.1 uses a different layout; lerobot
itself refuses to load it without conversion). See ``_layout`` for where each path and
column name comes from.

Frames come back as plain Python/numpy values, keyed by feature name:

* numeric features: a 1-D numpy array of the declared dtype and shape (scalars are
  shape ``(1,)``, as declared in ``info.json``);
* ``string`` features: ``str``;
* ``image`` / ``video`` features: an HWC ``uint8`` RGB numpy array (``decode_visual=True``);
* bookkeeping columns (``timestamp``, ``frame_index``, ``episode_index``, ``index``,
  ``task_index``) as Python scalars, plus ``"task"`` resolved through ``meta/tasks.parquet``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from avala.converters.lerobot_v3 import _layout as L

__all__ = ["LeRobotV3Dataset", "UnsupportedLeRobotVersion"]

_INSTALL_HINT = "LeRobot v3 reading requires the 'lerobot-core' extra: pip install 'avala[lerobot-core]'"
_VIDEO_HINT = "Decoding LeRobot v3 video features requires PyAV: pip install 'avala[lerobot-core-video]'"


class UnsupportedLeRobotVersion(ValueError):
    """The dataset is not in the LeRobot v3 layout."""


def _pq() -> Any:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without the extra
        raise ModuleNotFoundError(_INSTALL_HINT) from exc
    return pq


class _VideoCursor:
    """Sequential decoder over one mp4 that serves frames by presentation time.

    Requests must move forward in time; a request earlier than the last served frame
    reopens the file. Each request returns the decoded frame nearest to the requested
    time and fails if none lies within half a frame period.
    """

    def __init__(self, path: Path, fps: float) -> None:
        try:
            import av
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(_VIDEO_HINT) from exc
        self._av = av
        self.path = path
        self.tolerance = 0.5 / fps
        self._open()

    def _open(self) -> None:
        self._container = self._av.open(str(self.path))
        stream = self._container.streams.video[0]
        self._frames = self._container.decode(stream)
        self._prev: Optional[Tuple[float, Any]] = None
        self._next: Optional[Tuple[float, Any]] = None

    def _pull(self) -> Optional[Tuple[float, Any]]:
        try:
            frame = next(self._frames)
        except StopIteration:
            return None
        t = float(frame.time) if frame.time is not None else float(frame.pts * frame.time_base)
        return (t, frame)

    def get(self, t: float) -> Any:
        if self._prev is not None and t < self._prev[0] - self.tolerance:
            self.close()
            self._open()
        if self._next is None:
            self._next = self._pull()
        while self._next is not None and self._next[0] < t:
            self._prev = self._next
            self._next = self._pull()
        candidates = [c for c in (self._prev, self._next) if c is not None]
        if not candidates:
            raise ValueError(f"{self.path}: no decodable frames")
        best = min(candidates, key=lambda c: abs(c[0] - t))
        if abs(best[0] - t) > self.tolerance:
            raise ValueError(
                f"{self.path}: no frame within {self.tolerance:.4f}s of t={t:.4f}s (nearest {best[0]:.4f}s)"
            )
        return best[1].to_ndarray(format="rgb24")

    def close(self) -> None:
        self._container.close()


class LeRobotV3Dataset:
    """A LeRobot v3 dataset directory opened for reading."""

    def __init__(self, root: Union[str, Path]) -> None:
        self.root = Path(root)
        info_path = self.root / L.INFO_PATH
        if not info_path.is_file():
            raise FileNotFoundError(f"{info_path} not found; is {self.root} a LeRobot dataset?")
        with info_path.open(encoding="utf-8") as fh:
            self.info: Dict[str, Any] = json.load(fh)
        version = str(self.info.get("codebase_version", ""))
        if not version.startswith("v3."):
            raise UnsupportedLeRobotVersion(
                f"{self.root} is LeRobot {version or '<unknown version>'}; only the v3.x layout is supported "
                "(convert with `python -m lerobot.scripts.convert_dataset_v21_to_v30`)"
            )
        self.features: Dict[str, Dict[str, Any]] = self.info["features"]
        self.fps: float = float(self.info["fps"])
        self.robot_type: Optional[str] = self.info.get("robot_type")
        self.tasks: Dict[int, str] = self._load_tasks()
        self.episodes: List[Dict[str, Any]] = self._load_episodes()
        self._tables: Dict[Tuple[int, int], Any] = {}
        self._cursors: Dict[Path, _VideoCursor] = {}

    def close(self) -> None:
        """Release open video decoders."""
        for cursor in self._cursors.values():
            cursor.close()
        self._cursors.clear()

    def __enter__(self) -> "LeRobotV3Dataset":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ── metadata ──
    @property
    def total_episodes(self) -> int:
        return len(self.episodes)

    @property
    def camera_keys(self) -> List[str]:
        return [k for k, s in self.features.items() if s["dtype"] in L.VISUAL_DTYPES]

    @property
    def video_keys(self) -> List[str]:
        return [k for k, s in self.features.items() if s["dtype"] == "video"]

    @property
    def user_features(self) -> Dict[str, Dict[str, Any]]:
        """Features excluding the five bookkeeping columns."""
        return {k: s for k, s in self.features.items() if k not in L.BOOKKEEPING_KEYS}

    def _load_tasks(self) -> Dict[int, str]:
        path = self.root / L.TASKS_PATH
        if not path.is_file():
            return {}
        table = _pq().read_table(str(path))
        names = table.column_names
        index_col = "task_index" if "task_index" in names else None
        text_cols = [n for n in names if n != index_col]
        if index_col is None or not text_cols:
            raise ValueError(f"{path}: expected task_index + task columns, found {names}")
        text_col = "task" if "task" in text_cols else text_cols[0]
        return {int(i): str(t) for i, t in zip(table.column(index_col).to_pylist(), table.column(text_col).to_pylist())}

    def _load_episodes(self) -> List[Dict[str, Any]]:
        pq = _pq()
        paths = sorted((self.root / L.EPISODES_DIR).glob("*/*.parquet"))
        rows: List[Dict[str, Any]] = []
        for path in paths:
            table = pq.read_table(str(path))
            keep = [n for n in table.column_names if not n.startswith("stats/")]
            rows.extend(table.select(keep).to_pylist())
        rows.sort(key=lambda r: int(r["episode_index"]))
        return rows

    def episode(self, episode_index: int) -> Dict[str, Any]:
        for row in self.episodes:
            if int(row["episode_index"]) == episode_index:
                return row
        raise KeyError(f"episode {episode_index} not found (have {self.total_episodes})")

    def stats(self) -> Optional[Dict[str, Any]]:
        path = self.root / L.STATS_PATH
        if not path.is_file():
            return None
        with path.open(encoding="utf-8") as fh:
            stats: Dict[str, Any] = json.load(fh)
        return stats

    # ── frames ──
    def _data_table(self, chunk: int, file: int) -> Any:
        key = (chunk, file)
        if key not in self._tables:
            path = self.root / str(self.info.get("data_path") or L.DATA_PATH).format(chunk_index=chunk, file_index=file)
            self._tables = {key: _pq().read_table(str(path))}  # keep one file resident
        return self._tables[key]

    def _episode_rows(self, ep: Dict[str, Any]) -> List[Dict[str, Any]]:
        import pyarrow.compute as pc

        table = self._data_table(int(ep["data/chunk_index"]), int(ep["data/file_index"]))
        mask = pc.equal(table.column("episode_index"), int(ep["episode_index"]))
        rows: List[Dict[str, Any]] = table.filter(mask).to_pylist()
        rows.sort(key=lambda r: int(r["frame_index"]))
        expected = int(ep["length"])
        if len(rows) != expected:
            raise ValueError(f"episode {ep['episode_index']}: data has {len(rows)} rows, episode table says {expected}")
        return rows

    def _coerce(self, key: str, value: Any) -> Any:
        import numpy as np

        spec = self.features[key]
        dtype = spec["dtype"]
        if dtype == "string":
            return value
        if dtype == "image":
            from io import BytesIO

            from PIL import Image

            payload = value.get("bytes") if isinstance(value, dict) else value
            if payload is None:
                raise ValueError(f"{key!r}: image struct has no embedded bytes")
            return np.asarray(Image.open(BytesIO(payload)).convert("RGB"))
        arr = np.asarray(value if isinstance(value, list) else [value], dtype=np.dtype(dtype))
        return arr.reshape(tuple(int(x) for x in spec["shape"]))

    def iter_frames(
        self,
        episode_index: int,
        *,
        keys: Optional[Sequence[str]] = None,
        decode_visual: bool = True,
    ) -> Iterator[Dict[str, Any]]:
        """Yield the frames of one episode in order.

        ``keys`` restricts the user features returned (bookkeeping columns and ``task``
        are always included). ``decode_visual=False`` skips image/video decoding.
        """
        ep = self.episode(episode_index)
        wanted = list(self.user_features) if keys is None else list(keys)
        unknown = [k for k in wanted if k not in self.features]
        if unknown:
            raise KeyError(f"unknown features {unknown}; available: {sorted(self.user_features)}")
        cursors: Dict[str, _VideoCursor] = {}  # per key, shared across episodes via self._cursors
        video_from: Dict[str, float] = {}
        if decode_visual:
            for key in wanted:
                if self.features[key]["dtype"] == "video":
                    path = self.root / str(self.info["video_path"]).format(
                        video_key=key,
                        chunk_index=int(ep[f"videos/{key}/chunk_index"]),
                        file_index=int(ep[f"videos/{key}/file_index"]),
                    )
                    if path not in self._cursors:
                        self._cursors[path] = _VideoCursor(path, self.fps)
                    cursors[key] = self._cursors[path]
                    video_from[key] = float(ep[f"videos/{key}/from_timestamp"])
        for row in self._episode_rows(ep):
            frame: Dict[str, Any] = {k: row[k] for k in L.BOOKKEEPING_KEYS if k in row}
            frame[L.TASK_KEY] = self.tasks.get(int(row["task_index"]), "")
            for key in wanted:
                dtype = self.features[key]["dtype"]
                if dtype == "video":
                    if decode_visual:
                        frame[key] = cursors[key].get(video_from[key] + float(row["timestamp"]))
                    continue
                if dtype == "image" and not decode_visual:
                    continue
                frame[key] = self._coerce(key, row[key])
            yield frame
