"""The LeRobot v3.0 on-disk layout, as constants.

Every value here was read from the ``lerobot`` 0.5.1 wheel (Apache-2.0) and checked
against a dataset that library wrote (``tests/fixtures/lerobot_v3/ref_dataset``):

* ``lerobot/datasets/utils.py`` — ``INFO_PATH``, ``STATS_PATH``, ``EPISODES_DIR``,
  ``CHUNK_FILE_PATTERN``, ``DEFAULT_TASKS_PATH``, ``DEFAULT_EPISODES_PATH``,
  ``DEFAULT_DATA_PATH``, ``DEFAULT_VIDEO_PATH``, ``DEFAULT_CHUNK_SIZE``,
  ``DEFAULT_DATA_FILE_SIZE_IN_MB``, ``DEFAULT_VIDEO_FILE_SIZE_IN_MB``,
  ``DEFAULT_FEATURES`` (the five bookkeeping columns).
* ``lerobot/datasets/dataset_metadata.py`` — ``CODEBASE_VERSION = "v3.0"`` and the
  episode-row keys ``dataset_from_index`` / ``dataset_to_index`` /
  ``meta/episodes/chunk_index`` / ``meta/episodes/file_index``.
* ``lerobot/datasets/dataset_writer.py`` — ``data/chunk_index``, ``data/file_index``
  and ``videos/<key>/{chunk_index,file_index,from_timestamp,to_timestamp}``; per-frame
  ``timestamp = frame_index / fps``.
* ``lerobot/datasets/feature_utils.py`` — ``create_empty_dataset_info`` (``info.json``
  keys) and ``get_hf_features_from_features`` (parquet column types).
* ``lerobot/datasets/compute_stats.py`` — ``DEFAULT_QUANTILES`` and the stats keys.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

CODEBASE_VERSION = "v3.0"

INFO_PATH = "meta/info.json"
STATS_PATH = "meta/stats.json"
TASKS_PATH = "meta/tasks.parquet"
EPISODES_DIR = "meta/episodes"
CHUNK_FILE_PATTERN = "chunk-{chunk_index:03d}/file-{file_index:03d}"
EPISODES_PATH = EPISODES_DIR + "/" + CHUNK_FILE_PATTERN + ".parquet"
DATA_PATH = "data/" + CHUNK_FILE_PATTERN + ".parquet"
VIDEO_PATH = "videos/{video_key}/" + CHUNK_FILE_PATTERN + ".mp4"

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_DATA_FILE_SIZE_IN_MB = 100
DEFAULT_VIDEO_FILE_SIZE_IN_MB = 200

# Added to every dataset by ``LeRobotDatasetMetadata.create``; ``info.json`` lists them
# after the user features, in this order.
DEFAULT_FEATURES: Dict[str, Dict[str, object]] = {
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}
BOOKKEEPING_KEYS: Tuple[str, ...] = tuple(DEFAULT_FEATURES)

# ``task`` is not a column: it is resolved through ``task_index`` -> meta/tasks.parquet.
TASK_KEY = "task"

QUANTILES: List[Tuple[str, float]] = [("q01", 0.01), ("q10", 0.10), ("q50", 0.50), ("q90", 0.90), ("q99", 0.99)]

VISUAL_DTYPES = frozenset({"image", "video"})


def format_path(template: str, **kwargs: object) -> str:
    """Fill one of the path templates above (thin wrapper so callers read uniformly)."""
    return template.format(**kwargs)


# Image/video features declare a 3-D ``shape`` whose channel axis is NOT fixed by the
# format. lerobot 0.5.1 accepts both layouts (``feature_utils.py:479-500``,
# ``validate_feature_image_or_video`` takes ``(c, h, w)`` or ``(h, w, c)``) and itself
# writes channels-LAST for camera features (``feature_utils.py:129-134``,
# ``hw_to_dataset_features``: ``"names": ["height", "width", "channels"]``). Hub datasets
# such as ``lerobot/pusht`` declare ``[96, 96, 3]`` with ``["height", "width", "channel"]``.
# So the channel axis is read from ``names``; the shape itself is never rewritten.
_CHANNEL_AXIS_NAMES = frozenset({"channel", "channels", "c", "rgb", "color", "colour"})
_CHANNEL_SIZES = frozenset({1, 3, 4})
# What a newly inferred camera feature declares: lerobot's own convention (see above).
INFERRED_IMAGE_NAMES: Tuple[str, str, str] = ("height", "width", "channels")


def image_channel_axis(spec: Mapping[str, Any]) -> int:
    """Index of the channel axis in an image/video feature's 3-D ``shape``.

    Taken from ``names`` when it names exactly one channel axis; otherwise the single
    outer axis (0 or 2) of size 3 — or, failing that, of size 1 or 4. Ambiguous shapes raise instead of
    guessing, because a wrong guess silently swaps height, width and channels.
    """
    shape = [int(x) for x in spec["shape"]]
    if len(shape) != 3:
        raise ValueError(f"image/video features need a 3-D shape, got {shape}")
    names = spec.get("names")
    if isinstance(names, (list, tuple)) and len(names) == 3:
        hits = [i for i, n in enumerate(names) if str(n).strip().lower() in _CHANNEL_AXIS_NAMES]
        if len(hits) == 1:
            return hits[0]
    # RGB (3) first: frames are always written as RGB, so [3, 4, 4] is CHW, not RGBA HWC.
    for sizes in ({3}, _CHANNEL_SIZES):
        candidates = [i for i in (0, 2) if shape[i] in sizes]
        if len(candidates) == 1:
            return candidates[0]
        if candidates:
            break
    raise ValueError(
        f"cannot tell the channel axis of image shape {shape} (names={names!r}); "
        "declare names such as ['height', 'width', 'channels'] or ['channels', 'height', 'width']"
    )


def image_hw(spec: Mapping[str, Any]) -> Tuple[int, int]:
    """``(height, width)`` of an image/video feature, whichever axis holds the channels."""
    axis = image_channel_axis(spec)
    height, width = [int(x) for i, x in enumerate(spec["shape"]) if i != axis]
    return height, width
