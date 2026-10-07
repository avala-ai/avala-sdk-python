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

from typing import Dict, List, Tuple

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
