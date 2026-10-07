"""Per-episode and aggregate feature statistics in LeRobot v3's shape.

Mirrors ``lerobot/datasets/compute_stats.py`` (lerobot 0.5.1):

* numeric features reduce over the frame axis — shape ``(D,)`` for a ``(D,)`` vector and
  ``(1,)`` for a scalar column;
* image/video features are per-channel over every pixel, normalised to ``[0, 1]``,
  shape ``(3, 1, 1)``;
* ``string`` features have no statistics;
* aggregation across episodes uses lerobot's ``aggregate_feature_stats`` formula
  (count-weighted mean, parallel variance, count-weighted quantiles).

One deliberate difference: lerobot estimates quantiles from a 5000-bin histogram, we
compute them exactly with ``numpy.quantile``. The keys and shapes are identical; the
quantile values agree to within lerobot's histogram error.
"""

from __future__ import annotations

from typing import Any, Dict, List

from avala.converters.lerobot_v3._layout import QUANTILES

Stats = Dict[str, Any]


def _np() -> Any:
    import numpy as np

    return np


def numeric_stats(values: Any) -> Stats:
    """Stats for a stacked numeric column: ``(N,)`` scalars or ``(N, D)`` vectors."""
    np = _np()
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    as_float = arr.astype(np.float64)
    out: Stats = {
        "min": arr.min(axis=0),
        "max": arr.max(axis=0),
        "mean": as_float.mean(axis=0),
        "std": as_float.std(axis=0),
        "count": np.array([arr.shape[0]], dtype=np.int64),
    }
    for key, q in QUANTILES:
        out[key] = np.quantile(as_float, q, axis=0)
    return out


def image_stats(frames: List[Any]) -> Stats:
    """Per-channel stats over a list of HWC uint8 frames, in ``[0, 1]``, shape ``(3, 1, 1)``."""
    np = _np()
    stacked = np.stack([np.asarray(f) for f in frames]).astype(np.float64) / 255.0  # N,H,W,C
    if stacked.ndim == 3:  # grayscale N,H,W -> N,H,W,1
        stacked = stacked[..., None]
    per_channel = stacked.transpose(3, 0, 1, 2).reshape(stacked.shape[3], -1)  # C, N*H*W
    out: Stats = {
        "min": per_channel.min(axis=1).reshape(-1, 1, 1),
        "max": per_channel.max(axis=1).reshape(-1, 1, 1),
        "mean": per_channel.mean(axis=1).reshape(-1, 1, 1),
        "std": per_channel.std(axis=1).reshape(-1, 1, 1),
        "count": np.array([len(frames)], dtype=np.int64),
    }
    for key, q in QUANTILES:
        out[key] = np.quantile(per_channel, q, axis=1).reshape(-1, 1, 1)
    return out


def aggregate(stats_list: List[Dict[str, Stats]]) -> Dict[str, Stats]:
    """Aggregate per-episode stats dicts (``{feature: stats}``) into dataset stats."""
    np = _np()
    keys = sorted({key for stats in stats_list for key in stats})
    result: Dict[str, Stats] = {}
    for key in keys:
        items = [stats[key] for stats in stats_list if key in stats]
        means = np.stack([s["mean"] for s in items])
        variances = np.stack([np.asarray(s["std"]) ** 2 for s in items])
        counts = np.stack([s["count"] for s in items])
        total = counts.sum(axis=0)
        weights = counts.astype(np.float64)
        while weights.ndim < means.ndim:
            weights = np.expand_dims(weights, axis=-1)
        mean = (means * weights).sum(axis=0) / total
        variance = ((variances + (means - mean) ** 2) * weights).sum(axis=0) / total
        agg: Stats = {
            "min": np.min(np.stack([s["min"] for s in items]), axis=0),
            "max": np.max(np.stack([s["max"] for s in items]), axis=0),
            "mean": mean,
            "std": np.sqrt(variance),
            "count": total,
        }
        for qkey, _ in QUANTILES:
            if all(qkey in s for s in items):
                agg[qkey] = (np.stack([s[qkey] for s in items]) * weights).sum(axis=0) / total
        result[key] = agg
    return result


def to_jsonable(stats: Dict[str, Stats]) -> Dict[str, Dict[str, Any]]:
    """Convert numpy stats to plain lists for ``meta/stats.json``."""
    np = _np()
    out: Dict[str, Dict[str, Any]] = {}
    for key, feature_stats in stats.items():
        out[key] = {name: np.asarray(value).tolist() for name, value in feature_stats.items()}
    return out
