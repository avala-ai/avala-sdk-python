"""Value coercion shared by the reader, writer and MCAP bridge."""

from __future__ import annotations

from typing import Any, Sequence

NUMERIC_DTYPES = frozenset(
    {
        "bool",
        "float16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }
)


def to_hwc_uint8(value: Any) -> Any:
    """Normalise an image (CHW or HWC, float ``[0, 1]`` or uint8, numpy or torch) to HWC uint8."""
    import numpy as np

    arr = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
    # CHW -> HWC: a leading axis of 1/3/4 smaller than the trailing axis is a channel dim
    # (lerobot returns torch CHW tensors; PIL/our reader return HWC).
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[2]:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        peak = float(arr.max()) if arr.size else 0.0
        # float [0,1] -> [0,255]; otherwise assume [0,255]. Clip so values cannot wrap.
        arr = np.clip(arr, 0.0, 1.0) * 255.0 if peak <= 1.0 else np.clip(arr, 0.0, 255.0)
        arr = arr.round().astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return arr


def to_rgb(arr: Any) -> Any:
    """HWC uint8 (gray, RGB or RGBA) -> HWC uint8 RGB."""
    import numpy as np

    if arr.ndim == 2:
        return np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[2] == 4:
        return np.ascontiguousarray(arr[:, :, :3])
    return arr


def as_python_scalar(value: Any) -> Any:
    """Unwrap a 0-d / 1-element tensor or array into a Python scalar; pass strings through."""
    if isinstance(value, (str, bytes)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, RuntimeError):
            pass
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return as_python_scalar(value[0])
    return value


def flat_numbers(value: Any) -> Sequence[Any]:
    """Flatten a tensor/array/scalar of numbers to a flat Python list (bools stay bools)."""
    import numpy as np

    arr = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
    return [x for x in np.asarray(arr).reshape(-1).tolist()]
