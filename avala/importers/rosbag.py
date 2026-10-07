"""Import a ROS bag (ROS1 ``.bag`` or ROS2 ``.db3``) into Avala as an MCAP dataset.

Each bag becomes one ``.mcap`` (= one Avala MCAP episode). **Camera topics**
(``sensor_msgs/Image`` and ``sensor_msgs/CompressedImage``, ROS1 or ROS2) are
re-encoded as ``foxglove.CompressedImage`` (protobuf) so they render in the Mission
Control MCAP viewer.

**Numeric and text topics** — ``sensor_msgs/JointState``, ``std_msgs/Float64``,
``*MultiArray``, ``sensor_msgs/Imu``, ``geometry_msgs/*``, ``std_msgs/String`` and any
other message made only of numbers, booleans, strings, small numeric arrays and nested
messages of those — are carried on their original topic as a ``google.protobuf.Struct``
mirroring the message's fields (``header``, ``name``, ``position``, ``velocity``,
``effort`` for a JointState). Values are copied, never synthesised; ``Struct`` numbers
are doubles, so integers above 2**53 lose precision.

Everything else is **skipped and reported**: point clouds and other messages carrying a
binary blob (a ``uint8``/``int8`` array of more than 256 bytes), messages with more than
4096 numeric values, and depth/unsupported images. ``carry_non_image=False`` restores
the camera-only behaviour.

Reading uses the pure-Python ``rosbags`` library (no ROS install). Install the extra
with ``pip install 'avala[rosbag]'``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Set, Tuple

from avala.importers import register_importer

if TYPE_CHECKING:
    from avala._client import Client
    from avala.types.dataset import Dataset

__all__ = ["convert_bag", "import_ros_bag", "write_bag_mcap"]

# ROS1 (``pkg/Type``) and ROS2 (``pkg/msg/Type``) image message names.
_RAW_IMAGE_TYPES = frozenset({"sensor_msgs/Image", "sensor_msgs/msg/Image"})
_COMPRESSED_IMAGE_TYPES = frozenset({"sensor_msgs/CompressedImage", "sensor_msgs/msg/CompressedImage"})
_IMAGE_TYPES = _RAW_IMAGE_TYPES | _COMPRESSED_IMAGE_TYPES


# Guards for carrying arbitrary messages as Struct: a byte array this long is a binary blob
# (point cloud, serialized payload), not a numeric signal; beyond this many numbers per
# message a Struct is the wrong container.
_MAX_BYTE_ARRAY = 256
_MAX_NUMBERS_PER_MESSAGE = 4096


class _NotCarried(ValueError):
    """The message cannot be represented faithfully as a Struct."""


def _msg_to_struct(msg: Any) -> Any:
    """Convert a deserialized ROS message to Struct-compatible Python values (dict tree).

    Raises :class:`_NotCarried` for binary blobs and oversized messages.
    """
    budget = [_MAX_NUMBERS_PER_MESSAGE]

    def spend(n: int) -> None:
        budget[0] -= n
        if budget[0] < 0:
            raise _NotCarried(f"more than {_MAX_NUMBERS_PER_MESSAGE} numeric values")

    def convert(value: Any) -> Any:
        if isinstance(value, (bool, str)) or value is None:
            return value
        if isinstance(value, (int, float)):
            spend(1)
            return value
        if isinstance(value, (bytes, bytearray)):
            raise _NotCarried("binary payload")
        if hasattr(value, "dtype") and hasattr(value, "tolist"):  # numpy array / scalar
            kind = value.dtype.kind
            if kind not in "biuf":
                raise _NotCarried(f"array of dtype {value.dtype}")
            size = int(getattr(value, "size", 1))
            if value.dtype.itemsize == 1 and kind in "iu" and size > _MAX_BYTE_ARRAY:
                raise _NotCarried("binary blob")
            spend(size)
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        fields = getattr(value, "__dataclass_fields__", None)
        if fields is not None:
            return {name: convert(getattr(value, name)) for name in fields if not name.startswith("__")}
        raise _NotCarried(f"unsupported field type {type(value).__name__}")

    out = convert(msg)
    if not isinstance(out, dict):
        raise _NotCarried("not a message")
    return out


_BYTES_PER_PIXEL = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1, "mono16": 2}


def _raw_image_to_jpeg(msg: Any) -> bytes:
    """Encode a ``sensor_msgs/Image`` (raw) message as JPEG bytes.

    Honors ``msg.step`` (rows may be padded so ``step > width * bytes_per_pixel``) and
    ``msg.is_bigendian`` (for ``mono16``).
    """
    from io import BytesIO

    import numpy as np
    from PIL import Image

    encoding = str(msg.encoding).lower()
    bpp = _BYTES_PER_PIXEL.get(encoding)
    if bpp is None:
        raise ValueError(f"unsupported raw image encoding {msg.encoding!r}; supported: {', '.join(_BYTES_PER_PIXEL)}")

    height, width = int(msg.height), int(msg.width)
    row_bytes = width * bpp
    flat = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    step = int(getattr(msg, "step", 0)) or row_bytes
    nrows = min(height, flat.size // step) if step else height
    # Drop any per-row padding: keep the first row_bytes of each step-sized row.
    rows = np.ascontiguousarray(flat[: step * nrows].reshape(nrows, step)[:, :row_bytes])

    if encoding in ("rgb8", "bgr8"):
        arr = rows.reshape(nrows, width, 3)
        if encoding == "bgr8":
            arr = arr[:, :, ::-1]
        pil = Image.fromarray(np.ascontiguousarray(arr), "RGB")
    elif encoding in ("rgba8", "bgra8"):
        arr = rows.reshape(nrows, width, 4)
        if encoding == "bgra8":
            arr = arr[:, :, [2, 1, 0, 3]]
        pil = Image.fromarray(np.ascontiguousarray(arr), "RGBA").convert("RGB")
    elif encoding == "mono8":
        pil = Image.fromarray(rows.reshape(nrows, width), "L")
    else:  # mono16
        dtype = np.dtype(">u2") if getattr(msg, "is_bigendian", 0) else np.dtype("<u2")
        arr16 = np.frombuffer(rows.tobytes(), dtype=dtype).reshape(nrows, width)
        pil = Image.fromarray((arr16 // 256).astype(np.uint8), "L")

    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _frame_id(msg: Any, fallback: str) -> str:
    header = getattr(msg, "header", None)
    fid = getattr(header, "frame_id", "") if header is not None else ""
    return str(fid) if fid else fallback


def _header_stamp_ns(msg: Any, fallback_ns: int) -> int:
    """Return the message header acquisition time in ns, or ``fallback_ns`` if absent/zero."""
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None) if header is not None else None
    if stamp is not None:
        sec = getattr(stamp, "sec", None)
        nsec = getattr(stamp, "nanosec", None)
        if sec is not None and nsec is not None and (sec or nsec):
            return int(sec) * 1_000_000_000 + int(nsec)
    return fallback_ns


def write_bag_mcap(
    out_path: str,
    bag_path: str,
    *,
    image_topics: Optional[Sequence[str]] = None,
    carry_non_image: bool = True,
) -> Tuple[int, Set[str]]:
    """Convert a ROS bag to a foxglove ``.mcap``.

    Returns ``(images_written, skipped_topics)``. ``image_topics`` restricts the camera
    conversion to specific topics (default: all image topics). Numeric/text topics are
    carried as ``Struct`` (see the module docstring) unless ``carry_non_image=False``;
    :func:`convert_bag` also returns how many of those messages were written.
    """
    images, _structs, skipped = convert_bag(
        out_path, bag_path, image_topics=image_topics, carry_non_image=carry_non_image
    )
    return images, skipped


def convert_bag(
    out_path: str,
    bag_path: str,
    *,
    image_topics: Optional[Sequence[str]] = None,
    carry_non_image: bool = True,
) -> Tuple[int, int, Set[str]]:
    """Like :func:`write_bag_mcap`; returns ``(images_written, structs_written, skipped_topics)``."""
    from pathlib import Path

    from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
    from google.protobuf.struct_pb2 import Struct
    from mcap_protobuf.writer import Writer
    from rosbags.highlevel import AnyReader

    wanted = set(image_topics) if image_topics is not None else None
    written = 0
    structs = 0
    skipped: Set[str] = set()

    with AnyReader([Path(bag_path)]) as reader, open(out_path, "wb") as fh, Writer(fh) as writer:
        for conn, timestamp, raw in reader.messages():
            if conn.msgtype not in _IMAGE_TYPES:
                if not carry_non_image or conn.topic in skipped:
                    skipped.add(conn.topic)
                    continue
                try:
                    payload_dict = _msg_to_struct(reader.deserialize(raw, conn.msgtype))
                except _NotCarried:
                    skipped.add(conn.topic)
                    continue
                struct = Struct()
                struct.update(payload_dict)
                writer.write_message(
                    topic=conn.topic, message=struct, log_time=int(timestamp), publish_time=int(timestamp)
                )
                structs += 1
                continue
            if wanted is not None and conn.topic not in wanted:
                continue

            msg = reader.deserialize(raw, conn.msgtype)
            if conn.msgtype in _COMPRESSED_IMAGE_TYPES:
                fmt_raw = str(msg.format).lower()
                # compressedDepth payloads carry a depth transport ConfigHeader before the
                # PNG/RVL stream — the bytes are NOT a plain image. Skip rather than emit a
                # CompressedImage with invalid data.
                if "compresseddepth" in fmt_raw:
                    skipped.add(conn.topic)
                    continue
                fmt = "png" if "png" in fmt_raw else "jpeg"
                payload = bytes(msg.data)
            else:
                fmt = "jpeg"
                try:
                    payload = _raw_image_to_jpeg(msg)
                except ValueError:
                    # Unsupported raw encoding (e.g. depth 16UC1/32FC1) — skip this topic
                    # rather than aborting the whole import; other camera streams still go through.
                    skipped.add(conn.topic)
                    continue

            out_msg = CompressedImage()
            # foxglove timestamp = image acquisition time (header.stamp); MCAP
            # log_time/publish_time keep the bag record time below.
            out_msg.timestamp.FromNanoseconds(_header_stamp_ns(msg, int(timestamp)))
            out_msg.frame_id = _frame_id(msg, conn.topic.strip("/").replace("/", "."))
            out_msg.format = fmt
            out_msg.data = payload
            writer.write_message(
                topic=conn.topic, message=out_msg, log_time=int(timestamp), publish_time=int(timestamp)
            )
            written += 1

    return written, structs, skipped


def import_ros_bag(
    client: "Client",
    *,
    bag: str,
    name: str,
    slug: str,
    image_topics: Optional[Sequence[str]] = None,
    visibility: str = "private",
    owner_name: Optional[str] = None,
    industry: Optional[int] = None,
    license: Optional[int] = None,
    workers: int = 8,
    on_progress: "Optional[Callable[[str, int], None]]" = None,
    wait: bool = False,
    wait_timeout: float = 3600.0,
    carry_non_image: bool = True,
) -> "Dataset":
    """Import a ROS bag (``.bag`` / ``.db3``) into Avala as an MCAP dataset.

    Camera topics are re-encoded as ``foxglove.CompressedImage``; numeric/text topics
    (joint states, IMU, scalars, strings, ...) are carried as ``Struct`` unless
    ``carry_non_image=False``. Everything goes into a single ``.mcap`` uploaded with
    ``data_type="mcap"``; topics that could not be carried are listed in a warning.
    ``image_topics`` restricts the camera conversion to specific topics.
    """
    import os
    import tempfile
    import warnings

    try:
        import rosbags.highlevel  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without the extra
        raise ModuleNotFoundError(
            "ROS bag import requires the 'rosbag' extra. Install it with: pip install 'avala[rosbag]'"
        ) from exc

    with tempfile.TemporaryDirectory(prefix="avala-rosbag-") as tmp:
        # Fixed filename — never derive the local path from the user-facing slug (which
        # could contain '/', '..', or an absolute path and escape the temp dir).
        out_path = os.path.join(tmp, "data.mcap")
        written, structs, skipped = convert_bag(
            out_path, bag, image_topics=image_topics, carry_non_image=carry_non_image
        )
        if written == 0 and structs == 0:
            detail = f" Skipped topics (unsupported or binary): {sorted(skipped)}." if skipped else ""
            raise ValueError(
                "no camera images or numeric topics found in the bag; this importer carries camera "
                "topics (sensor_msgs/Image, sensor_msgs/CompressedImage) and numeric/text messages." + detail
            )
        if skipped:
            warnings.warn(
                f"skipped {len(skipped)} topic(s) not carried over (binary payloads such as point clouds, "
                f"oversized messages, or unsupported image formats such as compressedDepth): {sorted(skipped)}",
                stacklevel=2,
            )
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


register_importer("rosbag", import_ros_bag)
