"""Import a ROS bag (ROS1 ``.bag`` or ROS2 ``.db3``) into Avala as an MCAP dataset.

Each bag becomes one ``.mcap`` (= one Avala MCAP episode). **Camera topics**
(``sensor_msgs/Image`` and ``sensor_msgs/CompressedImage``, ROS1 or ROS2) are
re-encoded as ``foxglove.CompressedImage`` (protobuf) so they render in the Mission
Control MCAP viewer.

Scope (this increment): camera streams only. Non-image topics (point clouds, TF,
joint states, …) are **not** carried over yet — the MC viewer renders protobuf image
channels, and faithfully copying ROS-encoded messages through (preserving their
schemas) is a planned follow-up. ``import_ros_bag`` reports how many topics it skipped.

Reading uses the pure-Python ``rosbags`` library (no ROS install). Install the extra
with ``pip install 'avala[rosbag]'``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Set, Tuple

from avala.importers import register_importer

if TYPE_CHECKING:
    from avala._client import Client
    from avala.types.dataset import Dataset

__all__ = ["import_ros_bag", "write_bag_mcap"]

# ROS1 (``pkg/Type``) and ROS2 (``pkg/msg/Type``) image message names.
_RAW_IMAGE_TYPES = frozenset({"sensor_msgs/Image", "sensor_msgs/msg/Image"})
_COMPRESSED_IMAGE_TYPES = frozenset({"sensor_msgs/CompressedImage", "sensor_msgs/msg/CompressedImage"})
_IMAGE_TYPES = _RAW_IMAGE_TYPES | _COMPRESSED_IMAGE_TYPES


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
) -> Tuple[int, Set[str]]:
    """Convert the camera topics of a ROS bag to a foxglove ``.mcap``.

    Returns ``(images_written, skipped_topics)``. ``image_topics`` restricts the
    conversion to specific topics (default: all image topics).
    """
    from pathlib import Path

    from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
    from mcap_protobuf.writer import Writer
    from rosbags.highlevel import AnyReader

    wanted = set(image_topics) if image_topics is not None else None
    written = 0
    skipped: Set[str] = set()

    with AnyReader([Path(bag_path)]) as reader, open(out_path, "wb") as fh, Writer(fh) as writer:
        for conn, timestamp, raw in reader.messages():
            if conn.msgtype not in _IMAGE_TYPES:
                skipped.add(conn.topic)
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

    return written, skipped


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
) -> "Dataset":
    """Import a ROS bag (``.bag`` / ``.db3``) into Avala as an MCAP dataset.

    Camera topics are re-encoded as ``foxglove.CompressedImage`` and written to a single
    ``.mcap`` that is uploaded with ``data_type="mcap"``. Non-image topics are skipped
    (a warning lists them). ``image_topics`` restricts the conversion to specific topics.
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
        written, skipped = write_bag_mcap(out_path, bag, image_topics=image_topics)
        if written == 0:
            detail = f" Skipped topics (non-image or unsupported): {sorted(skipped)}." if skipped else ""
            raise ValueError(
                "no camera images found in the bag; this importer carries camera topics "
                "(sensor_msgs/Image, sensor_msgs/CompressedImage) only." + detail
            )
        if skipped:
            warnings.warn(
                f"skipped {len(skipped)} topic(s) not carried over (non-image, or unsupported "
                f"image format such as compressedDepth): {sorted(skipped)}",
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
