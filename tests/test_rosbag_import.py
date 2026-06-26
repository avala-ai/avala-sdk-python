from __future__ import annotations

from io import BytesIO

import pytest

# The ROS-bag importer depends on the optional ``avala[rosbag]`` extra
# (rosbags / mcap / mcap-protobuf-support / foxglove-schemas-protobuf / pillow / numpy).
pytest.importorskip("rosbags.highlevel")
pytest.importorskip("rosbags.rosbag2")
pytest.importorskip("mcap.reader")
pytest.importorskip("mcap_protobuf.writer")
pytest.importorskip("foxglove_schemas_protobuf")
pytest.importorskip("PIL")

import httpx  # noqa: E402
import numpy as np  # noqa: E402
import respx  # noqa: E402
from avala import Client  # noqa: E402
from avala.importers import available_importers, import_ros_bag  # noqa: E402
from avala.importers.rosbag import _raw_image_to_jpeg, write_bag_mcap  # noqa: E402
from PIL import Image  # noqa: E402
from rosbags.rosbag2 import Writer  # noqa: E402
from rosbags.typesys import Stores, get_typestore  # noqa: E402

BASE_URL = "https://api.avala.ai/api/v1"
PRESIGN_URL = f"{BASE_URL}/datasets/manual-upload/file-upload-url/"
FINALIZE_URL = f"{BASE_URL}/datasets/manual-upload/"
S3_URL = "https://s3.example.com/upload"


def _jpeg_bytes(color=(10, 20, 30)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="JPEG")
    return buf.getvalue()


def _make_bag(
    bag_dir,
    *,
    compressed=True,
    raw=False,
    string=True,
    depth=False,
    bad_raw=False,
    stamp_sec=1,
    bag_time_ns=1_000_000_000,
):
    """Write a small ROS2 bag with the requested topics. Returns the bag directory path."""
    ts = get_typestore(Stores.ROS2_HUMBLE)
    Header = ts.types["std_msgs/msg/Header"]
    Time = ts.types["builtin_interfaces/msg/Time"]

    # rosbags >=0.10 made Writer's `version` kwarg required (older versions don't
    # accept it); pass it only when present so this works across versions.
    import inspect

    wkw = {}
    if "version" in inspect.signature(Writer.__init__).parameters:
        wkw["version"] = getattr(Writer, "VERSION_LATEST", 9)
    writer = Writer(bag_dir, **wkw)
    writer.open()
    try:
        if compressed:
            CompressedImage = ts.types["sensor_msgs/msg/CompressedImage"]
            msg = CompressedImage(
                header=Header(stamp=Time(sec=stamp_sec, nanosec=0), frame_id="cam"),
                format="jpeg",
                data=np.frombuffer(_jpeg_bytes(), dtype=np.uint8),
            )
            conn = writer.add_connection("/camera/compressed", "sensor_msgs/msg/CompressedImage", typestore=ts)
            writer.write(conn, bag_time_ns, bytes(ts.serialize_cdr(msg, "sensor_msgs/msg/CompressedImage")))
        if raw:
            ImageMsg = ts.types["sensor_msgs/msg/Image"]
            h, w = 4, 6
            pixels = np.zeros((h, w, 3), dtype=np.uint8).reshape(-1)
            msg = ImageMsg(
                header=Header(stamp=Time(sec=2, nanosec=0), frame_id="raw_cam"),
                height=h,
                width=w,
                encoding="rgb8",
                is_bigendian=0,
                step=w * 3,
                data=pixels,
            )
            conn = writer.add_connection("/camera/raw", "sensor_msgs/msg/Image", typestore=ts)
            writer.write(conn, 2_000_000_000, bytes(ts.serialize_cdr(msg, "sensor_msgs/msg/Image")))
        if depth:
            CompressedImage = ts.types["sensor_msgs/msg/CompressedImage"]
            msg = CompressedImage(
                header=Header(stamp=Time(sec=3, nanosec=0), frame_id="depth"),
                format="16UC1; compressedDepth png",
                data=np.frombuffer(b"\x00" * 32, dtype=np.uint8),
            )
            conn = writer.add_connection(
                "/camera/depth/compressedDepth", "sensor_msgs/msg/CompressedImage", typestore=ts
            )
            writer.write(conn, 3_000_000_000, bytes(ts.serialize_cdr(msg, "sensor_msgs/msg/CompressedImage")))
        if bad_raw:
            ImageMsg = ts.types["sensor_msgs/msg/Image"]
            h, w = 2, 2
            msg = ImageMsg(
                header=Header(stamp=Time(sec=4, nanosec=0), frame_id="depth"),
                height=h,
                width=w,
                encoding="32FC1",  # unsupported by the importer
                is_bigendian=0,
                step=w * 4,
                data=np.zeros(h * w * 4, dtype=np.uint8),
            )
            conn = writer.add_connection("/camera/depth32", "sensor_msgs/msg/Image", typestore=ts)
            writer.write(conn, 4_000_000_000, bytes(ts.serialize_cdr(msg, "sensor_msgs/msg/Image")))
        if string:
            String = ts.types["std_msgs/msg/String"]
            conn = writer.add_connection("/chatter", "std_msgs/msg/String", typestore=ts)
            writer.write(conn, 1_500_000_000, bytes(ts.serialize_cdr(String(data="hi"), "std_msgs/msg/String")))
    finally:
        writer.close()
    return str(bag_dir)


def _read_summary(mcap_path):
    from mcap.reader import make_reader

    with open(mcap_path, "rb") as fh:
        return make_reader(fh).get_summary()


# ── registry ──
def test_rosbag_registered():
    assert "rosbag" in available_importers()


# ── write_bag_mcap ──
def test_write_bag_mcap_converts_compressed_and_skips_non_image(tmp_path):
    bag = _make_bag(tmp_path / "bag", compressed=True, string=True)
    out = tmp_path / "out.mcap"
    written, skipped = write_bag_mcap(str(out), bag)

    assert written == 1
    assert skipped == {"/chatter"}  # the String topic is reported, not carried

    summary = _read_summary(out)
    assert summary is not None
    assert "foxglove.CompressedImage" in {s.name for s in summary.schemas.values()}
    chan = next(c for c in summary.channels.values() if c.topic == "/camera/compressed")
    assert chan.message_encoding == "protobuf"
    assert summary.statistics.message_count == 1


def test_write_bag_mcap_encodes_raw_image(tmp_path):
    bag = _make_bag(tmp_path / "bag", compressed=False, raw=True, string=False)
    out = tmp_path / "out.mcap"
    written, skipped = write_bag_mcap(str(out), bag)
    assert written == 1 and skipped == set()
    assert "foxglove.CompressedImage" in {s.name for s in _read_summary(out).schemas.values()}


def test_write_bag_mcap_skips_compressed_depth(tmp_path):
    # compressedDepth payloads are not plain images -> reported, not emitted as bad PNG
    bag = _make_bag(tmp_path / "bag", compressed=False, string=False, depth=True)
    out = tmp_path / "out.mcap"
    written, skipped = write_bag_mcap(str(out), bag)
    assert written == 0
    assert "/camera/depth/compressedDepth" in skipped


def test_write_bag_mcap_skips_unsupported_raw_encoding(tmp_path):
    # a depth image (32FC1) must be skipped, not abort the whole import
    bag = _make_bag(tmp_path / "bag", compressed=True, string=False, bad_raw=True)
    out = tmp_path / "out.mcap"
    written, skipped = write_bag_mcap(str(out), bag)
    assert written == 1  # the valid CompressedImage still imported
    assert "/camera/depth32" in skipped


def test_write_bag_mcap_topic_filter(tmp_path):
    bag = _make_bag(tmp_path / "bag", compressed=True, raw=True, string=False)
    out = tmp_path / "out.mcap"
    written, _ = write_bag_mcap(str(out), bag, image_topics=["/camera/compressed"])
    assert written == 1  # only the selected topic converted


def test_write_bag_mcap_uses_header_stamp_for_timestamp(tmp_path):
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    # header stamp (5s) differs from the bag record time (9s); foxglove ts must use header
    bag = _make_bag(tmp_path / "bag", compressed=True, string=False, stamp_sec=5, bag_time_ns=9_000_000_000)
    out = tmp_path / "out.mcap"
    write_bag_mcap(str(out), bag)

    with open(out, "rb") as fh:
        decoded = list(make_reader(fh, decoder_factories=[DecoderFactory()]).iter_decoded_messages())
    assert len(decoded) == 1
    img = decoded[0].decoded_message
    assert img.timestamp.seconds == 5  # acquisition time from header.stamp, not bag time (9)


# ── raw image decoding (unit) ──
class _FakeImage:
    def __init__(self, height, width, encoding, data, step=None, is_bigendian=0):
        bpp = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1, "mono16": 2}[encoding]
        self.height, self.width, self.encoding = height, width, encoding
        self.data = data
        self.step = step if step is not None else width * bpp
        self.is_bigendian = is_bigendian


def _jpeg_size(jpeg: bytes):
    return Image.open(BytesIO(jpeg)).size  # (width, height)


def test_raw_image_honors_row_padding():
    # rgb8 2x2 with 2 bytes of per-row padding (step = 8 > width*3 = 6)
    row = bytes([10, 20, 30, 40, 50, 60, 0, 0])  # 2 pixels + 2 pad bytes
    img = _FakeImage(2, 2, "rgb8", row * 2, step=8)
    assert _jpeg_size(_raw_image_to_jpeg(img)) == (2, 2)


def test_raw_image_mono16_endianness_runs():
    data = bytes([0x01, 0x00, 0x02, 0x00])  # two uint16 values
    big = _FakeImage(1, 2, "mono16", data, is_bigendian=1)
    little = _FakeImage(1, 2, "mono16", data, is_bigendian=0)
    assert _jpeg_size(_raw_image_to_jpeg(big)) == (2, 1)
    assert _jpeg_size(_raw_image_to_jpeg(little)) == (2, 1)


def test_raw_image_unsupported_encoding_errors():
    bad = type(
        "Img", (), {"encoding": "yuv422", "height": 1, "width": 1, "data": b"\x00", "step": 1, "is_bigendian": 0}
    )()
    with pytest.raises(ValueError, match="unsupported raw image encoding"):
        _raw_image_to_jpeg(bad)


# ── import_ros_bag end-to-end (respx upload) ──
def _wire_upload(dataset_json):
    respx.post(PRESIGN_URL).mock(
        return_value=httpx.Response(200, json={"url": S3_URL, "fields": {"Content-Type": "application/octet-stream"}})
    )
    s3 = respx.post(S3_URL).mock(return_value=httpx.Response(204))
    respx.post(FINALIZE_URL).mock(return_value=httpx.Response(201, json=dataset_json))
    return s3


@respx.mock
def test_import_ros_bag_end_to_end(tmp_path):
    bag = _make_bag(tmp_path / "bag", compressed=True, string=True)
    s3 = _wire_upload({"uid": "d1", "name": "Bag", "slug": "bag", "data_type": "mcap", "item_count": 1})

    client = Client(api_key="test-key")
    with pytest.warns(UserWarning, match="skipped 1 topic"):
        ds = import_ros_bag(client, bag=bag, name="Bag", slug="bag")
    client.close()

    assert ds.uid == "d1"
    assert ds.data_type == "mcap"
    assert s3.call_count == 1


def test_import_ros_bag_errors_without_images(tmp_path):
    bag = _make_bag(tmp_path / "bag", compressed=False, string=True)
    client = Client(api_key="test-key")
    with pytest.raises(ValueError, match="no camera images"):
        import_ros_bag(client, bag=bag, name="x", slug="x")
    client.close()
