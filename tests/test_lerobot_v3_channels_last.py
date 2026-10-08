"""Channels-last image features: LeRobot v3 -> Avala MCAP -> LeRobot v3.

Hub datasets declare camera features channels-LAST. ``lerobot/pusht`` at revision
``7628202a2180972f291ba1bc6723834921e72c19`` (``meta/info.json`` sha256
``0b46becc21ad92b117da1e95daf44110bb21d0bde75638b4ad167c8ccd959d38``) has::

    "observation.image": {"dtype": "video", "shape": [96, 96, 3],
                          "names": ["height", "width", "channel"]}

and lerobot 0.5.1 writes its own camera features the same way
(``lerobot/datasets/feature_utils.py:129-134``, ``hw_to_dataset_features``). The export
used to read every image shape as ``(C, H, W)`` and failed with "frame is 96x96, feature
declares 3x96". Contract under test: the declared ``shape`` and ``names`` survive the
round trip exactly, and the pixels are not transposed.

Source: ``fixtures/lerobot_v3/ref_dataset_hwc``, written by the REAL lerobot 0.5.1
library (``generate_hwc_fixture.py``) with pusht's feature set, so the oracle is not our
own writer.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

for _mod in ("numpy", "pyarrow", "PIL", "av", "mcap.reader", "mcap_protobuf.writer", "foxglove_schemas_protobuf"):
    pytest.importorskip(_mod)

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from avala import Client  # noqa: E402
from avala.converters.lerobot_v3 import _layout as L  # noqa: E402
from avala.converters.lerobot_v3.mcap import mcap_to_lerobot  # noqa: E402
from avala.converters.lerobot_v3.reader import LeRobotV3Dataset  # noqa: E402
from avala.converters.lerobot_v3.writer import LeRobotV3Writer  # noqa: E402
from avala.importers import import_lerobot  # noqa: E402
from avala.types.dataset import Dataset  # noqa: E402

REF = Path(__file__).parent / "fixtures" / "lerobot_v3" / "ref_dataset_hwc"
CAM = "observation.image"
PIXEL_TOLERANCE = 6.0  # mean absolute error per channel, 0-255 (two lossy encodes)

# Copied verbatim from lerobot/pusht@7628202a meta/info.json (sha256 above).
PUSHT_IMAGE = {"dtype": "video", "shape": [96, 96, 3], "names": ["height", "width", "channel"]}
PUSHT_MOTORS = {"motors": ["motor_0", "motor_1"]}


@pytest.fixture(scope="module")
def roundtrip(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    try:
        tmp = tmp_path_factory.mktemp("hwc")
        dest = tmp / "mcap"
        client = Client(api_key="test-key")

        def _capture(*, source, name, slug, data_type, **_kwargs):
            shutil.copytree(source, dest)
            return Dataset(uid="d", name=name, slug=slug, data_type=data_type, item_count=0)

        mp.setattr(client.datasets, "create_from_local", _capture)
        import_lerobot(client, root=str(REF), name="pusht", slug="pusht", backend="core")
        client.close()
        mcaps = sorted(dest.glob("*.mcap"))
        out = mcap_to_lerobot(mcaps, tmp / "lerobot", repo_id="avala/pusht-roundtrip")
    finally:
        mp.undo()
    return mcaps, out


def test_fixture_declares_images_like_pusht():
    features = json.loads((REF / "meta/info.json").read_text())["features"]
    assert {k: features[CAM][k] for k in ("dtype", "shape", "names")} == PUSHT_IMAGE
    assert features["observation.state"]["names"] == PUSHT_MOTORS
    assert features["action"]["names"] == PUSHT_MOTORS


def test_roundtrip_preserves_declared_shape_names_and_values(roundtrip):
    mcaps, out = roundtrip
    assert len(mcaps) == 2
    src_info = json.loads((REF / "meta/info.json").read_text())
    dst_info = json.loads((out / "meta/info.json").read_text())
    assert list(src_info["features"]) == list(dst_info["features"])
    for name, spec in src_info["features"].items():
        for attr in ("dtype", "shape", "names"):
            assert spec.get(attr) == dst_info["features"][name].get(attr), f"{name}.{attr}"
    assert dst_info["features"][CAM]["info"]["video.height"] == 96
    assert dst_info["features"][CAM]["info"]["video.width"] == 96

    src_rows = pq.read_table(REF / "data/chunk-000/file-000.parquet").to_pylist()
    dst_rows = pq.read_table(out / "data/chunk-000/file-000.parquet").to_pylist()
    assert src_rows == dst_rows


def test_roundtrip_pixels_are_not_transposed(roundtrip):
    _mcaps, out = roundtrip
    with LeRobotV3Dataset(REF) as src, LeRobotV3Dataset(out) as dst:
        for ep in range(src.total_episodes):
            for a, b in zip(src.iter_frames(ep, keys=[CAM]), dst.iter_frames(ep, keys=[CAM])):
                assert a[CAM].shape == b[CAM].shape == (96, 96, 3)
                assert float(np.abs(a[CAM].astype(float) - b[CAM].astype(float)).mean()) <= PIXEL_TOLERANCE
                img = b[CAM].astype(float)
                # red follows the row and green the column (generate_hwc_fixture.py)
                assert img[-8:, :, 0].mean() - img[:8, :, 0].mean() > 200
                assert img[:, -8:, 1].mean() - img[:, :8, 1].mean() > 200


@pytest.mark.skipif(
    importlib.util.find_spec("lerobot") is None,
    reason="lerobot library not installed (needs Python 3.12 + avala[lerobot]); cross-check runs where it is",
)
def test_roundtrip_output_loads_in_lerobot(roundtrip):
    from lerobot.datasets import LeRobotDataset

    _mcaps, out = roundtrip
    ds = LeRobotDataset("avala/pusht-roundtrip", root=out, video_backend="pyav")
    assert len(ds) == 7
    assert ds.meta.features[CAM]["shape"] == (96, 96, 3)
    assert tuple(ds[0][CAM].shape) == (3, 96, 96)  # lerobot's torch tensors are CHW at load


def _image_only_mcap(path: Path, height: int, width: int, frames: int = 3) -> None:
    from io import BytesIO

    from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
    from mcap_protobuf.writer import Writer
    from PIL import Image

    with open(path, "wb") as fh:
        writer = Writer(fh)
        for i in range(frames):
            img = np.zeros((height, width, 3), dtype=np.uint8)
            img[:, :, 0] = 40 * i
            buf = BytesIO()
            Image.fromarray(img).save(buf, format="PNG")
            t = i * 100_000_000
            msg = CompressedImage(data=buf.getvalue(), format="png", frame_id="cam")
            writer.write_message("/camera/front/image", msg, log_time=t, publish_time=t)
        writer.finish()


def test_inferred_camera_feature_is_declared_channels_last(tmp_path):
    """An MCAP without our metadata record gets lerobot's own camera layout."""
    src = tmp_path / "ep.mcap"
    _image_only_mcap(src, height=24, width=40)
    out = mcap_to_lerobot([src], tmp_path / "out", repo_id="a/b", fps=10, use_videos=False)
    spec = json.loads((out / "meta/info.json").read_text())["features"]["observation.images.front"]
    assert spec["shape"] == [24, 40, 3]
    assert spec["names"] == ["height", "width", "channels"]
    with LeRobotV3Dataset(out) as ds:
        frame = next(iter(ds.iter_frames(0, keys=["observation.images.front"])))
    assert frame["observation.images.front"].shape == (24, 40, 3)


@pytest.mark.parametrize(
    "shape,names,axis",
    [
        ([96, 96, 3], ["height", "width", "channel"], 2),
        ([96, 96, 3], ["height", "width", "channels"], 2),
        ([3, 64, 64], ["channels", "height", "width"], 0),
        ([3, 4, 5], None, 0),
        ([4, 5, 3], None, 2),
        ([3, 4, 4], None, 0),
        ([3, 3, 3], ["height", "width", "channel"], 2),
        ([3, 3, 3], ["channel", "height", "width"], 0),
        ([96, 96, 3], {"axes": ["height", "width", "channel"]}, 2),
    ],
)
def test_channel_axis(shape, names, axis):
    assert L.image_channel_axis({"shape": shape, "names": names}) == axis


@pytest.mark.parametrize("shape", [[3, 96, 3], [4, 8, 4], [8, 8, 8]])
def test_ambiguous_unnamed_image_shape_is_refused(shape, tmp_path):
    with pytest.raises(ValueError, match="channel axis"):
        L.image_channel_axis({"shape": shape, "names": None})
    with pytest.raises(ValueError, match="channel axis"):
        LeRobotV3Writer.create(
            repo_id="a/b",
            fps=5,
            features={"observation.images.cam": {"dtype": "image", "shape": shape, "names": None}},
            root=tmp_path / "d",
        )


def test_writer_accepts_channels_last_frames_with_matching_height_and_width(tmp_path):
    features = {
        "observation.images.cam": {"dtype": "image", "shape": [4, 6, 3], "names": ["height", "width", "channel"]}
    }
    writer = LeRobotV3Writer.create(repo_id="a/b", fps=5, features=features, root=tmp_path / "d")
    writer.add_frame({"observation.images.cam": np.zeros((4, 6, 3), dtype=np.uint8), "task": "t"})
    with pytest.raises(ValueError, match="frame is 4x6, feature declares 6x4"):
        writer.add_frame({"observation.images.cam": np.zeros((6, 4, 3), dtype=np.uint8), "task": "t"})
