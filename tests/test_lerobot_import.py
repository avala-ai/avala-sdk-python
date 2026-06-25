from __future__ import annotations

import sys
import types

import pytest

# The MCAP writer + LeRobot adapter depend on the optional ``avala[lerobot]`` extra
# (mcap / mcap-protobuf-support / foxglove-schemas-protobuf / pillow / numpy). Skip the
# whole module on the default ``[dev,cli]`` install so CI doesn't fail to import.
pytest.importorskip("numpy")
pytest.importorskip("mcap.reader")
pytest.importorskip("mcap_protobuf.writer")
pytest.importorskip("foxglove_schemas_protobuf")
pytest.importorskip("PIL")

import httpx  # noqa: E402
import numpy as np  # noqa: E402
import respx  # noqa: E402
from avala import Client  # noqa: E402
from avala.importers import available_importers, import_dataset, import_lerobot  # noqa: E402
from avala.importers.lerobot import _to_hwc_uint8, _to_list, write_episode_mcap  # noqa: E402

BASE_URL = "https://api.avala.ai/api/v1"
PRESIGN_URL = f"{BASE_URL}/datasets/manual-upload/file-upload-url/"
FINALIZE_URL = f"{BASE_URL}/datasets/manual-upload/"
S3_URL = "https://s3.example.com/upload"


# ── pure MCAP writer ──
def test_write_episode_mcap_roundtrip(tmp_path):
    from mcap.reader import make_reader

    img = np.zeros((8, 8, 3), dtype=np.uint8)
    frames = [
        {
            "timestamp_ns": 0,
            "images": {"/observation/images/cam": img},
            "structs": {"/observation/state": {"data": [0.1, 0.2]}, "/action": {"data": [1.0]}},
        },
        {
            "timestamp_ns": 100_000_000,
            "images": {"/observation/images/cam": img},
            "structs": {"/observation/state": {"data": [0.3, 0.4]}, "/action": {"data": [2.0]}},
        },
    ]
    out = tmp_path / "episode_000000.mcap"
    count = write_episode_mcap(str(out), frames)
    assert count == 2

    with open(out, "rb") as fh:
        reader = make_reader(fh)
        summary = reader.get_summary()

    assert summary is not None  # server parser requires a summary section
    schema_names = {s.name for s in summary.schemas.values()}
    assert "foxglove.CompressedImage" in schema_names  # renders in the MC viewer
    assert "google.protobuf.Struct" in schema_names  # proprioception preserved
    topics = {c.topic for c in summary.channels.values()}
    assert topics == {"/observation/images/cam", "/observation/state", "/action"}
    # 2 frames x (1 image + 2 structs) = 6 messages
    assert summary.statistics.message_count == 6
    # image channel uses protobuf encoding (required for the viewer's decoder)
    image_channel = next(c for c in summary.channels.values() if c.topic == "/observation/images/cam")
    assert image_channel.message_encoding == "protobuf"


# ── tensor normalization helpers ──
def test_to_hwc_uint8_from_chw_float():
    chw = np.full((3, 4, 5), 0.5, dtype=np.float32)  # CHW float [0,1]
    out = _to_hwc_uint8(chw)
    assert out.shape == (4, 5, 3)
    assert out.dtype == np.uint8
    assert int(out[0, 0, 0]) == 128


def test_to_hwc_uint8_passthrough_hwc_uint8():
    hwc = np.zeros((4, 5, 3), dtype=np.uint8)
    out = _to_hwc_uint8(hwc)
    assert out.shape == (4, 5, 3) and out.dtype == np.uint8


def test_to_hwc_uint8_grayscale_squeeze():
    chw = np.zeros((1, 4, 5), dtype=np.uint8)
    out = _to_hwc_uint8(chw)
    assert out.shape == (4, 5)  # single channel squeezed


def test_to_list_flattens():
    assert _to_list(np.array([[1.0, 2.0], [3.0, 4.0]])) == [1.0, 2.0, 3.0, 4.0]


# ── registry ──
def test_lerobot_registered():
    assert "lerobot" in available_importers()


def test_import_lerobot_requires_source():
    with pytest.raises(ValueError, match=r"repo_id.*root"):
        import_lerobot(Client(api_key="k"), name="x", slug="x")


# ── full flow with a fake lerobot library injected ──
class _FakeMetadata:
    """Stand-in for lerobot.datasets.lerobot_dataset.LeRobotDatasetMetadata."""

    def __init__(self, repo_id, root=None):
        self.repo_id = repo_id
        self.root = root
        self.camera_keys = ["observation.images.cam"]
        self.fps = 10
        self.total_episodes = 1
        self.features = {"observation.state": 1, "action": 1, "observation.images.cam": 1}


class _FakeDataset:
    def __init__(self, repo_id, root=None, episodes=None):
        self.repo_id = repo_id
        self.root = root
        self.episodes = episodes  # records the subset that was requested

    def __len__(self):
        return 3  # 3 frames, all in episode 0

    def __getitem__(self, i):
        return {
            "observation.images.cam": np.zeros((3, 8, 8), dtype=np.uint8),  # CHW uint8
            "observation.state": np.array([0.1, 0.2, 0.3]),
            "action": np.array([1.0, 2.0]),
            "timestamp": float(i) / 10.0,
            "episode_index": 0,
        }


@pytest.fixture
def fake_lerobot(monkeypatch):
    pkg = types.ModuleType("lerobot")
    datasets = types.ModuleType("lerobot.datasets")
    mod = types.ModuleType("lerobot.datasets.lerobot_dataset")
    mod.LeRobotDataset = _FakeDataset
    mod.LeRobotDatasetMetadata = _FakeMetadata
    datasets.lerobot_dataset = mod
    pkg.datasets = datasets
    monkeypatch.setitem(sys.modules, "lerobot", pkg)
    monkeypatch.setitem(sys.modules, "lerobot.datasets", datasets)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", mod)
    yield


def _wire_upload(dataset_json):
    respx.post(PRESIGN_URL).mock(
        return_value=httpx.Response(200, json={"url": S3_URL, "fields": {"Content-Type": "application/octet-stream"}})
    )
    s3 = respx.post(S3_URL).mock(return_value=httpx.Response(204))
    respx.post(FINALIZE_URL).mock(return_value=httpx.Response(201, json=dataset_json))
    return s3


@respx.mock
def test_import_lerobot_end_to_end(fake_lerobot):
    s3 = _wire_upload({"uid": "d1", "name": "SO101", "slug": "so101", "data_type": "mcap", "item_count": 1})

    client = Client(api_key="test-key")
    ds = import_lerobot(client, repo_id="lerobot/svla_so101_pickplace", name="SO101", slug="so101")
    client.close()

    assert ds.uid == "d1"
    assert ds.data_type == "mcap"
    assert s3.call_count == 1  # one .mcap per episode


@respx.mock
def test_import_lerobot_dispatches_via_registry(fake_lerobot):
    _wire_upload({"uid": "d2", "name": "L", "slug": "l", "data_type": "mcap", "item_count": 1})
    client = Client(api_key="test-key")
    ds = import_dataset("lerobot", client, repo_id="lerobot/x", name="L", slug="l")
    client.close()
    assert ds.uid == "d2"


def test_import_lerobot_empty_selection(fake_lerobot):
    # episodes=[] selects nothing -> rejected before any download/upload
    client = Client(api_key="test-key")
    with pytest.raises(ValueError, match="no episodes selected"):
        import_lerobot(client, repo_id="lerobot/x", name="L", slug="l", episodes=[])
    client.close()


@respx.mock
def test_import_lerobot_passes_episode_subset_to_loader(fake_lerobot):
    # --episodes 0 must be forwarded to LeRobotDataset(episodes=...) so only that subset loads
    captured = {}
    original = sys.modules["lerobot.datasets.lerobot_dataset"].LeRobotDataset

    class _SpyDataset(original):  # type: ignore[misc, valid-type]
        def __init__(self, repo_id, root=None, episodes=None):
            captured["episodes"] = episodes
            super().__init__(repo_id, root=root, episodes=episodes)

    sys.modules["lerobot.datasets.lerobot_dataset"].LeRobotDataset = _SpyDataset
    try:
        _wire_upload({"uid": "d5", "name": "L", "slug": "l", "data_type": "mcap", "item_count": 1})
        client = Client(api_key="test-key")
        import_lerobot(client, repo_id="lerobot/x", name="L", slug="l", episodes=[0])
        client.close()
    finally:
        sys.modules["lerobot.datasets.lerobot_dataset"].LeRobotDataset = original
    assert captured["episodes"] == [0]


def test_import_lerobot_rejects_unknown_camera_key(fake_lerobot):
    # a typo'd camera key must fail loudly, not silently produce an MCAP with no image channel
    client = Client(api_key="test-key")
    with pytest.raises(ValueError, match="unknown camera keys"):
        import_lerobot(client, repo_id="lerobot/x", name="L", slug="l", camera_keys=["observation.images.cm"])
    client.close()


def test_import_lerobot_rejects_out_of_range_episode(fake_lerobot):
    # fake dataset has num_episodes=1; index 5 (and -1) must be rejected, not silently used
    client = Client(api_key="test-key")
    with pytest.raises(ValueError, match="out of range"):
        import_lerobot(client, repo_id="lerobot/x", name="L", slug="l", episodes=[5])
    with pytest.raises(ValueError, match="out of range"):
        import_lerobot(client, repo_id="lerobot/x", name="L", slug="l", episodes=[-1])
    client.close()
