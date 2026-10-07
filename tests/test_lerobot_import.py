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
# Must be a real S3 host: the uploader refuses to POST file bytes to anything
# outside the presigned-URL allow-list (``avala/_uploads.py``).
S3_URL = "https://s3.us-east-1.amazonaws.com/upload"


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
        self.robot_type = "fake_bot"
        self.info = {"codebase_version": "v3.0"}
        self.features = {
            "observation.images.cam": {"dtype": "video", "shape": (3, 8, 8), "names": None},
            "observation.state": {"dtype": "float32", "shape": (3,), "names": None},
            "action": {"dtype": "float32", "shape": (2,), "names": None},
            "annotation.vendor.control_source": {"dtype": "string", "shape": (1,), "names": None},
            "failure_type": {"dtype": "string", "shape": (1,), "names": None},
            "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
            "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
        }


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
            "annotation.vendor.control_source": ["policy", "teleop", "intervention"][i],
            "failure_type": "none",
            "task": "fold the towel",
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


# ── torch-free core backend (no lerobot library) against the lerobot-written fixture ──
REF = __import__("pathlib").Path(__file__).parent / "fixtures" / "lerobot_v3" / "ref_dataset"


def _capture_mcaps(monkeypatch, client, dest):
    """Replace the upload with a copy of the staged .mcap files into ``dest``."""
    import shutil

    from avala.types.dataset import Dataset

    def _fake_create_from_local(*, source, name, slug, data_type, **_kwargs):
        shutil.copytree(source, dest)
        return Dataset(uid="d", name=name, slug=slug, data_type=data_type, item_count=0)

    monkeypatch.setattr(client.datasets, "create_from_local", _fake_create_from_local)


def _decoded(path):
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    with open(path, "rb") as fh:
        reader = make_reader(fh, decoder_factories=[DecoderFactory()])
        return [(m.channel.topic, m.message.log_time, m.decoded_message) for m in reader.iter_decoded_messages()]


def test_core_backend_imports_v3_without_lerobot(monkeypatch, tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("av")
    monkeypatch.setitem(sys.modules, "lerobot", None)  # simulate: lerobot not installed
    client = Client(api_key="test-key")
    _capture_mcaps(monkeypatch, client, tmp_path / "out")

    ds = import_lerobot(client, root=str(REF), name="Ref", slug="ref")
    client.close()

    assert ds.data_type == "mcap"
    files = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert files == ["episode_000000.mcap", "episode_000001.mcap"]
    messages = _decoded(tmp_path / "out" / "episode_000001.mcap")
    state = [(t, list(m["data"])) for topic, t, m in messages if topic == "/observation/state"]
    # Struct layout and topic name unchanged; values exact (float32 -> double is lossless)
    assert [t for t, _ in state] == [0, 100_000_001, 200_000_002]
    assert state[0][1] == [1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
    images = [m for topic, _, m in messages if topic == "/observation/images/top"]
    assert len(images) == 3 and images[0].format == "jpeg"


def test_backend_lerobot_without_library_errors(monkeypatch):
    monkeypatch.setitem(sys.modules, "lerobot", None)
    monkeypatch.setitem(sys.modules, "lerobot.datasets", None)
    monkeypatch.setitem(sys.modules, "lerobot.datasets.lerobot_dataset", None)
    with pytest.raises(ModuleNotFoundError, match=r"avala\[lerobot\]"):
        import_lerobot(Client(api_key="k"), root=str(REF), name="x", slug="x", backend="lerobot")


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="backend"):
        import_lerobot(Client(api_key="k"), root=str(REF), name="x", slug="x", backend="torch")


@respx.mock
def test_library_backend_carries_extra_columns(fake_lerobot, monkeypatch, tmp_path):
    client = Client(api_key="test-key")
    _capture_mcaps(monkeypatch, client, tmp_path / "out")
    import_lerobot(client, repo_id="lerobot/x", name="L", slug="l")
    client.close()

    messages = _decoded(tmp_path / "out" / "episode_000000.mcap")
    by_topic = {}
    for topic, _t, msg in messages:
        by_topic.setdefault(topic, []).append(msg)
    assert set(by_topic) == {
        "/observation/images/cam",
        "/observation/state",
        "/action",
        "/lerobot/control_source",
        "/failure_type",
        "/lerobot/task",
    }
    assert [m["data"] for m in by_topic["/lerobot/control_source"]] == ["policy", "teleop", "intervention"]
    assert [m["data"] for m in by_topic["/failure_type"]] == ["none"] * 3
    assert [m["data"] for m in by_topic["/lerobot/task"]] == ["fold the towel"] * 3
    assert list(by_topic["/action"][0]["data"]) == [1.0, 2.0]  # unchanged Struct layout


def test_core_backend_carries_every_fixture_column(monkeypatch, tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("av")
    import json

    import pyarrow.parquet as pq
    from mcap.reader import make_reader

    client = Client(api_key="test-key")
    _capture_mcaps(monkeypatch, client, tmp_path / "out")
    import_lerobot(client, root=str(REF), name="Ref", slug="ref", backend="core")
    client.close()

    raw = pq.read_table(REF / "data/chunk-000/file-000.parquet").to_pylist()
    for ep in (0, 1):
        path = tmp_path / "out" / f"episode_{ep:06d}.mcap"
        rows = [r for r in raw if r["episode_index"] == ep]
        by_topic = {}
        for topic, _t, msg in _decoded(path):
            by_topic.setdefault(topic, []).append(msg)
        assert set(by_topic) == {
            "/observation/images/top",
            "/observation/state",
            "/action",
            "/next/reward",
            "/lerobot/control_source",
            "/episode_uuid",
            "/failure_type",
            "/lerobot/task",
        }
        assert [m["data"] for m in by_topic["/lerobot/control_source"]] == [
            r["annotation.vendor.control_source"] for r in rows
        ]
        assert [m["data"] for m in by_topic["/episode_uuid"]] == [r["episode_uuid"] for r in rows]
        assert [m["data"] for m in by_topic["/failure_type"]] == [r["failure_type"] for r in rows]
        assert [list(m["data"]) for m in by_topic["/next/reward"]] == [[r["next.reward"]] for r in rows]

        with open(path, "rb") as fh:
            records = {m.name: m.metadata for m in make_reader(fh).iter_metadata()}
        meta = records["avala.lerobot"]
        assert meta["episode_index"] == str(ep)
        assert json.loads(meta["topics"])["annotation.vendor.control_source"] == "/lerobot/control_source"
        info = json.loads((REF / "meta/info.json").read_text())
        recorded = json.loads(meta["features"])
        for key, spec in recorded.items():
            assert spec["dtype"] == info["features"][key]["dtype"]
            assert spec["shape"] == info["features"][key]["shape"]
            assert spec["names"] == info["features"][key]["names"]


@pytest.mark.parametrize("column", ["annotation.vendor_a.control_source", "annotation.vendor_b.control_source"])
def test_core_import_accepts_any_vendor_control_source_column(monkeypatch, tmp_path, column):
    pytest.importorskip("pyarrow")
    from avala.converters.lerobot_v3.writer import LeRobotV3Writer

    features = {
        "observation.images.cam": {"dtype": "image", "shape": (3, 8, 8), "names": None},
        column: {"dtype": "string", "shape": (1,), "names": None},
    }
    writer = LeRobotV3Writer.create(repo_id="a/b", fps=10, features=features, root=tmp_path / "src", use_videos=False)
    for source in ("policy", "intervention", "teleop", "hold"):
        writer.add_frame({"observation.images.cam": np.zeros((8, 8, 3), np.uint8), column: source, "task": "t"})
    writer.save_episode()
    writer.finalize()

    client = Client(api_key="test-key")
    _capture_mcaps(monkeypatch, client, tmp_path / "out")
    import_lerobot(client, root=str(tmp_path / "src"), name="x", slug="x", backend="core")
    client.close()
    messages = _decoded(tmp_path / "out" / "episode_000000.mcap")
    assert [m["data"] for topic, _t, m in messages if topic == "/lerobot/control_source"] == [
        "policy",
        "intervention",
        "teleop",
        "hold",
    ]
