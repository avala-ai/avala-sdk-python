from __future__ import annotations

import importlib.util
import re
from typing import Any

import httpx
import pytest
import respx
from avala import Client
from avala.types.dataset import DatasetSequence

import avala.lerobot as al
from avala.lerobot import build_features, discover_camera_specs, export_dataset, iter_frames

BASE_URL = "https://api.avala.ai/api/v1"
SEQ_LIST_RE = re.escape(f"{BASE_URL}/datasets/o/s/sequences/") + r"(\?.*)?$"
SEQ_GET_RE = re.escape(f"{BASE_URL}/datasets/o/s/sequences/") + r"[^/?]+/$"
CDN_RE = r"https://cdn\.example\.com/.*\.png"


def _frame(*, n_cams=1, urls=None, state=None, action=None, pos=None, heading=None):
    frame: dict[str, Any] = {
        "image_urls": urls if urls is not None else [f"https://cdn.example.com/cam{i}.png" for i in range(n_cams)]
    }
    if state is not None:
        frame["obs"] = {"state": state}
    if action is not None:
        frame["action_vec"] = action
    if pos is not None:
        frame["device_position"] = pos
    if heading is not None:
        frame["device_heading"] = heading
    return frame


def _sequence(uid="seq1", n=2, **frame_kwargs):
    return DatasetSequence(uid=uid, number_of_frames=n, frames=[_frame(**frame_kwargs) for _ in range(n)])


def _png_bytes(h: int = 6, w: int = 8) -> bytes:
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    import io

    buf = io.BytesIO()
    Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _png_response(*_a, h: int = 6, w: int = 8) -> httpx.Response:
    return httpx.Response(200, content=_png_bytes(h, w), headers={"content-type": "image/png"})


# ── build_features: pure logic over resolved camera specs (no network) ──
def test_build_features_from_specs():
    seq = _sequence(n_cams=2)
    features = build_features(seq, camera_specs=[("cam0", 6, 8), ("cam1", 6, 8)])
    assert set(features) == {"observation.images.cam0", "observation.images.cam1"}
    assert features["observation.images.cam0"] == {
        "dtype": "video",
        "shape": (3, 6, 8),
        "names": ["channels", "height", "width"],
    }


def test_build_features_no_video_uses_image_dtype():
    features = build_features(_sequence(), camera_specs=[("cam0", 6, 8)], use_videos=False)
    assert features["observation.images.cam0"]["dtype"] == "image"


def test_build_features_state_and_action_when_keys_resolve():
    seq = _sequence(state=[1.0, 2.0, 3.0], action=[0.1, 0.2])
    features = build_features(seq, camera_specs=[("cam0", 6, 8)], state_key="obs.state", action_key="action_vec")
    assert features["observation.state"]["shape"] == (3,)
    assert features["action"]["shape"] == (2,)


def test_build_features_omits_state_when_absent():
    features = build_features(_sequence(), camera_specs=[("cam0", 6, 8)])
    assert "observation.state" not in features and "action" not in features


def test_build_features_ego_pose_is_7dim():
    seq = _sequence(pos={"x": 1, "y": 2, "z": 3}, heading={"x": 0, "y": 0, "z": 0, "w": 1})
    features = build_features(seq, camera_specs=[("cam0", 6, 8)], include_ego_pose=True)
    assert features["observation.state"]["shape"] == (7,)
    assert features["observation.state"]["names"] == ["x", "y", "z", "qw", "qx", "qy", "qz"]


def test_build_features_state_key_and_ego_pose_conflict():
    with pytest.raises(ValueError, match="not both"):
        build_features(
            _sequence(state=[1.0]), camera_specs=[("cam0", 6, 8)], state_key="obs.state", include_ego_pose=True
        )


def test_build_features_missing_state_key_errors():
    with pytest.raises(KeyError):
        build_features(_sequence(), camera_specs=[("cam0", 6, 8)], state_key="obs.state")


def test_build_features_requires_specs():
    with pytest.raises(ValueError, match="no camera_specs"):
        build_features(_sequence(), camera_specs=[])


def test_build_features_ego_pose_requires_pose_on_frame0():
    with pytest.raises(ValueError, match="device_position"):
        build_features(_sequence(), camera_specs=[("cam0", 6, 8)], include_ego_pose=True)


def test_build_features_rejects_colliding_specs():
    with pytest.raises(ValueError, match="collide"):
        build_features(_sequence(), camera_specs=[("cam0", 6, 8), ("cam0", 6, 8)])


# ── discover_camera_specs: probes image_urls for dimensions ──
@respx.mock
def test_discover_camera_specs_probes_dimensions():
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    respx.get(url__regex=CDN_RE).mock(side_effect=lambda *_: _png_response(h=6, w=8))
    frame = _frame(n_cams=2)
    media = httpx.Client()
    specs = discover_camera_specs(frame, media)
    media.close()
    assert specs == [("cam0", 6, 8), ("cam1", 6, 8)]


@respx.mock
def test_discover_camera_specs_filters_by_camera_keys():
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    respx.get(url__regex=CDN_RE).mock(side_effect=lambda *_: _png_response())
    media = httpx.Client()
    specs = discover_camera_specs(_frame(n_cams=3), media, camera_keys=["cam1"])
    media.close()
    assert specs == [("cam1", 6, 8)]


def test_discover_camera_specs_no_images_errors():
    media = httpx.Client()
    with pytest.raises(ValueError, match="no image_urls"):
        discover_camera_specs({"image_urls": []}, media)
    media.close()


# ── iter_frames ──
@respx.mock
def test_iter_frames_decodes_images_and_state():
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    respx.get(url__regex=CDN_RE).mock(side_effect=lambda *_: _png_response())
    seq = _sequence(n=2, n_cams=1, state=[1.0, 2.0])
    features = build_features(seq, camera_specs=[("cam0", 6, 8)], state_key="obs.state")
    media = httpx.Client()
    samples = list(iter_frames(seq, media_client=media, features=features, state_key="obs.state", task="grab"))
    media.close()

    assert len(samples) == 2
    assert samples[0]["task"] == "grab"
    assert samples[0]["observation.images.cam0"].shape == (6, 8, 3)
    assert list(samples[0]["observation.state"]) == [1.0, 2.0]


@respx.mock
def test_iter_frames_detects_resolution_drift():
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")

    def _handler(request: httpx.Request) -> httpx.Response:
        # frame 1's image is a different size than frame 0 -> must raise.
        return _png_response(h=4, w=4) if "f1" in str(request.url) else _png_response(h=6, w=8)

    respx.get(url__regex=CDN_RE).mock(side_effect=_handler)
    seq = DatasetSequence(
        uid="s",
        frames=[
            _frame(urls=["https://cdn.example.com/f0-cam0.png"]),
            _frame(urls=["https://cdn.example.com/f1-cam0.png"]),
        ],
    )
    features = build_features(seq, camera_specs=[("cam0", 6, 8)])
    media = httpx.Client()
    with pytest.raises(ValueError, match="resolution must be constant"):
        list(iter_frames(seq, media_client=media, features=features, task="t"))
    media.close()


@respx.mock
def test_iter_frames_detects_camera_count_change():
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    respx.get(url__regex=CDN_RE).mock(side_effect=lambda *_: _png_response())
    seq = DatasetSequence(uid="s", frames=[_frame(n_cams=2), _frame(n_cams=1)])
    features = build_features(seq, camera_specs=[("cam0", 6, 8), ("cam1", 6, 8)])
    media = httpx.Client()
    with pytest.raises(ValueError, match="do not match the dataset cameras"):
        list(iter_frames(seq, media_client=media, features=features, task="t"))
    media.close()


@respx.mock
def test_iter_frames_detects_state_dim_drift():
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    respx.get(url__regex=CDN_RE).mock(side_effect=lambda *_: _png_response())
    seq = DatasetSequence(uid="s", frames=[_frame(state=[1.0, 2.0]), _frame(state=[1.0, 2.0, 3.0])])
    features = build_features(seq, camera_specs=[("cam0", 6, 8)], state_key="obs.state")
    media = httpx.Client()
    with pytest.raises(ValueError, match="expected 2"):
        list(iter_frames(seq, media_client=media, features=features, state_key="obs.state", task="t"))
    media.close()


@respx.mock
def test_iter_frames_wraps_non_image_decode_error():
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    respx.get(url__regex=CDN_RE).mock(
        return_value=httpx.Response(200, content=b"<html>not an image</html>", headers={"content-type": "text/html"})
    )
    seq = _sequence(n=1)
    features = build_features(seq, camera_specs=[("cam0", 6, 8)])
    media = httpx.Client()
    with pytest.raises(ValueError, match="failed to decode image"):
        list(iter_frames(seq, media_client=media, features=features, task="t"))
    media.close()


class _FakeLeRobotDataset:
    """Duck-typed stand-in for lerobot's LeRobotDataset (no heavy dep needed)."""

    last_instance: "_FakeLeRobotDataset | None" = None

    def __init__(self, **kwargs):
        self.create_kwargs = kwargs
        self.frames: list = []
        self.episodes = 0
        self.finalized = False
        self.pushed = False
        self.push_kwargs: dict = {}
        _FakeLeRobotDataset.last_instance = self

    @classmethod
    def create(cls, **kwargs):
        return cls(**kwargs)

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.episodes += 1

    def finalize(self):
        self.finalized = True

    def push_to_hub(self, *args, **kwargs):
        self.pushed = True
        self.push_kwargs = kwargs


def _wire_seq_routes(list_uids, seq_handler):
    respx.get(url__regex=SEQ_LIST_RE).mock(
        return_value=httpx.Response(
            200, json={"results": [{"uid": u} for u in list_uids], "next": None, "previous": None}
        )
    )
    respx.get(url__regex=SEQ_GET_RE).mock(side_effect=seq_handler)
    respx.get(url__regex=CDN_RE).mock(side_effect=lambda *_: _png_response())


@respx.mock
def test_export_dataset_orchestration_no_export_step(monkeypatch, tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    monkeypatch.setattr(al, "_lerobot_dataset_cls", lambda: _FakeLeRobotDataset)
    full = _sequence(uid="seqX", n=2, n_cams=1).model_dump(mode="json")
    _wire_seq_routes(["seq1", "seq2"], lambda *_: httpx.Response(200, json=full))
    export_route = respx.post(f"{BASE_URL}/exports/").mock(return_value=httpx.Response(201, json={}))

    client = Client(api_key="test-key")
    out = export_dataset(client, "o", "s", repo_id="user/ds", output_dir=tmp_path, fps=30)
    client.close()

    ds = _FakeLeRobotDataset.last_instance
    assert ds is not None
    assert ds.finalized is True
    assert ds.episodes == 2
    assert len(ds.frames) == 4
    assert "observation.images.cam0" in ds.create_kwargs["features"]
    assert ds.create_kwargs["features"]["observation.images.cam0"]["shape"] == (3, 6, 8)
    assert out == tmp_path
    assert not export_route.called


@respx.mock
def test_export_dataset_finalizes_even_on_mid_batch_error(monkeypatch, tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    monkeypatch.setattr(al, "_lerobot_dataset_cls", lambda: _FakeLeRobotDataset)

    def _seq_handler(request: httpx.Request) -> httpx.Response:
        uid = request.url.path.rstrip("/").split("/")[-1]
        n_cams = 2 if uid == "seq2" else 1  # seq2 has an extra camera -> schema mismatch mid-batch
        return httpx.Response(200, json=_sequence(uid=uid, n=1, n_cams=n_cams).model_dump(mode="json"))

    _wire_seq_routes(["seq1", "seq2"], _seq_handler)

    client = Client(api_key="test-key")
    with pytest.raises(ValueError, match="do not match the dataset cameras"):
        export_dataset(client, "o", "s", repo_id="user/ds", output_dir=tmp_path, fps=30)
    client.close()

    ds = _FakeLeRobotDataset.last_instance
    assert ds is not None
    assert ds.finalized is True  # already-saved episode footered despite the failure
    assert ds.episodes == 1


@respx.mock
def test_export_dataset_skips_empty_leading_sequence(monkeypatch, tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    monkeypatch.setattr(al, "_lerobot_dataset_cls", lambda: _FakeLeRobotDataset)

    def _seq_handler(request: httpx.Request) -> httpx.Response:
        uid = request.url.path.rstrip("/").split("/")[-1]
        if uid == "empty":
            return httpx.Response(200, json=DatasetSequence(uid="empty", frames=[]).model_dump(mode="json"))
        return httpx.Response(200, json=_sequence(uid="good", n=2, n_cams=1).model_dump(mode="json"))

    _wire_seq_routes(["empty", "good"], _seq_handler)

    client = Client(api_key="test-key")
    export_dataset(client, "o", "s", repo_id="user/ds", output_dir=tmp_path, fps=30)
    client.close()

    ds = _FakeLeRobotDataset.last_instance
    assert ds is not None
    assert ds.episodes == 1
    assert ds.finalized is True


@respx.mock
def test_export_dataset_push_brands_tags_and_license(monkeypatch, tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    monkeypatch.setattr(al, "_lerobot_dataset_cls", lambda: _FakeLeRobotDataset)
    full = _sequence(uid="seq1", n=1, n_cams=1).model_dump(mode="json")
    _wire_seq_routes(["seq1"], lambda *_: httpx.Response(200, json=full))

    client = Client(api_key="test-key")
    export_dataset(
        client,
        "o",
        "s",
        repo_id="user/ds",
        output_dir=tmp_path,
        fps=30,
        push=True,
        tags=["robotics-pilot"],
        repo_license="mit",
    )
    client.close()

    ds = _FakeLeRobotDataset.last_instance
    assert ds is not None and ds.pushed is True
    assert ds.push_kwargs["tags"] == ["avala", "robotics-pilot"]
    assert ds.push_kwargs["license"] == "mit"


def test_export_dataset_rejects_bad_repo_id(monkeypatch):
    monkeypatch.setattr(al, "_lerobot_dataset_cls", lambda: _FakeLeRobotDataset)
    with pytest.raises(ValueError, match="repo_id"):
        export_dataset(Client(api_key="k"), "o", "s", repo_id="no-slash", output_dir="/tmp/x")


def test_actionable_error_when_lerobot_missing():
    if importlib.util.find_spec("lerobot") is not None:
        pytest.skip("lerobot is installed; cannot test the missing-dependency path")
    with pytest.raises(ModuleNotFoundError, match=r"avala\[lerobot\]"):
        al._lerobot_dataset_cls()
