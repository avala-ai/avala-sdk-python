from __future__ import annotations

import json
import re
import sys
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
    monkeypatch.setattr(al, "_lerobot_dataset_cls", lambda *_a, **_k: _FakeLeRobotDataset)
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
    monkeypatch.setattr(al, "_lerobot_dataset_cls", lambda *_a, **_k: _FakeLeRobotDataset)

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
    monkeypatch.setattr(al, "_lerobot_dataset_cls", lambda *_a, **_k: _FakeLeRobotDataset)

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
    monkeypatch.setattr(al, "_lerobot_dataset_cls", lambda *_a, **_k: _FakeLeRobotDataset)
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
    monkeypatch.setattr(al, "_lerobot_dataset_cls", lambda *_a, **_k: _FakeLeRobotDataset)
    with pytest.raises(ValueError, match="repo_id"):
        export_dataset(Client(api_key="k"), "o", "s", repo_id="no-slash", output_dir="/tmp/x")


def test_actionable_error_when_no_writer_is_installed(monkeypatch):
    # Neither the lerobot library nor pyarrow (the torch-free writer's dependency).
    monkeypatch.setitem(sys.modules, "lerobot", None)
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    with pytest.raises(ModuleNotFoundError, match=r"avala\[lerobot-core-video\].*avala\[lerobot\]"):
        al._lerobot_dataset_cls()


def test_backend_lerobot_requires_the_library(monkeypatch):
    monkeypatch.setitem(sys.modules, "lerobot", None)
    with pytest.raises(ModuleNotFoundError, match=r"avala\[lerobot\]"):
        al._lerobot_dataset_cls("lerobot")


def test_falls_back_to_core_writer_without_lerobot(monkeypatch):
    pytest.importorskip("pyarrow")
    from avala.converters.lerobot_v3.writer import LeRobotV3Writer

    monkeypatch.setitem(sys.modules, "lerobot", None)
    assert al._lerobot_dataset_cls() is LeRobotV3Writer
    assert al._lerobot_dataset_cls("core") is LeRobotV3Writer
    with pytest.raises(ValueError, match="backend"):
        al._lerobot_dataset_cls("torch")


@respx.mock
def test_export_dataset_with_core_writer_is_readable(tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("PIL")
    pytest.importorskip("av")
    from avala.converters.lerobot_v3.reader import LeRobotV3Dataset

    seq = _sequence(uid="seq1", n=3, n_cams=1, state=[1.0, 2.0], action=[0.5]).model_dump(mode="json")
    _wire_seq_routes(["seq1", "seq2"], lambda *_: httpx.Response(200, json=seq))

    client = Client(api_key="test-key")
    out = export_dataset(
        client,
        "o",
        "s",
        repo_id="user/ds",
        output_dir=tmp_path / "ds",
        fps=10,
        task="pick",
        state_key="obs.state",
        action_key="action_vec",
        backend="core",
    )
    client.close()

    ds = LeRobotV3Dataset(out)
    assert ds.total_episodes == 2
    assert [e["length"] for e in ds.episodes] == [3, 3]
    frames = list(ds.iter_frames(1))
    assert [f["frame_index"] for f in frames] == [0, 1, 2]
    assert frames[0]["observation.state"].tolist() == [1.0, 2.0]
    assert frames[0]["action"].tolist() == [0.5]
    assert frames[0]["task"] == "pick"
    assert frames[0]["observation.images.cam0"].shape == (6, 8, 3)


def test_export_push_with_core_writer_fails_before_writing(tmp_path):
    pytest.importorskip("pyarrow")
    with pytest.raises(ModuleNotFoundError, match="push=True needs the lerobot library"):
        export_dataset(
            Client(api_key="k"), "o", "s", repo_id="u/d", output_dir=tmp_path / "x", push=True, backend="core"
        )
    assert not (tmp_path / "x").exists()


def _outcome_json(sequence_uid: str, **overrides):
    body = {
        "uid": f"o-{sequence_uid}",
        "sequence_uid": sequence_uid,
        "version": 3,
        "is_current": True,
        "outcome": "mistake_and_recovery",
        "progress": 0.5,
        "quality": None,
        "speed": None,
        "subtasks": [{"label": "regrasp", "start_ts": 1.0, "end_ts": 2.0, "outcome": None}],
        "mistake_type": "grasp_slip",
        "recovery_type": "",
        "failure_stage": "",
        "autonomy_level": "teleoperation",
        "model_version": "",
        "evaluation_membership": "held_out_eval",
        "leakage_groups": {"location": "kitchen-3"},
        "source": "human",
        "labeled_by": None,
        "confidence": None,
    }
    body.update(overrides)
    return body


@respx.mock
def test_export_dataset_writes_outcome_sidecar_per_episode(monkeypatch, tmp_path):
    """Avala sequence -> LeRobot episode: one sidecar row per SAVED episode, in save order."""
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    monkeypatch.setattr(al, "_lerobot_dataset_cls", lambda *_a, **_k: _FakeLeRobotDataset)
    full = _sequence(uid="seqX", n=1, n_cams=1).model_dump(mode="json")
    _wire_seq_routes(["seq1", "seq2"], lambda *_: httpx.Response(200, json=full))
    hand_actions = {
        "left": [
            {
                "start_ts": 0.0,
                "end_ts": 2.0,
                "action": "holding the piston",
                "object": "piston",
                "verb": "hold",
                "contact": True,
            }
        ],
        "right": [],
    }
    respx.get(f"{BASE_URL}/datasets/o/s/sequences/seq1/outcome/").mock(
        return_value=httpx.Response(
            200, json=_outcome_json("seq1", hand_actions=hand_actions, source_metadata={"run_id": "run-42"})
        )
    )
    respx.get(f"{BASE_URL}/datasets/o/s/sequences/seq2/outcome/").mock(
        return_value=httpx.Response(404, json={"detail": "This sequence has no outcome label."})
    )

    client = Client(api_key="test-key")
    export_dataset(client, "o", "s", repo_id="user/ds", output_dir=tmp_path, fps=30, include_outcomes=True)
    client.close()

    rows = [json.loads(line) for line in (tmp_path / al.OUTCOMES_SIDECAR).read_text().splitlines()]
    assert [(r["episode_index"], r["avala_sequence_uid"]) for r in rows] == [(0, "seq1"), (1, "seq2")]
    assert rows[0]["outcome"] == "mistake_and_recovery"
    assert rows[0]["evaluation_membership"] == "held_out_eval"
    assert rows[0]["recovery_type"] is None  # empty strings map to null
    assert rows[0]["subtasks"][0]["label"] == "regrasp"
    # Per-hand streams ride along unchanged; timestamps stay seconds from episode start.
    assert rows[0]["hand_actions"] == hand_actions
    assert rows[0]["outcome_source_metadata"] == {"run_id": "run-42"}
    assert rows[1] == {"episode_index": 1, "avala_sequence_uid": "seq2", "outcome": None}


def test_outcome_episode_metadata_maps_missing_hand_actions_to_empty_streams():
    from avala.types.sequence_outcome import SequenceOutcome

    row = al.outcome_episode_metadata(4, "seqZ", SequenceOutcome.model_validate(_outcome_json("seqZ")))
    assert row["episode_index"] == 4
    assert row["hand_actions"] == {"left": [], "right": []}


@respx.mock
def test_export_dataset_skips_outcomes_unless_requested(monkeypatch, tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    monkeypatch.setattr(al, "_lerobot_dataset_cls", lambda *_a, **_k: _FakeLeRobotDataset)
    full = _sequence(uid="seqX", n=1, n_cams=1).model_dump(mode="json")
    _wire_seq_routes(["seq1"], lambda *_: httpx.Response(200, json=full))
    outcome_route = respx.get(f"{BASE_URL}/datasets/o/s/sequences/seq1/outcome/").mock(
        return_value=httpx.Response(200, json=_outcome_json("seq1"))
    )

    client = Client(api_key="test-key")
    export_dataset(client, "o", "s", repo_id="user/ds", output_dir=tmp_path, fps=30)
    client.close()

    assert not outcome_route.called
    assert not (tmp_path / al.OUTCOMES_SIDECAR).exists()


# ── control_source column on export ──
def _seq_with_sources(sources, uid="seqC"):
    frames = []
    for src in sources:
        frame = _frame(n_cams=1)
        if src is not None:
            frame["control"] = {"source": src}
        frames.append(frame)
    return DatasetSequence(uid=uid, number_of_frames=len(frames), frames=frames)


def test_build_features_adds_control_source_column():
    seq = _seq_with_sources(["policy", "teleop"])
    features = build_features(seq, camera_specs=[("cam0", 6, 8)], control_source_key="control.source")
    assert features["annotation.avala.control_source"] == {"dtype": "string", "shape": (1,), "names": None}
    custom = build_features(
        seq,
        camera_specs=[("cam0", 6, 8)],
        control_source_key="control.source",
        control_source_column="annotation.vendor.control_source",
    )
    assert "annotation.vendor.control_source" in custom


def test_build_features_control_source_must_resolve_to_string():
    with pytest.raises(KeyError, match="control.source"):
        build_features(_seq_with_sources([None]), camera_specs=[("cam0", 6, 8)], control_source_key="control.source")
    seq = _sequence(n=1)
    seq.frames[0]["control"] = {"source": 3}
    with pytest.raises(ValueError, match="non-empty string"):
        build_features(seq, camera_specs=[("cam0", 6, 8)], control_source_key="control.source")


@respx.mock
def test_iter_frames_emits_control_source_per_frame():
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    respx.get(url__regex=CDN_RE).mock(side_effect=lambda *_: _png_response())
    seq = _seq_with_sources(["policy", "intervention", "hold"])
    features = build_features(seq, camera_specs=[("cam0", 6, 8)], control_source_key="control.source")
    media = httpx.Client()
    samples = list(
        iter_frames(seq, media_client=media, features=features, task="t", control_source_key="control.source")
    )
    media.close()
    assert [s["annotation.avala.control_source"] for s in samples] == ["policy", "intervention", "hold"]


@respx.mock
def test_iter_frames_control_source_missing_mid_sequence_errors():
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    respx.get(url__regex=CDN_RE).mock(side_effect=lambda *_: _png_response())
    seq = _seq_with_sources(["policy", None])
    features = build_features(seq, camera_specs=[("cam0", 6, 8)], control_source_key="control.source")
    media = httpx.Client()
    with pytest.raises(KeyError, match="control.source"):
        list(iter_frames(seq, media_client=media, features=features, task="t", control_source_key="control.source"))
    media.close()


@respx.mock
def test_export_with_core_writer_writes_control_source_and_outcomes(tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("av")
    from avala.converters.lerobot_v3.reader import LeRobotV3Dataset

    seq = _seq_with_sources(["policy", "teleop"], uid="seq1").model_dump(mode="json")
    _wire_seq_routes(["seq1"], lambda *_: httpx.Response(200, json=seq))
    respx.get(url__regex=re.escape(f"{BASE_URL}/datasets/o/s/sequences/seq1/outcome/") + r".*").mock(
        return_value=httpx.Response(200, json=_outcome_json("seq1"))
    )

    client = Client(api_key="test-key")
    with pytest.warns(UserWarning, match="perception-only"):
        out = export_dataset(
            client,
            "o",
            "s",
            repo_id="u/d",
            output_dir=tmp_path / "ds",
            fps=10,
            control_source_key="control.source",
            include_outcomes=True,
            backend="core",
        )
    client.close()

    ds = LeRobotV3Dataset(out)
    assert [f["annotation.avala.control_source"] for f in ds.iter_frames(0, decode_visual=False)] == [
        "policy",
        "teleop",
    ]
    rows = [json.loads(line) for line in (out / "meta/avala_sequence_outcomes.jsonl").read_text().splitlines()]
    assert rows[0]["episode_index"] == 0 and rows[0]["outcome"] == "mistake_and_recovery"


# ── outcome task text -> LeRobot's native per-frame task ──
def test_outcome_task_text_reads_source_metadata():
    from avala.types.sequence_outcome import SequenceOutcome

    assert al.outcome_task_text(None) is None
    labelled = SequenceOutcome.model_validate(_outcome_json("s", source_metadata={"task": "  stack the cups "}))
    assert al.outcome_task_text(labelled) == "stack the cups"
    alt = SequenceOutcome.model_validate(_outcome_json("s", source_metadata={"language_instruction": "open the door"}))
    assert al.outcome_task_text(alt) == "open the door"
    assert al.outcome_task_text(SequenceOutcome.model_validate(_outcome_json("s"))) is None


@respx.mock
def test_export_writes_outcome_task_as_lerobot_task(tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("av")
    from avala.converters.lerobot_v3.reader import LeRobotV3Dataset

    seq = _sequence(uid="x", n=2, n_cams=1).model_dump(mode="json")
    _wire_seq_routes(["seq1", "seq2"], lambda *_: httpx.Response(200, json=seq))
    respx.get(url__regex=re.escape(f"{BASE_URL}/datasets/o/s/sequences/seq1/outcome/") + r".*").mock(
        return_value=httpx.Response(200, json=_outcome_json("seq1", source_metadata={"task": "stack the cups"}))
    )
    respx.get(url__regex=re.escape(f"{BASE_URL}/datasets/o/s/sequences/seq2/outcome/") + r".*").mock(
        return_value=httpx.Response(200, json=_outcome_json("seq2"))
    )

    client = Client(api_key="test-key")
    with pytest.warns(UserWarning, match="perception-only"):
        out = export_dataset(
            client,
            "o",
            "s",
            repo_id="u/d",
            output_dir=tmp_path / "ds",
            task="default",
            include_outcomes=True,
            backend="core",
        )
    client.close()

    ds = LeRobotV3Dataset(out)
    assert [f["task"] for f in ds.iter_frames(0, decode_visual=False)] == ["stack the cups"] * 2
    assert [f["task"] for f in ds.iter_frames(1, decode_visual=False)] == ["default"] * 2  # no task text: caller's
    rows = [json.loads(line) for line in (out / "meta/avala_sequence_outcomes.jsonl").read_text().splitlines()]
    assert [r["avala_sequence_uid"] for r in rows] == ["seq1", "seq2"]  # sidecar still written


@respx.mock
def test_export_ignores_outcome_task_unless_outcomes_requested(tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("av")
    from avala.converters.lerobot_v3.reader import LeRobotV3Dataset

    seq = _sequence(uid="x", n=1, n_cams=1).model_dump(mode="json")
    _wire_seq_routes(["seq1"], lambda *_: httpx.Response(200, json=seq))
    route = respx.get(url__regex=r".*/outcome/.*").mock(return_value=httpx.Response(500))
    client = Client(api_key="test-key")
    with pytest.warns(UserWarning, match="perception-only"):
        out = export_dataset(client, "o", "s", repo_id="u/d", output_dir=tmp_path / "ds", task="t", backend="core")
    client.close()
    assert not route.called
    assert [f["task"] for f in LeRobotV3Dataset(out).iter_frames(0, decode_visual=False)] == ["t"]
