"""Torch-free LeRobot v3 reader/writer (``avala.converters.lerobot_v3``).

The oracle is ``tests/fixtures/lerobot_v3/ref_dataset``, written by the real ``lerobot``
0.5.1 library (see ``generate_fixture.py`` there). Reader tests compare against values
read straight out of that fixture's parquet with pyarrow, and writer tests compare our
output's layout, schemas and ``info.json`` against it — so neither side is checked only
against itself.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pyarrow")
pytest.importorskip("PIL")

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from avala.converters.lerobot_v3 import _layout  # noqa: E402
from avala.converters.lerobot_v3.reader import LeRobotV3Dataset, UnsupportedLeRobotVersion  # noqa: E402
from avala.converters.lerobot_v3.writer import LeRobotV3Writer  # noqa: E402

REF = Path(__file__).parent / "fixtures" / "lerobot_v3" / "ref_dataset"
HAS_AV = importlib.util.find_spec("av") is not None
needs_av = pytest.mark.skipif(not HAS_AV, reason="PyAV not installed (avala[lerobot-core-video])")

CAM = "observation.images.top"
CONTROL = "annotation.vendor.control_source"


def _raw_rows():
    return pq.read_table(REF / "data/chunk-000/file-000.parquet").to_pylist()


# ── layout constants match what lerobot wrote ──
def test_fixture_layout_matches_layout_constants():
    info = json.loads((REF / _layout.INFO_PATH).read_text())
    assert info["codebase_version"] == _layout.CODEBASE_VERSION
    assert info["data_path"] == _layout.DATA_PATH
    assert info["video_path"] == _layout.VIDEO_PATH
    assert info["chunks_size"] == _layout.DEFAULT_CHUNK_SIZE
    assert list(info["features"])[-5:] == list(_layout.DEFAULT_FEATURES)
    for rel in (_layout.TASKS_PATH, _layout.STATS_PATH, _layout.EPISODES_PATH.format(chunk_index=0, file_index=0)):
        assert (REF / rel).is_file(), rel


# ── reader ──
def test_reader_metadata_and_episode_boundaries():
    ds = LeRobotV3Dataset(REF)
    assert ds.fps == 10
    assert ds.robot_type == "so101"
    assert ds.tasks == {0: "pick up the red cube", 1: "place the cube in the bin"}
    assert [(e["episode_index"], e["length"], e["dataset_from_index"], e["dataset_to_index"]) for e in ds.episodes] == [
        (0, 4, 0, 4),
        (1, 3, 4, 7),
    ]
    assert ds.camera_keys == [CAM]


def test_reader_values_equal_raw_parquet():
    ds = LeRobotV3Dataset(REF)
    raw = _raw_rows()
    frames = [f for ep in range(ds.total_episodes) for f in ds.iter_frames(ep, decode_visual=False)]
    assert len(frames) == len(raw) == 7
    for frame, row in zip(frames, raw):
        for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            assert frame[key] == row[key]
        assert frame["observation.state"].dtype == np.float32
        assert frame["observation.state"].tolist() == row["observation.state"]
        assert frame["action"].tolist() == row["action"]
        assert frame["next.reward"].shape == (1,)
        assert frame["next.reward"].tolist() == [row["next.reward"]]
        assert frame[CONTROL] == row[CONTROL]
        assert frame["episode_uuid"] == row["episode_uuid"]
        assert frame["failure_type"] == row["failure_type"]
        assert CAM not in frame
    assert [f[CONTROL] for f in frames] == ["policy", "policy", "intervention", "teleop", "hold", "policy", "policy"]
    assert [f["task"] for f in frames] == ["pick up the red cube"] * 4 + ["place the cube in the bin"] * 3


@needs_av
def test_reader_decodes_video_frames_at_episode_offsets():
    # The fixture paints frame i of episode e with R=30+60i, G=255-R, B=40e. h264 is lossy,
    # so allow a few levels of error; picking the wrong frame would be off by ~60.
    with LeRobotV3Dataset(REF) as ds:
        for ep in range(ds.total_episodes):
            for frame in ds.iter_frames(ep, keys=[CAM]):
                img = frame[CAM]
                assert img.shape == (64, 64, 3) and img.dtype == np.uint8
                level = 30 + 60 * frame["frame_index"]
                expected = np.array([level, 255 - level, 40 * ep])
                assert np.abs(img.reshape(-1, 3).mean(axis=0) - expected).max() < 6


def test_reader_rejects_v21(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text(json.dumps({"codebase_version": "v2.1", "features": {}, "fps": 30}))
    with pytest.raises(UnsupportedLeRobotVersion, match="v2.1"):
        LeRobotV3Dataset(tmp_path)


def test_reader_requires_info(tmp_path):
    with pytest.raises(FileNotFoundError):
        LeRobotV3Dataset(tmp_path)


# ── writer ──
def _copy_fixture(out: Path, *, as_images: bool) -> Path:
    src = LeRobotV3Dataset(REF)
    features = {k: dict(v) for k, v in src.user_features.items()}
    for spec in features.values():
        spec.pop("info", None)
    if as_images:
        features[CAM]["dtype"] = "image"
    writer = LeRobotV3Writer.create(
        repo_id="avala/test",
        fps=int(src.fps),
        features=features,
        root=out,
        robot_type=src.robot_type,
        use_videos=not as_images,
        vcodec="libx264",
    )
    for ep in range(src.total_episodes):
        for frame in src.iter_frames(ep, decode_visual=HAS_AV):
            sample = {k: frame[k] for k in features if k in frame}
            if CAM not in sample:
                sample[CAM] = np.zeros((64, 64, 3), dtype=np.uint8)
            sample["task"] = frame["task"]
            writer.add_frame(sample)
        writer.save_episode()
    writer.finalize()
    return out


@needs_av
def test_writer_layout_and_schemas_match_lerobot(tmp_path):
    out = _copy_fixture(tmp_path / "ours", as_images=False)
    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    ref_files = sorted(str(p.relative_to(REF)) for p in REF.rglob("*") if p.is_file())
    assert files == ref_files

    ours_info = json.loads((out / _layout.INFO_PATH).read_text())
    ref_info = json.loads((REF / _layout.INFO_PATH).read_text())
    assert ours_info == ref_info  # byte-for-byte equal JSON content, incl. the video info block

    for rel in ("data/chunk-000/file-000.parquet", "meta/tasks.parquet"):
        ours = pq.read_schema(out / rel).remove_metadata()
        ref = pq.read_schema(REF / rel).remove_metadata()
        assert ours.equals(ref), rel
    # pandas reads tasks.parquet with `task` as the index (lerobot's load_tasks relies on it)
    pandas_meta = json.loads(pq.read_schema(out / _layout.TASKS_PATH).metadata[b"pandas"])
    assert pandas_meta["index_columns"] == ["task"]

    ours_ep = pq.read_schema(out / "meta/episodes/chunk-000/file-000.parquet")
    ref_ep = pq.read_schema(REF / "meta/episodes/chunk-000/file-000.parquet")
    assert ours_ep.names == ref_ep.names
    for name in ours_ep.names:
        if name.startswith("stats/") and name.endswith(("/min", "/max")):
            continue  # value type follows the feature dtype on both sides; checked below
        assert ours_ep.field(name).type == ref_ep.field(name).type, name

    ours_rows = pq.read_table(out / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    ref_rows = pq.read_table(REF / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    for o, r in zip(ours_rows, ref_rows):
        for key in r:
            if key.startswith("stats/"):
                continue
            if key.endswith("_timestamp"):
                assert o[key] == pytest.approx(r[key]), key
            else:
                assert o[key] == r[key], key
        for feat in ("observation.state", "action", "next.reward", "timestamp", "index"):
            for stat in ("min", "max", "mean", "std", "count"):
                # lerobot accumulates in float32, we in float64: agree to ~1e-4 relative
                assert np.allclose(o[f"stats/{feat}/{stat}"], r[f"stats/{feat}/{stat}"], rtol=1e-4), (feat, stat)

    ours_stats = json.loads((out / _layout.STATS_PATH).read_text())
    ref_stats = json.loads((REF / _layout.STATS_PATH).read_text())
    assert set(ours_stats) == set(ref_stats)
    for feat in ("observation.state", "action", "next.reward"):
        for stat in ("min", "max", "mean", "std", "count"):
            assert np.allclose(ours_stats[feat][stat], ref_stats[feat][stat], rtol=1e-4), (feat, stat)


@needs_av
def test_writer_output_reads_back_identically(tmp_path):
    out = _copy_fixture(tmp_path / "ours", as_images=False)
    raw_ours = pq.read_table(out / "data/chunk-000/file-000.parquet").to_pylist()
    assert raw_ours == _raw_rows()  # every non-video value, bit-exact


def test_writer_image_mode_is_lossless(tmp_path):
    feats = {
        CAM: {"dtype": "image", "shape": (3, 4, 5), "names": ["channels", "height", "width"]},
        "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
        CONTROL: {"dtype": "string", "shape": (1,), "names": None},
    }
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 255, size=(4, 5, 3), dtype=np.uint8) for _ in range(3)]
    writer = LeRobotV3Writer.create(repo_id="a/b", fps=5, features=feats, root=tmp_path / "ds", use_videos=False)
    for i, img in enumerate(images):
        writer.add_frame({CAM: img, "observation.state": [i, -i], CONTROL: "teleop", "task": "t"})
    writer.save_episode()
    writer.finalize()

    info = json.loads((tmp_path / "ds/meta/info.json").read_text())
    assert info["video_path"] is None
    back = list(LeRobotV3Dataset(tmp_path / "ds").iter_frames(0))
    for img, frame in zip(images, back):
        assert np.array_equal(frame[CAM], img)
    assert [f["timestamp"] for f in back] == pytest.approx([0.0, 0.2, 0.4])
    raw = pq.read_table(tmp_path / "ds/data/chunk-000/file-000.parquet").to_pylist()
    assert raw[0][CAM]["path"] == "frame-000000.png"


def test_writer_accepts_chw_float_images(tmp_path):
    feats = {CAM: {"dtype": "image", "shape": (3, 4, 5), "names": None}}
    writer = LeRobotV3Writer.create(repo_id="a/b", fps=5, features=feats, root=tmp_path / "ds", use_videos=False)
    writer.add_frame({CAM: np.full((3, 4, 5), 1.0, dtype=np.float32), "task": "t"})
    writer.save_episode()
    writer.finalize()
    (frame,) = LeRobotV3Dataset(tmp_path / "ds").iter_frames(0)
    assert int(frame[CAM].min()) == 255


@pytest.mark.parametrize(
    "frame, match",
    [
        ({"x": [1.0], "task": "t", "timestamp": 0.0}, "computed automatically"),
        ({"task": "t"}, "missing features"),
        ({"x": [1.0, 2.0], "task": "t"}, "expected shape"),
        ({"x": [1.0]}, "task"),
        ({"x": [1.0], "y": 1, "task": "t"}, "not features"),
    ],
)
def test_writer_rejects_bad_frames(tmp_path, frame, match):
    writer = LeRobotV3Writer.create(
        repo_id="a/b", fps=5, features={"x": {"dtype": "float32", "shape": (1,), "names": None}}, root=tmp_path / "d"
    )
    with pytest.raises(ValueError, match=match):
        writer.add_frame(frame)


def test_writer_rejects_reserved_and_bad_features(tmp_path):
    with pytest.raises(ValueError, match="reserved"):
        LeRobotV3Writer.create(
            repo_id="a/b", fps=5, features={"index": {"dtype": "int64", "shape": (1,)}}, root=tmp_path / "a"
        )
    with pytest.raises(ValueError, match="'/'"):
        LeRobotV3Writer.create(
            repo_id="a/b", fps=5, features={"a/b": {"dtype": "int64", "shape": (1,)}}, root=tmp_path / "b"
        )
    with pytest.raises(ValueError, match="use_videos"):
        LeRobotV3Writer.create(
            repo_id="a/b",
            fps=5,
            features={CAM: {"dtype": "video", "shape": (3, 4, 4)}},
            root=tmp_path / "c",
            use_videos=False,
        )


def test_writer_refuses_non_empty_root(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "f").write_text("")
    with pytest.raises(FileExistsError):
        LeRobotV3Writer.create(repo_id="a/b", fps=5, features={}, root=tmp_path / "x")


def test_writer_finalize_drops_unsaved_frames_but_keeps_saved_episodes(tmp_path):
    writer = LeRobotV3Writer.create(
        repo_id="a/b", fps=5, features={"x": {"dtype": "float32", "shape": (1,), "names": None}}, root=tmp_path / "d"
    )
    writer.add_frame({"x": [1.0], "task": "t"})
    writer.save_episode()
    writer.add_frame({"x": [2.0], "task": "t"})
    with pytest.warns(UserWarning, match="discarding 1 unsaved"):
        writer.finalize()
    ds = LeRobotV3Dataset(tmp_path / "d")
    assert ds.total_episodes == 1
    assert [f["x"].tolist() for f in ds.iter_frames(0)] == [[1.0]]


def test_core_imports_without_torch_or_lerobot():
    code = (
        "import sys, avala.converters.lerobot_v3 as m;"
        "import avala.converters.lerobot_v3.reader, avala.converters.lerobot_v3.writer;"
        "bad = [n for n in ('torch', 'lerobot', 'datasets') if n in sys.modules];"
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# ── cross-check against the real library (Python 3.12 + lerobot only) ──
@needs_av
@pytest.mark.skipif(
    importlib.util.find_spec("lerobot") is None,
    reason="lerobot library not installed (needs Python 3.12 + avala[lerobot]); cross-check runs where it is",
)
@pytest.mark.parametrize("as_images", [False, True])
def test_writer_output_loads_in_lerobot(tmp_path, as_images):
    from lerobot.datasets import LeRobotDataset

    out = _copy_fixture(tmp_path / "ours", as_images=as_images)
    ds = LeRobotDataset("avala/test", root=out, video_backend="pyav")
    assert len(ds) == 7
    assert ds.meta.total_episodes == 2
    sample = ds[5]
    assert int(sample["episode_index"]) == 1 and int(sample["frame_index"]) == 1
    assert sample[CONTROL] == "policy"
    assert sample["failure_type"] == "grasp_slip"
    assert sample["task"] == "place the cube in the bin"
    assert tuple(sample[CAM].shape) == (3, 64, 64)
    assert sample["observation.state"].tolist() == pytest.approx([1.1, 1.35, 1.6, 1.85, 2.1, 2.35])
