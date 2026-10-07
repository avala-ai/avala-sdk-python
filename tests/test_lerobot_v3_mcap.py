"""Topic planning for the LeRobot <-> MCAP bridge (``avala.converters.lerobot_v3.mcap``)."""

from __future__ import annotations

import pytest
from avala.converters.lerobot_v3.mcap import (
    CONTROL_SOURCE_TOPIC,
    TASK_TOPIC,
    build_frame,
    plan_topics,
)

pytest.importorskip("numpy")

import numpy as np  # noqa: E402

F = {
    "observation.images.top": {"dtype": "video", "shape": (3, 4, 4)},
    "observation.images.wrist": {"dtype": "video", "shape": (3, 4, 4)},
    "observation.state": {"dtype": "float32", "shape": (2,)},
    "action": {"dtype": "float32", "shape": (2,)},
    "annotation.vendor.control_source": {"dtype": "string", "shape": (1,)},
    "annotation.other.control_source": {"dtype": "string", "shape": (1,)},
    "is_success": {"dtype": "bool", "shape": (1,)},
    "timestamp": {"dtype": "float32", "shape": (1,)},
    "index": {"dtype": "int64", "shape": (1,)},
}


def test_plan_keeps_state_topics_and_carries_everything_else():
    plan = plan_topics(F, ["observation.images.top"], ["observation.state", "action"])
    assert plan.cameras == {"observation.images.top": "/observation/images/top"}  # unselected camera not carried
    assert plan.numeric == {"observation.state": "/observation/state", "action": "/action", "is_success": "/is_success"}
    # first control_source column -> stable topic, the second keeps its own path
    assert plan.strings == {
        "annotation.vendor.control_source": CONTROL_SOURCE_TOPIC,
        "annotation.other.control_source": "/annotation/other/control_source",
    }
    assert "timestamp" not in plan.topics and "index" not in plan.topics


def test_plan_records_features_in_source_order():
    # state keys listed in a different order than the dataset declares them
    plan = plan_topics(F, ["observation.images.top"], ["action", "observation.state"])
    assert list(plan.specs) == [
        "observation.images.top",
        "observation.state",
        "action",
        "annotation.vendor.control_source",
        "annotation.other.control_source",
        "is_success",
    ]


def test_plan_rejects_non_numeric_state_key_and_topic_collisions():
    with pytest.raises(ValueError, match="string feature"):
        plan_topics(F, [], ["annotation.vendor.control_source"])
    with pytest.raises(ValueError, match="topic '/lerobot/task'"):
        plan_topics({"lerobot.task": {"dtype": "string", "shape": (1,)}}, [])
    with pytest.raises(ValueError, match="no dtype"):
        plan_topics({"x": 1}, [])


def test_build_frame_encodes_values():
    plan = plan_topics(F, [], ["observation.state"])
    frame = build_frame(
        {
            "observation.state": np.array([0.5, 1.5], dtype=np.float32),
            "action": np.array([1, 2]),
            "annotation.vendor.control_source": "teleop",
            "annotation.other.control_source": np.array("hold"),
            "is_success": np.array([True]),
            "timestamp": np.float32(0.2),
            "task": "stack",
        },
        plan,
        fps=10,
    )
    assert frame["timestamp_ns"] == int(float(np.float32(0.2)) * 1e9)
    assert frame["structs"]["/observation/state"] == {"data": [0.5, 1.5]}
    assert frame["structs"]["/action"] == {"data": [1.0, 2.0]}
    assert frame["structs"][CONTROL_SOURCE_TOPIC] == {"data": "teleop"}
    assert frame["structs"]["/annotation/other/control_source"] == {"data": "hold"}
    assert frame["structs"]["/is_success"] == {"data": [True]}
    assert frame["structs"][TASK_TOPIC] == {"data": "stack"}


def test_build_frame_rejects_non_string_categorical():
    plan = plan_topics(F, [], [])
    with pytest.raises(ValueError, match="non-string"):
        build_frame({"annotation.vendor.control_source": 3, "timestamp": 0.0}, plan, fps=10)


# ── mcap_to_lerobot on MCAPs that were not written by the importer ──
def _mcap_deps():
    for mod in ("pyarrow", "PIL", "av", "mcap.reader", "mcap_protobuf.writer", "foxglove_schemas_protobuf"):
        pytest.importorskip(mod)


def _img(level):
    return np.full((8, 8, 3), level, dtype=np.uint8)


def test_mcap_to_lerobot_infers_schema_and_holds_values(tmp_path):
    _mcap_deps()
    from avala.converters.lerobot_v3.mcap import mcap_to_lerobot, write_episode_mcap
    from avala.converters.lerobot_v3.reader import LeRobotV3Dataset

    ms = 1_000_000
    frames = [
        {"timestamp_ns": 0, "images": {"/cam/front": _img(10)}, "structs": {CONTROL_SOURCE_TOPIC: {"data": "policy"}}},
        {"timestamp_ns": 50 * ms, "structs": {"/joints": {"data": [0.5, 1.0]}, "/noise": {"other": 1}}},
        {"timestamp_ns": 100 * ms, "images": {"/cam/front": _img(80)}},
        {"timestamp_ns": 150 * ms, "structs": {"/joints": {"data": [1.5, 2.0]}}},
        {
            "timestamp_ns": 200 * ms,
            "images": {"/cam/front": _img(150)},
            "structs": {CONTROL_SOURCE_TOPIC: {"data": "teleop"}},
        },
        {"timestamp_ns": 300 * ms, "images": {"/cam/front": _img(220)}},
    ]
    src = tmp_path / "rec.mcap"
    write_episode_mcap(src, frames)

    with pytest.warns(UserWarning, match="/noise"):
        root = mcap_to_lerobot([src], tmp_path / "out", repo_id="a/b", task="drive")
    ds = LeRobotV3Dataset(root)
    assert ds.fps == 10  # median camera interval
    assert set(ds.user_features) == {"observation.images.front", "joints", "annotation.avala.control_source"}
    got = list(ds.iter_frames(0))
    # t=0 is dropped (joints had not published yet); later frames hold the latest value
    assert [f["joints"].tolist() for f in got] == [[0.5, 1.0], [1.5, 2.0], [1.5, 2.0]]
    assert [f["annotation.avala.control_source"] for f in got] == ["policy", "teleop", "teleop"]
    assert [f["task"] for f in got] == ["drive"] * 3
    assert [int(f["observation.images.front"].mean()) for f in got] == pytest.approx([80, 150, 220], abs=4)


def test_mcap_to_lerobot_exact_mode_refuses_ragged_frames(tmp_path):
    _mcap_deps()
    from avala.converters.lerobot_v3.mcap import episode_metadata, mcap_to_lerobot, write_episode_mcap

    feats = {"x": {"dtype": "float32", "shape": (1,), "names": None}, "y": {"dtype": "float32", "shape": (1,)}}
    plan = plan_topics(feats, [])
    meta = episode_metadata(plan, fps=10, episode_index=0, robot_type=None, codebase_version="v3.0")
    frames = [
        {"timestamp_ns": 0, "structs": {"/x": {"data": [1.0]}, "/y": {"data": [2.0]}}},
        {"timestamp_ns": 100_000_000, "structs": {"/x": {"data": [3.0]}}},  # y missing
    ]
    src = tmp_path / "e.mcap"
    write_episode_mcap(src, frames, metadata=meta)
    # rejected before any frame is written (frames are aligned first)
    with pytest.raises(ValueError, match="no value for \\['y'\\]"):
        mcap_to_lerobot([src], tmp_path / "out", repo_id="a/b")


def test_mcap_to_lerobot_rejects_schema_change_between_episodes(tmp_path):
    _mcap_deps()
    from avala.converters.lerobot_v3.mcap import mcap_to_lerobot, write_episode_mcap

    a, b = tmp_path / "a.mcap", tmp_path / "b.mcap"
    write_episode_mcap(
        a, [{"timestamp_ns": t * 100_000_000, "structs": {"/q": {"data": [1.0, 2.0]}}} for t in range(3)]
    )
    write_episode_mcap(b, [{"timestamp_ns": t * 100_000_000, "structs": {"/q": {"data": [1.0]}}} for t in range(3)])
    with pytest.raises(ValueError, match="different feature schema"):
        mcap_to_lerobot([a, b], tmp_path / "out", repo_id="a/b")


@pytest.mark.parametrize("column", ["annotation.vendor_a.control_source", "annotation.labs-b.control_source"])
def test_any_vendor_segment_maps_to_the_stable_topic(column):
    plan = plan_topics({column: {"dtype": "string", "shape": (1,)}}, [])
    assert plan.strings == {column: CONTROL_SOURCE_TOPIC}
