"""MCAP -> LeRobot v3 for the teleop recording layout, generated in-test.

Modelled on Avala's robot-rig recordings: three H.264 ``foxglove.CompressedVideo`` cameras
(``/camera/{high,left_wrist,right_wrist}/video``, Annex-B, keyframes carry SPS/PPS, no
B-frames, 30 fps), ``/joint_states`` as ROS 2 ``sensor_msgs/msg/JointState`` (CDR), an
optional ``/joint_commands`` topic, and the task text as a ``foxglove.Log`` named ``task``.
"""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import pytest

for _mod in ("numpy", "pyarrow", "PIL", "av", "mcap.writer", "mcap_protobuf.writer", "foxglove_schemas_protobuf"):
    pytest.importorskip(_mod)
pytest.importorskip("mcap_ros2.decoder")

import av  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from avala.converters.lerobot_v3.mcap import mcap_to_lerobot  # noqa: E402
from avala.converters.lerobot_v3.reader import LeRobotV3Dataset  # noqa: E402

FPS = 30
W, H = 64, 48
CAMERAS = ("high", "left_wrist", "right_wrist")
JOINTS = ["left_shoulder", "left_elbow", "right_shoulder", "right_elbow"]
JOINT_STATE_MSG = """std_msgs/Header header
string[] name
float64[] position
float64[] velocity
float64[] effort
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"""
T0 = 1_700_000_000_000_000_000


def _h264_access_units(n: int, cam: int) -> list:
    """Encode ``n`` distinct frames with x264 (no B-frames) and split into Annex-B access units."""
    buf = io.BytesIO()
    out = av.open(buf, "w", format="h264")
    stream = out.add_stream(
        "libx264", rate=FPS, options={"crf": "18", "x264-params": "keyint=10:min-keyint=10:scenecut=0:bframes=0"}
    )
    stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
    for i in range(n):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[:, :, 0] = (20 + 9 * i) % 256
        img[:, :, 1] = 60 * cam
        img[:, : (i % W) + 1, 2] = 200
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        frame.pts = i
        for packet in stream.encode(frame):
            out.mux(packet)
    for packet in stream.encode():
        out.mux(packet)
    out.close()
    parser = av.CodecContext.create("h264", "r")
    units = [bytes(p) for p in parser.parse(buf.getvalue())] + [bytes(p) for p in parser.parse(None)]
    assert len(units) == n
    return units


def _decode(units: list) -> list:
    ctx = av.CodecContext.create("h264", "r")
    frames = []
    for unit in units:
        frames += [f.to_ndarray(format="rgb24") for f in ctx.decode(av.Packet(unit))]
    return frames + [f.to_ndarray(format="rgb24") for f in ctx.decode(None)]


def _joint_value(i: int, kind: str) -> list:
    base = {"position": 0.25, "velocity": 0.5, "command": -0.125}[kind]
    return [base * i + 0.0625 * j for j in range(len(JOINTS))]


def write_teleop_mcap(
    path: Path,
    *,
    n: int = 20,
    joint_offset_ns: int = -1_000_000,
    with_commands: bool = True,
    camera_jitter_ns: int = 0,
    frame_interval_ns: int = 1_000_000_000 // FPS,
    shuffled_joint_names: bool = False,
) -> dict:
    """Write one teleop-style episode; returns the source values the export must reproduce."""
    from foxglove_schemas_protobuf.CompressedVideo_pb2 import CompressedVideo
    from foxglove_schemas_protobuf.Log_pb2 import Log
    from google.protobuf.struct_pb2 import Struct
    from mcap_protobuf.writer import Writer
    from mcap_ros2._dynamic import serialize_dynamic

    encode_joint = serialize_dynamic("sensor_msgs/msg/JointState", JOINT_STATE_MSG)["sensor_msgs/msg/JointState"]
    units = {cam: _h264_access_units(n, k) for k, cam in enumerate(CAMERAS)}
    times = [T0 + i * frame_interval_ns + (camera_jitter_ns if i % 2 else 0) for i in range(n)]
    with open(path, "wb") as fh, Writer(fh) as writer:
        raw = writer._writer
        schema_id = raw.register_schema("sensor_msgs/msg/JointState", "ros2msg", JOINT_STATE_MSG.encode())
        joint_channel = raw.register_channel("/joint_states", "cdr", schema_id)
        log = Log()
        log.timestamp.FromNanoseconds(T0)
        log.name, log.message = "task", "fold the towel"
        writer.write_message("/annotations/text", log, log_time=T0, publish_time=T0)
        for i, t in enumerate(times):
            for cam in CAMERAS:
                msg = CompressedVideo()
                msg.timestamp.FromNanoseconds(t)
                msg.frame_id, msg.format, msg.data = cam, "h264", units[cam][i]
                writer.write_message(f"/camera/{cam}/video", msg, log_time=t, publish_time=t)
            tj = t + joint_offset_ns
            names = list(reversed(JOINTS)) if shuffled_joint_names and i % 2 else JOINTS
            order = [JOINTS.index(nm) for nm in names]
            position = [_joint_value(i, "position")[k] for k in order]
            velocity = [_joint_value(i, "velocity")[k] for k in order]
            payload = {
                "header": {"stamp": {"sec": tj // 10**9, "nanosec": tj % 10**9}, "frame_id": "base"},
                "name": names,
                "position": position,
                "velocity": velocity,
                "effort": [],
            }
            raw.add_message(joint_channel, log_time=tj, data=encode_joint(payload), publish_time=tj)
            if with_commands:
                cmd = Struct()  # the Struct mirror the Avala ROS importer writes
                cmd.update({"name": JOINTS, "position": _joint_value(i, "command"), "velocity": [], "effort": []})
                writer.write_message("/joint_commands", cmd, log_time=tj, publish_time=tj)
    return {"units": units, "times": times}


def _rows(out: Path) -> list:
    return pq.read_table(out / "data/chunk-000/file-000.parquet").to_pylist()


def test_teleop_mcap_exports_state_action_task_and_copies_h264(tmp_path):
    src = write_teleop_mcap(tmp_path / "ep0.mcap")
    out = mcap_to_lerobot([tmp_path / "ep0.mcap"], tmp_path / "lerobot", repo_id="avala/teleop")

    info = json.loads((out / "meta/info.json").read_text())
    feats = info["features"]
    assert info["fps"] == FPS
    assert [k for k in feats if k.startswith("observation.images.")] == [
        "observation.images.high",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    ]
    assert feats["observation.state"] == {"dtype": "float32", "shape": [4], "names": JOINTS}
    assert feats["observation.velocity"]["shape"] == [4]
    assert "observation.effort" not in feats  # empty arrays are not invented into a feature
    assert feats["action"] == {"dtype": "float32", "shape": [4], "names": JOINTS}
    assert feats["observation.images.high"]["info"]["video.codec"] == "h264"

    rows = _rows(out)
    assert len(rows) == 20
    for i, row in enumerate(rows):
        assert row["observation.state"] == pytest.approx(_joint_value(i, "position"), abs=0)
        assert row["observation.velocity"] == pytest.approx(_joint_value(i, "velocity"), abs=0)
        assert row["action"] == pytest.approx(_joint_value(i, "command"), abs=0)
    tasks = pq.read_table(out / "meta/tasks.parquet").to_pylist()
    assert tasks == [{"task_index": 0, "task": "fold the towel"}]

    report = json.loads((out / "meta/avala_export_report.json").read_text())
    (episode,) = report["episodes"]
    assert episode["video"] == {f"observation.images.{c}": "passthrough" for c in CAMERAS}
    assert episode["dropped_leading_frames"] == 0
    assert episode["max_timing_skew_s"] < 1e-6

    # Copied, not re-encoded: decoding the mp4 gives exactly the source bitstream's pictures.
    with LeRobotV3Dataset(out) as ds:
        for k, cam in enumerate(CAMERAS):
            expected = _decode(src["units"][cam])
            got = [f[f"observation.images.{cam}"] for f in ds.iter_frames(0, keys=[f"observation.images.{cam}"])]
            assert len(got) == len(expected)
            assert all(np.array_equal(a, b) for a, b in zip(got, expected)), cam


def test_no_command_topic_means_no_action(tmp_path):
    write_teleop_mcap(tmp_path / "ep.mcap", with_commands=False)
    out = mcap_to_lerobot([tmp_path / "ep.mcap"], tmp_path / "lerobot", repo_id="a/b")
    feats = json.loads((out / "meta/info.json").read_text())["features"]
    assert "observation.state" in feats
    assert "action" not in feats  # never derived from the state


def test_joint_names_reordered_by_name(tmp_path):
    write_teleop_mcap(tmp_path / "ep.mcap", shuffled_joint_names=True)
    out = mcap_to_lerobot([tmp_path / "ep.mcap"], tmp_path / "lerobot", repo_id="a/b")
    for i, row in enumerate(_rows(out)):
        assert row["observation.state"] == pytest.approx(_joint_value(i, "position"), abs=0)


def test_leading_frames_without_state_are_dropped_and_video_reencoded(tmp_path):
    # joints start 2 ms after the first camera frame: frame 0 has no state and is dropped,
    # so the first kept packet is not a keyframe and the cameras must be re-encoded.
    write_teleop_mcap(tmp_path / "ep.mcap", joint_offset_ns=2_000_000)
    with pytest.warns(UserWarning, match="re-encoding /camera/high/video: the first kept packet is not a keyframe"):
        out = mcap_to_lerobot([tmp_path / "ep.mcap"], tmp_path / "lerobot", repo_id="a/b")
    rows = _rows(out)
    assert len(rows) == 19
    # frame k holds the state published just before camera frame k+1 (sample-and-hold)
    assert rows[0]["observation.state"] == pytest.approx(_joint_value(0, "position"), abs=0)
    report = json.loads((out / "meta/avala_export_report.json").read_text())["episodes"][0]
    assert report["dropped_leading_frames"] == 1
    assert set(report["video"].values()) == {"reencoded"}


def test_recorded_jitter_is_measured_and_large_drift_warned(tmp_path):
    write_teleop_mcap(tmp_path / "jitter.mcap", camera_jitter_ns=4_000_000)
    out = mcap_to_lerobot([tmp_path / "jitter.mcap"], tmp_path / "a", repo_id="a/b", fps=FPS)
    skew = json.loads((out / "meta/avala_export_report.json").read_text())["episodes"][0]["max_timing_skew_s"]
    assert skew == pytest.approx(0.004, abs=1e-6)

    # recorded at 25 fps but exported as 30 fps: the grid drifts past half a frame
    write_teleop_mcap(tmp_path / "slow.mcap", frame_interval_ns=40_000_000)
    with pytest.warns(UserWarning, match="deviates from frame_index/fps"):
        out = mcap_to_lerobot([tmp_path / "slow.mcap"], tmp_path / "b", repo_id="a/b", fps=FPS)
    skew = json.loads((out / "meta/avala_export_report.json").read_text())["episodes"][0]["max_timing_skew_s"]
    assert skew == pytest.approx(19 * (0.040 - 1 / FPS), abs=1e-6)


def test_passthrough_can_be_disabled(tmp_path):
    write_teleop_mcap(tmp_path / "ep.mcap", n=8)
    with pytest.warns(UserWarning, match="pass-through disabled"):
        out = mcap_to_lerobot([tmp_path / "ep.mcap"], tmp_path / "lerobot", repo_id="a/b", passthrough_video=False)
    report = json.loads((out / "meta/avala_export_report.json").read_text())["episodes"][0]
    assert set(report["video"].values()) == {"reencoded"}


def test_two_episodes_each_get_their_own_passthrough_file(tmp_path):
    for k in range(2):
        write_teleop_mcap(tmp_path / f"ep{k}.mcap", n=6)
    out = mcap_to_lerobot([tmp_path / "ep0.mcap", tmp_path / "ep1.mcap"], tmp_path / "lerobot", repo_id="a/b")
    eps = pq.read_table(out / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    key = "videos/observation.images.high"
    assert [(e[f"{key}/file_index"], e[f"{key}/from_timestamp"], e[f"{key}/to_timestamp"]) for e in eps] == [
        (0, 0.0, pytest.approx(6 / FPS)),
        (1, 0.0, pytest.approx(6 / FPS)),
    ]
    with LeRobotV3Dataset(out) as ds:
        assert [len(list(ds.iter_frames(e, keys=["observation.images.high"]))) for e in (0, 1)] == [6, 6]


@pytest.mark.skipif(
    importlib.util.find_spec("lerobot") is None,
    reason="lerobot library not installed (needs Python 3.12 + avala[lerobot]); cross-check runs where it is",
)
def test_teleop_export_loads_in_lerobot(tmp_path):
    from lerobot.datasets import LeRobotDataset

    for k in range(2):
        write_teleop_mcap(tmp_path / f"ep{k}.mcap", n=10)
    out = mcap_to_lerobot([tmp_path / "ep0.mcap", tmp_path / "ep1.mcap"], tmp_path / "lerobot", repo_id="a/b")
    ds = LeRobotDataset("avala/teleop", root=out, video_backend="pyav")
    assert len(ds) == 20
    sample = ds[13]
    assert int(sample["episode_index"]) == 1 and int(sample["frame_index"]) == 3
    assert sample["observation.state"].tolist() == pytest.approx(_joint_value(3, "position"))
    assert sample["action"].tolist() == pytest.approx(_joint_value(3, "command"))
    assert sample["task"] == "fold the towel"
    assert tuple(sample["observation.images.left_wrist"].shape) == (3, H, W)
