"""Round-trip fidelity: LeRobot v3 -> Avala MCAP -> LeRobot v3 (target: >= 95% of checks).

Source: ``fixtures/lerobot_v3/ref_dataset``, written by the real ``lerobot`` 0.5.1 library
with a per-frame ``annotation.vendor.control_source`` column plus ``episode_uuid``
and ``failure_type``. Path under test: ``avala.importers.lerobot.import_lerobot``
(backend="core", the real import path up to the upload) -> ``.mcap`` per episode ->
``avala.converters.lerobot_v3.mcap.mcap_to_lerobot`` -> a new v3 dataset.

Both sides are compared through pyarrow/json directly (not through our reader), field by
field. Every lossless field must match exactly; the camera is h264 on both ends (and JPEG
in the MCAP), so pixels are checked against a tolerance instead and counted separately.
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
from avala.converters.lerobot_v3.mcap import mcap_to_lerobot  # noqa: E402
from avala.converters.lerobot_v3.reader import LeRobotV3Dataset  # noqa: E402
from avala.importers import import_lerobot  # noqa: E402
from avala.types.dataset import Dataset  # noqa: E402

REF = Path(__file__).parent / "fixtures" / "lerobot_v3" / "ref_dataset"
CAM = "observation.images.top"
PIXEL_TOLERANCE = 6.0  # mean absolute error per channel, 0-255 scale (two lossy encodes)


def _import_to_mcaps(monkeypatch, dest: Path) -> list:
    client = Client(api_key="test-key")

    def _capture(*, source, name, slug, data_type, **_kwargs):
        shutil.copytree(source, dest)
        return Dataset(uid="d", name=name, slug=slug, data_type=data_type, item_count=0)

    monkeypatch.setattr(client.datasets, "create_from_local", _capture)
    import_lerobot(client, root=str(REF), name="ref", slug="ref", backend="core")
    client.close()
    return sorted(dest.glob("*.mcap"))


class _Ledger:
    def __init__(self) -> None:
        self.passed = 0
        self.failed: list = []

    def check(self, name: str, ok: bool) -> None:
        if ok:
            self.passed += 1
        else:
            self.failed.append(name)

    @property
    def total(self) -> int:
        return self.passed + len(self.failed)


@pytest.fixture(scope="module")
def roundtrip(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    try:
        tmp = tmp_path_factory.mktemp("rt")
        mcaps = _import_to_mcaps(mp, tmp / "mcap")
        out = mcap_to_lerobot(mcaps, tmp / "lerobot", repo_id="avala/roundtrip")
    finally:
        mp.undo()
    return mcaps, out


def test_roundtrip_fidelity(roundtrip):
    mcaps, out = roundtrip
    assert [p.name for p in mcaps] == ["episode_000000.mcap", "episode_000001.mcap"]
    ledger = _Ledger()
    src_info = json.loads((REF / "meta/info.json").read_text())
    dst_info = json.loads((out / "meta/info.json").read_text())

    # declared metadata
    for key in ("codebase_version", "fps", "robot_type", "total_episodes", "total_frames", "total_tasks", "splits"):
        ledger.check(f"info.{key}", src_info[key] == dst_info[key])
    ledger.check("feature order", list(src_info["features"]) == list(dst_info["features"]))
    for name, spec in src_info["features"].items():
        other = dst_info["features"].get(name, {})
        for attr in ("dtype", "shape", "names"):
            ledger.check(f"feature {name}.{attr}", spec.get(attr) == other.get(attr))

    # tasks
    src_tasks = pq.read_table(REF / "meta/tasks.parquet").to_pylist()
    dst_tasks = pq.read_table(out / "meta/tasks.parquet").to_pylist()
    ledger.check("tasks", src_tasks == dst_tasks)

    # episode boundaries
    src_eps = pq.read_table(REF / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    dst_eps = pq.read_table(out / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    ledger.check("episode count", len(src_eps) == len(dst_eps))
    for s, d in zip(src_eps, dst_eps):
        for key in ("episode_index", "tasks", "length", "dataset_from_index", "dataset_to_index"):
            ledger.check(f"episode {s['episode_index']}.{key}", s[key] == d[key])
        for key in (f"videos/{CAM}/from_timestamp", f"videos/{CAM}/to_timestamp"):
            ledger.check(f"episode {s['episode_index']}.{key}", abs(s[key] - d[key]) < 1e-6)

    # every lossless per-frame value, bit-exact (timestamps, indices, numeric, categorical)
    src_rows = pq.read_table(REF / "data/chunk-000/file-000.parquet").to_pylist()
    dst_rows = pq.read_table(out / "data/chunk-000/file-000.parquet").to_pylist()
    ledger.check("frame count", len(src_rows) == len(dst_rows))
    for s, d in zip(src_rows, dst_rows):
        for key, value in s.items():
            ledger.check(f"frame {s['index']}.{key}", d.get(key) == value)
    lossless_total = ledger.total

    # camera pixels (lossy): decoded frame vs decoded source frame
    pixel_errors = []
    with LeRobotV3Dataset(REF) as src, LeRobotV3Dataset(out) as dst:
        for ep in range(src.total_episodes):
            for a, b in zip(src.iter_frames(ep, keys=[CAM]), dst.iter_frames(ep, keys=[CAM])):
                err = float(np.abs(a[CAM].astype(float) - b[CAM].astype(float)).mean())
                pixel_errors.append(err)
                ledger.check(
                    f"pixels ep{ep} f{a['frame_index']}", a[CAM].shape == b[CAM].shape and err <= PIXEL_TOLERANCE
                )

    fidelity = ledger.passed / ledger.total
    print(
        f"\nLeRobot v3 round-trip fidelity: {ledger.passed}/{ledger.total} = {fidelity:.1%} "
        f"({lossless_total} lossless checks, {len(pixel_errors)} pixel checks, "
        f"max mean-abs pixel error {max(pixel_errors):.2f}/255)"
    )
    assert not ledger.failed, ledger.failed
    assert fidelity >= 0.95


def test_roundtrip_preserves_control_source_through_the_mcap(roundtrip):
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    mcaps, _out = roundtrip
    raw = pq.read_table(REF / "data/chunk-000/file-000.parquet").to_pylist()
    seen = []
    for path in mcaps:
        with open(path, "rb") as fh:
            reader = make_reader(fh, decoder_factories=[DecoderFactory()])
            seen += [
                m.decoded_message["data"] for m in reader.iter_decoded_messages(topics=["/lerobot/control_source"])
            ]
    assert seen == [r["annotation.vendor.control_source"] for r in raw]


@pytest.mark.skipif(
    importlib.util.find_spec("lerobot") is None,
    reason="lerobot library not installed (needs Python 3.12 + avala[lerobot]); cross-check runs where it is",
)
def test_roundtrip_output_loads_in_lerobot(roundtrip):
    from lerobot.datasets import LeRobotDataset

    _mcaps, out = roundtrip
    ds = LeRobotDataset("avala/roundtrip", root=out, video_backend="pyav")
    assert len(ds) == 7
    assert [ds[i]["annotation.vendor.control_source"] for i in range(7)] == [
        "policy",
        "policy",
        "intervention",
        "teleop",
        "hold",
        "policy",
        "policy",
    ]
    assert ds[4]["failure_type"] == "grasp_slip"
