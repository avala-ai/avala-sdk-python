"""Regenerate the committed LeRobot v3 reference fixture with the REAL ``lerobot`` library.

The fixture under ``ref_dataset/`` is the independent oracle for the torch-free
``avala.converters.lerobot_v3`` reader/writer: it was written by
``lerobot.datasets.LeRobotDataset`` (lerobot 0.5.1, codebase_version v3.0), not by
our own writer, so tests that read it are not self-confirming.

It mimics a third-party recording with a per-frame ``control_source`` column stored as
``annotation.vendor.control_source`` plus ``episode_uuid`` and ``failure_type``.

Run (needs Python 3.12 + ``pip install 'lerobot>=0.5,<0.6'``)::

    python tests/fixtures/lerobot_v3/generate_fixture.py

h264 is used instead of lerobot's default ``libsvtav1`` because it keeps the file tiny
and decodes everywhere; the layout is identical.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from lerobot.datasets import LeRobotDataset

ROOT = Path(__file__).parent / "ref_dataset"

FEATURES = {
    "observation.images.top": {"dtype": "video", "shape": (3, 64, 64), "names": ["channels", "height", "width"]},
    "observation.state": {"dtype": "float32", "shape": (6,), "names": [f"joint_{i}" for i in range(6)]},
    "action": {"dtype": "float32", "shape": (6,), "names": [f"joint_{i}" for i in range(6)]},
    "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
    "annotation.vendor.control_source": {"dtype": "string", "shape": (1,), "names": None},
    "episode_uuid": {"dtype": "string", "shape": (1,), "names": None},
    "failure_type": {"dtype": "string", "shape": (1,), "names": None},
}

# (task, control sources per frame, episode uuid, failure type)
EPISODES = [
    ("pick up the red cube", ["policy", "policy", "intervention", "teleop"], "6f1c7a52-ep0", "none"),
    ("place the cube in the bin", ["hold", "policy", "policy"], "a93be0d1-ep1", "grasp_slip"),
]


def main() -> None:
    shutil.rmtree(ROOT, ignore_errors=True)
    ds = LeRobotDataset.create(
        repo_id="avala-fixtures/lerobot-v3-ref",
        fps=10,
        features=FEATURES,
        root=ROOT,
        robot_type="so101",
        use_videos=True,
        vcodec="h264",
    )
    for ep, (task, sources, uuid, failure) in enumerate(EPISODES):
        for i, source in enumerate(sources):
            level = 30 + 60 * i
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            frame[:, :, 0] = level
            frame[:, :, 1] = 255 - level
            frame[:, :, 2] = 40 * ep
            ds.add_frame(
                {
                    "observation.images.top": frame,
                    "observation.state": np.arange(6, dtype=np.float32) * 0.25 + ep + i * 0.1,
                    "action": -np.arange(6, dtype=np.float32) * 0.5 + i,
                    "next.reward": np.array([0.5 * i], dtype=np.float32),
                    "annotation.vendor.control_source": source,
                    "episode_uuid": uuid,
                    "failure_type": failure,
                    "task": task,
                }
            )
        ds.save_episode()
    ds.finalize()


if __name__ == "__main__":
    main()
