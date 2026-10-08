"""Regenerate the committed channels-last fixture with the REAL ``lerobot`` library.

``ref_dataset_hwc/`` is shaped like the Hub dataset ``lerobot/pusht`` at revision
``7628202a2180972f291ba1bc6723834921e72c19``, whose ``meta/info.json`` (sha256
``0b46becc21ad92b117da1e95daf44110bb21d0bde75638b4ad167c8ccd959d38``) declares::

    "observation.image": {"dtype": "video", "shape": [96, 96, 3],
                          "names": ["height", "width", "channel"]}

i.e. channels-LAST, unlike ``ref_dataset/`` (``[3, 64, 64]``). lerobot 0.5.1 writes
camera features this way itself (``lerobot/datasets/feature_utils.py:129-134``). The
feature set, names and dtypes below copy pusht's; only the length and codec differ.

The image is deliberately asymmetric (red follows the row, green the column), so a
height/width swap or a channel/axis transpose anywhere in a round trip changes pixels.

Run (needs Python 3.12 + ``pip install 'lerobot>=0.5,<0.6'``)::

    python tests/fixtures/lerobot_v3/generate_hwc_fixture.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from lerobot.datasets import LeRobotDataset

ROOT = Path(__file__).parent / "ref_dataset_hwc"
SIZE = 96
MOTORS = {"motors": ["motor_0", "motor_1"]}

FEATURES = {
    "observation.image": {"dtype": "video", "shape": (SIZE, SIZE, 3), "names": ["height", "width", "channel"]},
    "observation.state": {"dtype": "float32", "shape": (2,), "names": MOTORS},
    "action": {"dtype": "float32", "shape": (2,), "names": MOTORS},
    "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
    "next.done": {"dtype": "bool", "shape": (1,), "names": None},
    "next.success": {"dtype": "bool", "shape": (1,), "names": None},
}

EPISODES = [
    ("Push the T-shaped block onto the T-shaped target.", 4),
    ("Push the T-shaped block onto the T-shaped target.", 3),
]


def frame_image(ep: int, i: int) -> np.ndarray:
    rows = np.linspace(0, 255, SIZE, dtype=np.float32)[:, None]
    cols = np.linspace(0, 255, SIZE, dtype=np.float32)[None, :]
    img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    img[:, :, 0] = np.broadcast_to(rows, (SIZE, SIZE)).astype(np.uint8)
    img[:, :, 1] = np.broadcast_to(cols, (SIZE, SIZE)).astype(np.uint8)
    img[:, :, 2] = 40 * ep + 20 * i
    return img


def main() -> None:
    shutil.rmtree(ROOT, ignore_errors=True)
    ds = LeRobotDataset.create(
        repo_id="avala-fixtures/lerobot-v3-hwc",
        fps=10,
        features=FEATURES,
        root=ROOT,
        use_videos=True,
        vcodec="h264",
    )
    for ep, (task, length) in enumerate(EPISODES):
        for i in range(length):
            last = i == length - 1
            ds.add_frame(
                {
                    "observation.image": frame_image(ep, i),
                    "observation.state": np.array([100.0 + 10 * i, 200.0 - 5 * ep], dtype=np.float32),
                    "action": np.array([110.0 + 10 * i, 190.0 + ep], dtype=np.float32),
                    "next.reward": np.array([0.25 * i], dtype=np.float32),
                    "next.done": np.array([last], dtype=bool),
                    "next.success": np.array([last and ep == 0], dtype=bool),
                    "task": task,
                }
            )
        ds.save_episode()
    ds.finalize()


if __name__ == "__main__":
    main()
