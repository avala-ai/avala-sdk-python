"""Regenerate the Inspect Robots eval-log fixture used by ``tests/test_inspect_robots_import.py``.

Not collected by pytest and never run in CI. It runs Inspect Robots' own offline mock
(the ``cubepick`` toy world and the ``ScriptedPolicy`` oracle) through ``inspect_robots.eval``,
so every byte of the fixture is written by Inspect Robots itself, not hand-built.

Usage (Inspect Robots needs Python >= 3.10)::

    python -m venv /tmp/ir && /tmp/ir/bin/pip install 'inspect-robots==0.60.0'
    cd sdks/python/tests/fixtures/inspect_robots
    /tmp/ir/bin/python generate_fixture.py

Each run writes into its own directory exactly as Inspect Robots lays it out (the JSON log,
``actions/<run_id>/<trial>.jsonl`` side-cars and, for ``run_frames``, ``frames/<run_id>/*.npy``).
The script only renames the log file to ``eval_log.json`` and blanks ``git_commit``. Runs are
deterministic apart from timestamps and the run id.

Scenes are chosen to cover every outcome branch of the importer:

* ``reach-fast``   the oracle reaches the cube          -> success
* ``reach-slow``   half speed, runs out of steps        -> failure (``task_progress`` ~0.75)
* ``reach-stuck``  zero speed, never moves              -> failure
* ``reach-flaky``  epoch 1 raises ``PolicyError``       -> epoch 0 success, epoch 1 errored
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from inspect_robots import Scene, Score, Task, eval
from inspect_robots.errors import PolicyError
from inspect_robots.mock import CubePickEmbodiment, ScriptedPolicy

HERE = Path(__file__).resolve().parent


class _TaskProgress:
    """Continuous progress in [0, 1]: how much of the start-to-cube distance was closed."""

    name = "task_progress"

    def __call__(self, record, target):  # type: ignore[no-untyped-def]
        if not record.steps:
            return Score(value=0.0)
        first = record.steps[0].observation.state
        start = float(np.linalg.norm(np.asarray(first["eef_pos"]) - np.asarray(first["cube_pos"])))
        best = min(float(s.result.info["distance"]) for s in record.steps)
        return Score(value=float(np.clip(1.0 - best / start, 0.0, 1.0)) if start > 0 else 1.0)


class _VariableSpeedPolicy(ScriptedPolicy):
    """The Inspect Robots scripted oracle, slowed (or broken) per scene via scene metadata."""

    def __init__(self) -> None:
        super().__init__(chunk_size=4, max_step=0.1)
        self._epoch_of_scene: dict[str, int] = {}
        self._scene: Scene | None = None

    def reset(self, scene: Scene) -> None:
        super().reset(scene)
        self._scene = scene
        self._epoch_of_scene[scene.id] = self._epoch_of_scene.get(scene.id, -1) + 1
        self.max_step = float(scene.metadata.get("speed", 0.1))

    def act(self, observation):  # type: ignore[no-untyped-def]
        assert self._scene is not None
        if self._scene.metadata.get("fail_epoch") == self._epoch_of_scene[self._scene.id]:
            raise PolicyError("simulated inference server timeout")
        return super().act(observation)


def _run(name: str, task: Task, **kwargs: object) -> None:
    """Run one eval into ``<name>/`` and rename its log to ``<name>/eval_log.json``."""
    shutil.rmtree(HERE / name, ignore_errors=True)
    [log] = eval(
        task,
        _VariableSpeedPolicy(),
        CubePickEmbodiment(),
        log_dir=name,  # relative, as a user's CLI run would record it
        seed=0,
        policy_checkpoint="cubepick-oracle-v1",
        environment_id="cubepick-mock",
        **kwargs,  # type: ignore[arg-type]
    )
    [written] = sorted((HERE / name).glob("*.json"))
    data = json.loads(written.read_text())
    data["eval"]["git_commit"] = None  # the generator's checkout is not part of the fixture
    (HERE / name / "eval_log.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    written.unlink()
    print(f"{name}: status={log.status} trials={log.results.total_trials} errored={log.results.errored_trials}")


def main() -> None:
    import os

    os.chdir(HERE)
    scorers = ["success_at_end", _TaskProgress(), "episode_length"]
    _run(
        "run_main",
        Task(
            name="cubepick-reach-eval",
            scenes=[
                Scene(id="reach-fast", instruction="reach the cube", init_seed=1, metadata={"speed": 0.1}),
                Scene(id="reach-slow", instruction="reach the cube", init_seed=2, metadata={"speed": 0.05}),
                Scene(id="reach-stuck", instruction="reach the cube", init_seed=3, metadata={"speed": 0.0}),
                Scene(
                    id="reach-flaky",
                    instruction="reach the cube",
                    init_seed=4,
                    metadata={"speed": 0.1, "fail_epoch": 1},
                ),
            ],
            scorer=scorers,
            epochs=2,
            max_steps=12,
        ),
    )
    # A second, tiny run with camera frames stored, to cover image conversion.
    _run(
        "run_frames",
        Task(
            name="cubepick-reach-frames",
            scenes=[Scene(id="reach-fast", instruction="reach the cube", init_seed=1, metadata={"speed": 0.1})],
            scorer=scorers,
            max_steps=3,
        ),
        store_frames=True,
    )


if __name__ == "__main__":
    main()
