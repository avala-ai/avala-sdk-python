"""Import Inspect Robots evaluation logs as outcome-labelled sequences.

`Inspect Robots <https://github.com/robocurve/inspect-robots>`_ runs a policy on an
embodiment against a benchmark and writes one JSON ``EvalLog`` per run (schema version 1,
read back with ``inspect_robots.read_eval_log``). This importer turns every *trial* in that
log, i.e. one scene x one epoch, into an Avala sequence carrying a
:class:`~avala.types.sequence_outcome.SequenceOutcome` label, so failed trials can be
diagnosed and turned into collection work.

What an Inspect Robots log holds (verified against ``inspect-robots`` 0.60.0):

* ``eval``: the run spec: ``task``, ``policy``, ``policy_checkpoint``, ``embodiment``,
  ``embodiment_info.control_hz``, ``seed``, ``max_steps``, ``inspect_robots_version``.
* ``samples``: one ``SceneResult`` per scene. Its ``epochs`` list is strictly parallel to
  ``termination_reasons``, ``operator_judgements`` and ``trial_metadata``: entry *i* is
  trial (scene, epoch *i*). ``epochs[i]`` maps scorer name -> float. An errored or
  cancelled trial is recorded but never scored, so its entry is ``{}``; the scene's
  ``status`` (``success`` / ``error`` / ``cancelled``) and ``error`` say why.
* Trajectories are NOT in the log. ``trial_metadata[i]["actions"]`` points at a JSONL
  side-car ``actions/<run_id>/<trial>.jsonl`` (relative to the log directory) with one
  executed action per control step; camera frames, when the run stored them, are
  ``.npy`` files under ``stats.frames_dir`` (``frames/<run_id>/``). The log carries no
  run id field of its own: the ``<run_id>`` path segment is the run id.

Mapping (one row per trial):

==========================================  ==========================================
Inspect Robots                              SequenceOutcome
==========================================  ==========================================
success score >= ``success_threshold``,     ``expert_success``
score >= ``expert_threshold``
success score >= ``success_threshold``,     ``partial_success``
score < ``expert_threshold``
success score < ``success_threshold``       ``failure``
trial not scored (errored / cancelled)      ``aborted``
scored, but the success scorer has no       not labelled (``unscored``)
value (scorer failed / non-finite)
``eval.policy`` [+ ``@policy_checkpoint``]  ``model_version``
(always)                                    ``source="imported"``,
                                            ``evaluation_membership="held_out_eval"``,
                                            ``autonomy_level="autonomous"``
``eval.task``                               one subtask label spanning the trial
``progress_key`` score in [0, 1]            ``progress`` (only when requested)
==========================================  ==========================================

Every label carries its run provenance in ``source_metadata`` (see
:meth:`TrialMapping.source_metadata`): ``importer``, ``importer_version``, ``run_id``,
``task``, ``trial_id``, ``log_file`` (the log's file name, never a local path), ``epoch``
and ``scene``. The fuller per-trial record (scores, termination reason, operator
judgement, ...) is the row's ``metadata``; it is too large and too nested for the label,
so it goes into the trial's MCAP (``/inspect_robots/trial`` topic) when the importer
creates sequences, into the optional JSON receipt, and is shown by ``--dry-run``.

Sequences are matched or created in one of two ways:

* **attach** (default): the dataset already holds one sequence per trial. A sequence
  matches a trial when the last path segment of its ``key`` (minus ``.mcap``) equals the
  trial id ``<scene_id>-e<epoch>``, the same stem Inspect Robots gives the trial's action
  side-car. Trials without exactly one match are reported and skipped.
* **create**: each trial's action side-car (and camera frames, when stored) is converted
  to one ``<trial>/<trial>.mcap`` and uploaded as a new MCAP dataset, then attached as
  above. A trial with no recorded steps has nothing to convert; it stays a metadata-only
  row (``no_trace``) and is not labelled.

Reading the log needs the ``inspect`` extra: ``pip install 'avala[inspect]'``.
"""

from __future__ import annotations

import importlib
import json
import math
import re
import zlib
from dataclasses import dataclass, field, replace
from importlib.metadata import PackageNotFoundError, version as _package_version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:
    from avala._client import Client
    from avala.types.sequence_outcome import SequenceOutcome

__all__ = [
    "DEFAULT_SUCCESS_KEYS",
    "InspectRobotsImportResult",
    "TrialMapping",
    "import_inspect_robots",
    "load_inspect_robots_log",
    "map_eval_log",
]

#: Scorers tried, in order, when ``success_key`` is not given. These are the Inspect
#: Robots builtins whose value is a pass/fail verdict.
DEFAULT_SUCCESS_KEYS: Tuple[str, ...] = ("success_at_end", "reached_goal_state", "operator")

_INSTALL_HINT = "Importing Inspect Robots logs requires the 'inspect' extra: pip install 'avala[inspect]'"
_MAX_MODEL_VERSION = 255  # SequenceOutcome.model_version max_length on the server
_MAX_SUBTASK_LABEL = 500  # SequenceOutcomeSubtask.label max_length on the server
# SequenceOutcome.source_metadata limits on the server (``sequence_outcome_serializers.py``).
_MAX_SOURCE_METADATA_KEYS = 20
_MAX_SOURCE_METADATA_KEY = 64
_MAX_SOURCE_METADATA_VALUE = 512
_IMPORTER = "inspect-robots"
# Provenance keys that describe the software that wrote a label rather than the label itself.
# A re-import that differs only in these is the same label, so it must not add a version:
# otherwise every SDK upgrade would re-stack a new version on every previously imported trial.
_WRITER_ONLY_SOURCE_METADATA_KEYS = frozenset({"importer_version"})

SourceMetadata = Dict[str, Union[bool, int, float, str]]

# Same rule as ``inspect_robots.frames._safe`` (0.60.0), which names the action and frame
# side-cars. Copied rather than imported because it is private to Inspect Robots.
_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_FRAME_RE = re.compile(r"_(\d{6})\.npy$")


def _importer_version() -> str:
    try:
        return _package_version("avala")
    except PackageNotFoundError:
        return "unknown"


def _clean_source_metadata(raw: Mapping[str, Any]) -> SourceMetadata:
    """Fit ``raw`` to the server's ``source_metadata`` contract instead of failing the write.

    Null values are dropped (the server rejects null), as are keys that are empty or longer
    than 64 characters, non-finite numbers, and anything that is not a string, number or
    boolean. Strings are cut to 512 characters. At most 20 keys are kept, in ``raw``'s
    order, so callers list the most important keys first.
    """
    clean: SourceMetadata = {}
    for key, value in raw.items():
        if len(clean) >= _MAX_SOURCE_METADATA_KEYS:
            break
        if not isinstance(key, str) or not key or len(key) > _MAX_SOURCE_METADATA_KEY or value is None:
            continue
        if isinstance(value, bool):
            clean[key] = value
        elif isinstance(value, (int, float)):
            if math.isfinite(value):
                clean[key] = value
        elif isinstance(value, str):
            clean[key] = value[:_MAX_SOURCE_METADATA_VALUE]
    return clean


def _safe(name: str) -> str:
    safe = _SAFE_RE.sub("-", name)
    if safe != name:
        safe = f"{safe}-{zlib.crc32(name.encode()) & 0xFFFFFFFF:08x}"
    return safe


@dataclass(frozen=True)
class TrialMapping:
    """One Inspect Robots trial and the outcome label it maps to.

    ``outcome`` is ``None`` when the trial cannot be labelled honestly (``status`` says
    why). ``status`` is ``planned`` after mapping, then one of ``labelled``,
    ``unchanged``, ``kept_existing``, ``unmatched``, ``ambiguous``, ``no_trace`` or
    ``unscored`` after an import.
    """

    trial_id: str
    scene_id: str
    epoch: int
    outcome: Optional[str]
    status: str
    reason: str
    success_value: Optional[float]
    score_value: Optional[float]
    progress: Optional[float]
    model_version: str
    subtasks: Tuple[Dict[str, Any], ...]
    metadata: Dict[str, Any]
    actions_path: Optional[str] = None
    sequence_uid: Optional[str] = None

    def source_metadata(self) -> SourceMetadata:
        """Run provenance stored on the outcome label, fitted to the server's limits.

        ``log_file`` is the log's file name only: a local path would leak the importing
        machine's directory layout to everyone who can read the dataset.
        """
        log_file = self.metadata.get("log_file")
        return _clean_source_metadata(
            {
                "importer": _IMPORTER,
                "importer_version": _importer_version(),
                "run_id": self.metadata.get("inspect_robots_run_id"),
                "task": self.metadata.get("inspect_robots_task") or None,
                "trial_id": self.trial_id,
                "log_file": Path(log_file).name if isinstance(log_file, str) and log_file else None,
                "epoch": self.epoch,
                "scene": self.scene_id,
            }
        )

    def outcome_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments for ``client.sequence_outcomes.set``."""
        return {
            "outcome": self.outcome,
            "progress": self.progress,
            "subtasks": [dict(subtask) for subtask in self.subtasks],
            "autonomy_level": "autonomous",
            "model_version": self.model_version,
            "evaluation_membership": "held_out_eval",
            "source": "imported",
            "source_metadata": self.source_metadata(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe form, used for the receipt and ``--output json``."""
        return {
            "trial_id": self.trial_id,
            "scene_id": self.scene_id,
            "epoch": self.epoch,
            "outcome": self.outcome,
            "status": self.status,
            "reason": self.reason,
            "success_value": self.success_value,
            "score_value": self.score_value,
            "progress": self.progress,
            "model_version": self.model_version,
            "subtasks": [dict(subtask) for subtask in self.subtasks],
            "sequence_uid": self.sequence_uid,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class InspectRobotsImportResult:
    """What an import did (or, with ``dry_run``, would do)."""

    owner: str
    slug: str
    run_id: str
    task: str
    dry_run: bool
    created_dataset_uid: Optional[str] = None
    rows: Tuple[TrialMapping, ...] = field(default_factory=tuple)

    def count(self, status: str) -> int:
        return sum(1 for row in self.rows if row.status == status)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": f"{self.owner}/{self.slug}",
            "run_id": self.run_id,
            "task": self.task,
            "dry_run": self.dry_run,
            "created_dataset_uid": self.created_dataset_uid,
            "rows": [row.to_dict() for row in self.rows],
        }


# ──────────────────────────────────────────────────────────────────────────────
# Reading
# ──────────────────────────────────────────────────────────────────────────────
def _require_inspect_robots() -> Any:
    try:
        return importlib.import_module("inspect_robots")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(_INSTALL_HINT) from exc


def load_inspect_robots_log(path: str) -> Dict[str, Any]:
    """Read and validate an Inspect Robots log with Inspect Robots' own reader.

    Returns the log as plain dicts (``EvalLog.to_dict()``). Raises
    ``ModuleNotFoundError`` with an install hint when the ``inspect`` extra is missing,
    and ``ValueError`` for a file that is not an Inspect Robots log.
    """
    log_path = Path(path)
    if log_path.suffix == ".eval":
        raise ValueError(
            f"{path} is an Inspect AI '.eval' archive. Inspect Robots writes JSON eval logs "
            "(<task>_<id>.json in the run's log directory); pass that file instead."
        )
    if log_path.name.endswith(".live.json"):
        raise ValueError(f"{path} is an Inspect Robots live snapshot; import the final <task>_<id>.json log instead")
    inspect_robots = _require_inspect_robots()
    try:
        log = inspect_robots.read_eval_log(str(log_path))
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} is not an Inspect Robots eval log: {exc}") from exc
    data: Dict[str, Any] = log.to_dict()
    return data


# ──────────────────────────────────────────────────────────────────────────────
# Mapping (pure: no network, no optional dependencies)
# ──────────────────────────────────────────────────────────────────────────────
def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _run_id(data: Mapping[str, Any], log_path: Optional[Path]) -> Tuple[str, str]:
    """Return ``(run_id, where_it_came_from)``.

    Inspect Robots names its per-run side-car directories ``actions/<run_id>/`` and
    ``frames/<run_id>/``; the JSON log has no run id field of its own.
    """
    for sample in data.get("samples") or ():
        for meta in sample.get("trial_metadata") or ():
            actions = (meta or {}).get("actions")
            if isinstance(actions, str):
                parts = Path(actions).parts
                if len(parts) >= 3 and parts[-3] == "actions":
                    return parts[-2], "actions_sidecar"
    frames_dir = (data.get("stats") or {}).get("frames_dir")
    if isinstance(frames_dir, str) and frames_dir:
        return Path(frames_dir).name, "frames_dir"
    if log_path is not None:
        return log_path.stem, "log_filename"
    return "unknown", "none"


def _model_version(spec: Mapping[str, Any]) -> str:
    policy = str(spec.get("policy") or "")
    checkpoint = spec.get("policy_checkpoint")
    version = f"{policy}@{checkpoint}" if checkpoint else policy
    return version[:_MAX_MODEL_VERSION]


def _pick_success_key(epochs: Sequence[Mapping[str, Any]], success_key: Optional[str]) -> str:
    scorer_names = sorted({name for epoch in epochs for name in epoch})
    if not scorer_names:
        # Every trial errored: nothing was scored, so every row is ``aborted`` regardless.
        return success_key or ""
    if success_key is not None:
        if success_key not in scorer_names:
            raise ValueError(f"success scorer {success_key!r} is not in this log; scorers: {scorer_names}")
        return success_key
    for name in DEFAULT_SUCCESS_KEYS:
        if name in scorer_names:
            return name
    if len(scorer_names) == 1:
        return scorer_names[0]
    raise ValueError(
        f"cannot tell which scorer decides success (scorers: {scorer_names}); pass success_key / --success-key"
    )


def _count_steps(actions_file: Optional[Path]) -> Optional[int]:
    if actions_file is None or not actions_file.is_file():
        return None
    steps = 0
    with actions_file.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip() and json.loads(line).get("kind") != "header":
                steps += 1
    return steps


def map_eval_log(
    data: Mapping[str, Any],
    *,
    log_path: Optional[str] = None,
    success_key: Optional[str] = None,
    success_threshold: float = 0.5,
    score_key: Optional[str] = None,
    expert_threshold: float = 1.0,
    progress_key: Optional[str] = None,
) -> List[TrialMapping]:
    """Map every trial of an Inspect Robots log (as dicts) to a planned outcome label.

    ``success_key`` names the scorer whose value decides success (default: the first of
    :data:`DEFAULT_SUCCESS_KEYS` present, or the only scorer). A trial succeeds when that
    value is ``>= success_threshold`` (0.5, the cut Inspect Robots' own ``pass_at_k``
    uses). A success is ``expert_success`` when ``score_key`` (default: the success
    scorer) is ``>= expert_threshold`` and ``partial_success`` otherwise. ``progress_key``
    copies a scorer in [0, 1] into ``progress``.
    """
    if not 0.0 <= success_threshold <= 1.0 or not math.isfinite(success_threshold):
        raise ValueError("success_threshold must be in [0, 1]")
    if not math.isfinite(expert_threshold):
        raise ValueError("expert_threshold must be a finite number")
    path = Path(log_path) if log_path is not None else None
    base_dir = path.parent if path is not None else None
    spec: Mapping[str, Any] = data.get("eval") or {}
    task = str(spec.get("task") or "")
    run_id, run_id_source = _run_id(data, path)
    model_version = _model_version(spec)
    control_hz = _number((spec.get("embodiment_info") or {}).get("control_hz"))
    samples: Sequence[Mapping[str, Any]] = data.get("samples") or ()
    all_epochs = [epoch for sample in samples for epoch in (sample.get("epochs") or ())]
    chosen_success = _pick_success_key(all_epochs, success_key)
    chosen_score = score_key or chosen_success

    rows: List[TrialMapping] = []
    for sample in samples:
        scene_id = str(sample["scene_id"])
        epochs: Sequence[Mapping[str, Any]] = sample.get("epochs") or ()
        reasons = list(sample.get("termination_reasons") or ())
        judgements = list(sample.get("operator_judgements") or ())
        trial_metas = list(sample.get("trial_metadata") or ())
        scene_status = str(sample.get("status") or "")
        for epoch_index, scores in enumerate(epochs):
            # Same stem Inspect Robots gives the action side-car (eval.py ``_write_action_log``).
            trial_id = f"{_safe(scene_id)}-e{epoch_index}"
            trial_meta = dict(trial_metas[epoch_index]) if epoch_index < len(trial_metas) else {}
            actions_rel = trial_meta.get("actions") if isinstance(trial_meta.get("actions"), str) else None
            actions_file = base_dir / actions_rel if (base_dir is not None and actions_rel) else None
            steps = _count_steps(actions_file)
            if steps is None:
                steps = int(scores["episode_length"]) if _number(scores.get("episode_length")) is not None else None
            duration = round(steps / control_hz, 6) if (steps is not None and control_hz) else 0.0
            termination = reasons[epoch_index] if epoch_index < len(reasons) else None
            success_value = _number(scores.get(chosen_success))
            score_value = _number(scores.get(chosen_score))

            outcome: Optional[str]
            if not scores:
                outcome = "aborted"
                status = "planned"
                reason = f"trial not scored (scene status: {scene_status or 'unknown'})"
            elif success_value is None:
                outcome = None
                status = "unscored"
                reason = f"no finite {chosen_success!r} value; not labelled"
            elif success_value >= success_threshold:
                expert = score_value is not None and score_value >= expert_threshold
                outcome = "expert_success" if expert else "partial_success"
                status = "planned"
                reason = (
                    f"{chosen_success}={success_value:g} >= {success_threshold:g}; "
                    f"{chosen_score}={'none' if score_value is None else format(score_value, 'g')} "
                    f"{'>=' if expert else '<'} {expert_threshold:g}"
                )
            else:
                outcome = "failure"
                status = "planned"
                reason = f"{chosen_success}={success_value:g} < {success_threshold:g}"

            progress: Optional[float] = None
            if progress_key is not None and outcome is not None:
                candidate = _number(scores.get(progress_key))
                if candidate is not None and 0.0 <= candidate <= 1.0:
                    progress = candidate

            subtasks: Tuple[Dict[str, Any], ...] = ()
            if task and outcome is not None:
                subtasks = (
                    {"label": task[:_MAX_SUBTASK_LABEL], "start_ts": 0.0, "end_ts": duration, "outcome": outcome},
                )
            metadata: Dict[str, Any] = {
                "inspect_robots_run_id": run_id,
                "inspect_robots_run_id_source": run_id_source,
                "inspect_robots_task": task,
                "inspect_robots_version": spec.get("inspect_robots_version"),
                "scene_id": scene_id,
                "epoch": epoch_index,
                "instruction": sample.get("instruction"),
                "policy": spec.get("policy"),
                "policy_checkpoint": spec.get("policy_checkpoint"),
                "embodiment": spec.get("embodiment"),
                "scene_status": scene_status,
                "scene_error": sample.get("error"),
                "termination_reason": termination,
                "operator_judgement": judgements[epoch_index] if epoch_index < len(judgements) else None,
                "scores": {name: _number(value) for name, value in scores.items()},
                "steps": steps,
                "log_file": path.name if path is not None else None,
            }
            rows.append(
                TrialMapping(
                    trial_id=trial_id,
                    scene_id=scene_id,
                    epoch=epoch_index,
                    outcome=outcome,
                    status=status,
                    reason=reason,
                    success_value=success_value,
                    score_value=score_value,
                    progress=progress,
                    model_version=model_version,
                    subtasks=subtasks,
                    metadata=metadata,
                    actions_path=str(actions_file) if actions_file is not None else None,
                )
            )
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Sequence creation (MCAP) and matching
# ──────────────────────────────────────────────────────────────────────────────
def _frames_dir(data: Mapping[str, Any], log_path: Path) -> Optional[Path]:
    recorded = (data.get("stats") or {}).get("frames_dir")
    if not recorded:
        return None
    candidates = [Path(recorded), log_path.parent / recorded, log_path.parent / "frames" / Path(recorded).name]
    return next((candidate for candidate in candidates if candidate.is_dir()), None)


def _trial_frames(frames_dir: Optional[Path], scene_id: str, epoch: int) -> Dict[int, Dict[str, Path]]:
    """``{t: {camera: path}}`` for one trial.

    ``FrameStore`` names frames ``<_safe(scene-eN)>_<_safe(camera)>_<t:06d>.npy``. Note the
    trial part is ``_safe`` of the whole ``scene-eN`` string, unlike the action side-car.
    """
    by_t: Dict[int, Dict[str, Path]] = {}
    if frames_dir is None:
        return by_t
    trial_id = _safe(f"{scene_id}-e{epoch}")
    prefix = f"{trial_id}_"
    for path in frames_dir.glob(f"{trial_id}_*.npy"):
        match = _FRAME_RE.search(path.name)
        if match is None:
            continue
        camera = path.name[len(prefix) : match.start()]
        by_t.setdefault(int(match.group(1)), {})[camera] = path
    return by_t


def _trial_frame_dicts(
    row: TrialMapping, frames: Mapping[int, Mapping[str, Path]], hz: float
) -> Iterator[Dict[str, Any]]:
    np = importlib.import_module("numpy")
    actions: Dict[int, List[float]] = {}
    labels: Optional[List[str]] = None
    if row.actions_path and Path(row.actions_path).is_file():
        with open(row.actions_path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("kind") == "header":
                    labels = entry.get("labels")
                else:
                    actions[int(entry["t"])] = [float(value) for value in entry["action"]]
    for t in sorted(set(actions) | set(frames)):
        structs: Dict[str, Any] = {}
        if t == 0:
            structs["/inspect_robots/trial"] = {**row.metadata, "outcome": row.outcome}
        if t in actions:
            payload: Dict[str, Any] = {"data": actions[t]}
            if labels:
                payload["labels"] = list(labels)
            structs["/inspect_robots/action"] = payload
        images = {f"/inspect_robots/camera/{camera}": np.load(path) for camera, path in frames.get(t, {}).items()}
        yield {"timestamp_ns": int(round(t / hz * 1_000_000_000)), "images": images, "structs": structs}


def _write_trial_mcaps(
    rows: Sequence[TrialMapping], data: Mapping[str, Any], log_path: Path, out_dir: Path
) -> List[TrialMapping]:
    """Write ``<trial>/<trial>.mcap`` per trial with a trace; mark the rest ``no_trace``."""
    try:
        lerobot_importer = importlib.import_module("avala.importers.lerobot")
        for module in ("numpy", "mcap_protobuf.writer", "foxglove_schemas_protobuf", "PIL"):
            importlib.import_module(module)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(f"{_INSTALL_HINT} (MCAP conversion needs: {exc.name})") from exc
    hz = _number(((data.get("eval") or {}).get("embodiment_info") or {}).get("control_hz")) or 1.0
    frames_dir = _frames_dir(data, log_path)
    updated: List[TrialMapping] = []
    for row in rows:
        frames = _trial_frames(frames_dir, row.scene_id, row.epoch)
        has_actions = bool(row.metadata.get("steps")) and row.actions_path is not None
        if not has_actions and not frames:
            updated.append(replace(row, status="no_trace", reason=f"{row.reason}; no recorded steps to convert"))
            continue
        trial_dir = out_dir / row.trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)
        lerobot_importer.write_episode_mcap(
            str(trial_dir / f"{row.trial_id}.mcap"), _trial_frame_dicts(row, frames, hz)
        )
        updated.append(row)
    return updated


def _sequence_stem(key: Optional[str]) -> Optional[str]:
    if not key:
        return None
    stem = key.rstrip("/").split("/")[-1]
    return stem[: -len(".mcap")] if stem.endswith(".mcap") else stem


def _list_all(fetch: Any) -> Iterator[Any]:
    cursor: Optional[str] = None
    while True:
        page = fetch(cursor)
        yield from page.items
        if not page.next_cursor:
            return
        cursor = page.next_cursor


def _attach(client: "Client", owner: str, slug: str, rows: Sequence[TrialMapping]) -> List[TrialMapping]:
    by_stem: Dict[str, List[str]] = {}
    for sequence in _list_all(lambda cursor: client.datasets.list_sequences(owner, slug, limit=100, cursor=cursor)):
        stem = _sequence_stem(sequence.key)
        if stem is not None:
            by_stem.setdefault(stem, []).append(sequence.uid)
    attached: List[TrialMapping] = []
    for row in rows:
        if row.status != "planned":
            attached.append(row)
            continue
        matches = by_stem.get(row.trial_id, [])
        if len(matches) == 1:
            attached.append(replace(row, sequence_uid=matches[0]))
        elif not matches:
            attached.append(replace(row, status="unmatched", reason=f"no sequence keyed {row.trial_id!r}"))
        else:
            attached.append(replace(row, status="ambiguous", reason=f"{len(matches)} sequences keyed {row.trial_id!r}"))
    return attached


def _label_provenance(source_metadata: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in source_metadata.items() if key not in _WRITER_ONLY_SOURCE_METADATA_KEYS}


def _same_label(current: "SequenceOutcome", wanted: Mapping[str, Any]) -> bool:
    """Whether writing ``wanted`` would only repeat ``current``.

    Compares every field the importer sets, ``source_metadata`` included, so a trial
    re-imported from a different run (or with a different task, log file, ...) gets a new
    label version while a plain re-run does not. ``importer_version`` is left out: it says
    which SDK wrote the label, not what the label says.
    """
    subtasks = [subtask.model_dump() for subtask in current.subtasks]
    return bool(
        current.outcome == wanted["outcome"]
        and current.progress == wanted["progress"]
        and subtasks == wanted["subtasks"]
        and current.autonomy_level == wanted["autonomy_level"]
        and current.model_version == wanted["model_version"]
        and current.evaluation_membership == wanted["evaluation_membership"]
        and current.source == wanted["source"]
        and _label_provenance(current.source_metadata) == _label_provenance(wanted["source_metadata"])
    )


def _write_outcomes(
    client: "Client", owner: str, slug: str, rows: Sequence[TrialMapping], overwrite: bool
) -> List[TrialMapping]:
    current = {
        label.sequence_uid: label
        for label in _list_all(lambda cursor: client.sequence_outcomes.list(owner, slug, limit=100, cursor=cursor))
    }
    written: List[TrialMapping] = []
    for row in rows:
        if row.status != "planned" or row.sequence_uid is None:
            written.append(row)
            continue
        wanted = row.outcome_kwargs()
        existing = current.get(row.sequence_uid)
        if existing is not None and _same_label(existing, wanted):
            written.append(replace(row, status="unchanged"))
            continue
        if existing is not None and existing.source != "imported" and not overwrite:
            written.append(
                replace(
                    row,
                    status="kept_existing",
                    reason=f"sequence already has a {existing.source} label ({existing.outcome}); pass overwrite",
                )
            )
            continue
        client.sequence_outcomes.set(owner, slug, row.sequence_uid, **wanted)
        written.append(replace(row, status="labelled"))
    return written


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def _split_dataset(dataset: str) -> Tuple[str, str]:
    owner, sep, slug = dataset.partition("/")
    if not sep or not owner or not slug or "/" in slug:
        raise ValueError(f"dataset must be 'owner/slug', got {dataset!r}")
    return owner, slug


def import_inspect_robots(
    client: Optional["Client"],
    *,
    log: str,
    dataset: str,
    dry_run: bool = False,
    create: bool = False,
    name: Optional[str] = None,
    organization_uid: Optional[str] = None,
    success_key: Optional[str] = None,
    success_threshold: float = 0.5,
    score_key: Optional[str] = None,
    expert_threshold: float = 1.0,
    progress_key: Optional[str] = None,
    overwrite: bool = False,
    workers: int = 8,
    wait_timeout: float = 3600.0,
) -> InspectRobotsImportResult:
    """Label one sequence per Inspect Robots trial with its evaluation outcome.

    ``dataset`` is ``owner/slug``. By default the trials attach to sequences that already
    exist in that dataset (see the module docstring for the key rule). With
    ``create=True`` the trials' recorded traces are converted to MCAP and uploaded as a
    new dataset ``owner/slug`` (named ``name`` or the slug) first.

    ``dry_run=True`` maps the log and returns the plan without any network call (the
    client may be ``None``). Existing labels from another source (``human`` / ``model``)
    are kept unless ``overwrite=True``; a label identical to the one being imported is
    left alone, so re-running an import does not stack duplicate versions.
    """
    import tempfile

    owner, slug = _split_dataset(dataset)
    data = load_inspect_robots_log(log)
    rows = map_eval_log(
        data,
        log_path=log,
        success_key=success_key,
        success_threshold=success_threshold,
        score_key=score_key,
        expert_threshold=expert_threshold,
        progress_key=progress_key,
    )
    run_id, _source = _run_id(data, Path(log))
    task = str((data.get("eval") or {}).get("task") or "")
    if dry_run:
        return InspectRobotsImportResult(
            owner=owner, slug=slug, run_id=run_id, task=task, dry_run=True, rows=tuple(rows)
        )
    if client is None:
        raise ValueError("a client is required unless dry_run=True")

    created_uid: Optional[str] = None
    if create:
        with tempfile.TemporaryDirectory(prefix="avala-inspect-robots-") as tmp:
            rows = _write_trial_mcaps(rows, data, Path(log), Path(tmp))
            if not any(Path(tmp).iterdir()):
                raise ValueError("no trial in this log has recorded steps to convert; nothing to create")
            created = client.datasets.create_from_local(
                source=tmp,
                name=name or slug,
                slug=slug,
                data_type="mcap",
                owner_name=None if organization_uid else owner,
                organization_uid=organization_uid,
                workers=workers,
                wait=True,
                wait_timeout=wait_timeout,
            )
            created_uid = created.uid

    rows = _attach(client, owner, slug, rows)
    rows = _write_outcomes(client, owner, slug, rows, overwrite)
    return InspectRobotsImportResult(
        owner=owner,
        slug=slug,
        run_id=run_id,
        task=task,
        dry_run=False,
        created_dataset_uid=created_uid,
        rows=tuple(rows),
    )
