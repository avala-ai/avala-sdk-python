"""Tests for ``avala import inspect-robots`` (``avala.importers.inspect_robots``).

Contracts these tests pin, and where each comes from:

* Log format: Inspect Robots 0.60.0 ``EvalLog`` schema version 1 (``inspect_robots/log.py``).
  ``SceneResult.epochs[i]`` is trial (scene, epoch i); an errored/cancelled trial is recorded
  but never scored, so its entry is ``{}`` (``inspect_robots/eval.py``: "Errored trials are
  recorded ... but never scored ... an empty entry in SceneResult.epochs"). Non-finite scores
  are written as JSON ``null`` (``inspect_robots/logging/json_log.py``). Per-trial actions live
  in ``actions/<run_id>/<scene>-e<epoch>.jsonl`` relative to the log dir.
* Fixtures: ``tests/fixtures/inspect_robots/run_*`` were written by ``inspect_robots.eval`` on
  its offline ``cubepick`` mock (see ``generate_fixture.py`` there), not by hand.
* Outcome mapping (accepted task spec for this importer): success -> ``expert_success`` or
  ``partial_success`` by a configurable score threshold; failure -> ``failure``;
  error/abort -> ``aborted``; ``source=imported``; ``evaluation_membership=held_out_eval``;
  ``model_version`` from the eval's model; the task name kept as a subtask label. The default
  success cut of 0.5 is Inspect Robots' own (``scorer.pass_at_k``: "success = value >= 0.5").
* Server contract for the writes: ``PUT datasets/<owner>/<slug>/sequences/<uid>/outcome/`` and
  ``GET .../sequence-outcomes/`` (``server/server/apps/dataset/api_sequence_outcomes.py``);
  sequence keys are the item key's first path segment under the dataset prefix
  (``server/server/apps/dataset/serializers.py::_sequence_key_for``).
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import httpx
import pytest
import respx

from avala import Client
from avala.importers import inspect_robots as ir

FIXTURES = Path(__file__).parent / "fixtures" / "inspect_robots"
MAIN_LOG = FIXTURES / "run_main" / "eval_log.json"
FRAMES_LOG = FIXTURES / "run_frames" / "eval_log.json"
BASE_URL = "https://api.avala.ai/api/v1"
OWNER, SLUG = "acme", "cubepick-eval"
SEQ_URL = f"{BASE_URL}/datasets/{OWNER}/{SLUG}/sequences/"
LIST_URL = f"{BASE_URL}/datasets/{OWNER}/{SLUG}/sequence-outcomes/"


def _log(path: Path = MAIN_LOG) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _by_trial(rows: List[ir.TrialMapping]) -> Dict[str, ir.TrialMapping]:
    return {row.trial_id: row for row in rows}


@pytest.fixture
def plain_json_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read logs as plain JSON so orchestration tests run without the optional extra.

    ``test_real_reader_matches_plain_json`` pins that Inspect Robots' own reader returns the
    same data for these fixtures, which is what makes this substitution faithful.
    """
    monkeypatch.setattr(ir, "load_inspect_robots_log", lambda path: json.loads(Path(path).read_text()))


# ──────────────────────────────────────────────────────────────────────────────
# Mapping table
# ──────────────────────────────────────────────────────────────────────────────
def test_default_mapping_table() -> None:
    rows = _by_trial(ir.map_eval_log(_log(), log_path=str(MAIN_LOG)))

    assert {trial: row.outcome for trial, row in rows.items()} == {
        "reach-fast-e0": "expert_success",
        "reach-fast-e1": "expert_success",
        "reach-slow-e0": "failure",
        "reach-slow-e1": "failure",
        "reach-stuck-e0": "failure",
        "reach-stuck-e1": "failure",
        "reach-flaky-e0": "expert_success",
        "reach-flaky-e1": "aborted",  # PolicyError: recorded, never scored
    }
    assert {row.status for row in rows.values()} == {"planned"}
    assert "scene status: error" in rows["reach-flaky-e1"].reason


def test_every_row_carries_the_fixed_import_fields() -> None:
    for row in ir.map_eval_log(_log(), log_path=str(MAIN_LOG)):
        kwargs = row.outcome_kwargs()
        assert kwargs["source"] == "imported"
        assert kwargs["evaluation_membership"] == "held_out_eval"
        assert kwargs["autonomy_level"] == "autonomous"
        # eval.policy + eval.policy_checkpoint from the log
        assert kwargs["model_version"] == "scripted@cubepick-oracle-v1"


def test_task_name_is_the_subtask_label_spanning_the_trial() -> None:
    rows = _by_trial(ir.map_eval_log(_log(), log_path=str(MAIN_LOG)))
    fast = rows["reach-fast-e0"]
    # 8 recorded actions at embodiment_info.control_hz = 10 -> 0.8 s
    assert fast.subtasks == (
        {"label": "cubepick-reach-eval", "start_ts": 0.0, "end_ts": 0.8, "outcome": "expert_success"},
    )
    # The errored trial recorded no steps: a zero-length span, still labelled.
    assert rows["reach-flaky-e1"].subtasks[0]["end_ts"] == 0.0


def test_run_id_and_task_are_kept_in_metadata() -> None:
    data = _log()
    rows = _by_trial(ir.map_eval_log(data, log_path=str(MAIN_LOG)))
    actions = data["samples"][0]["trial_metadata"][0]["actions"]  # actions/<run_id>/reach-fast-e0.jsonl
    run_id = Path(actions).parts[1]

    meta = rows["reach-slow-e1"].metadata
    assert meta["inspect_robots_run_id"] == run_id
    assert meta["inspect_robots_run_id_source"] == "actions_sidecar"
    assert meta["inspect_robots_task"] == "cubepick-reach-eval"
    assert (meta["scene_id"], meta["epoch"]) == ("reach-slow", 1)
    assert meta["termination_reason"] == "max_steps"
    assert meta["steps"] == 12
    assert set(meta["scores"]) == {"success_at_end", "task_progress", "episode_length"}
    assert rows["reach-flaky-e1"].metadata["scene_error"] == "PolicyError: simulated inference server timeout"


def test_run_id_falls_back_to_frames_dir_then_log_name() -> None:
    data = _log(FRAMES_LOG)
    frames_run = Path(data["stats"]["frames_dir"]).name
    for sample in data["samples"]:
        sample["trial_metadata"] = [{} for _ in sample["trial_metadata"]]
    assert ir.map_eval_log(data, log_path=str(FRAMES_LOG))[0].metadata["inspect_robots_run_id"] == frames_run

    data["stats"]["frames_dir"] = None
    row = ir.map_eval_log(data, log_path=str(FRAMES_LOG))[0]
    assert (row.metadata["inspect_robots_run_id"], row.metadata["inspect_robots_run_id_source"]) == (
        "eval_log",
        "log_filename",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Thresholds and scorer choice
# ──────────────────────────────────────────────────────────────────────────────
def test_continuous_scorer_splits_expert_and_partial_success() -> None:
    rows = _by_trial(
        ir.map_eval_log(
            _log(),
            log_path=str(MAIN_LOG),
            success_key="task_progress",
            expert_threshold=0.95,
            progress_key="task_progress",
        )
    )
    assert rows["reach-fast-e0"].outcome == "expert_success"  # progress 1.0
    assert rows["reach-flaky-e0"].outcome == "expert_success"  # progress ~0.97
    assert rows["reach-slow-e0"].outcome == "partial_success"  # progress ~0.75: >= 0.5, < 0.95
    assert rows["reach-stuck-e0"].outcome == "failure"  # progress 0.0
    assert rows["reach-slow-e0"].progress == pytest.approx(0.7522, abs=1e-4)
    assert rows["reach-flaky-e1"].progress is None  # aborted: nothing measured


def test_success_threshold_is_configurable() -> None:
    rows = _by_trial(ir.map_eval_log(_log(), success_key="task_progress", success_threshold=0.8))
    assert rows["reach-slow-e0"].outcome == "failure"  # ~0.75 < 0.8
    assert rows["reach-flaky-e0"].outcome == "partial_success"  # ~0.97 >= 0.8, < expert 1.0


def test_score_key_grades_success_independently_of_the_success_scorer() -> None:
    rows = _by_trial(ir.map_eval_log(_log(), score_key="task_progress", expert_threshold=0.99))
    assert rows["reach-fast-e0"].outcome == "expert_success"  # success_at_end 1, progress 1.0
    assert rows["reach-flaky-e0"].outcome == "partial_success"  # success_at_end 1, progress ~0.97


def test_progress_is_unset_unless_requested() -> None:
    assert {row.progress for row in ir.map_eval_log(_log())} == {None}


@pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan")])
def test_success_threshold_must_be_a_probability(bad: float) -> None:
    with pytest.raises(ValueError, match="success_threshold"):
        ir.map_eval_log(_log(), success_threshold=bad)


def test_unknown_success_key_is_refused() -> None:
    with pytest.raises(ValueError, match="not in this log"):
        ir.map_eval_log(_log(), success_key="does_not_exist")


def test_ambiguous_scorers_require_an_explicit_success_key() -> None:
    data = _log()
    for sample in data["samples"]:
        for epoch in sample["epochs"]:
            epoch.pop("success_at_end", None)
    with pytest.raises(ValueError, match="--success-key"):
        ir.map_eval_log(data)


def test_non_finite_success_score_is_left_unlabelled() -> None:
    # JsonLogSink writes inf/nan as null; that is "no value", not a failure.
    data = copy.deepcopy(_log())
    data["samples"][1]["epochs"][0]["success_at_end"] = None
    row = _by_trial(ir.map_eval_log(data))["reach-slow-e0"]
    assert (row.outcome, row.status, row.subtasks) == (None, "unscored", ())


def test_all_errored_run_maps_every_trial_to_aborted() -> None:
    data = _log()
    for sample in data["samples"]:
        sample["epochs"] = [{} for _ in sample["epochs"]]
    assert {row.outcome for row in ir.map_eval_log(data)} == {"aborted"}


# ──────────────────────────────────────────────────────────────────────────────
# Reading / optional dependency
# ──────────────────────────────────────────────────────────────────────────────
def test_missing_optional_dependency_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "inspect_robots", None)  # makes the import fail
    with pytest.raises(ModuleNotFoundError, match=r"avala\[inspect\]"):
        ir.load_inspect_robots_log(str(MAIN_LOG))


def test_inspect_ai_eval_archives_are_refused_with_guidance(tmp_path: Path) -> None:
    archive = tmp_path / "run.eval"
    archive.write_bytes(b"PK")
    with pytest.raises(ValueError, match="Inspect AI '.eval' archive"):
        ir.load_inspect_robots_log(str(archive))


def test_real_reader_matches_plain_json() -> None:
    pytest.importorskip("inspect_robots")
    for path in (MAIN_LOG, FRAMES_LOG):
        via_reader = json.loads(json.dumps(ir.load_inspect_robots_log(str(path))))
        assert via_reader == _log(path)


def test_trial_naming_matches_inspect_robots() -> None:
    """The copied ``_safe`` must agree with Inspect Robots' own, or keys stop matching."""
    frames = pytest.importorskip("inspect_robots.frames")
    for name in ("reach-fast", "pick/place", "kitchen counter #2", "a-b", "a/b"):
        assert ir._safe(name) == frames._safe(name)
    data = _log()
    data["samples"][0]["scene_id"] = "pick/place"
    row = ir.map_eval_log(data)[0]
    assert row.trial_id == f"{frames._safe('pick/place')}-e0"  # the action side-car stem


def test_real_reader_rejects_a_non_inspect_robots_json(tmp_path: Path) -> None:
    pytest.importorskip("inspect_robots")
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"version": 2, "status": "success", "eval": {"task_id": "x"}}))
    with pytest.raises(ValueError):
        ir.load_inspect_robots_log(str(other))


# ──────────────────────────────────────────────────────────────────────────────
# Dry run: no network
# ──────────────────────────────────────────────────────────────────────────────
@respx.mock(assert_all_mocked=True)
def test_dry_run_makes_no_network_calls(plain_json_reader: None) -> None:
    result = ir.import_inspect_robots(None, log=str(MAIN_LOG), dataset=f"{OWNER}/{SLUG}", dry_run=True)
    assert result.dry_run is True
    assert len(result.rows) == 8
    assert respx.calls.call_count == 0


@respx.mock(assert_all_mocked=True)
def test_dry_run_with_a_client_still_makes_no_network_calls(plain_json_reader: None) -> None:
    result = ir.import_inspect_robots(
        Client(api_key="k"), log=str(MAIN_LOG), dataset=f"{OWNER}/{SLUG}", dry_run=True, create=True
    )
    assert {row.status for row in result.rows} == {"planned"}
    assert respx.calls.call_count == 0


def test_dataset_must_be_owner_slug(plain_json_reader: None) -> None:
    with pytest.raises(ValueError, match="owner/slug"):
        ir.import_inspect_robots(None, log=str(MAIN_LOG), dataset="no-owner", dry_run=True)


def test_cli_dry_run_prints_the_mapping_without_an_api_key(
    plain_json_reader: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("click")
    from click.testing import CliRunner

    from avala.cli import main

    monkeypatch.delenv("AVALA_API_KEY", raising=False)
    with respx.mock(assert_all_mocked=True) as mock:
        result = CliRunner().invoke(
            main,
            ["import", "inspect-robots", str(MAIN_LOG), "--dataset", f"{OWNER}/{SLUG}", "--dry-run"],
            env={"COLUMNS": "200"},
        )
        assert mock.calls.call_count == 0
    assert result.exit_code == 0, result.output
    assert "reach-flaky-e1" in result.output and "aborted" in result.output
    assert "scripted@cubepick-oracle-v1" in result.output


def test_cli_dry_run_json_includes_metadata(plain_json_reader: None, tmp_path: Path) -> None:
    pytest.importorskip("click")
    from click.testing import CliRunner

    from avala.cli import main

    receipt = tmp_path / "receipt.json"
    result = CliRunner().invoke(
        main,
        ["-o", "json", "import", "inspect-robots", str(MAIN_LOG), "--dataset", f"{OWNER}/{SLUG}", "--dry-run"]
        + ["--receipt", str(receipt)],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == json.loads(receipt.read_text())
    assert payload["rows"][0]["metadata"]["inspect_robots_task"] == "cubepick-reach-eval"


def test_cli_reports_the_missing_extra_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("click")
    from click.testing import CliRunner

    from avala.cli import main

    monkeypatch.setitem(sys.modules, "inspect_robots", None)
    result = CliRunner().invoke(
        main, ["import", "inspect-robots", str(MAIN_LOG), "--dataset", f"{OWNER}/{SLUG}", "--dry-run"]
    )
    assert result.exit_code == 1
    assert "pip install 'avala[inspect]'" in result.output
    assert "Traceback" not in result.output


# ──────────────────────────────────────────────────────────────────────────────
# Attach and write
# ──────────────────────────────────────────────────────────────────────────────
def _seq(uid: str, stem: str) -> Dict[str, Any]:
    return {"uid": uid, "key": f"orgs/acme/cubepick-eval/{stem}"}


def _label(seq_uid: str, **overrides: Any) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "uid": f"o-{seq_uid}",
        "sequence_uid": seq_uid,
        "version": 1,
        "is_current": True,
        "outcome": "failure",
        "source": "human",
    }
    body.update(overrides)
    return body


def _mock_sequences(stems: Dict[str, str]) -> None:
    sequences = [_seq(uid, stem) for stem, uid in stems.items()]
    half = len(sequences) // 2
    route = respx.get(SEQ_URL)
    route.side_effect = [
        httpx.Response(200, json={"results": sequences[:half], "next": f"{SEQ_URL}?cursor=p2", "previous": None}),
        httpx.Response(200, json={"results": sequences[half:], "next": None, "previous": None}),
    ]


@respx.mock
def test_attach_writes_one_imported_label_per_matched_trial(plain_json_reader: None) -> None:
    stems = {
        "reach-fast-e0": "s-fast0",
        "reach-fast-e1.mcap": "s-fast1",  # an uploaded .mcap at the dataset root
        "reach-slow-e0": "s-slow0",
        "reach-slow-e1": "s-slow1",
        "reach-stuck-e0": "s-stuck0",
        "reach-stuck-e1": "s-stuck1",
        "reach-flaky-e1": "s-flaky1",
        # reach-flaky-e0 has no sequence -> unmatched
    }
    _mock_sequences(stems)
    respx.get(LIST_URL).mock(return_value=httpx.Response(200, json={"results": [], "next": None, "previous": None}))
    put = respx.put(url__regex=rf"{SEQ_URL}[^/]+/outcome/").mock(
        side_effect=lambda request: httpx.Response(
            200,
            json=_label(request.url.path.split("/")[-3], **json.loads(request.content)),
        )
    )

    result = ir.import_inspect_robots(Client(api_key="k"), log=str(MAIN_LOG), dataset=f"{OWNER}/{SLUG}")

    rows = _by_trial(list(result.rows))
    assert rows["reach-flaky-e0"].status == "unmatched"
    assert result.count("labelled") == 7
    assert put.call_count == 7
    sent = {call.request.url.path.split("/")[-3]: json.loads(call.request.content) for call in put.calls}
    assert sent["s-fast1"]["outcome"] == "expert_success"
    assert sent["s-flaky1"]["outcome"] == "aborted"
    assert sent["s-slow0"] == {
        "outcome": "failure",
        "subtasks": [{"label": "cubepick-reach-eval", "start_ts": 0.0, "end_ts": 1.2, "outcome": "failure"}],
        "autonomy_level": "autonomous",
        "model_version": "scripted@cubepick-oracle-v1",
        "evaluation_membership": "held_out_eval",
        "source": "imported",
    }


@respx.mock
def test_existing_labels_are_respected_and_reruns_do_not_stack_versions(plain_json_reader: None) -> None:
    _mock_sequences({"reach-fast-e0": "s-fast0", "reach-slow-e0": "s-slow0", "reach-stuck-e0": "s-stuck0"})
    same = ir.map_eval_log(_log(), log_path=str(MAIN_LOG))[0].outcome_kwargs()  # reach-fast-e0
    respx.get(LIST_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    _label("s-fast0", **same),  # identical imported label -> unchanged
                    _label("s-slow0", outcome="partial_success", source="human"),  # human -> kept
                ],
                "next": None,
                "previous": None,
            },
        )
    )
    put = respx.put(url__regex=rf"{SEQ_URL}[^/]+/outcome/").mock(
        side_effect=lambda request: httpx.Response(200, json=_label(request.url.path.split("/")[-3]))
    )

    rows = _by_trial(
        list(ir.import_inspect_robots(Client(api_key="k"), log=str(MAIN_LOG), dataset=f"{OWNER}/{SLUG}").rows)
    )

    assert rows["reach-fast-e0"].status == "unchanged"
    assert rows["reach-slow-e0"].status == "kept_existing"
    assert rows["reach-stuck-e0"].status == "labelled"
    assert [call.request.url.path.split("/")[-3] for call in put.calls] == ["s-stuck0"]


@respx.mock
def test_overwrite_replaces_a_human_label(plain_json_reader: None) -> None:
    _mock_sequences({"reach-slow-e0": "s-slow0", "reach-slow-e1": "s-slow1"})
    respx.get(LIST_URL).mock(
        return_value=httpx.Response(
            200, json={"results": [_label("s-slow0", outcome="partial_success")], "next": None, "previous": None}
        )
    )
    put = respx.put(url__regex=rf"{SEQ_URL}[^/]+/outcome/").mock(
        side_effect=lambda request: httpx.Response(200, json=_label(request.url.path.split("/")[-3]))
    )
    ir.import_inspect_robots(Client(api_key="k"), log=str(MAIN_LOG), dataset=f"{OWNER}/{SLUG}", overwrite=True)
    assert sorted(call.request.url.path.split("/")[-3] for call in put.calls) == ["s-slow0", "s-slow1"]


@respx.mock
def test_duplicate_sequence_keys_are_ambiguous_not_guessed(plain_json_reader: None) -> None:
    respx.get(SEQ_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [_seq("a", "reach-fast-e0"), _seq("b", "x/reach-fast-e0")],
                "next": None,
                "previous": None,
            },
        )
    )
    respx.get(LIST_URL).mock(return_value=httpx.Response(200, json={"results": [], "next": None, "previous": None}))
    put = respx.put(url__regex=rf"{SEQ_URL}[^/]+/outcome/")
    rows = _by_trial(
        list(ir.import_inspect_robots(Client(api_key="k"), log=str(MAIN_LOG), dataset=f"{OWNER}/{SLUG}").rows)
    )
    assert rows["reach-fast-e0"].status == "ambiguous"
    assert put.call_count == 0


# ──────────────────────────────────────────────────────────────────────────────
# Create: trial traces -> MCAP
# ──────────────────────────────────────────────────────────────────────────────
def _read_mcap(path: Path) -> Dict[str, List[Any]]:
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    by_topic: Dict[str, List[Any]] = {}
    with path.open("rb") as fh:
        for _schema, channel, _message, decoded in make_reader(
            fh, decoder_factories=[DecoderFactory()]
        ).iter_decoded_messages():
            by_topic.setdefault(channel.topic, []).append(decoded)
    return by_topic


@pytest.mark.parametrize("log_path", [MAIN_LOG, FRAMES_LOG], ids=["actions", "actions+frames"])
def test_create_converts_each_trial_trace_to_one_mcap(
    plain_json_reader: None, monkeypatch: pytest.MonkeyPatch, log_path: Path
) -> None:
    for module in (
        "numpy",
        "mcap.reader",
        "mcap_protobuf.writer",
        "mcap_protobuf.decoder",
        "foxglove_schemas_protobuf",
        "PIL",
    ):
        pytest.importorskip(module)
    from google.protobuf.json_format import MessageToDict

    client = Client(api_key="k")
    uploaded: Dict[str, Dict[str, List[Any]]] = {}

    def fake_create_from_local(**kwargs: Any) -> Any:
        assert (kwargs["data_type"], kwargs["slug"], kwargs["owner_name"], kwargs["wait"]) == (
            "mcap",
            SLUG,
            OWNER,
            True,
        )
        for mcap in sorted(Path(kwargs["source"]).rglob("*.mcap")):
            assert mcap.parent.name == mcap.stem  # <trial>/<trial>.mcap -> sequence key <trial>
            uploaded[mcap.stem] = _read_mcap(mcap)
        return type("Created", (), {"uid": "ds-new"})()

    monkeypatch.setattr(client.datasets, "create_from_local", fake_create_from_local)
    with respx.mock:
        respx.get(SEQ_URL).mock(return_value=httpx.Response(200, json={"results": [], "next": None, "previous": None}))
        respx.get(LIST_URL).mock(return_value=httpx.Response(200, json={"results": [], "next": None, "previous": None}))
        result = ir.import_inspect_robots(client, log=str(log_path), dataset=f"{OWNER}/{SLUG}", create=True)

    assert result.created_dataset_uid == "ds-new"
    rows = _by_trial(list(result.rows))
    if log_path == MAIN_LOG:
        assert rows["reach-flaky-e1"].status == "no_trace"  # zero recorded steps: nothing to convert
        assert sorted(uploaded) == sorted(trial for trial in rows if trial != "reach-flaky-e1")
    fast = uploaded["reach-fast-e0"]
    steps = rows["reach-fast-e0"].metadata["steps"]
    assert len(fast["/inspect_robots/action"]) == steps
    assert MessageToDict(fast["/inspect_robots/action"][0])["labels"] == ["dx", "dy"]
    [trial_meta] = fast["/inspect_robots/trial"]
    meta = MessageToDict(trial_meta)
    assert meta["inspect_robots_task"] == rows["reach-fast-e0"].metadata["inspect_robots_task"]
    assert meta["inspect_robots_run_id"] == rows["reach-fast-e0"].metadata["inspect_robots_run_id"]
    assert meta["outcome"] == rows["reach-fast-e0"].outcome  # failure in the 3-step frames run
    if log_path == FRAMES_LOG:
        # frame t=0 is the reset observation, t=1..3 follow each of the 3 actions
        assert len(fast["/inspect_robots/camera/top"]) == steps + 1
    else:
        assert "/inspect_robots/camera/top" not in fast
