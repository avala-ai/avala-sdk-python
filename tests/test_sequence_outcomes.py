"""Tests for the SequenceOutcomes resource (sync, async and CLI).

Contract: server routes ``datasets/<owner>/<slug>/sequences/<uid>/outcome/`` (GET/PUT),
``.../outcome/history/`` (GET, plain array) and ``datasets/<owner>/<slug>/sequence-outcomes/``
(GET, cursor page, comma-separated enum filters) — see
``server/server/apps/dataset/api_sequence_outcomes.py``.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from avala import AsyncClient, Client
from avala.errors import NotFoundError

BASE_URL = "https://api.avala.ai/api/v1"
OWNER = "acme"
SLUG = "pick-place"
SEQ = "6f1c2a8e-9a4b-4f0e-8b1d-2f6c1e9d0a11"
OUTCOME_URL = f"{BASE_URL}/datasets/{OWNER}/{SLUG}/sequences/{SEQ}/outcome/"
LIST_URL = f"{BASE_URL}/datasets/{OWNER}/{SLUG}/sequence-outcomes/"


def _outcome(**overrides: object) -> dict:
    body = {
        "uid": "o-1",
        "sequence_uid": SEQ,
        "version": 1,
        "is_current": True,
        "outcome": "mistake_and_recovery",
        "progress": 0.75,
        "quality": 4,
        "speed": 2,
        "subtasks": [{"label": "regrasp", "start_ts": 1.5, "end_ts": 4.0, "outcome": None}],
        "mistake_type": "grasp_slip",
        "recovery_type": "regrasp",
        "failure_stage": "",
        "autonomy_level": "teleoperation",
        "model_version": "",
        "evaluation_membership": "held_out_eval",
        "leakage_groups": {"location": "kitchen-3"},
        "source": "human",
        "labeled_by": None,
        "confidence": None,
        "created_at": "2026-09-30T00:00:00Z",
        "updated_at": "2026-09-30T00:00:00Z",
    }
    body.update(overrides)
    return body


@respx.mock
def test_get_returns_current_label():
    respx.get(OUTCOME_URL).mock(return_value=httpx.Response(200, json=_outcome()))
    client = Client(api_key="test-key")
    outcome = client.sequence_outcomes.get(OWNER, SLUG, SEQ)
    assert outcome.outcome == "mistake_and_recovery"
    assert outcome.subtasks[0].label == "regrasp" and outcome.subtasks[0].outcome is None
    assert outcome.leakage_groups == {"location": "kitchen-3"}
    client.close()


@respx.mock
def test_get_unlabeled_sequence_raises_not_found():
    respx.get(OUTCOME_URL).mock(
        return_value=httpx.Response(404, json={"detail": "This sequence has no outcome label."})
    )
    client = Client(api_key="test-key")
    with pytest.raises(NotFoundError):
        client.sequence_outcomes.get(OWNER, SLUG, SEQ)
    client.close()


@respx.mock
def test_set_puts_only_supplied_fields():
    route = respx.put(OUTCOME_URL).mock(
        return_value=httpx.Response(200, json=_outcome(version=2, outcome="slow_success", source="model"))
    )
    client = Client(api_key="test-key")
    outcome = client.sequence_outcomes.set(
        OWNER,
        SLUG,
        SEQ,
        outcome="slow_success",
        source="model",
        confidence=0.6,
        model_version="policy-v3",
        subtasks=[{"label": "reach", "start_ts": 0.0, "end_ts": 1.0}],
        leakage_groups={"operator": "op-1"},
    )
    assert outcome.version == 2
    body = json.loads(route.calls.last.request.content)
    assert body == {
        "outcome": "slow_success",
        "source": "model",
        "confidence": 0.6,
        "model_version": "policy-v3",
        "subtasks": [{"label": "reach", "start_ts": 0.0, "end_ts": 1.0}],
        "leakage_groups": {"operator": "op-1"},
    }
    client.close()


@respx.mock
def test_history_returns_versions():
    respx.get(f"{OUTCOME_URL}history/").mock(
        return_value=httpx.Response(200, json=[_outcome(version=2, uid="o-2"), _outcome(is_current=False)])
    )
    client = Client(api_key="test-key")
    versions = client.sequence_outcomes.history(OWNER, SLUG, SEQ)
    assert [v.version for v in versions] == [2, 1]
    assert [v.is_current for v in versions] == [True, False]
    client.close()


@respx.mock
def test_list_sends_comma_joined_filters():
    route = respx.get(LIST_URL).mock(
        return_value=httpx.Response(200, json={"results": [_outcome()], "next": None, "previous": None})
    )
    client = Client(api_key="test-key")
    page = client.sequence_outcomes.list(
        OWNER, SLUG, outcome=["failure", "aborted"], evaluation_membership="held_out_eval", limit=10
    )
    assert len(page.items) == 1 and page.has_more is False
    params = route.calls.last.request.url.params
    assert params["outcome"] == "failure,aborted"
    assert params["evaluation_membership"] == "held_out_eval"
    assert params["limit"] == "10"
    assert "source" not in params
    client.close()


@pytest.mark.asyncio
@respx.mock
async def test_async_get_and_list():
    respx.get(OUTCOME_URL).mock(return_value=httpx.Response(200, json=_outcome()))
    respx.get(LIST_URL).mock(return_value=httpx.Response(200, json={"results": [], "next": None, "previous": None}))
    async with AsyncClient(api_key="test-key") as client:
        outcome = await client.sequence_outcomes.get(OWNER, SLUG, SEQ)
        assert outcome.evaluation_membership == "held_out_eval"
        page = await client.sequence_outcomes.list(OWNER, SLUG, outcome="failure")
        assert page.items == []


@respx.mock
def test_cli_set_parses_leakage_groups_and_subtasks(tmp_path):
    pytest.importorskip("click", reason="CLI dependencies not installed (pip install avala[cli])")
    from click.testing import CliRunner

    from avala.cli import main

    route = respx.put(OUTCOME_URL).mock(return_value=httpx.Response(200, json=_outcome()))
    subtasks = tmp_path / "subtasks.json"
    subtasks.write_text(json.dumps([{"label": "reach", "start_ts": 0, "end_ts": 1}]))
    result = CliRunner().invoke(
        main,
        [
            "--api-key",
            "test-key",
            "sequence-outcomes",
            "set",
            OWNER,
            SLUG,
            SEQ,
            "--outcome",
            "mistake_and_recovery",
            "--progress",
            "0.75",
            "--leakage-group",
            "location=kitchen-3",
            "--subtasks-file",
            str(subtasks),
        ],
    )
    assert result.exit_code == 0, result.output
    body = json.loads(route.calls.last.request.content)
    assert body["leakage_groups"] == {"location": "kitchen-3"}
    assert body["subtasks"] == [{"label": "reach", "start_ts": 0, "end_ts": 1}]
    assert body["progress"] == 0.75


def test_cli_set_rejects_unknown_outcome_before_any_request():
    pytest.importorskip("click", reason="CLI dependencies not installed (pip install avala[cli])")
    from click.testing import CliRunner

    from avala.cli import main

    result = CliRunner().invoke(
        main, ["--api-key", "test-key", "sequence-outcomes", "set", OWNER, SLUG, SEQ, "--outcome", "great"]
    )
    assert result.exit_code != 0
    assert "great" in result.output


@respx.mock
def test_cli_list_json_output():
    pytest.importorskip("click", reason="CLI dependencies not installed (pip install avala[cli])")
    from click.testing import CliRunner

    from avala.cli import main

    respx.get(LIST_URL).mock(
        return_value=httpx.Response(200, json={"results": [_outcome()], "next": None, "previous": None})
    )
    result = CliRunner().invoke(
        main, ["--api-key", "test-key", "--output", "json", "sequence-outcomes", "list", OWNER, SLUG]
    )
    assert result.exit_code == 0, result.output
    rows = json.loads(result.output)
    assert rows[0]["outcome"] == "mistake_and_recovery"
