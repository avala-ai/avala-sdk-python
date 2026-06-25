"""Tests for QualityTargets resource (project-scoped)."""

from __future__ import annotations

import json

import httpx
import respx

from avala import Client

BASE_URL = "https://api.avala.ai/api/v1"
PROJECT_UID = "proj-uid-001"


@respx.mock
def test_list_quality_targets():
    """QualityTargets.list() returns a CursorPage of QualityTarget objects."""
    respx.get(f"{BASE_URL}/projects/{PROJECT_UID}/quality-targets/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "uid": "qt-001",
                        "name": "Min Accuracy",
                        "metric": "accuracy",
                        "operator": "gte",
                        "threshold": 0.95,
                        "severity": "critical",
                        "is_active": True,
                        "notify_webhook": True,
                        "notify_emails": ["alerts@example.com"],
                        "is_breached": False,
                        "breach_count": 0,
                        "created_at": "2026-01-01T00:00:00Z",
                        "updated_at": "2026-01-02T00:00:00Z",
                    }
                ],
                "next": None,
                "previous": None,
            },
        )
    )
    client = Client(api_key="test-key")
    page = client.quality_targets.list(PROJECT_UID)
    assert len(page.items) == 1
    assert page.items[0].uid == "qt-001"
    assert page.items[0].name == "Min Accuracy"
    assert page.items[0].metric == "accuracy"
    assert page.items[0].operator == "gte"
    assert page.items[0].threshold == 0.95
    assert page.items[0].severity == "critical"
    assert page.items[0].notify_emails == ["alerts@example.com"]
    assert page.has_more is False
    client.close()


@respx.mock
def test_get_quality_target():
    """QualityTargets.get() returns a single QualityTarget by uid."""
    uid = "qt-001"
    respx.get(f"{BASE_URL}/projects/{PROJECT_UID}/quality-targets/{uid}/").mock(
        return_value=httpx.Response(
            200,
            json={
                "uid": uid,
                "name": "Min Accuracy",
                "metric": "accuracy",
                "operator": "gte",
                "threshold": 0.95,
                "severity": "critical",
                "is_active": True,
                "is_breached": True,
                "breach_count": 3,
                "last_value": 0.91,
                "last_evaluated_at": "2026-01-03T00:00:00Z",
                "last_breached_at": "2026-01-03T00:00:00Z",
            },
        )
    )
    client = Client(api_key="test-key")
    target = client.quality_targets.get(PROJECT_UID, uid)
    assert target.uid == uid
    assert target.name == "Min Accuracy"
    assert target.is_breached is True
    assert target.breach_count == 3
    assert target.last_value == 0.91
    client.close()


@respx.mock
def test_create_quality_target():
    """QualityTargets.create() sends correct payload and returns QualityTarget."""
    route = respx.post(f"{BASE_URL}/projects/{PROJECT_UID}/quality-targets/").mock(
        return_value=httpx.Response(
            201,
            json={
                "uid": "qt-new-001",
                "name": "Min Accuracy",
                "metric": "accuracy",
                "operator": "gte",
                "threshold": 0.95,
                "severity": "critical",
                "is_active": True,
                "notify_webhook": False,
                "notify_emails": ["alerts@example.com"],
            },
        )
    )
    client = Client(api_key="test-key")
    target = client.quality_targets.create(
        PROJECT_UID,
        name="Min Accuracy",
        metric="accuracy",
        threshold=0.95,
        operator="gte",
        severity="critical",
        is_active=True,
        notify_webhook=False,
        notify_emails=["alerts@example.com"],
    )
    assert target.uid == "qt-new-001"
    assert target.name == "Min Accuracy"
    assert target.notify_webhook is False
    request_body = json.loads(route.calls.last.request.content)
    assert request_body["name"] == "Min Accuracy"
    assert request_body["metric"] == "accuracy"
    assert request_body["threshold"] == 0.95
    assert request_body["operator"] == "gte"
    assert request_body["severity"] == "critical"
    assert request_body["is_active"] is True
    assert request_body["notify_webhook"] is False
    assert request_body["notify_emails"] == ["alerts@example.com"]
    client.close()


@respx.mock
def test_create_quality_target_minimal_payload():
    """QualityTargets.create() omits optional fields when not provided."""
    route = respx.post(f"{BASE_URL}/projects/{PROJECT_UID}/quality-targets/").mock(
        return_value=httpx.Response(
            201,
            json={
                "uid": "qt-new-002",
                "name": "Max Error Rate",
                "metric": "error_rate",
                "operator": "lte",
                "threshold": 0.05,
            },
        )
    )
    client = Client(api_key="test-key")
    target = client.quality_targets.create(
        PROJECT_UID,
        name="Max Error Rate",
        metric="error_rate",
        threshold=0.05,
    )
    assert target.uid == "qt-new-002"
    request_body = json.loads(route.calls.last.request.content)
    assert request_body == {"name": "Max Error Rate", "metric": "error_rate", "threshold": 0.05}
    assert "operator" not in request_body
    assert "severity" not in request_body
    client.close()


@respx.mock
def test_update_quality_target():
    """QualityTargets.update() sends only provided fields and returns QualityTarget."""
    uid = "qt-001"
    route = respx.patch(f"{BASE_URL}/projects/{PROJECT_UID}/quality-targets/{uid}/").mock(
        return_value=httpx.Response(
            200,
            json={
                "uid": uid,
                "name": "Updated Accuracy",
                "metric": "accuracy",
                "operator": "gte",
                "threshold": 0.99,
                "severity": "warning",
                "is_active": False,
            },
        )
    )
    client = Client(api_key="test-key")
    target = client.quality_targets.update(
        PROJECT_UID,
        uid,
        name="Updated Accuracy",
        threshold=0.99,
        severity="warning",
        is_active=False,
    )
    assert target.uid == uid
    assert target.name == "Updated Accuracy"
    assert target.threshold == 0.99
    assert target.severity == "warning"
    assert target.is_active is False
    request_body = json.loads(route.calls.last.request.content)
    assert request_body["name"] == "Updated Accuracy"
    assert request_body["threshold"] == 0.99
    assert request_body["severity"] == "warning"
    assert request_body["is_active"] is False
    assert "metric" not in request_body
    assert "operator" not in request_body
    client.close()


@respx.mock
def test_delete_quality_target():
    """QualityTargets.delete() issues a DELETE and returns None."""
    uid = "qt-001"
    route = respx.delete(f"{BASE_URL}/projects/{PROJECT_UID}/quality-targets/{uid}/").mock(
        return_value=httpx.Response(204)
    )
    client = Client(api_key="test-key")
    result = client.quality_targets.delete(PROJECT_UID, uid)
    assert result is None
    assert route.called
    client.close()


@respx.mock
def test_evaluate_quality_targets():
    """QualityTargets.evaluate() returns a list of QualityTargetEvaluation objects."""
    route = respx.post(f"{BASE_URL}/projects/{PROJECT_UID}/quality-targets/evaluate/").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "uid": "qt-001",
                    "name": "Min Accuracy",
                    "metric": "accuracy",
                    "threshold": 0.95,
                    "operator": "gte",
                    "current_value": 0.91,
                    "is_breached": True,
                    "severity": "critical",
                },
                {
                    "uid": "qt-002",
                    "name": "Max Error Rate",
                    "metric": "error_rate",
                    "threshold": 0.05,
                    "operator": "lte",
                    "current_value": 0.02,
                    "is_breached": False,
                    "severity": None,
                },
            ],
        )
    )
    client = Client(api_key="test-key")
    evaluations = client.quality_targets.evaluate(PROJECT_UID)
    assert len(evaluations) == 2
    assert evaluations[0].uid == "qt-001"
    assert evaluations[0].current_value == 0.91
    assert evaluations[0].is_breached is True
    assert evaluations[0].severity == "critical"
    assert evaluations[1].uid == "qt-002"
    assert evaluations[1].is_breached is False
    assert evaluations[1].severity is None
    assert route.called
    client.close()
