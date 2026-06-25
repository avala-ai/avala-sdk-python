"""Tests for Consensus resource (project-scoped)."""

from __future__ import annotations

import json

import httpx
import respx

from avala import Client

BASE_URL = "https://api.avala.ai/api/v1"
PROJECT_UID = "proj-uid-001"


@respx.mock
def test_get_summary():
    """Consensus.get_summary() returns a ConsensusSummary."""
    respx.get(f"{BASE_URL}/projects/{PROJECT_UID}/consensus/").mock(
        return_value=httpx.Response(
            200,
            json={
                "mean_score": 0.82,
                "median_score": 0.85,
                "min_score": 0.4,
                "max_score": 1.0,
                "total_items": 120,
                "items_with_consensus": 98,
                "score_distribution": {"0.0-0.5": 5, "0.5-1.0": 115},
                "by_task_name": [{"task_name": "bbox", "mean_score": 0.9}],
            },
        )
    )
    client = Client(api_key="test-key")
    summary = client.consensus.get_summary(PROJECT_UID)
    assert summary.mean_score == 0.82
    assert summary.median_score == 0.85
    assert summary.min_score == 0.4
    assert summary.max_score == 1.0
    assert summary.total_items == 120
    assert summary.items_with_consensus == 98
    assert summary.score_distribution == {"0.0-0.5": 5, "0.5-1.0": 115}
    assert summary.by_task_name == [{"task_name": "bbox", "mean_score": 0.9}]
    client.close()


@respx.mock
def test_list_scores():
    """Consensus.list_scores() returns a CursorPage of ConsensusScore objects."""
    respx.get(f"{BASE_URL}/projects/{PROJECT_UID}/consensus/scores/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "uid": "score-001",
                        "dataset_item_uid": "item-001",
                        "task_name": "bbox",
                        "score_type": "iou",
                        "score": 0.95,
                        "annotator_count": 3,
                        "details": {"matched": 2},
                        "created_at": "2026-01-01T00:00:00Z",
                    }
                ],
                "next": None,
                "previous": None,
            },
        )
    )
    client = Client(api_key="test-key")
    page = client.consensus.list_scores(PROJECT_UID)
    assert len(page.items) == 1
    score = page.items[0]
    assert score.uid == "score-001"
    assert score.dataset_item_uid == "item-001"
    assert score.task_name == "bbox"
    assert score.score_type == "iou"
    assert score.score == 0.95
    assert score.annotator_count == 3
    assert score.details == {"matched": 2}
    assert page.has_more is False
    client.close()


@respx.mock
def test_list_scores_with_params():
    """Consensus.list_scores() forwards limit and cursor query params."""
    route = respx.get(f"{BASE_URL}/projects/{PROJECT_UID}/consensus/scores/").mock(
        return_value=httpx.Response(
            200,
            json={"results": [], "next": None, "previous": None},
        )
    )
    client = Client(api_key="test-key")
    page = client.consensus.list_scores(PROJECT_UID, limit=25, cursor="abc123")
    assert page.items == []
    request_url = route.calls.last.request.url
    assert request_url.params["limit"] == "25"
    assert request_url.params["cursor"] == "abc123"
    client.close()


@respx.mock
def test_compute():
    """Consensus.compute() POSTs and returns a ConsensusComputeResult."""
    route = respx.post(f"{BASE_URL}/projects/{PROJECT_UID}/consensus/compute/").mock(
        return_value=httpx.Response(
            200,
            json={"status": "started", "message": "Consensus computation queued"},
        )
    )
    client = Client(api_key="test-key")
    result = client.consensus.compute(PROJECT_UID)
    assert result.status == "started"
    assert result.message == "Consensus computation queued"
    assert route.calls.last.request.method == "POST"
    client.close()


@respx.mock
def test_get_config():
    """Consensus.get_config() returns a ConsensusConfig."""
    respx.get(f"{BASE_URL}/projects/{PROJECT_UID}/consensus/config/").mock(
        return_value=httpx.Response(
            200,
            json={
                "uid": "config-001",
                "iou_threshold": 0.6,
                "min_agreement_ratio": 0.75,
                "min_annotations": 3,
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-02T00:00:00Z",
            },
        )
    )
    client = Client(api_key="test-key")
    config = client.consensus.get_config(PROJECT_UID)
    assert config.uid == "config-001"
    assert config.iou_threshold == 0.6
    assert config.min_agreement_ratio == 0.75
    assert config.min_annotations == 3
    client.close()


@respx.mock
def test_update_config():
    """Consensus.update_config() PUTs the payload and returns a ConsensusConfig."""
    route = respx.put(f"{BASE_URL}/projects/{PROJECT_UID}/consensus/config/").mock(
        return_value=httpx.Response(
            200,
            json={
                "uid": "config-001",
                "iou_threshold": 0.7,
                "min_agreement_ratio": 0.8,
                "min_annotations": 4,
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-03T00:00:00Z",
            },
        )
    )
    client = Client(api_key="test-key")
    config = client.consensus.update_config(
        PROJECT_UID,
        iou_threshold=0.7,
        min_agreement_ratio=0.8,
        min_annotations=4,
    )
    assert config.iou_threshold == 0.7
    assert config.min_agreement_ratio == 0.8
    assert config.min_annotations == 4
    assert route.calls.last.request.method == "PUT"
    request_body = json.loads(route.calls.last.request.content)
    assert request_body == {
        "iou_threshold": 0.7,
        "min_agreement_ratio": 0.8,
        "min_annotations": 4,
    }
    client.close()


@respx.mock
def test_update_config_partial():
    """Consensus.update_config() only sends provided fields in the payload."""
    route = respx.put(f"{BASE_URL}/projects/{PROJECT_UID}/consensus/config/").mock(
        return_value=httpx.Response(
            200,
            json={
                "uid": "config-001",
                "iou_threshold": 0.9,
                "min_agreement_ratio": 0.5,
                "min_annotations": 2,
            },
        )
    )
    client = Client(api_key="test-key")
    config = client.consensus.update_config(PROJECT_UID, iou_threshold=0.9)
    assert config.iou_threshold == 0.9
    request_body = json.loads(route.calls.last.request.content)
    assert request_body == {"iou_threshold": 0.9}
    assert "min_agreement_ratio" not in request_body
    assert "min_annotations" not in request_body
    client.close()
