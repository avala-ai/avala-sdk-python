"""Tests for AutoLabelJobs resource."""

from __future__ import annotations

import json

import httpx
import respx

from avala import Client

BASE_URL = "https://api.avala.ai/api/v1"


@respx.mock
def test_list_auto_label_jobs():
    """AutoLabelJobs.list() returns a CursorPage of AutoLabelJob objects."""
    respx.get(f"{BASE_URL}/auto-label-jobs/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "uid": "alj-001",
                        "status": "completed",
                        "model_type": "yolo",
                        "confidence_threshold": 0.75,
                        "labels": ["cat", "dog"],
                        "dry_run": False,
                        "total_items": 100,
                        "processed_items": 100,
                        "successful_items": 95,
                        "failed_items": 5,
                        "skipped_items": 0,
                        "progress_pct": 100.0,
                        "created_at": "2026-01-01T00:00:00Z",
                    }
                ],
                "next": None,
                "previous": None,
            },
        )
    )
    client = Client(api_key="test-key")
    page = client.auto_label_jobs.list()
    assert len(page.items) == 1
    assert page.items[0].uid == "alj-001"
    assert page.items[0].status == "completed"
    assert page.items[0].model_type == "yolo"
    assert page.items[0].confidence_threshold == 0.75
    assert page.items[0].labels == ["cat", "dog"]
    assert page.items[0].successful_items == 95
    assert page.has_more is False
    client.close()


@respx.mock
def test_list_auto_label_jobs_with_project_filter():
    """AutoLabelJobs.list(project=...) sends the project query param."""
    route = respx.get(f"{BASE_URL}/auto-label-jobs/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "uid": "alj-002",
                        "status": "running",
                        "model_type": "sam",
                        "labels": [],
                    }
                ],
                "next": None,
                "previous": None,
            },
        )
    )
    client = Client(api_key="test-key")
    page = client.auto_label_jobs.list(project="proj-uid-001")
    assert len(page.items) == 1
    assert page.items[0].uid == "alj-002"
    assert page.items[0].status == "running"
    assert route.calls.last.request.url.params["project"] == "proj-uid-001"
    client.close()


@respx.mock
def test_get_auto_label_job():
    """AutoLabelJobs.get() returns a single AutoLabelJob by uid."""
    uid = "alj-001"
    respx.get(f"{BASE_URL}/auto-label-jobs/{uid}/").mock(
        return_value=httpx.Response(
            200,
            json={
                "uid": uid,
                "status": "completed",
                "model_type": "yolo",
                "confidence_threshold": 0.5,
                "labels": ["person"],
                "total_items": 10,
                "processed_items": 10,
                "successful_items": 10,
                "summary": {"detections": 42},
                "started_at": "2026-01-01T00:00:00Z",
                "completed_at": "2026-01-01T01:00:00Z",
                "created_at": "2026-01-01T00:00:00Z",
            },
        )
    )
    client = Client(api_key="test-key")
    job = client.auto_label_jobs.get(uid)
    assert job.uid == uid
    assert job.status == "completed"
    assert job.confidence_threshold == 0.5
    assert job.labels == ["person"]
    assert job.summary == {"detections": 42}
    client.close()


@respx.mock
def test_create_auto_label_job():
    """AutoLabelJobs.create() sends correct payload and returns AutoLabelJob."""
    project_uid = "proj-uid-001"
    route = respx.post(f"{BASE_URL}/projects/{project_uid}/auto-label/").mock(
        return_value=httpx.Response(
            201,
            json={
                "uid": "alj-new-001",
                "status": "pending",
                "model_type": "yolo",
                "confidence_threshold": 0.8,
                "labels": ["car", "truck"],
                "dry_run": False,
            },
        )
    )
    client = Client(api_key="test-key")
    job = client.auto_label_jobs.create(
        project_uid,
        model_type="yolo",
        confidence_threshold=0.8,
        labels=["car", "truck"],
        dry_run=False,
    )
    assert job.uid == "alj-new-001"
    assert job.status == "pending"
    assert job.model_type == "yolo"
    assert job.labels == ["car", "truck"]
    request_body = json.loads(route.calls.last.request.content)
    assert request_body["model_type"] == "yolo"
    assert request_body["confidence_threshold"] == 0.8
    assert request_body["labels"] == ["car", "truck"]
    assert request_body["dry_run"] is False
    client.close()


@respx.mock
def test_create_auto_label_job_minimal_payload():
    """AutoLabelJobs.create() omits unset optional fields from the payload."""
    project_uid = "proj-uid-002"
    route = respx.post(f"{BASE_URL}/projects/{project_uid}/auto-label/").mock(
        return_value=httpx.Response(
            201,
            json={"uid": "alj-new-002", "status": "pending"},
        )
    )
    client = Client(api_key="test-key")
    job = client.auto_label_jobs.create(project_uid)
    assert job.uid == "alj-new-002"
    assert job.status == "pending"
    request_body = json.loads(route.calls.last.request.content)
    assert request_body == {}
    assert "model_type" not in request_body
    assert "confidence_threshold" not in request_body
    client.close()


@respx.mock
def test_cancel_auto_label_job():
    """AutoLabelJobs.cancel() issues a DELETE and returns None."""
    uid = "alj-001"
    route = respx.delete(f"{BASE_URL}/auto-label-jobs/{uid}/").mock(return_value=httpx.Response(204))
    client = Client(api_key="test-key")
    result = client.auto_label_jobs.cancel(uid)
    assert result is None
    assert route.called
    client.close()
