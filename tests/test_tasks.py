"""Tests for Tasks resource."""

from __future__ import annotations

import httpx
import respx

from avala import Client

BASE_URL = "https://api.avala.ai/api/v1"


@respx.mock
def test_list_tasks():
    """Tasks.list() returns a CursorPage of Task objects."""
    respx.get(f"{BASE_URL}/tasks/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "uid": "task-001",
                        "type": "annotation",
                        "name": "Task One",
                        "status": "pending",
                        "project": "proj-001",
                        "created_at": "2024-01-01T00:00:00Z",
                        "updated_at": "2024-01-02T00:00:00Z",
                    }
                ],
                "next": None,
                "previous": None,
            },
        )
    )
    client = Client(api_key="test-key")
    page = client.tasks.list()
    assert len(page.items) == 1
    assert page.items[0].uid == "task-001"
    assert page.items[0].type == "annotation"
    assert page.has_more is False
    client.close()


@respx.mock
def test_get_task():
    """Tasks.get() returns a single Task by uid."""
    uid = "task-001"
    respx.get(f"{BASE_URL}/tasks/{uid}/").mock(
        return_value=httpx.Response(
            200,
            json={
                "uid": uid,
                "type": "annotation",
                "name": "Task One",
                "status": "completed",
                "project": "proj-001",
            },
        )
    )
    client = Client(api_key="test-key")
    task = client.tasks.get(uid)
    assert task.uid == uid
    assert task.status == "completed"
    client.close()


@respx.mock
def test_list_tasks_with_project_filter():
    """Tasks.list(project=...) passes project param to the API."""
    route = respx.get(f"{BASE_URL}/tasks/").mock(
        return_value=httpx.Response(
            200,
            json={"results": [], "next": None, "previous": None},
        )
    )
    client = Client(api_key="test-key")
    page = client.tasks.list(project="proj-001")
    assert len(page.items) == 0
    assert "project=proj-001" in str(route.calls.last.request.url)
    client.close()


@respx.mock
def test_list_tasks_with_status_filter():
    """Tasks.list(status=...) passes status param to the API."""
    route = respx.get(f"{BASE_URL}/tasks/").mock(
        return_value=httpx.Response(
            200,
            json={"results": [], "next": None, "previous": None},
        )
    )
    client = Client(api_key="test-key")
    page = client.tasks.list(status="pending")
    assert len(page.items) == 0
    assert "status=pending" in str(route.calls.last.request.url)
    client.close()
