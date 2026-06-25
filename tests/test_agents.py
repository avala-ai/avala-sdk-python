"""Tests for Agents resource."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from avala import Client

BASE_URL = "https://api.avala.ai/api/v1"


@respx.mock
def test_list_agents():
    """Agents.list() returns a CursorPage of Agent objects."""
    respx.get(f"{BASE_URL}/agents/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "uid": "agent-001",
                        "name": "Webhook Agent",
                        "description": "Listens for task events",
                        "events": ["task.completed"],
                        "callback_url": "https://example.com/hook",
                        "is_active": True,
                        "project": "proj-001",
                        "task_types": ["bounding_box"],
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
    page = client.agents.list()
    assert len(page.items) == 1
    assert page.items[0].uid == "agent-001"
    assert page.items[0].name == "Webhook Agent"
    assert page.items[0].events == ["task.completed"]
    assert page.items[0].is_active is True
    assert page.has_more is False
    client.close()


@respx.mock
def test_list_agents_pagination():
    """Agents.list() exposes the next cursor when more pages exist."""
    route = respx.get(f"{BASE_URL}/agents/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"uid": "agent-001", "name": "Agent One"},
                    {"uid": "agent-002", "name": "Agent Two"},
                ],
                "next": "https://api.avala.ai/api/v1/agents/?cursor=next-page-token&limit=2",
                "previous": None,
            },
        )
    )
    client = Client(api_key="test-key")
    page = client.agents.list(limit=2)
    assert len(page.items) == 2
    assert page.has_more is True
    assert page.next_cursor == "next-page-token"
    assert route.calls.last.request.url.params["limit"] == "2"
    client.close()


@respx.mock
def test_get_agent():
    """Agents.get() returns a single Agent by uid."""
    uid = "agent-001"
    respx.get(f"{BASE_URL}/agents/{uid}/").mock(
        return_value=httpx.Response(
            200,
            json={
                "uid": uid,
                "name": "Webhook Agent",
                "events": ["task.completed", "task.created"],
                "is_active": True,
                "secret": "whsec_abc123",
                "execution_stats": {"total": 5, "failed": 1},
            },
        )
    )
    client = Client(api_key="test-key")
    agent = client.agents.get(uid)
    assert agent.uid == uid
    assert agent.name == "Webhook Agent"
    assert agent.secret == "whsec_abc123"
    assert agent.execution_stats == {"total": 5, "failed": 1}
    client.close()


@respx.mock
def test_create_agent():
    """Agents.create() sends the correct payload and returns an Agent."""
    route = respx.post(f"{BASE_URL}/agents/").mock(
        return_value=httpx.Response(
            201,
            json={
                "uid": "agent-new-001",
                "name": "New Agent",
                "description": "A freshly created agent",
                "events": ["task.completed"],
                "callback_url": "https://example.com/hook",
                "is_active": True,
                "project": "proj-001",
                "task_types": ["bounding_box"],
            },
        )
    )
    client = Client(api_key="test-key")
    agent = client.agents.create(
        name="New Agent",
        events=["task.completed"],
        description="A freshly created agent",
        callback_url="https://example.com/hook",
        is_active=True,
        project="proj-001",
        task_types=["bounding_box"],
    )
    assert agent.uid == "agent-new-001"
    assert agent.name == "New Agent"

    request_body = json.loads(route.calls.last.request.content)
    assert request_body == {
        "name": "New Agent",
        "events": ["task.completed"],
        "description": "A freshly created agent",
        "callback_url": "https://example.com/hook",
        "is_active": True,
        "project": "proj-001",
        "task_types": ["bounding_box"],
    }
    client.close()


@respx.mock
def test_create_agent_minimal_payload():
    """Agents.create() only sends provided fields (name only)."""
    route = respx.post(f"{BASE_URL}/agents/").mock(
        return_value=httpx.Response(
            201,
            json={"uid": "agent-new-002", "name": "Minimal Agent"},
        )
    )
    client = Client(api_key="test-key")
    agent = client.agents.create(name="Minimal Agent")
    assert agent.uid == "agent-new-002"

    request_body = json.loads(route.calls.last.request.content)
    assert request_body == {"name": "Minimal Agent"}
    assert "description" not in request_body
    assert "events" not in request_body
    client.close()


@respx.mock
def test_update_agent():
    """Agents.update() sends only provided fields and returns the updated Agent."""
    uid = "agent-001"
    route = respx.patch(f"{BASE_URL}/agents/{uid}/").mock(
        return_value=httpx.Response(
            200,
            json={
                "uid": uid,
                "name": "Renamed Agent",
                "is_active": False,
            },
        )
    )
    client = Client(api_key="test-key")
    agent = client.agents.update(uid, name="Renamed Agent", is_active=False)
    assert agent.uid == uid
    assert agent.name == "Renamed Agent"
    assert agent.is_active is False

    request_body = json.loads(route.calls.last.request.content)
    assert request_body == {"name": "Renamed Agent", "is_active": False}
    assert "description" not in request_body
    client.close()


@respx.mock
def test_delete_agent():
    """Agents.delete() issues a DELETE and returns None."""
    uid = "agent-001"
    route = respx.delete(f"{BASE_URL}/agents/{uid}/").mock(return_value=httpx.Response(204))
    client = Client(api_key="test-key")
    result = client.agents.delete(uid)
    assert result is None
    assert route.called
    client.close()


@respx.mock
def test_list_agent_executions():
    """Agents.list_executions() returns a CursorPage of AgentExecution objects."""
    uid = "agent-001"
    respx.get(f"{BASE_URL}/agents/{uid}/executions/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "uid": "exec-001",
                        "registration": uid,
                        "event_type": "task.completed",
                        "task": "task-123",
                        "status": "success",
                        "action": "label",
                        "event_payload": {"task_id": "task-123"},
                        "response_payload": {"ok": True},
                    }
                ],
                "next": None,
                "previous": None,
            },
        )
    )
    client = Client(api_key="test-key")
    page = client.agents.list_executions(uid)
    assert len(page.items) == 1
    assert page.items[0].uid == "exec-001"
    assert page.items[0].registration == uid
    assert page.items[0].event_type == "task.completed"
    assert page.items[0].status == "success"
    assert page.has_more is False
    client.close()


@respx.mock
def test_test_agent():
    """Agents.test() POSTs to the test endpoint and returns the raw dict."""
    uid = "agent-001"
    respx.post(f"{BASE_URL}/agents/{uid}/test/").mock(
        return_value=httpx.Response(200, json={"delivered": True, "status_code": 200})
    )
    client = Client(api_key="test-key")
    result = client.agents.test(uid)
    assert result["delivered"] is True
    assert result["status_code"] == 200
    client.close()


@respx.mock
def test_get_agent_not_found():
    """Agents.get() raises on a 404 response."""
    uid = "missing-agent"
    respx.get(f"{BASE_URL}/agents/{uid}/").mock(return_value=httpx.Response(404, json={"detail": "Not found."}))
    client = Client(api_key="test-key")
    with pytest.raises(Exception):
        client.agents.get(uid)
    client.close()
