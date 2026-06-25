"""Tests for InferenceProviders resource."""

from __future__ import annotations

import json

import httpx
import respx

from avala import Client

BASE_URL = "https://api.avala.ai/api/v1"


@respx.mock
def test_list_inference_providers():
    """InferenceProviders.list() returns a CursorPage of InferenceProvider objects."""
    respx.get(f"{BASE_URL}/inference-providers/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "uid": "ip-001",
                        "name": "OpenAI GPT-4",
                        "description": "Primary inference provider",
                        "provider_type": "openai",
                        "config": {"model": "gpt-4o", "base_url": "https://api.openai.com/v1"},
                        "is_active": True,
                        "project": "proj-001",
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
    page = client.inference_providers.list()
    assert len(page.items) == 1
    assert page.items[0].uid == "ip-001"
    assert page.items[0].name == "OpenAI GPT-4"
    assert page.items[0].provider_type == "openai"
    assert page.items[0].config == {"model": "gpt-4o", "base_url": "https://api.openai.com/v1"}
    assert page.items[0].is_active is True
    assert page.has_more is False
    client.close()


@respx.mock
def test_get_inference_provider():
    """InferenceProviders.get() returns a single InferenceProvider by uid."""
    uid = "ip-001"
    respx.get(f"{BASE_URL}/inference-providers/{uid}/").mock(
        return_value=httpx.Response(
            200,
            json={
                "uid": uid,
                "name": "OpenAI GPT-4",
                "provider_type": "openai",
                "config": {"model": "gpt-4o"},
                "is_active": True,
            },
        )
    )
    client = Client(api_key="test-key")
    provider = client.inference_providers.get(uid)
    assert provider.uid == uid
    assert provider.name == "OpenAI GPT-4"
    assert provider.config == {"model": "gpt-4o"}
    client.close()


@respx.mock
def test_create_inference_provider():
    """InferenceProviders.create() sends correct payload and returns InferenceProvider."""
    route = respx.post(f"{BASE_URL}/inference-providers/").mock(
        return_value=httpx.Response(
            201,
            json={
                "uid": "ip-new-001",
                "name": "Anthropic Claude",
                "description": "Claude inference",
                "provider_type": "anthropic",
                "config": {"model": "claude-3-5-sonnet"},
                "is_active": True,
                "project": "proj-001",
            },
        )
    )
    client = Client(api_key="test-key")
    provider = client.inference_providers.create(
        name="Anthropic Claude",
        provider_type="anthropic",
        config={"model": "claude-3-5-sonnet", "api_key": "TEST_KEY_NOT_REAL"},
        description="Claude inference",
        is_active=True,
        project="proj-001",
    )
    assert provider.uid == "ip-new-001"
    assert provider.name == "Anthropic Claude"
    assert provider.provider_type == "anthropic"
    request_body = json.loads(route.calls.last.request.content)
    assert request_body["name"] == "Anthropic Claude"
    assert request_body["provider_type"] == "anthropic"
    assert request_body["config"] == {"model": "claude-3-5-sonnet", "api_key": "TEST_KEY_NOT_REAL"}
    assert request_body["description"] == "Claude inference"
    assert request_body["is_active"] is True
    assert request_body["project"] == "proj-001"
    client.close()


@respx.mock
def test_create_inference_provider_minimal():
    """InferenceProviders.create() omits optional fields when not provided."""
    route = respx.post(f"{BASE_URL}/inference-providers/").mock(
        return_value=httpx.Response(
            201,
            json={
                "uid": "ip-new-002",
                "name": "Local Model",
                "provider_type": "custom",
                "config": {"base_url": "http://localhost:8000"},
                "is_active": True,
            },
        )
    )
    client = Client(api_key="test-key")
    provider = client.inference_providers.create(
        name="Local Model",
        provider_type="custom",
        config={"base_url": "http://localhost:8000"},
    )
    assert provider.uid == "ip-new-002"
    request_body = json.loads(route.calls.last.request.content)
    assert request_body == {
        "name": "Local Model",
        "provider_type": "custom",
        "config": {"base_url": "http://localhost:8000"},
    }
    assert "description" not in request_body
    assert "is_active" not in request_body
    assert "project" not in request_body
    client.close()


@respx.mock
def test_update_inference_provider():
    """InferenceProviders.update() sends only provided fields and returns InferenceProvider."""
    uid = "ip-001"
    route = respx.patch(f"{BASE_URL}/inference-providers/{uid}/").mock(
        return_value=httpx.Response(
            200,
            json={
                "uid": uid,
                "name": "Renamed Provider",
                "provider_type": "openai",
                "config": {"model": "gpt-4o-mini"},
                "is_active": False,
            },
        )
    )
    client = Client(api_key="test-key")
    provider = client.inference_providers.update(
        uid,
        name="Renamed Provider",
        config={"model": "gpt-4o-mini"},
        is_active=False,
    )
    assert provider.uid == uid
    assert provider.name == "Renamed Provider"
    assert provider.is_active is False
    request_body = json.loads(route.calls.last.request.content)
    assert request_body == {
        "name": "Renamed Provider",
        "config": {"model": "gpt-4o-mini"},
        "is_active": False,
    }
    assert "provider_type" not in request_body
    assert "description" not in request_body
    client.close()


@respx.mock
def test_delete_inference_provider():
    """InferenceProviders.delete() issues a DELETE request and returns None."""
    uid = "ip-001"
    route = respx.delete(f"{BASE_URL}/inference-providers/{uid}/").mock(return_value=httpx.Response(204))
    client = Client(api_key="test-key")
    result = client.inference_providers.delete(uid)
    assert result is None
    assert route.called
    client.close()


@respx.mock
def test_test_inference_provider():
    """InferenceProviders.test() posts to the test endpoint and returns the raw payload."""
    uid = "ip-001"
    respx.post(f"{BASE_URL}/inference-providers/{uid}/test/").mock(
        return_value=httpx.Response(200, json={"success": True, "latency_ms": 120})
    )
    client = Client(api_key="test-key")
    result = client.inference_providers.test(uid)
    assert result["success"] is True
    assert result["latency_ms"] == 120
    client.close()
