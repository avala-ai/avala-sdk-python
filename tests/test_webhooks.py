"""Tests for Webhooks and WebhookDeliveries resources."""

from __future__ import annotations

import json

import httpx
import respx

from avala import Client

BASE_URL = "https://api.avala.ai/api/v1"


@respx.mock
def test_list_webhooks():
    """Webhooks.list() returns a CursorPage of Webhook objects."""
    respx.get(f"{BASE_URL}/webhooks/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "uid": "webhook-001",
                        "target_url": "https://example.com/hook",
                        "events": ["task.completed"],
                        "is_active": True,
                        "secret": "shhh",
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
    page = client.webhooks.list()
    assert len(page.items) == 1
    assert page.items[0].uid == "webhook-001"
    assert page.items[0].target_url == "https://example.com/hook"
    assert page.items[0].events == ["task.completed"]
    assert page.items[0].is_active is True
    assert page.has_more is False
    client.close()


@respx.mock
def test_get_webhook():
    """Webhooks.get() returns a single Webhook by uid."""
    uid = "webhook-001"
    respx.get(f"{BASE_URL}/webhooks/{uid}/").mock(
        return_value=httpx.Response(
            200,
            json={
                "uid": uid,
                "target_url": "https://example.com/hook",
                "events": ["task.completed", "task.failed"],
                "is_active": True,
            },
        )
    )
    client = Client(api_key="test-key")
    webhook = client.webhooks.get(uid)
    assert webhook.uid == uid
    assert webhook.events == ["task.completed", "task.failed"]
    client.close()


@respx.mock
def test_create_webhook():
    """Webhooks.create() sends correct payload and returns Webhook."""
    route = respx.post(f"{BASE_URL}/webhooks/").mock(
        return_value=httpx.Response(
            201,
            json={
                "uid": "webhook-new-001",
                "target_url": "https://example.com/new",
                "events": ["task.completed"],
                "is_active": True,
                "secret": "topsecret",
            },
        )
    )
    client = Client(api_key="test-key")
    webhook = client.webhooks.create(
        target_url="https://example.com/new",
        events=["task.completed"],
        is_active=True,
        secret="topsecret",
    )
    assert webhook.uid == "webhook-new-001"
    assert webhook.target_url == "https://example.com/new"
    request_body = json.loads(route.calls.last.request.content)
    assert request_body["target_url"] == "https://example.com/new"
    assert request_body["events"] == ["task.completed"]
    assert request_body["is_active"] is True
    assert request_body["secret"] == "topsecret"
    client.close()


@respx.mock
def test_update_webhook():
    """Webhooks.update() sends correct payload and returns updated Webhook."""
    uid = "webhook-001"
    route = respx.patch(f"{BASE_URL}/webhooks/{uid}/").mock(
        return_value=httpx.Response(
            200,
            json={
                "uid": uid,
                "target_url": "https://example.com/updated",
                "events": ["task.failed"],
                "is_active": False,
            },
        )
    )
    client = Client(api_key="test-key")
    webhook = client.webhooks.update(
        uid,
        target_url="https://example.com/updated",
        events=["task.failed"],
        is_active=False,
    )
    assert webhook.target_url == "https://example.com/updated"
    assert webhook.is_active is False
    request_body = json.loads(route.calls.last.request.content)
    assert request_body["target_url"] == "https://example.com/updated"
    assert request_body["events"] == ["task.failed"]
    assert request_body["is_active"] is False
    client.close()


@respx.mock
def test_delete_webhook():
    """Webhooks.delete() issues a DELETE request."""
    uid = "webhook-001"
    respx.delete(f"{BASE_URL}/webhooks/{uid}/").mock(return_value=httpx.Response(204))
    client = Client(api_key="test-key")
    client.webhooks.delete(uid)
    client.close()


@respx.mock
def test_test_webhook():
    """Webhooks.test() posts to the test endpoint and returns the response body."""
    uid = "webhook-001"
    respx.post(f"{BASE_URL}/webhooks/{uid}/test/").mock(return_value=httpx.Response(200, json={"delivered": True}))
    client = Client(api_key="test-key")
    result = client.webhooks.test(uid)
    assert result["delivered"] is True
    client.close()


@respx.mock
def test_list_webhook_deliveries():
    """WebhookDeliveries.list() returns a CursorPage of WebhookDelivery objects."""
    respx.get(f"{BASE_URL}/webhook-deliveries/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "uid": "delivery-001",
                        "subscription": "webhook-001",
                        "event_type": "task.completed",
                        "payload": {"task": "task-001"},
                        "response_status": 200,
                        "response_body": "ok",
                        "attempts": 1,
                        "status": "delivered",
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
    page = client.webhook_deliveries.list()
    assert len(page.items) == 1
    assert page.items[0].uid == "delivery-001"
    assert page.items[0].subscription == "webhook-001"
    assert page.items[0].event_type == "task.completed"
    assert page.items[0].response_status == 200
    assert page.has_more is False
    client.close()


@respx.mock
def test_get_webhook_delivery():
    """WebhookDeliveries.get() returns a single WebhookDelivery by uid."""
    uid = "delivery-001"
    respx.get(f"{BASE_URL}/webhook-deliveries/{uid}/").mock(
        return_value=httpx.Response(
            200,
            json={
                "uid": uid,
                "subscription": "webhook-001",
                "event_type": "task.failed",
                "attempts": 3,
                "status": "failed",
            },
        )
    )
    client = Client(api_key="test-key")
    delivery = client.webhook_deliveries.get(uid)
    assert delivery.uid == uid
    assert delivery.event_type == "task.failed"
    assert delivery.attempts == 3
    assert delivery.status == "failed"
    client.close()
