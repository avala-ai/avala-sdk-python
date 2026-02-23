"""Tests for Exports resource."""

from __future__ import annotations

import json

import httpx
import respx

from avala import Client

BASE_URL = "https://server.avala.ai/api/v1"


@respx.mock
def test_list_exports():
    """Exports.list() returns a CursorPage of Export objects."""
    respx.get(f"{BASE_URL}/exports/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "uid": "export-001",
                        "status": "completed",
                        "download_url": "https://example.com/export.zip",
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
    page = client.exports.list()
    assert len(page.items) == 1
    assert page.items[0].uid == "export-001"
    assert page.items[0].status == "completed"
    assert page.has_more is False
    client.close()


@respx.mock
def test_get_export():
    """Exports.get() returns a single Export by uid."""
    uid = "export-001"
    respx.get(f"{BASE_URL}/exports/{uid}/").mock(
        return_value=httpx.Response(
            200,
            json={
                "uid": uid,
                "status": "completed",
                "download_url": "https://example.com/export.zip",
            },
        )
    )
    client = Client(api_key="test-key")
    export = client.exports.get(uid)
    assert export.uid == uid
    assert export.download_url == "https://example.com/export.zip"
    client.close()


@respx.mock
def test_create_export_with_project():
    """Exports.create() with project sends correct payload and returns Export."""
    route = respx.post(f"{BASE_URL}/exports/").mock(
        return_value=httpx.Response(
            201,
            json={
                "uid": "export-new-001",
                "status": "pending",
                "download_url": None,
            },
        )
    )
    client = Client(api_key="test-key")
    export = client.exports.create(project="proj-uid-001")
    assert export.uid == "export-new-001"
    assert export.status == "pending"
    request_body = json.loads(route.calls.last.request.content)
    assert request_body["project"] == "proj-uid-001"
    assert "dataset" not in request_body
    client.close()


@respx.mock
def test_create_export_with_dataset():
    """Exports.create() with dataset sends correct payload and returns Export."""
    route = respx.post(f"{BASE_URL}/exports/").mock(
        return_value=httpx.Response(
            201,
            json={
                "uid": "export-new-002",
                "status": "pending",
                "download_url": None,
            },
        )
    )
    client = Client(api_key="test-key")
    export = client.exports.create(dataset="dataset-uid-001")
    assert export.uid == "export-new-002"
    request_body = json.loads(route.calls.last.request.content)
    assert request_body["dataset"] == "dataset-uid-001"
    assert "project" not in request_body
    client.close()
