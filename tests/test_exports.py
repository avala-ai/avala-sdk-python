"""Tests for Exports resource."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest
import respx

from avala import Client

BASE_URL = "https://api.avala.ai/api/v1"


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


@respx.mock
@patch("avala.resources.exports.time.sleep", return_value=None)
def test_wait_export_immediate_completion(mock_sleep):
    """Exports.wait() returns immediately when export is already in terminal status."""
    uid = "export-done"
    respx.get(f"{BASE_URL}/exports/{uid}/").mock(
        return_value=httpx.Response(
            200,
            json={"uid": uid, "status": "exported", "download_url": "https://example.com/export.zip"},
        )
    )
    client = Client(api_key="test-key")
    export = client.exports.wait(uid)
    assert export.uid == uid
    assert export.status == "exported"
    mock_sleep.assert_not_called()
    client.close()


@respx.mock
@patch("avala.resources.exports.time.sleep", return_value=None)
def test_wait_export_polls_until_complete(mock_sleep):
    """Exports.wait() polls multiple times until export reaches terminal status."""
    uid = "export-poll"
    respx.get(f"{BASE_URL}/exports/{uid}/").mock(
        side_effect=[
            httpx.Response(200, json={"uid": uid, "status": "pending"}),
            httpx.Response(200, json={"uid": uid, "status": "processing"}),
            httpx.Response(
                200,
                json={"uid": uid, "status": "exported", "download_url": "https://example.com/out.zip"},
            ),
        ]
    )
    client = Client(api_key="test-key")
    export = client.exports.wait(uid, interval=0.1)
    assert export.status == "exported"
    assert mock_sleep.call_count == 2
    client.close()


@respx.mock
@patch("avala.resources.exports.time.sleep", return_value=None)
@patch("avala.resources.exports.time.monotonic")
def test_wait_export_timeout(mock_monotonic, mock_sleep):
    """Exports.wait() raises TimeoutError when timeout exceeded."""
    uid = "export-slow"
    # monotonic returns: 0 (deadline calc), 1 (first check), 999 (exceeds deadline)
    mock_monotonic.side_effect = [0.0, 1.0, 999.0]
    respx.get(f"{BASE_URL}/exports/{uid}/").mock(
        side_effect=[
            httpx.Response(200, json={"uid": uid, "status": "pending"}),
            httpx.Response(200, json={"uid": uid, "status": "pending"}),
        ]
    )
    client = Client(api_key="test-key")
    with pytest.raises(TimeoutError, match="did not complete"):
        client.exports.wait(uid, timeout=10.0)
    client.close()


@respx.mock
@patch("avala.resources.exports.time.sleep", return_value=None)
def test_wait_export_failed_status(mock_sleep):
    """Exports.wait() returns on 'failed' terminal status."""
    uid = "export-fail"
    respx.get(f"{BASE_URL}/exports/{uid}/").mock(
        return_value=httpx.Response(200, json={"uid": uid, "status": "failed"})
    )
    client = Client(api_key="test-key")
    export = client.exports.wait(uid)
    assert export.status == "failed"
    mock_sleep.assert_not_called()
    client.close()


@respx.mock
@patch("avala.resources.exports.time.sleep", return_value=None)
def test_wait_export_on_poll_callback(mock_sleep):
    """Exports.wait() calls _on_poll callback for each non-terminal poll."""
    uid = "export-cb"
    respx.get(f"{BASE_URL}/exports/{uid}/").mock(
        side_effect=[
            httpx.Response(200, json={"uid": uid, "status": "pending"}),
            httpx.Response(
                200,
                json={"uid": uid, "status": "exported", "download_url": "https://example.com/x.zip"},
            ),
        ]
    )
    callback_count = 0

    def on_poll():
        nonlocal callback_count
        callback_count += 1

    client = Client(api_key="test-key")
    export = client.exports.wait(uid, interval=0.1, _on_poll=on_poll)
    assert export.status == "exported"
    assert callback_count == 1
    client.close()
