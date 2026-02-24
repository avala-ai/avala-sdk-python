"""Tests for Slices resource."""

from __future__ import annotations

import json

import httpx
import respx

from avala import Client

BASE_URL = "https://api.avala.ai/api/v1"


@respx.mock
def test_list_slices():
    """Slices.list() returns a CursorPage of Slice objects."""
    owner = "test-org"
    respx.get(f"{BASE_URL}/slices/{owner}/list/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "uid": "slice-001",
                        "name": "Test Slice",
                        "slug": "test-slice",
                        "owner_name": owner,
                        "visibility": "public",
                        "item_count": 50,
                    }
                ],
                "next": None,
                "previous": None,
            },
        )
    )
    client = Client(api_key="test-key")
    page = client.slices.list(owner)
    assert len(page.items) == 1
    assert page.items[0].uid == "slice-001"
    assert page.items[0].name == "Test Slice"
    assert page.items[0].item_count == 50
    assert page.has_more is False
    client.close()


@respx.mock
def test_get_slice():
    """Slices.get() returns a single Slice by owner and slug."""
    owner = "test-org"
    slug = "test-slice"
    respx.get(f"{BASE_URL}/slices/{owner}/{slug}/").mock(
        return_value=httpx.Response(
            200,
            json={
                "uid": "slice-001",
                "name": "Test Slice",
                "slug": slug,
                "owner_name": owner,
                "visibility": "public",
                "item_count": 50,
            },
        )
    )
    client = Client(api_key="test-key")
    s = client.slices.get(owner, slug)
    assert s.uid == "slice-001"
    assert s.slug == slug
    assert s.owner_name == owner
    client.close()


@respx.mock
def test_create_slice():
    """Slices.create() sends correct payload and returns Slice."""
    route = respx.post(f"{BASE_URL}/slices/").mock(
        return_value=httpx.Response(
            201,
            json={
                "uid": "slice-new-001",
                "name": "New Slice",
                "slug": "new-slice",
                "visibility": "private",
                "item_count": 0,
            },
        )
    )
    client = Client(api_key="test-key")
    s = client.slices.create(
        name="New Slice",
        visibility="private",
        sub_slices=[{"dataset": "ds-001", "query": "label=car"}],
    )
    assert s.uid == "slice-new-001"
    assert s.name == "New Slice"
    request_body = json.loads(route.calls.last.request.content)
    assert request_body["name"] == "New Slice"
    assert request_body["visibility"] == "private"
    assert len(request_body["sub_slices"]) == 1
    client.close()


@respx.mock
def test_list_slice_items():
    """Slices.list_items() returns a CursorPage of SliceItem objects."""
    owner = "test-org"
    slug = "test-slice"
    respx.get(f"{BASE_URL}/slices/{owner}/{slug}/items/list/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "uid": "item-001",
                        "key": "image_001.png",
                        "dataset": "ds-001",
                        "url": "https://example.com/image_001.png",
                    }
                ],
                "next": None,
                "previous": None,
            },
        )
    )
    client = Client(api_key="test-key")
    page = client.slices.list_items(owner, slug)
    assert len(page.items) == 1
    assert page.items[0].uid == "item-001"
    assert page.items[0].key == "image_001.png"
    assert page.has_more is False
    client.close()
