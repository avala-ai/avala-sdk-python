"""Item response compatibility for the nullable detail field accepted in PR #20318."""

from __future__ import annotations

from typing import Optional

import httpx
import pytest
import respx

from avala import Client

BASE_URL = "https://api.avala.ai/api/v1"


@pytest.mark.parametrize("resource_name", ["datasets", "slices"])
@pytest.mark.parametrize("fingerprint", ["masks-revision-001", None])
@respx.mock
def test_detail_fingerprint_survives_response_parsing(resource_name: str, fingerprint: Optional[str]) -> None:
    """PR #20318's DatasetItemSerializer exposes a read-only string-or-null fingerprint."""
    respx.get(f"{BASE_URL}/{resource_name}/test-org/test-data/items/item-001/").mock(
        return_value=httpx.Response(200, json={"uid": "item-001", "masks_fingerprint": fingerprint})
    )

    with Client(api_key="test-key") as client:
        item = getattr(client, resource_name).get_item("test-org", "test-data", "item-001")

    assert item.masks_fingerprint == fingerprint
    assert "masks_fingerprint" in item.model_fields_set


@pytest.mark.parametrize("resource_name", ["datasets", "slices"])
@pytest.mark.parametrize("response_shape", ["detail", "list"])
@respx.mock
def test_missing_fingerprint_remains_valid(resource_name: str, response_shape: str) -> None:
    """The accepted field is detail-only; list and older detail responses may omit it."""
    payload = {"uid": "item-001"}
    if response_shape == "detail":
        path = f"/{resource_name}/test-org/test-data/items/item-001/"
        response = payload
    else:
        suffix = "list/" if resource_name == "slices" else ""
        path = f"/{resource_name}/test-org/test-data/items/{suffix}"
        response = {"results": [payload], "next": None, "previous": None}
    respx.get(BASE_URL + path).mock(return_value=httpx.Response(200, json=response))

    with Client(api_key="test-key") as client:
        resource = getattr(client, resource_name)
        item = (
            resource.get_item("test-org", "test-data", "item-001")
            if response_shape == "detail"
            else resource.list_items("test-org", "test-data").items[0]
        )

    assert item.masks_fingerprint is None
    assert "masks_fingerprint" not in item.model_fields_set
