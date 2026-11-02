"""Sequence discovery compatibility: DatasetSequenceItemRetrieveSerializer/lidar_r2.discovery."""

from __future__ import annotations

from typing import Any, Dict, Optional

import httpx
import pytest
import respx

from avala import AsyncClient, Client

BASE_URL = "https://api.avala.ai/api/v1/datasets/test-org/test-data/sequences/"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "descriptor",
    [
        {"provider": "s3", "protocol": "lidar-aws-post-v1", "ground_upload_supported": True},
        {"provider": "r2", "protocol": "lidar-r2-v1", "ground_upload_supported": False},
        None,
    ],
)
@respx.mock
async def test_detail_discovery_survives_decoding(asynchronous: bool, descriptor: Optional[Dict[str, Any]]) -> None:
    """Preserve both existing server descriptors and the SDK's optional-dictionary null compatibility."""
    respx.get(BASE_URL + "sequence-001/").mock(
        return_value=httpx.Response(200, json={"uid": "sequence-001", "annotation_storage": descriptor})
    )

    if asynchronous:
        async with AsyncClient(api_key="test-key") as client:
            sequence = await client.datasets.get_sequence("test-org", "test-data", "sequence-001")
    else:
        with Client(api_key="test-key") as sync_client:
            sequence = sync_client.datasets.get_sequence("test-org", "test-data", "sequence-001")

    assert sequence.annotation_storage == descriptor
    assert "annotation_storage" in sequence.model_fields_set


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("response_shape", ["detail", "list"])
@respx.mock
async def test_omitted_discovery_remains_compatible(asynchronous: bool, response_shape: str) -> None:
    """The server field is detail-only; list/API-key list and older detail payloads omit it."""
    payload = {"uid": "sequence-001"}
    response: Dict[str, Any] = (
        payload if response_shape == "detail" else {"results": [payload], "next": None, "previous": None}
    )
    path = BASE_URL + ("sequence-001/" if response_shape == "detail" else "")
    respx.get(path).mock(return_value=httpx.Response(200, json=response))

    if asynchronous:
        async with AsyncClient(api_key="test-key") as client:
            sequence = (
                await client.datasets.get_sequence("test-org", "test-data", "sequence-001")
                if response_shape == "detail"
                else (await client.datasets.list_sequences("test-org", "test-data")).items[0]
            )
    else:
        with Client(api_key="test-key") as sync_client:
            sequence = (
                sync_client.datasets.get_sequence("test-org", "test-data", "sequence-001")
                if response_shape == "detail"
                else sync_client.datasets.list_sequences("test-org", "test-data").items[0]
            )

    assert sequence.annotation_storage is None
    assert "annotation_storage" not in sequence.model_fields_set
