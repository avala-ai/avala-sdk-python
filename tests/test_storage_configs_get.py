"""Tests for StorageConfigs.get() and AsyncStorageConfigs.get()."""

from __future__ import annotations

import httpx
import respx

from avala import AsyncClient, Client

BASE_URL = "https://api.avala.ai/api/v1"

STORAGE_CONFIG = {
    "uid": "sc-001",
    "name": "Test S3",
    "provider": "aws_s3",
    "s3_bucket_name": "test-bucket",
    "s3_bucket_region": "us-west-2",
    "s3_bucket_prefix": "data/",
    "s3_auth_method": "access_key",
    "s3_is_accelerated": False,
    "gc_storage_bucket_name": None,
    "gc_storage_prefix": None,
    "is_verified": True,
    "last_verified_at": "2024-01-01T00:00:00Z",
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-02T00:00:00Z",
}


@respx.mock
def test_get_storage_config():
    """StorageConfigs.get() returns a single StorageConfig by UID."""
    respx.get(f"{BASE_URL}/storage-configs/sc-001/").mock(return_value=httpx.Response(200, json=STORAGE_CONFIG))
    client = Client(api_key="test-key")
    sc = client.storage_configs.get("sc-001")
    assert sc.uid == "sc-001"
    assert sc.name == "Test S3"
    assert sc.provider == "aws_s3"
    assert sc.s3_bucket_name == "test-bucket"
    assert sc.s3_bucket_region == "us-west-2"
    assert sc.is_verified is True
    client.close()


@respx.mock
def test_get_storage_config_404():
    """StorageConfigs.get() raises NotFoundError for unknown UID."""
    import pytest

    from avala.errors import NotFoundError

    respx.get(f"{BASE_URL}/storage-configs/sc-missing/").mock(
        return_value=httpx.Response(404, json={"detail": "Not found."})
    )
    client = Client(api_key="test-key")
    with pytest.raises(NotFoundError):
        client.storage_configs.get("sc-missing")
    client.close()


@respx.mock
async def test_async_get_storage_config():
    """AsyncStorageConfigs.get() returns a StorageConfig."""
    respx.get(f"{BASE_URL}/storage-configs/sc-001/").mock(return_value=httpx.Response(200, json=STORAGE_CONFIG))
    async with AsyncClient(api_key="test-key") as client:
        sc = await client.storage_configs.get("sc-001")
    assert sc.uid == "sc-001"
    assert sc.provider == "aws_s3"
