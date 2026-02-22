"""Storage configs resource."""

from __future__ import annotations

from typing import Any

from avala._pagination import CursorPage
from avala.resources._base import BaseAsyncResource, BaseSyncResource
from avala.types.storage_config import StorageConfig


class StorageConfigs(BaseSyncResource):
    def list(self, *, limit: int | None = None, cursor: str | None = None) -> CursorPage[StorageConfig]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page("/storage-configs/", StorageConfig, params=params or None)

    def create(
        self,
        *,
        name: str,
        provider: str,
        s3_bucket_name: str | None = None,
        s3_bucket_region: str | None = None,
        s3_bucket_prefix: str | None = None,
        s3_access_key_id: str | None = None,
        s3_secret_access_key: str | None = None,
        s3_is_accelerated: bool = False,
        gc_storage_bucket_name: str | None = None,
        gc_storage_prefix: str | None = None,
        gc_storage_auth_json_content: str | None = None,
    ) -> StorageConfig:
        payload: dict[str, Any] = {"name": name, "provider": provider}
        if s3_bucket_name is not None:
            payload["s3_bucket_name"] = s3_bucket_name
        if s3_bucket_region is not None:
            payload["s3_bucket_region"] = s3_bucket_region
        if s3_bucket_prefix is not None:
            payload["s3_bucket_prefix"] = s3_bucket_prefix
        if s3_access_key_id is not None:
            payload["s3_access_key_id"] = s3_access_key_id
        if s3_secret_access_key is not None:
            payload["s3_secret_access_key"] = s3_secret_access_key
        if s3_is_accelerated:
            payload["s3_is_accelerated"] = s3_is_accelerated
        if gc_storage_bucket_name is not None:
            payload["gc_storage_bucket_name"] = gc_storage_bucket_name
        if gc_storage_prefix is not None:
            payload["gc_storage_prefix"] = gc_storage_prefix
        if gc_storage_auth_json_content is not None:
            payload["gc_storage_auth_json_content"] = gc_storage_auth_json_content
        data = self._transport.request("POST", "/storage-configs/", json=payload)
        return StorageConfig.model_validate(data)

    def test(self, uid: str) -> dict[str, Any]:
        data = self._transport.request("POST", f"/storage-configs/{uid}/test/")
        return data  # type: ignore[no-any-return]

    def delete(self, uid: str) -> None:
        self._transport.request("DELETE", f"/storage-configs/{uid}/")


class AsyncStorageConfigs(BaseAsyncResource):
    async def list(self, *, limit: int | None = None, cursor: str | None = None) -> CursorPage[StorageConfig]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page("/storage-configs/", StorageConfig, params=params or None)

    async def create(
        self,
        *,
        name: str,
        provider: str,
        s3_bucket_name: str | None = None,
        s3_bucket_region: str | None = None,
        s3_bucket_prefix: str | None = None,
        s3_access_key_id: str | None = None,
        s3_secret_access_key: str | None = None,
        s3_is_accelerated: bool = False,
        gc_storage_bucket_name: str | None = None,
        gc_storage_prefix: str | None = None,
        gc_storage_auth_json_content: str | None = None,
    ) -> StorageConfig:
        payload: dict[str, Any] = {"name": name, "provider": provider}
        if s3_bucket_name is not None:
            payload["s3_bucket_name"] = s3_bucket_name
        if s3_bucket_region is not None:
            payload["s3_bucket_region"] = s3_bucket_region
        if s3_bucket_prefix is not None:
            payload["s3_bucket_prefix"] = s3_bucket_prefix
        if s3_access_key_id is not None:
            payload["s3_access_key_id"] = s3_access_key_id
        if s3_secret_access_key is not None:
            payload["s3_secret_access_key"] = s3_secret_access_key
        if s3_is_accelerated:
            payload["s3_is_accelerated"] = s3_is_accelerated
        if gc_storage_bucket_name is not None:
            payload["gc_storage_bucket_name"] = gc_storage_bucket_name
        if gc_storage_prefix is not None:
            payload["gc_storage_prefix"] = gc_storage_prefix
        if gc_storage_auth_json_content is not None:
            payload["gc_storage_auth_json_content"] = gc_storage_auth_json_content
        data = await self._transport.request("POST", "/storage-configs/", json=payload)
        return StorageConfig.model_validate(data)

    async def test(self, uid: str) -> dict[str, Any]:
        data = await self._transport.request("POST", f"/storage-configs/{uid}/test/")
        return data  # type: ignore[no-any-return]

    async def delete(self, uid: str) -> None:
        await self._transport.request("DELETE", f"/storage-configs/{uid}/")
