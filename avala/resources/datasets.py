"""Datasets resource."""

from __future__ import annotations

from typing import Any

from avala._pagination import CursorPage
from avala.resources._base import BaseAsyncResource, BaseSyncResource
from avala.types.dataset import Dataset, DatasetItem, DatasetSequence


class Datasets(BaseSyncResource):
    def list(
        self,
        *,
        data_type: str | None = None,
        name: str | None = None,
        status: str | None = None,
        visibility: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> CursorPage[Dataset]:
        params: dict[str, Any] = {}
        if data_type is not None:
            params["data_type"] = data_type
        if name is not None:
            params["name"] = name
        if status is not None:
            params["status"] = status
        if visibility is not None:
            params["visibility"] = visibility
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page("/datasets/", Dataset, params=params or None)

    def get(self, uid: str) -> Dataset:
        data = self._transport.request("GET", f"/datasets/{uid}/")
        return Dataset.model_validate(data)

    def create(
        self,
        *,
        name: str,
        slug: str,
        data_type: str,
        is_sequence: bool = False,
        visibility: str = "private",
        create_metadata: bool = True,
        provider_config: dict[str, Any] | None = None,
        owner_name: str | None = None,
    ) -> Dataset:
        payload: dict[str, Any] = {
            "name": name,
            "slug": slug,
            "data_type": data_type,
            "is_sequence": is_sequence,
            "visibility": visibility,
            "create_metadata": create_metadata,
        }
        if provider_config is not None:
            payload["provider_config"] = provider_config
        if owner_name is not None:
            payload["owner_name"] = owner_name
        data = self._transport.request("POST", "/datasets/", json=payload)
        return Dataset.model_validate(data)

    def list_items(
        self, owner: str, slug: str, *, limit: int | None = None, cursor: str | None = None
    ) -> CursorPage[DatasetItem]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page(f"/datasets/{owner}/{slug}/items/", DatasetItem, params=params or None)

    def get_item(self, owner: str, slug: str, item_uid: str) -> DatasetItem:
        data = self._transport.request("GET", f"/datasets/{owner}/{slug}/items/{item_uid}/")
        return DatasetItem.model_validate(data)

    def list_sequences(
        self, owner: str, slug: str, *, limit: int | None = None, cursor: str | None = None
    ) -> CursorPage[DatasetSequence]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page(
            f"/datasets/{owner}/{slug}/sequences/", DatasetSequence, params=params or None
        )

    def get_sequence(self, owner: str, slug: str, sequence_uid: str) -> DatasetSequence:
        data = self._transport.request("GET", f"/datasets/{owner}/{slug}/sequences/{sequence_uid}/")
        return DatasetSequence.model_validate(data)


class AsyncDatasets(BaseAsyncResource):
    async def list(
        self,
        *,
        data_type: str | None = None,
        name: str | None = None,
        status: str | None = None,
        visibility: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> CursorPage[Dataset]:
        params: dict[str, Any] = {}
        if data_type is not None:
            params["data_type"] = data_type
        if name is not None:
            params["name"] = name
        if status is not None:
            params["status"] = status
        if visibility is not None:
            params["visibility"] = visibility
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page("/datasets/", Dataset, params=params or None)

    async def get(self, uid: str) -> Dataset:
        data = await self._transport.request("GET", f"/datasets/{uid}/")
        return Dataset.model_validate(data)

    async def create(
        self,
        *,
        name: str,
        slug: str,
        data_type: str,
        is_sequence: bool = False,
        visibility: str = "private",
        create_metadata: bool = True,
        provider_config: dict[str, Any] | None = None,
        owner_name: str | None = None,
    ) -> Dataset:
        payload: dict[str, Any] = {
            "name": name,
            "slug": slug,
            "data_type": data_type,
            "is_sequence": is_sequence,
            "visibility": visibility,
            "create_metadata": create_metadata,
        }
        if provider_config is not None:
            payload["provider_config"] = provider_config
        if owner_name is not None:
            payload["owner_name"] = owner_name
        data = await self._transport.request("POST", "/datasets/", json=payload)
        return Dataset.model_validate(data)

    async def list_items(
        self, owner: str, slug: str, *, limit: int | None = None, cursor: str | None = None
    ) -> CursorPage[DatasetItem]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page(
            f"/datasets/{owner}/{slug}/items/", DatasetItem, params=params or None
        )

    async def get_item(self, owner: str, slug: str, item_uid: str) -> DatasetItem:
        data = await self._transport.request("GET", f"/datasets/{owner}/{slug}/items/{item_uid}/")
        return DatasetItem.model_validate(data)

    async def list_sequences(
        self, owner: str, slug: str, *, limit: int | None = None, cursor: str | None = None
    ) -> CursorPage[DatasetSequence]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page(
            f"/datasets/{owner}/{slug}/sequences/", DatasetSequence, params=params or None
        )

    async def get_sequence(self, owner: str, slug: str, sequence_uid: str) -> DatasetSequence:
        data = await self._transport.request("GET", f"/datasets/{owner}/{slug}/sequences/{sequence_uid}/")
        return DatasetSequence.model_validate(data)
