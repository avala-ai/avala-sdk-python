"""Datasets resource."""

from __future__ import annotations

from typing import Any

from avala._pagination import CursorPage
from avala.resources._base import BaseAsyncResource, BaseSyncResource
from avala.types.dataset import Dataset, DatasetItem, DatasetSequence


class Datasets(BaseSyncResource):
    def list(self, *, limit: int | None = None, cursor: str | None = None) -> CursorPage[Dataset]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page("/datasets/", Dataset, params=params or None)

    def get(self, uid: str) -> Dataset:
        data = self._transport.request("GET", f"/datasets/{uid}/")
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
    async def list(self, *, limit: int | None = None, cursor: str | None = None) -> CursorPage[Dataset]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page("/datasets/", Dataset, params=params or None)

    async def get(self, uid: str) -> Dataset:
        data = await self._transport.request("GET", f"/datasets/{uid}/")
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
