"""Datasets resource."""

from __future__ import annotations

from typing import Any

from avala._pagination import CursorPage
from avala.resources._base import BaseAsyncResource, BaseSyncResource
from avala.types.dataset import Dataset


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
