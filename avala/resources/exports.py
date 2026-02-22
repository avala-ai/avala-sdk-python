"""Exports resource."""

from __future__ import annotations

from typing import Any

from avala._pagination import CursorPage
from avala.resources._base import BaseAsyncResource, BaseSyncResource
from avala.types.export import Export


class Exports(BaseSyncResource):
    def list(self, *, limit: int | None = None, cursor: str | None = None) -> CursorPage[Export]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page("/exports/", Export, params=params or None)

    def create(self, *, project: str | None = None, dataset: str | None = None) -> Export:
        payload: dict[str, Any] = {}
        if project is not None:
            payload["project"] = project
        if dataset is not None:
            payload["dataset"] = dataset
        data = self._transport.request("POST", "/exports/", json=payload)
        return Export.model_validate(data)

    def get(self, uid: str) -> Export:
        data = self._transport.request("GET", f"/exports/{uid}/")
        return Export.model_validate(data)


class AsyncExports(BaseAsyncResource):
    async def list(self, *, limit: int | None = None, cursor: str | None = None) -> CursorPage[Export]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page("/exports/", Export, params=params or None)

    async def create(self, *, project: str | None = None, dataset: str | None = None) -> Export:
        payload: dict[str, Any] = {}
        if project is not None:
            payload["project"] = project
        if dataset is not None:
            payload["dataset"] = dataset
        data = await self._transport.request("POST", "/exports/", json=payload)
        return Export.model_validate(data)

    async def get(self, uid: str) -> Export:
        data = await self._transport.request("GET", f"/exports/{uid}/")
        return Export.model_validate(data)
