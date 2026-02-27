"""Fleet recordings resource."""

from __future__ import annotations

from typing import Any, List

from avala._pagination import CursorPage
from avala.resources._base import BaseAsyncResource, BaseSyncResource
from avala.types.fleet import Recording


class FleetRecordings(BaseSyncResource):
    def list(
        self,
        *,
        device_id: str | None = None,
        status: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> CursorPage[Recording]:
        params: dict[str, Any] = {}
        if device_id is not None:
            params["device"] = device_id
        if status is not None:
            params["status"] = status
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page("/fleet/recordings/", Recording, params=params or None)

    def get(self, recording_id: str) -> Recording:
        data = self._transport.request("GET", f"/fleet/recordings/{recording_id}/")
        return Recording.model_validate(data)

    def update(
        self,
        recording_id: str,
        *,
        status: str | None = None,
        tags: List[str] | None = None,
    ) -> Recording:
        payload: dict[str, Any] = {}
        if status is not None:
            payload["status"] = status
        if tags is not None:
            payload["tags"] = tags
        data = self._transport.request("PATCH", f"/fleet/recordings/{recording_id}/", json=payload)
        return Recording.model_validate(data)

    def delete(self, recording_id: str) -> None:
        self._transport.request("DELETE", f"/fleet/recordings/{recording_id}/")


class AsyncFleetRecordings(BaseAsyncResource):
    async def list(
        self,
        *,
        device_id: str | None = None,
        status: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> CursorPage[Recording]:
        params: dict[str, Any] = {}
        if device_id is not None:
            params["device"] = device_id
        if status is not None:
            params["status"] = status
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page("/fleet/recordings/", Recording, params=params or None)

    async def get(self, recording_id: str) -> Recording:
        data = await self._transport.request("GET", f"/fleet/recordings/{recording_id}/")
        return Recording.model_validate(data)

    async def update(
        self,
        recording_id: str,
        *,
        status: str | None = None,
        tags: List[str] | None = None,
    ) -> Recording:
        payload: dict[str, Any] = {}
        if status is not None:
            payload["status"] = status
        if tags is not None:
            payload["tags"] = tags
        data = await self._transport.request("PATCH", f"/fleet/recordings/{recording_id}/", json=payload)
        return Recording.model_validate(data)

    async def delete(self, recording_id: str) -> None:
        await self._transport.request("DELETE", f"/fleet/recordings/{recording_id}/")
