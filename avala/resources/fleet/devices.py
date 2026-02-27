"""Fleet devices resource."""

from __future__ import annotations

from typing import Any, Dict, List

from avala._pagination import CursorPage
from avala.resources._base import BaseAsyncResource, BaseSyncResource
from avala.types.fleet import Device


class FleetDevices(BaseSyncResource):
    def list(
        self,
        *,
        status: str | None = None,
        type: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> CursorPage[Device]:
        params: dict[str, Any] = {}
        if status is not None:
            params["status"] = status
        if type is not None:
            params["type"] = type
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page("/fleet/devices/", Device, params=params or None)

    def register(
        self,
        *,
        name: str,
        type: str,
        firmware_version: str | None = None,
        tags: List[str] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Device:
        payload: dict[str, Any] = {"name": name, "type": type}
        if firmware_version is not None:
            payload["firmware_version"] = firmware_version
        if tags is not None:
            payload["tags"] = tags
        if metadata is not None:
            payload["metadata"] = metadata
        data = self._transport.request("POST", "/fleet/devices/", json=payload)
        return Device.model_validate(data)

    def get(self, device_id: str) -> Device:
        data = self._transport.request("GET", f"/fleet/devices/{device_id}/")
        return Device.model_validate(data)

    def update(
        self,
        device_id: str,
        *,
        name: str | None = None,
        status: str | None = None,
        firmware_version: str | None = None,
        tags: List[str] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Device:
        payload: dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if status is not None:
            payload["status"] = status
        if firmware_version is not None:
            payload["firmware_version"] = firmware_version
        if tags is not None:
            payload["tags"] = tags
        if metadata is not None:
            payload["metadata"] = metadata
        data = self._transport.request("PATCH", f"/fleet/devices/{device_id}/", json=payload)
        return Device.model_validate(data)

    def delete(self, device_id: str) -> None:
        self._transport.request("DELETE", f"/fleet/devices/{device_id}/")


class AsyncFleetDevices(BaseAsyncResource):
    async def list(
        self,
        *,
        status: str | None = None,
        type: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> CursorPage[Device]:
        params: dict[str, Any] = {}
        if status is not None:
            params["status"] = status
        if type is not None:
            params["type"] = type
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page("/fleet/devices/", Device, params=params or None)

    async def register(
        self,
        *,
        name: str,
        type: str,
        firmware_version: str | None = None,
        tags: List[str] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Device:
        payload: dict[str, Any] = {"name": name, "type": type}
        if firmware_version is not None:
            payload["firmware_version"] = firmware_version
        if tags is not None:
            payload["tags"] = tags
        if metadata is not None:
            payload["metadata"] = metadata
        data = await self._transport.request("POST", "/fleet/devices/", json=payload)
        return Device.model_validate(data)

    async def get(self, device_id: str) -> Device:
        data = await self._transport.request("GET", f"/fleet/devices/{device_id}/")
        return Device.model_validate(data)

    async def update(
        self,
        device_id: str,
        *,
        name: str | None = None,
        status: str | None = None,
        firmware_version: str | None = None,
        tags: List[str] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Device:
        payload: dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if status is not None:
            payload["status"] = status
        if firmware_version is not None:
            payload["firmware_version"] = firmware_version
        if tags is not None:
            payload["tags"] = tags
        if metadata is not None:
            payload["metadata"] = metadata
        data = await self._transport.request("PATCH", f"/fleet/devices/{device_id}/", json=payload)
        return Device.model_validate(data)

    async def delete(self, device_id: str) -> None:
        await self._transport.request("DELETE", f"/fleet/devices/{device_id}/")
