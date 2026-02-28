"""Fleet events resource."""

from __future__ import annotations

from typing import Any, Dict, List

from avala._pagination import CursorPage
from avala.resources._base import BaseAsyncResource, BaseSyncResource
from avala.types.fleet import BatchEventParams, FleetEvent


class FleetEvents(BaseSyncResource):
    def list(
        self,
        *,
        recording_id: str | None = None,
        device_id: str | None = None,
        type: str | None = None,
        severity: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> CursorPage[FleetEvent]:
        params: dict[str, Any] = {}
        if recording_id is not None:
            params["recording"] = recording_id
        if device_id is not None:
            params["device"] = device_id
        if type is not None:
            params["type"] = type
        if severity is not None:
            params["severity"] = severity
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page("/fleet/events/", FleetEvent, params=params or None)

    def create(
        self,
        *,
        recording: str,
        device: str,
        label: str,
        type: str,
        timestamp: str,
        severity: str | None = None,
        description: str | None = None,
        duration_ms: int | None = None,
        tags: List[str] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> FleetEvent:
        payload: dict[str, Any] = {
            "recording": recording,
            "device": device,
            "label": label,
            "type": type,
            "timestamp": timestamp,
        }
        if severity is not None:
            payload["severity"] = severity
        if description is not None:
            payload["description"] = description
        if duration_ms is not None:
            payload["duration_ms"] = duration_ms
        if tags is not None:
            payload["tags"] = tags
        if metadata is not None:
            payload["metadata"] = metadata
        data = self._transport.request("POST", "/fleet/events/", json=payload)
        return FleetEvent.model_validate(data)

    def create_batch(self, *, events: List[BatchEventParams]) -> int:
        data = self._transport.request("POST", "/fleet/events/batch/", json={"events": events})
        return int(data["created"])

    def get(self, event_id: str) -> FleetEvent:
        data = self._transport.request("GET", f"/fleet/events/{event_id}/")
        return FleetEvent.model_validate(data)

    def delete(self, event_id: str) -> None:
        self._transport.request("DELETE", f"/fleet/events/{event_id}/")


class AsyncFleetEvents(BaseAsyncResource):
    async def list(
        self,
        *,
        recording_id: str | None = None,
        device_id: str | None = None,
        type: str | None = None,
        severity: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> CursorPage[FleetEvent]:
        params: dict[str, Any] = {}
        if recording_id is not None:
            params["recording"] = recording_id
        if device_id is not None:
            params["device"] = device_id
        if type is not None:
            params["type"] = type
        if severity is not None:
            params["severity"] = severity
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page("/fleet/events/", FleetEvent, params=params or None)

    async def create(
        self,
        *,
        recording: str,
        device: str,
        label: str,
        type: str,
        timestamp: str,
        severity: str | None = None,
        description: str | None = None,
        duration_ms: int | None = None,
        tags: List[str] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> FleetEvent:
        payload: dict[str, Any] = {
            "recording": recording,
            "device": device,
            "label": label,
            "type": type,
            "timestamp": timestamp,
        }
        if severity is not None:
            payload["severity"] = severity
        if description is not None:
            payload["description"] = description
        if duration_ms is not None:
            payload["duration_ms"] = duration_ms
        if tags is not None:
            payload["tags"] = tags
        if metadata is not None:
            payload["metadata"] = metadata
        data = await self._transport.request("POST", "/fleet/events/", json=payload)
        return FleetEvent.model_validate(data)

    async def create_batch(self, *, events: List[BatchEventParams]) -> int:
        data = await self._transport.request("POST", "/fleet/events/batch/", json={"events": events})
        return int(data["created"])

    async def get(self, event_id: str) -> FleetEvent:
        data = await self._transport.request("GET", f"/fleet/events/{event_id}/")
        return FleetEvent.model_validate(data)

    async def delete(self, event_id: str) -> None:
        await self._transport.request("DELETE", f"/fleet/events/{event_id}/")
