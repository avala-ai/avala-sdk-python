"""Fleet alerts and alert channels resources."""

from __future__ import annotations

from typing import Any, Dict

from avala._pagination import CursorPage
from avala.resources._base import BaseAsyncResource, BaseSyncResource
from avala.types.fleet import Alert, AlertChannel


class FleetAlerts(BaseSyncResource):
    def list(
        self,
        *,
        status: str | None = None,
        severity: str | None = None,
        device_id: str | None = None,
        rule_id: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> CursorPage[Alert]:
        params: dict[str, Any] = {}
        if status is not None:
            params["status"] = status
        if severity is not None:
            params["severity"] = severity
        if device_id is not None:
            params["device"] = device_id
        if rule_id is not None:
            params["rule"] = rule_id
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page("/fleet/alerts/", Alert, params=params or None)

    def get(self, alert_id: str) -> Alert:
        data = self._transport.request("GET", f"/fleet/alerts/{alert_id}/")
        return Alert.model_validate(data)

    def acknowledge(self, alert_id: str) -> Alert:
        data = self._transport.request("POST", f"/fleet/alerts/{alert_id}/acknowledge/")
        return Alert.model_validate(data)

    def resolve(self, alert_id: str, *, resolution_note: str | None = None) -> Alert:
        payload: dict[str, Any] = {}
        if resolution_note is not None:
            payload["resolution_note"] = resolution_note
        data = self._transport.request("POST", f"/fleet/alerts/{alert_id}/resolve/", json=payload or None)
        return Alert.model_validate(data)


class AsyncFleetAlerts(BaseAsyncResource):
    async def list(
        self,
        *,
        status: str | None = None,
        severity: str | None = None,
        device_id: str | None = None,
        rule_id: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> CursorPage[Alert]:
        params: dict[str, Any] = {}
        if status is not None:
            params["status"] = status
        if severity is not None:
            params["severity"] = severity
        if device_id is not None:
            params["device"] = device_id
        if rule_id is not None:
            params["rule"] = rule_id
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page("/fleet/alerts/", Alert, params=params or None)

    async def get(self, alert_id: str) -> Alert:
        data = await self._transport.request("GET", f"/fleet/alerts/{alert_id}/")
        return Alert.model_validate(data)

    async def acknowledge(self, alert_id: str) -> Alert:
        data = await self._transport.request("POST", f"/fleet/alerts/{alert_id}/acknowledge/")
        return Alert.model_validate(data)

    async def resolve(self, alert_id: str, *, resolution_note: str | None = None) -> Alert:
        payload: dict[str, Any] = {}
        if resolution_note is not None:
            payload["resolution_note"] = resolution_note
        data = await self._transport.request("POST", f"/fleet/alerts/{alert_id}/resolve/", json=payload or None)
        return Alert.model_validate(data)


class FleetAlertChannels(BaseSyncResource):
    def list(
        self,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> CursorPage[AlertChannel]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page("/fleet/alert-channels/", AlertChannel, params=params or None)

    def create(
        self,
        *,
        name: str,
        type: str,
        config: Dict[str, Any],
    ) -> AlertChannel:
        payload: dict[str, Any] = {"name": name, "type": type, "config": config}
        data = self._transport.request("POST", "/fleet/alert-channels/", json=payload)
        return AlertChannel.model_validate(data)

    def delete(self, channel_id: str) -> None:
        self._transport.request("DELETE", f"/fleet/alert-channels/{channel_id}/")

    def test(self, channel_id: str) -> dict[str, Any]:
        data = self._transport.request("POST", f"/fleet/alert-channels/{channel_id}/test/")
        return data  # type: ignore[no-any-return]


class AsyncFleetAlertChannels(BaseAsyncResource):
    async def list(
        self,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> CursorPage[AlertChannel]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page("/fleet/alert-channels/", AlertChannel, params=params or None)

    async def create(
        self,
        *,
        name: str,
        type: str,
        config: Dict[str, Any],
    ) -> AlertChannel:
        payload: dict[str, Any] = {"name": name, "type": type, "config": config}
        data = await self._transport.request("POST", "/fleet/alert-channels/", json=payload)
        return AlertChannel.model_validate(data)

    async def delete(self, channel_id: str) -> None:
        await self._transport.request("DELETE", f"/fleet/alert-channels/{channel_id}/")

    async def test(self, channel_id: str) -> dict[str, Any]:
        data = await self._transport.request("POST", f"/fleet/alert-channels/{channel_id}/test/")
        return data  # type: ignore[no-any-return]
