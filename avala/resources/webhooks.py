"""Webhooks resource."""

from __future__ import annotations

from typing import Any, List

from avala._pagination import CursorPage
from avala.resources._base import BaseAsyncResource, BaseSyncResource
from avala.types.webhook import Webhook, WebhookDelivery


class Webhooks(BaseSyncResource):
    def list(self, *, limit: int | None = None, cursor: str | None = None) -> CursorPage[Webhook]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page("/webhooks/", Webhook, params=params or None)

    def get(self, uid: str) -> Webhook:
        data = self._transport.request("GET", f"/webhooks/{uid}/")
        return Webhook.model_validate(data)

    def create(
        self,
        *,
        target_url: str,
        events: List[str],
        is_active: bool | None = None,
        secret: str | None = None,
    ) -> Webhook:
        payload: dict[str, Any] = {"target_url": target_url, "events": events}
        if is_active is not None:
            payload["is_active"] = is_active
        if secret is not None:
            payload["secret"] = secret
        data = self._transport.request("POST", "/webhooks/", json=payload)
        return Webhook.model_validate(data)

    def update(
        self,
        uid: str,
        *,
        target_url: str | None = None,
        events: List[str] | None = None,
        is_active: bool | None = None,
    ) -> Webhook:
        payload: dict[str, Any] = {}
        if target_url is not None:
            payload["target_url"] = target_url
        if events is not None:
            payload["events"] = events
        if is_active is not None:
            payload["is_active"] = is_active
        data = self._transport.request("PATCH", f"/webhooks/{uid}/", json=payload)
        return Webhook.model_validate(data)

    def delete(self, uid: str) -> None:
        self._transport.request("DELETE", f"/webhooks/{uid}/")

    def test(self, uid: str) -> dict[str, Any]:
        data = self._transport.request("POST", f"/webhooks/{uid}/test/")
        return data  # type: ignore[no-any-return]


class AsyncWebhooks(BaseAsyncResource):
    async def list(self, *, limit: int | None = None, cursor: str | None = None) -> CursorPage[Webhook]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page("/webhooks/", Webhook, params=params or None)

    async def get(self, uid: str) -> Webhook:
        data = await self._transport.request("GET", f"/webhooks/{uid}/")
        return Webhook.model_validate(data)

    async def create(
        self,
        *,
        target_url: str,
        events: List[str],
        is_active: bool | None = None,
        secret: str | None = None,
    ) -> Webhook:
        payload: dict[str, Any] = {"target_url": target_url, "events": events}
        if is_active is not None:
            payload["is_active"] = is_active
        if secret is not None:
            payload["secret"] = secret
        data = await self._transport.request("POST", "/webhooks/", json=payload)
        return Webhook.model_validate(data)

    async def update(
        self,
        uid: str,
        *,
        target_url: str | None = None,
        events: List[str] | None = None,
        is_active: bool | None = None,
    ) -> Webhook:
        payload: dict[str, Any] = {}
        if target_url is not None:
            payload["target_url"] = target_url
        if events is not None:
            payload["events"] = events
        if is_active is not None:
            payload["is_active"] = is_active
        data = await self._transport.request("PATCH", f"/webhooks/{uid}/", json=payload)
        return Webhook.model_validate(data)

    async def delete(self, uid: str) -> None:
        await self._transport.request("DELETE", f"/webhooks/{uid}/")

    async def test(self, uid: str) -> dict[str, Any]:
        data = await self._transport.request("POST", f"/webhooks/{uid}/test/")
        return data  # type: ignore[no-any-return]


class WebhookDeliveries(BaseSyncResource):
    def list(self, *, limit: int | None = None, cursor: str | None = None) -> CursorPage[WebhookDelivery]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page("/webhook-deliveries/", WebhookDelivery, params=params or None)

    def get(self, uid: str) -> WebhookDelivery:
        data = self._transport.request("GET", f"/webhook-deliveries/{uid}/")
        return WebhookDelivery.model_validate(data)


class AsyncWebhookDeliveries(BaseAsyncResource):
    async def list(self, *, limit: int | None = None, cursor: str | None = None) -> CursorPage[WebhookDelivery]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page("/webhook-deliveries/", WebhookDelivery, params=params or None)

    async def get(self, uid: str) -> WebhookDelivery:
        data = await self._transport.request("GET", f"/webhook-deliveries/{uid}/")
        return WebhookDelivery.model_validate(data)
