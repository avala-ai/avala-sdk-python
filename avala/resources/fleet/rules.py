"""Fleet rules resource."""

from __future__ import annotations

from typing import Any, Dict, List

from avala._pagination import CursorPage
from avala.resources._base import BaseAsyncResource, BaseSyncResource
from avala.types.fleet import Rule


class FleetRules(BaseSyncResource):
    def list(
        self,
        *,
        enabled: bool | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> CursorPage[Rule]:
        params: dict[str, Any] = {}
        if enabled is not None:
            params["enabled"] = enabled
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page("/fleet/rules/", Rule, params=params or None)

    def create(
        self,
        *,
        name: str,
        condition: Dict[str, Any],
        actions: List[Dict[str, Any]],
        description: str | None = None,
        enabled: bool | None = None,
        scope: Dict[str, Any] | None = None,
    ) -> Rule:
        payload: dict[str, Any] = {"name": name, "condition": condition, "actions": actions}
        if description is not None:
            payload["description"] = description
        if enabled is not None:
            payload["enabled"] = enabled
        if scope is not None:
            payload["scope"] = scope
        data = self._transport.request("POST", "/fleet/rules/", json=payload)
        return Rule.model_validate(data)

    def get(self, rule_id: str) -> Rule:
        data = self._transport.request("GET", f"/fleet/rules/{rule_id}/")
        return Rule.model_validate(data)

    def update(
        self,
        rule_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        enabled: bool | None = None,
        condition: Dict[str, Any] | None = None,
        actions: List[Dict[str, Any]] | None = None,
        scope: Dict[str, Any] | None = None,
    ) -> Rule:
        payload: dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if enabled is not None:
            payload["enabled"] = enabled
        if condition is not None:
            payload["condition"] = condition
        if actions is not None:
            payload["actions"] = actions
        if scope is not None:
            payload["scope"] = scope
        data = self._transport.request("PATCH", f"/fleet/rules/{rule_id}/", json=payload)
        return Rule.model_validate(data)

    def delete(self, rule_id: str) -> None:
        self._transport.request("DELETE", f"/fleet/rules/{rule_id}/")


class AsyncFleetRules(BaseAsyncResource):
    async def list(
        self,
        *,
        enabled: bool | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> CursorPage[Rule]:
        params: dict[str, Any] = {}
        if enabled is not None:
            params["enabled"] = enabled
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page("/fleet/rules/", Rule, params=params or None)

    async def create(
        self,
        *,
        name: str,
        condition: Dict[str, Any],
        actions: List[Dict[str, Any]],
        description: str | None = None,
        enabled: bool | None = None,
        scope: Dict[str, Any] | None = None,
    ) -> Rule:
        payload: dict[str, Any] = {"name": name, "condition": condition, "actions": actions}
        if description is not None:
            payload["description"] = description
        if enabled is not None:
            payload["enabled"] = enabled
        if scope is not None:
            payload["scope"] = scope
        data = await self._transport.request("POST", "/fleet/rules/", json=payload)
        return Rule.model_validate(data)

    async def get(self, rule_id: str) -> Rule:
        data = await self._transport.request("GET", f"/fleet/rules/{rule_id}/")
        return Rule.model_validate(data)

    async def update(
        self,
        rule_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        enabled: bool | None = None,
        condition: Dict[str, Any] | None = None,
        actions: List[Dict[str, Any]] | None = None,
        scope: Dict[str, Any] | None = None,
    ) -> Rule:
        payload: dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if enabled is not None:
            payload["enabled"] = enabled
        if condition is not None:
            payload["condition"] = condition
        if actions is not None:
            payload["actions"] = actions
        if scope is not None:
            payload["scope"] = scope
        data = await self._transport.request("PATCH", f"/fleet/rules/{rule_id}/", json=payload)
        return Rule.model_validate(data)

    async def delete(self, rule_id: str) -> None:
        await self._transport.request("DELETE", f"/fleet/rules/{rule_id}/")
