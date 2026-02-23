"""Agents resource."""

from __future__ import annotations

from typing import Any, List

from avala._pagination import CursorPage
from avala.resources._base import BaseAsyncResource, BaseSyncResource
from avala.types.agent import Agent, AgentExecution


class Agents(BaseSyncResource):
    def list(self, *, limit: int | None = None, cursor: str | None = None) -> CursorPage[Agent]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page("/agents/", Agent, params=params or None)

    def get(self, uid: str) -> Agent:
        data = self._transport.request("GET", f"/agents/{uid}/")
        return Agent.model_validate(data)

    def create(
        self,
        *,
        name: str,
        events: List[str] | None = None,
        description: str | None = None,
        callback_url: str | None = None,
        is_active: bool | None = None,
        project: str | None = None,
        task_types: List[str] | None = None,
    ) -> Agent:
        payload: dict[str, Any] = {"name": name}
        if events is not None:
            payload["events"] = events
        if description is not None:
            payload["description"] = description
        if callback_url is not None:
            payload["callback_url"] = callback_url
        if is_active is not None:
            payload["is_active"] = is_active
        if project is not None:
            payload["project"] = project
        if task_types is not None:
            payload["task_types"] = task_types
        data = self._transport.request("POST", "/agents/", json=payload)
        return Agent.model_validate(data)

    def update(
        self,
        uid: str,
        *,
        name: str | None = None,
        events: List[str] | None = None,
        description: str | None = None,
        callback_url: str | None = None,
        is_active: bool | None = None,
        project: str | None = None,
        task_types: List[str] | None = None,
    ) -> Agent:
        payload: dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if events is not None:
            payload["events"] = events
        if description is not None:
            payload["description"] = description
        if callback_url is not None:
            payload["callback_url"] = callback_url
        if is_active is not None:
            payload["is_active"] = is_active
        if project is not None:
            payload["project"] = project
        if task_types is not None:
            payload["task_types"] = task_types
        data = self._transport.request("PATCH", f"/agents/{uid}/", json=payload)
        return Agent.model_validate(data)

    def delete(self, uid: str) -> None:
        self._transport.request("DELETE", f"/agents/{uid}/")

    def list_executions(
        self, uid: str, *, limit: int | None = None, cursor: str | None = None
    ) -> CursorPage[AgentExecution]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page(f"/agents/{uid}/executions/", AgentExecution, params=params or None)

    def test(self, uid: str) -> dict[str, Any]:
        data = self._transport.request("POST", f"/agents/{uid}/test/")
        return data  # type: ignore[no-any-return]


class AsyncAgents(BaseAsyncResource):
    async def list(self, *, limit: int | None = None, cursor: str | None = None) -> CursorPage[Agent]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page("/agents/", Agent, params=params or None)

    async def get(self, uid: str) -> Agent:
        data = await self._transport.request("GET", f"/agents/{uid}/")
        return Agent.model_validate(data)

    async def create(
        self,
        *,
        name: str,
        events: List[str] | None = None,
        description: str | None = None,
        callback_url: str | None = None,
        is_active: bool | None = None,
        project: str | None = None,
        task_types: List[str] | None = None,
    ) -> Agent:
        payload: dict[str, Any] = {"name": name}
        if events is not None:
            payload["events"] = events
        if description is not None:
            payload["description"] = description
        if callback_url is not None:
            payload["callback_url"] = callback_url
        if is_active is not None:
            payload["is_active"] = is_active
        if project is not None:
            payload["project"] = project
        if task_types is not None:
            payload["task_types"] = task_types
        data = await self._transport.request("POST", "/agents/", json=payload)
        return Agent.model_validate(data)

    async def update(
        self,
        uid: str,
        *,
        name: str | None = None,
        events: List[str] | None = None,
        description: str | None = None,
        callback_url: str | None = None,
        is_active: bool | None = None,
        project: str | None = None,
        task_types: List[str] | None = None,
    ) -> Agent:
        payload: dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if events is not None:
            payload["events"] = events
        if description is not None:
            payload["description"] = description
        if callback_url is not None:
            payload["callback_url"] = callback_url
        if is_active is not None:
            payload["is_active"] = is_active
        if project is not None:
            payload["project"] = project
        if task_types is not None:
            payload["task_types"] = task_types
        data = await self._transport.request("PATCH", f"/agents/{uid}/", json=payload)
        return Agent.model_validate(data)

    async def delete(self, uid: str) -> None:
        await self._transport.request("DELETE", f"/agents/{uid}/")

    async def list_executions(
        self, uid: str, *, limit: int | None = None, cursor: str | None = None
    ) -> CursorPage[AgentExecution]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page(f"/agents/{uid}/executions/", AgentExecution, params=params or None)

    async def test(self, uid: str) -> dict[str, Any]:
        data = await self._transport.request("POST", f"/agents/{uid}/test/")
        return data  # type: ignore[no-any-return]
