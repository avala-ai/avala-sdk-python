"""Tasks resource."""

from __future__ import annotations

from typing import Any

from avala._pagination import CursorPage
from avala.resources._base import BaseAsyncResource, BaseSyncResource
from avala.types.task import Task


class Tasks(BaseSyncResource):
    def list(
        self,
        *,
        project: str | None = None,
        status: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> CursorPage[Task]:
        params: dict[str, Any] = {}
        if project is not None:
            params["project"] = project
        if status is not None:
            params["status"] = status
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page("/tasks/", Task, params=params or None)

    def get(self, uid: str) -> Task:
        data = self._transport.request("GET", f"/tasks/{uid}/")
        return Task.model_validate(data)


class AsyncTasks(BaseAsyncResource):
    async def list(
        self,
        *,
        project: str | None = None,
        status: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> CursorPage[Task]:
        params: dict[str, Any] = {}
        if project is not None:
            params["project"] = project
        if status is not None:
            params["status"] = status
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page("/tasks/", Task, params=params or None)

    async def get(self, uid: str) -> Task:
        data = await self._transport.request("GET", f"/tasks/{uid}/")
        return Task.model_validate(data)
