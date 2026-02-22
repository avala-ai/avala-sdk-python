"""Projects resource."""

from __future__ import annotations

from typing import Any

from avala._pagination import CursorPage
from avala.resources._base import BaseAsyncResource, BaseSyncResource
from avala.types.project import Project


class Projects(BaseSyncResource):
    def list(self, *, limit: int | None = None, cursor: str | None = None) -> CursorPage[Project]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page("/projects/", Project, params=params or None)

    def get(self, uid: str) -> Project:
        data = self._transport.request("GET", f"/projects/{uid}/")
        return Project.model_validate(data)


class AsyncProjects(BaseAsyncResource):
    async def list(self, *, limit: int | None = None, cursor: str | None = None) -> CursorPage[Project]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page("/projects/", Project, params=params or None)

    async def get(self, uid: str) -> Project:
        data = await self._transport.request("GET", f"/projects/{uid}/")
        return Project.model_validate(data)
