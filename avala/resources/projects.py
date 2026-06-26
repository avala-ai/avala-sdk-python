"""Projects resource."""

from __future__ import annotations

from typing import Any

from avala._pagination import CursorPage
from avala.resources._base import BaseAsyncResource, BaseSyncResource
from avala.types.project import Project


# The API key owner's projects live under the user-scoped route. ``me`` resolves to
# the authenticated key owner, and the response is filtered to the projects that owner
# can access. (The bare ``/projects/`` collection is a staff/admin listing of every
# project and is not reachable with a customer API key.)
_LIST_PATH = "/users/me/projects/"


def _detail_path(uid: str) -> str:
    return f"/users/me/projects/{uid}/"


class Projects(BaseSyncResource):
    def list(self, *, limit: int | None = None, cursor: str | None = None) -> CursorPage[Project]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page(_LIST_PATH, Project, params=params or None)

    def get(self, uid: str) -> Project:
        data = self._transport.request("GET", _detail_path(uid))
        return Project.model_validate(data)


class AsyncProjects(BaseAsyncResource):
    async def list(self, *, limit: int | None = None, cursor: str | None = None) -> CursorPage[Project]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page(_LIST_PATH, Project, params=params or None)

    async def get(self, uid: str) -> Project:
        data = await self._transport.request("GET", _detail_path(uid))
        return Project.model_validate(data)
