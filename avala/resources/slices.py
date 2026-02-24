"""Slices resource."""

from __future__ import annotations

from typing import Any, Dict, List

from avala._pagination import CursorPage
from avala.resources._base import BaseAsyncResource, BaseSyncResource
from avala.types.slice import Slice, SliceItem


class Slices(BaseSyncResource):
    def list(self, owner: str, *, limit: int | None = None, cursor: str | None = None) -> CursorPage[Slice]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page(f"/slices/{owner}/list/", Slice, params=params or None)

    def get(self, owner: str, slug: str) -> Slice:
        data = self._transport.request("GET", f"/slices/{owner}/{slug}/")
        return Slice.model_validate(data)

    def create(
        self,
        *,
        name: str,
        visibility: str,
        sub_slices: List[Dict[str, Any]],
        organization: str | None = None,
        random_selection_count: int | None = None,
    ) -> Slice:
        payload: dict[str, Any] = {
            "name": name,
            "visibility": visibility,
            "sub_slices": sub_slices,
        }
        if organization is not None:
            payload["organization"] = organization
        if random_selection_count is not None:
            payload["random_selection_count"] = random_selection_count
        data = self._transport.request("POST", "/slices/", json=payload)
        return Slice.model_validate(data)

    def list_items(
        self, owner: str, slug: str, *, limit: int | None = None, cursor: str | None = None
    ) -> CursorPage[SliceItem]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page(f"/slices/{owner}/{slug}/items/list/", SliceItem, params=params or None)

    def get_item(self, owner: str, slug: str, item_uid: str) -> SliceItem:
        data = self._transport.request("GET", f"/slices/{owner}/{slug}/items/{item_uid}/")
        return SliceItem.model_validate(data)


class AsyncSlices(BaseAsyncResource):
    async def list(self, owner: str, *, limit: int | None = None, cursor: str | None = None) -> CursorPage[Slice]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page(f"/slices/{owner}/list/", Slice, params=params or None)

    async def get(self, owner: str, slug: str) -> Slice:
        data = await self._transport.request("GET", f"/slices/{owner}/{slug}/")
        return Slice.model_validate(data)

    async def create(
        self,
        *,
        name: str,
        visibility: str,
        sub_slices: List[Dict[str, Any]],
        organization: str | None = None,
        random_selection_count: int | None = None,
    ) -> Slice:
        payload: dict[str, Any] = {
            "name": name,
            "visibility": visibility,
            "sub_slices": sub_slices,
        }
        if organization is not None:
            payload["organization"] = organization
        if random_selection_count is not None:
            payload["random_selection_count"] = random_selection_count
        data = await self._transport.request("POST", "/slices/", json=payload)
        return Slice.model_validate(data)

    async def list_items(
        self, owner: str, slug: str, *, limit: int | None = None, cursor: str | None = None
    ) -> CursorPage[SliceItem]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page(
            f"/slices/{owner}/{slug}/items/list/", SliceItem, params=params or None
        )

    async def get_item(self, owner: str, slug: str, item_uid: str) -> SliceItem:
        data = await self._transport.request("GET", f"/slices/{owner}/{slug}/items/{item_uid}/")
        return SliceItem.model_validate(data)
