"""Inference providers resource."""

from __future__ import annotations

from typing import Any

from avala._pagination import CursorPage
from avala.resources._base import BaseAsyncResource, BaseSyncResource
from avala.types.inference_provider import InferenceProvider


class InferenceProviders(BaseSyncResource):
    def list(self, *, limit: int | None = None, cursor: str | None = None) -> CursorPage[InferenceProvider]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page("/inference-providers/", InferenceProvider, params=params or None)

    def get(self, uid: str) -> InferenceProvider:
        data = self._transport.request("GET", f"/inference-providers/{uid}/")
        return InferenceProvider.model_validate(data)

    def create(
        self,
        *,
        name: str,
        provider_type: str,
        config: dict[str, Any],
        description: str | None = None,
        is_active: bool | None = None,
        project: str | None = None,
    ) -> InferenceProvider:
        payload: dict[str, Any] = {"name": name, "provider_type": provider_type, "config": config}
        if description is not None:
            payload["description"] = description
        if is_active is not None:
            payload["is_active"] = is_active
        if project is not None:
            payload["project"] = project
        data = self._transport.request("POST", "/inference-providers/", json=payload)
        return InferenceProvider.model_validate(data)

    def update(
        self,
        uid: str,
        *,
        name: str | None = None,
        description: str | None = None,
        provider_type: str | None = None,
        config: dict[str, Any] | None = None,
        is_active: bool | None = None,
        project: str | None = None,
    ) -> InferenceProvider:
        payload: dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if provider_type is not None:
            payload["provider_type"] = provider_type
        if config is not None:
            payload["config"] = config
        if is_active is not None:
            payload["is_active"] = is_active
        if project is not None:
            payload["project"] = project
        data = self._transport.request("PATCH", f"/inference-providers/{uid}/", json=payload)
        return InferenceProvider.model_validate(data)

    def delete(self, uid: str) -> None:
        self._transport.request("DELETE", f"/inference-providers/{uid}/")

    def test(self, uid: str) -> dict[str, Any]:
        data = self._transport.request("POST", f"/inference-providers/{uid}/test/")
        return data  # type: ignore[no-any-return]


class AsyncInferenceProviders(BaseAsyncResource):
    async def list(self, *, limit: int | None = None, cursor: str | None = None) -> CursorPage[InferenceProvider]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page("/inference-providers/", InferenceProvider, params=params or None)

    async def get(self, uid: str) -> InferenceProvider:
        data = await self._transport.request("GET", f"/inference-providers/{uid}/")
        return InferenceProvider.model_validate(data)

    async def create(
        self,
        *,
        name: str,
        provider_type: str,
        config: dict[str, Any],
        description: str | None = None,
        is_active: bool | None = None,
        project: str | None = None,
    ) -> InferenceProvider:
        payload: dict[str, Any] = {"name": name, "provider_type": provider_type, "config": config}
        if description is not None:
            payload["description"] = description
        if is_active is not None:
            payload["is_active"] = is_active
        if project is not None:
            payload["project"] = project
        data = await self._transport.request("POST", "/inference-providers/", json=payload)
        return InferenceProvider.model_validate(data)

    async def update(
        self,
        uid: str,
        *,
        name: str | None = None,
        description: str | None = None,
        provider_type: str | None = None,
        config: dict[str, Any] | None = None,
        is_active: bool | None = None,
        project: str | None = None,
    ) -> InferenceProvider:
        payload: dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if provider_type is not None:
            payload["provider_type"] = provider_type
        if config is not None:
            payload["config"] = config
        if is_active is not None:
            payload["is_active"] = is_active
        if project is not None:
            payload["project"] = project
        data = await self._transport.request("PATCH", f"/inference-providers/{uid}/", json=payload)
        return InferenceProvider.model_validate(data)

    async def delete(self, uid: str) -> None:
        await self._transport.request("DELETE", f"/inference-providers/{uid}/")

    async def test(self, uid: str) -> dict[str, Any]:
        data = await self._transport.request("POST", f"/inference-providers/{uid}/test/")
        return data  # type: ignore[no-any-return]
