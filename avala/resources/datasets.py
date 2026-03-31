"""Datasets resource."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from avala._pagination import CursorPage
from avala.resources._base import BaseAsyncResource, BaseSyncResource
from avala.types.dataset import Dataset, DatasetItem, DatasetSequence

_MIN_INTERVAL = 1.0


class Datasets(BaseSyncResource):
    def list(
        self,
        *,
        data_type: str | None = None,
        name: str | None = None,
        status: str | None = None,
        visibility: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> CursorPage[Dataset]:
        params: dict[str, Any] = {}
        if data_type is not None:
            params["data_type"] = data_type
        if name is not None:
            params["name"] = name
        if status is not None:
            params["status"] = status
        if visibility is not None:
            params["visibility"] = visibility
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page("/datasets/", Dataset, params=params or None)

    def get(self, uid: str) -> Dataset:
        data = self._transport.request("GET", f"/datasets/{uid}/")
        return Dataset.model_validate(data)

    def create(
        self,
        *,
        name: str,
        slug: str,
        data_type: str,
        is_sequence: bool = False,
        visibility: str = "private",
        create_metadata: bool = True,
        provider_config: dict[str, Any] | None = None,
        owner_name: str | None = None,
        organization_id: int | None = None,
        gpu_texture_format: str | None = None,
        metadata: dict[str, Any] | None = None,
        industry: int | None = None,
        license: int | None = None,
    ) -> Dataset:
        payload: dict[str, Any] = {
            "name": name,
            "slug": slug,
            "data_type": data_type,
            "is_sequence": is_sequence,
            "visibility": visibility,
            "create_metadata": create_metadata,
        }
        if provider_config is not None:
            payload["provider_config"] = provider_config
        if owner_name is not None:
            payload["owner_name"] = owner_name
        if organization_id is not None:
            payload["organization_id"] = organization_id
        if gpu_texture_format is not None:
            payload["gpu_texture_format"] = gpu_texture_format
        if metadata is not None:
            payload["metadata"] = metadata
        if industry is not None:
            payload["industry"] = industry
        if license is not None:
            payload["license"] = license
        data = self._transport.request("POST", "/datasets/", json=payload)
        return Dataset.model_validate(data)

    def list_items(
        self, owner: str, slug: str, *, limit: int | None = None, cursor: str | None = None
    ) -> CursorPage[DatasetItem]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page(f"/datasets/{owner}/{slug}/items/", DatasetItem, params=params or None)

    def get_item(self, owner: str, slug: str, item_uid: str) -> DatasetItem:
        data = self._transport.request("GET", f"/datasets/{owner}/{slug}/items/{item_uid}/")
        return DatasetItem.model_validate(data)

    def list_sequences(
        self, owner: str, slug: str, *, limit: int | None = None, cursor: str | None = None
    ) -> CursorPage[DatasetSequence]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page(
            f"/datasets/{owner}/{slug}/sequences/", DatasetSequence, params=params or None
        )

    def get_sequence(self, owner: str, slug: str, sequence_uid: str) -> DatasetSequence:
        data = self._transport.request("GET", f"/datasets/{owner}/{slug}/sequences/{sequence_uid}/")
        return DatasetSequence.model_validate(data)

    def wait(
        self,
        uid: str,
        *,
        status: str = "created",
        interval: float = 10.0,
        timeout: float = 3600.0,
        _on_poll: Callable[[Dataset], None] | None = None,
    ) -> Dataset:
        """Poll a dataset until it reaches the target status.

        Args:
            uid: The dataset UID to poll.
            status: Target status to wait for (default ``"created"``).
            interval: Seconds between polls (default 10, minimum 1).
            timeout: Maximum seconds to wait before raising ``TimeoutError`` (default 3600).
            _on_poll: Optional callback invoked after each non-terminal poll with the current dataset.

        Returns:
            The Dataset object once it reaches the target status.

        Raises:
            TimeoutError: If the dataset does not reach the target status within *timeout* seconds.
        """
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        if interval < 0:
            raise ValueError("interval must be non-negative")
        interval = max(interval, _MIN_INTERVAL)
        deadline = time.monotonic() + timeout
        while True:
            dataset = self.get(uid)
            if dataset.status == status:
                return dataset
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Dataset {uid} did not reach status '{status}' within {timeout}s (last status: {dataset.status})"
                )
            if _on_poll is not None:
                _on_poll(dataset)
            time.sleep(interval)


class AsyncDatasets(BaseAsyncResource):
    async def list(
        self,
        *,
        data_type: str | None = None,
        name: str | None = None,
        status: str | None = None,
        visibility: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> CursorPage[Dataset]:
        params: dict[str, Any] = {}
        if data_type is not None:
            params["data_type"] = data_type
        if name is not None:
            params["name"] = name
        if status is not None:
            params["status"] = status
        if visibility is not None:
            params["visibility"] = visibility
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page("/datasets/", Dataset, params=params or None)

    async def get(self, uid: str) -> Dataset:
        data = await self._transport.request("GET", f"/datasets/{uid}/")
        return Dataset.model_validate(data)

    async def create(
        self,
        *,
        name: str,
        slug: str,
        data_type: str,
        is_sequence: bool = False,
        visibility: str = "private",
        create_metadata: bool = True,
        provider_config: dict[str, Any] | None = None,
        owner_name: str | None = None,
        organization_id: int | None = None,
        gpu_texture_format: str | None = None,
        metadata: dict[str, Any] | None = None,
        industry: int | None = None,
        license: int | None = None,
    ) -> Dataset:
        payload: dict[str, Any] = {
            "name": name,
            "slug": slug,
            "data_type": data_type,
            "is_sequence": is_sequence,
            "visibility": visibility,
            "create_metadata": create_metadata,
        }
        if provider_config is not None:
            payload["provider_config"] = provider_config
        if owner_name is not None:
            payload["owner_name"] = owner_name
        if organization_id is not None:
            payload["organization_id"] = organization_id
        if gpu_texture_format is not None:
            payload["gpu_texture_format"] = gpu_texture_format
        if metadata is not None:
            payload["metadata"] = metadata
        if industry is not None:
            payload["industry"] = industry
        if license is not None:
            payload["license"] = license
        data = await self._transport.request("POST", "/datasets/", json=payload)
        return Dataset.model_validate(data)

    async def list_items(
        self, owner: str, slug: str, *, limit: int | None = None, cursor: str | None = None
    ) -> CursorPage[DatasetItem]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page(
            f"/datasets/{owner}/{slug}/items/", DatasetItem, params=params or None
        )

    async def get_item(self, owner: str, slug: str, item_uid: str) -> DatasetItem:
        data = await self._transport.request("GET", f"/datasets/{owner}/{slug}/items/{item_uid}/")
        return DatasetItem.model_validate(data)

    async def list_sequences(
        self, owner: str, slug: str, *, limit: int | None = None, cursor: str | None = None
    ) -> CursorPage[DatasetSequence]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page(
            f"/datasets/{owner}/{slug}/sequences/", DatasetSequence, params=params or None
        )

    async def get_sequence(self, owner: str, slug: str, sequence_uid: str) -> DatasetSequence:
        data = await self._transport.request("GET", f"/datasets/{owner}/{slug}/sequences/{sequence_uid}/")
        return DatasetSequence.model_validate(data)

    async def wait(
        self,
        uid: str,
        *,
        status: str = "created",
        interval: float = 10.0,
        timeout: float = 3600.0,
        _on_poll: Callable[[Dataset], None] | None = None,
    ) -> Dataset:
        """Poll a dataset until it reaches the target status.

        Args:
            uid: The dataset UID to poll.
            status: Target status to wait for (default ``"created"``).
            interval: Seconds between polls (default 10, minimum 1).
            timeout: Maximum seconds to wait before raising ``TimeoutError`` (default 3600).
            _on_poll: Optional callback invoked after each non-terminal poll with the current dataset.

        Returns:
            The Dataset object once it reaches the target status.

        Raises:
            TimeoutError: If the dataset does not reach the target status within *timeout* seconds.
        """
        import asyncio

        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        if interval < 0:
            raise ValueError("interval must be non-negative")
        interval = max(interval, _MIN_INTERVAL)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            dataset = await self.get(uid)
            if dataset.status == status:
                return dataset
            if loop.time() >= deadline:
                raise TimeoutError(
                    f"Dataset {uid} did not reach status '{status}' within {timeout}s (last status: {dataset.status})"
                )
            if _on_poll is not None:
                _on_poll(dataset)
            await asyncio.sleep(interval)
