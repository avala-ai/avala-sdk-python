"""Exports resource."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from avala._pagination import CursorPage
from avala.resources._base import BaseAsyncResource, BaseSyncResource
from avala.types.export import Export

TERMINAL_STATUSES = frozenset({"exported", "failed"})
_MIN_INTERVAL = 0.5


class Exports(BaseSyncResource):
    def list(self, *, limit: int | None = None, cursor: str | None = None) -> CursorPage[Export]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page("/exports/", Export, params=params or None)

    def create(self, *, project: str | None = None, dataset: str | None = None) -> Export:
        payload: dict[str, Any] = {}
        if project is not None:
            payload["project"] = project
        if dataset is not None:
            payload["dataset"] = dataset
        data = self._transport.request("POST", "/exports/", json=payload)
        return Export.model_validate(data)

    def get(self, uid: str) -> Export:
        data = self._transport.request("GET", f"/exports/{uid}/")
        return Export.model_validate(data)

    def wait(
        self,
        uid: str,
        *,
        interval: float = 2.0,
        timeout: float = 300.0,
        _on_poll: Callable[[], None] | None = None,
    ) -> Export:
        """Poll an export until it reaches a terminal status.

        Args:
            uid: The export UID to poll.
            interval: Seconds between polls (default 2.0, minimum 0.5).
            timeout: Maximum seconds to wait before raising TimeoutError (default 300).
            _on_poll: Optional callback invoked after each non-terminal poll (internal use).

        Returns:
            The final Export object once it reaches a terminal status.

        Raises:
            TimeoutError: If the export does not complete within *timeout* seconds.
            ValueError: If *interval* or *timeout* is negative.
        """
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        if interval < 0:
            raise ValueError("interval must be non-negative")
        interval = max(interval, _MIN_INTERVAL)
        deadline = time.monotonic() + timeout
        while True:
            export = self.get(uid)
            if export.status in TERMINAL_STATUSES:
                return export
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Export {uid} did not complete within {timeout}s (last status: {export.status})")
            if _on_poll is not None:
                _on_poll()
            time.sleep(interval)


class AsyncExports(BaseAsyncResource):
    async def list(self, *, limit: int | None = None, cursor: str | None = None) -> CursorPage[Export]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page("/exports/", Export, params=params or None)

    async def create(self, *, project: str | None = None, dataset: str | None = None) -> Export:
        payload: dict[str, Any] = {}
        if project is not None:
            payload["project"] = project
        if dataset is not None:
            payload["dataset"] = dataset
        data = await self._transport.request("POST", "/exports/", json=payload)
        return Export.model_validate(data)

    async def get(self, uid: str) -> Export:
        data = await self._transport.request("GET", f"/exports/{uid}/")
        return Export.model_validate(data)

    async def wait(
        self,
        uid: str,
        *,
        interval: float = 2.0,
        timeout: float = 300.0,
    ) -> Export:
        """Poll an export until it reaches a terminal status.

        Args:
            uid: The export UID to poll.
            interval: Seconds between polls (default 2.0, minimum 0.5).
            timeout: Maximum seconds to wait before raising TimeoutError (default 300).

        Returns:
            The final Export object once it reaches a terminal status.

        Raises:
            TimeoutError: If the export does not complete within *timeout* seconds.
            ValueError: If *interval* or *timeout* is negative.
        """
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        if interval < 0:
            raise ValueError("interval must be non-negative")
        interval = max(interval, _MIN_INTERVAL)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            export = await self.get(uid)
            if export.status in TERMINAL_STATUSES:
                return export
            if loop.time() >= deadline:
                raise TimeoutError(f"Export {uid} did not complete within {timeout}s (last status: {export.status})")
            await asyncio.sleep(interval)
