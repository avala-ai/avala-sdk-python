"""Base resource classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from avala._async_http import AsyncHTTPTransport
    from avala._http import SyncHTTPTransport


class BaseSyncResource:
    def __init__(self, transport: SyncHTTPTransport) -> None:
        self._transport = transport


class BaseAsyncResource:
    def __init__(self, transport: AsyncHTTPTransport) -> None:
        self._transport = transport
