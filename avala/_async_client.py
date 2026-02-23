"""Asynchronous Avala client."""

from __future__ import annotations

from types import TracebackType
from typing import Optional, Type

from avala._async_http import AsyncHTTPTransport
from avala._config import ClientConfig
from avala.resources.datasets import AsyncDatasets
from avala.resources.exports import AsyncExports
from avala.resources.projects import AsyncProjects
from avala.resources.storage_configs import AsyncStorageConfigs
from avala.resources.tasks import AsyncTasks


class AsyncClient:
    """Asynchronous client for the Avala API."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        config = ClientConfig.from_params(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._transport = AsyncHTTPTransport(config)
        self.datasets = AsyncDatasets(self._transport)
        self.projects = AsyncProjects(self._transport)
        self.exports = AsyncExports(self._transport)
        self.tasks = AsyncTasks(self._transport)
        self.storage_configs = AsyncStorageConfigs(self._transport)

    @property
    def rate_limit_info(self) -> dict[str, str | None]:
        """Return rate limit headers from the last API response."""
        return self._transport.last_rate_limit

    async def close(self) -> None:
        await self._transport.close()

    async def __aenter__(self) -> AsyncClient:
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        await self.close()
