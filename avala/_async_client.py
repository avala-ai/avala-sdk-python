"""Asynchronous Avala client."""

from __future__ import annotations

from types import TracebackType
from typing import Optional, Type

from avala._async_http import AsyncHTTPTransport
from avala._config import ClientConfig
from avala.resources.agents import AsyncAgents
from avala.resources.auto_label_jobs import AsyncAutoLabelJobs
from avala.resources.consensus import AsyncConsensus
from avala.resources.datasets import AsyncDatasets
from avala.resources.exports import AsyncExports
from avala.resources.inference_providers import AsyncInferenceProviders
from avala.resources.projects import AsyncProjects
from avala.resources.quality_targets import AsyncQualityTargets
from avala.resources.storage_configs import AsyncStorageConfigs
from avala.resources.tasks import AsyncTasks
from avala.resources.webhooks import AsyncWebhookDeliveries, AsyncWebhooks


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
        self.agents = AsyncAgents(self._transport)
        self.auto_label_jobs = AsyncAutoLabelJobs(self._transport)
        self.consensus = AsyncConsensus(self._transport)
        self.datasets = AsyncDatasets(self._transport)
        self.exports = AsyncExports(self._transport)
        self.inference_providers = AsyncInferenceProviders(self._transport)
        self.projects = AsyncProjects(self._transport)
        self.quality_targets = AsyncQualityTargets(self._transport)
        self.storage_configs = AsyncStorageConfigs(self._transport)
        self.tasks = AsyncTasks(self._transport)
        self.webhooks = AsyncWebhooks(self._transport)
        self.webhook_deliveries = AsyncWebhookDeliveries(self._transport)

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
