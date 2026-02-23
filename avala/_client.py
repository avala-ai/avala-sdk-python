"""Synchronous Avala client."""

from __future__ import annotations

from avala._config import ClientConfig
from avala._http import SyncHTTPTransport
from avala.resources.datasets import Datasets
from avala.resources.exports import Exports
from avala.resources.projects import Projects
from avala.resources.storage_configs import StorageConfigs
from avala.resources.tasks import Tasks


class Client:
    """Synchronous client for the Avala API."""

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
        self._transport = SyncHTTPTransport(config)
        self.datasets = Datasets(self._transport)
        self.projects = Projects(self._transport)
        self.exports = Exports(self._transport)
        self.tasks = Tasks(self._transport)
        self.storage_configs = StorageConfigs(self._transport)

    @property
    def rate_limit_info(self) -> dict[str, str | None]:
        """Return rate limit headers from the last API response."""
        return self._transport.last_rate_limit

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
