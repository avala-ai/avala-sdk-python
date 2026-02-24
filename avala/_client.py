"""Synchronous Avala client."""

from __future__ import annotations

from avala._config import ClientConfig
from avala._http import SyncHTTPTransport
from avala.resources.agents import Agents
from avala.resources.annotation_issues import AnnotationIssues
from avala.resources.auto_label_jobs import AutoLabelJobs
from avala.resources.consensus import Consensus
from avala.resources.datasets import Datasets
from avala.resources.exports import Exports
from avala.resources.inference_providers import InferenceProviders
from avala.resources.organizations import Organizations
from avala.resources.projects import Projects
from avala.resources.quality_targets import QualityTargets
from avala.resources.slices import Slices
from avala.resources.storage_configs import StorageConfigs
from avala.resources.tasks import Tasks
from avala.resources.webhooks import WebhookDeliveries, Webhooks


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
        self.agents = Agents(self._transport)
        self.annotation_issues = AnnotationIssues(self._transport)
        self.auto_label_jobs = AutoLabelJobs(self._transport)
        self.consensus = Consensus(self._transport)
        self.datasets = Datasets(self._transport)
        self.exports = Exports(self._transport)
        self.inference_providers = InferenceProviders(self._transport)
        self.organizations = Organizations(self._transport)
        self.projects = Projects(self._transport)
        self.quality_targets = QualityTargets(self._transport)
        self.slices = Slices(self._transport)
        self.storage_configs = StorageConfigs(self._transport)
        self.tasks = Tasks(self._transport)
        self.webhooks = Webhooks(self._transport)
        self.webhook_deliveries = WebhookDeliveries(self._transport)

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
