"""Resource modules for Avala API."""

from avala.resources.agents import Agents, AsyncAgents
from avala.resources.auto_label_jobs import AsyncAutoLabelJobs, AutoLabelJobs
from avala.resources.consensus import AsyncConsensus, Consensus
from avala.resources.datasets import AsyncDatasets, Datasets
from avala.resources.exports import AsyncExports, Exports
from avala.resources.inference_providers import AsyncInferenceProviders, InferenceProviders
from avala.resources.projects import AsyncProjects, Projects
from avala.resources.quality_targets import AsyncQualityTargets, QualityTargets
from avala.resources.tasks import AsyncTasks, Tasks
from avala.resources.webhooks import AsyncWebhookDeliveries, AsyncWebhooks, WebhookDeliveries, Webhooks

__all__ = [
    "Agents",
    "AsyncAgents",
    "AutoLabelJobs",
    "AsyncAutoLabelJobs",
    "Consensus",
    "AsyncConsensus",
    "Datasets",
    "AsyncDatasets",
    "Exports",
    "AsyncExports",
    "InferenceProviders",
    "AsyncInferenceProviders",
    "Projects",
    "AsyncProjects",
    "QualityTargets",
    "AsyncQualityTargets",
    "Tasks",
    "AsyncTasks",
    "Webhooks",
    "AsyncWebhooks",
    "WebhookDeliveries",
    "AsyncWebhookDeliveries",
]
