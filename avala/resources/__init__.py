"""Resource modules for Avala API."""

from avala.resources.agents import Agents, AsyncAgents
from avala.resources.annotation_issues import AnnotationIssues, AsyncAnnotationIssues
from avala.resources.auto_label_jobs import AsyncAutoLabelJobs, AutoLabelJobs
from avala.resources.consensus import AsyncConsensus, Consensus
from avala.resources.datasets import AsyncDatasets, Datasets
from avala.resources.exports import AsyncExports, Exports
from avala.resources.inference_providers import AsyncInferenceProviders, InferenceProviders
from avala.resources.organizations import AsyncOrganizations, Organizations
from avala.resources.projects import AsyncProjects, Projects
from avala.resources.quality_targets import AsyncQualityTargets, QualityTargets
from avala.resources.slices import AsyncSlices, Slices
from avala.resources.storage_configs import AsyncStorageConfigs, StorageConfigs
from avala.resources.tasks import AsyncTasks, Tasks
from avala.resources.webhooks import AsyncWebhookDeliveries, AsyncWebhooks, WebhookDeliveries, Webhooks

__all__ = [
    "Agents",
    "AsyncAgents",
    "AnnotationIssues",
    "AsyncAnnotationIssues",
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
    "Organizations",
    "AsyncOrganizations",
    "Projects",
    "AsyncProjects",
    "QualityTargets",
    "AsyncQualityTargets",
    "Slices",
    "AsyncSlices",
    "StorageConfigs",
    "AsyncStorageConfigs",
    "Tasks",
    "AsyncTasks",
    "Webhooks",
    "AsyncWebhooks",
    "WebhookDeliveries",
    "AsyncWebhookDeliveries",
]
