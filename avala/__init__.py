"""Official Python SDK for the Avala API."""

from avala._async_client import AsyncClient
from avala._client import Client
from avala.datasets import (
    AsyncResolvedDataset,
    AsyncResolvedDatasetObject,
    ResolvedDataset,
    ResolvedDatasetObject,
    async_load,
    load,
)
from avala.errors import MutableDatasetAliasWarning
from avala.signup import async_signup, signup

# Alias for consistency with the TypeScript SDK (``new Avala(...)``).
Avala = Client

__all__ = [
    "Avala",
    "AsyncClient",
    "AsyncResolvedDataset",
    "AsyncResolvedDatasetObject",
    "Client",
    "MutableDatasetAliasWarning",
    "ResolvedDataset",
    "ResolvedDatasetObject",
    "async_load",
    "async_signup",
    "load",
    "signup",
]
