"""Official Python SDK for the Avala API."""

from avala._async_client import AsyncClient
from avala._client import Client
from avala.signup import async_signup, signup

# Alias for consistency with the TypeScript SDK (``new Avala(...)``).
Avala = Client

__all__ = ["Avala", "Client", "AsyncClient", "signup", "async_signup"]
