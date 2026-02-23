"""Tests for AsyncClient."""

from __future__ import annotations

import pytest

from avala._async_client import AsyncClient


def test_async_client_requires_api_key(monkeypatch: pytest.MonkeyPatch):
    """AsyncClient raises ValueError when no API key is provided."""
    monkeypatch.delenv("AVALA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No API key"):
        AsyncClient()


def test_async_client_accepts_api_key():
    """AsyncClient can be created with an explicit API key."""
    client = AsyncClient(api_key="test-key")
    assert client.datasets is not None
    assert client.projects is not None
    assert client.exports is not None
    assert client.tasks is not None


@pytest.mark.asyncio
async def test_async_client_context_manager():
    """AsyncClient can be used as an async context manager."""
    async with AsyncClient(api_key="test-key") as client:
        assert client.datasets is not None
        assert client.projects is not None


@pytest.mark.asyncio
async def test_async_client_close():
    """AsyncClient.close() can be awaited without error."""
    client = AsyncClient(api_key="test-key")
    await client.close()
