"""Tests for AsyncHTTPTransport — mirrors sync tests using pytest-asyncio + respx."""

from __future__ import annotations

import pytest
import httpx
import respx

from avala._config import ClientConfig
from avala._async_http import AsyncHTTPTransport
from avala.errors import (
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
)
from avala.types.dataset import Dataset

BASE_URL = "https://server.avala.ai/api/v1"


def make_transport(api_key: str = "test-key", base_url: str = BASE_URL) -> AsyncHTTPTransport:
    config = ClientConfig.from_params(api_key=api_key, base_url=base_url)
    return AsyncHTTPTransport(config)


@respx.mock
@pytest.mark.asyncio
async def test_async_api_key_header_sent():
    """X-Avala-Api-Key header is included in async requests."""
    route = respx.get(f"{BASE_URL}/datasets/").mock(
        return_value=httpx.Response(200, json={"results": [], "next": None, "previous": None})
    )
    transport = make_transport(api_key="async-key")
    await transport.request_page("/datasets/", Dataset)
    request = route.calls.last.request
    assert request.headers["X-Avala-Api-Key"] == "async-key"
    await transport.close()


@respx.mock
@pytest.mark.asyncio
async def test_async_get_request():
    """Async GET request returns parsed JSON body."""
    uid = "ds-uid"
    respx.get(f"{BASE_URL}/datasets/{uid}/").mock(
        return_value=httpx.Response(200, json={"uid": uid, "name": "DS", "slug": "ds", "item_count": 0})
    )
    transport = make_transport()
    data = await transport.request("GET", f"/datasets/{uid}/")
    assert data["uid"] == uid
    await transport.close()


@respx.mock
@pytest.mark.asyncio
async def test_async_204_returns_none():
    """Async 204 No Content response returns None."""
    respx.delete(f"{BASE_URL}/exports/exp-uid/").mock(return_value=httpx.Response(204))
    transport = make_transport()
    result = await transport.request("DELETE", "/exports/exp-uid/")
    assert result is None
    await transport.close()


@respx.mock
@pytest.mark.asyncio
async def test_async_request_page_returns_cursor_page():
    """Async request_page returns a CursorPage with parsed items."""
    respx.get(f"{BASE_URL}/datasets/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"uid": "ds-1", "name": "DS 1", "slug": "ds-1", "item_count": 5},
                ],
                "next": f"{BASE_URL}/datasets/?cursor=async-cursor",
                "previous": None,
            },
        )
    )
    transport = make_transport()
    page = await transport.request_page("/datasets/", Dataset)
    assert len(page.items) == 1
    assert page.items[0].uid == "ds-1"
    assert page.has_more is True
    assert page.next_cursor == "async-cursor"
    await transport.close()


@respx.mock
@pytest.mark.asyncio
async def test_async_401_raises_authentication_error():
    respx.get(f"{BASE_URL}/datasets/").mock(return_value=httpx.Response(401, json={"detail": "Invalid API key."}))
    transport = make_transport()
    with pytest.raises(AuthenticationError) as exc_info:
        await transport.request_page("/datasets/", Dataset)
    assert exc_info.value.status_code == 401
    await transport.close()


@respx.mock
@pytest.mark.asyncio
async def test_async_404_raises_not_found_error():
    respx.get(f"{BASE_URL}/datasets/missing/").mock(return_value=httpx.Response(404, json={"detail": "Not found."}))
    transport = make_transport()
    with pytest.raises(NotFoundError):
        await transport.request("GET", "/datasets/missing/")
    await transport.close()


@respx.mock
@pytest.mark.asyncio
async def test_async_429_raises_rate_limit_error_with_retry_after():
    respx.get(f"{BASE_URL}/datasets/").mock(
        return_value=httpx.Response(
            429,
            json={"detail": "Rate limit exceeded."},
            headers={"Retry-After": "45"},
        )
    )
    transport = make_transport()
    with pytest.raises(RateLimitError) as exc_info:
        await transport.request_page("/datasets/", Dataset)
    assert exc_info.value.retry_after == 45.0
    await transport.close()


@respx.mock
@pytest.mark.asyncio
async def test_async_500_raises_server_error():
    respx.get(f"{BASE_URL}/datasets/").mock(return_value=httpx.Response(500, json={"detail": "Internal server error."}))
    transport = make_transport()
    with pytest.raises(ServerError) as exc_info:
        await transport.request_page("/datasets/", Dataset)
    assert exc_info.value.status_code == 500
    await transport.close()


@respx.mock
@pytest.mark.asyncio
async def test_async_non_json_body_raises_server_error():
    """Non-JSON error responses still raise the correct error class."""
    respx.get(f"{BASE_URL}/datasets/").mock(
        return_value=httpx.Response(500, content=b"Gateway Timeout", headers={"Content-Type": "text/plain"})
    )
    transport = make_transport()
    with pytest.raises(ServerError):
        await transport.request_page("/datasets/", Dataset)
    await transport.close()


@respx.mock
@pytest.mark.asyncio
async def test_async_400_raises_validation_error():
    respx.post(f"{BASE_URL}/exports/").mock(return_value=httpx.Response(400, json={"detail": "Bad request."}))
    transport = make_transport()
    with pytest.raises(ValidationError) as exc_info:
        await transport.request("POST", "/exports/", json={})
    assert exc_info.value.status_code == 400
    await transport.close()
