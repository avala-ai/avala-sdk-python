from __future__ import annotations

import pytest
import httpx
import respx

from avala import Client
from avala.errors import AuthenticationError, AvalaError, NotFoundError, RateLimitError, ServerError, ValidationError

BASE_URL = "https://api.avala.ai/api/v1"


@respx.mock
def test_authentication_error():
    respx.get(f"{BASE_URL}/datasets/").mock(return_value=httpx.Response(401, json={"detail": "Invalid API key."}))
    client = Client(api_key="bad-key")
    with pytest.raises(AuthenticationError, match="Invalid API key"):
        client.datasets.list()
    client.close()


@respx.mock
def test_not_found_error():
    respx.get(f"{BASE_URL}/datasets/nonexistent/").mock(return_value=httpx.Response(404, json={"detail": "Not found."}))
    client = Client(api_key="test-key")
    with pytest.raises(NotFoundError, match="Not found"):
        client.datasets.get("nonexistent")
    client.close()


@respx.mock
def test_rate_limit_error():
    respx.get(f"{BASE_URL}/datasets/").mock(
        return_value=httpx.Response(
            429,
            json={"detail": "Rate limit exceeded."},
            headers={"Retry-After": "30"},
        )
    )
    client = Client(api_key="test-key")
    with pytest.raises(RateLimitError) as exc_info:
        client.datasets.list()
    assert exc_info.value.retry_after == 30.0
    client.close()


@respx.mock
def test_400_validation_error():
    """400 response raises ValidationError with correct status code."""
    respx.post(f"{BASE_URL}/exports/").mock(return_value=httpx.Response(400, json={"detail": "Invalid input."}))
    client = Client(api_key="test-key")
    with pytest.raises(ValidationError) as exc_info:
        client.exports.create()
    assert exc_info.value.status_code == 400
    assert "Invalid input" in str(exc_info.value)
    client.close()


@respx.mock
def test_422_validation_error_with_details():
    """422 response raises ValidationError; list body is stored in .details."""
    error_details = [{"loc": ["body", "project"], "msg": "field required", "type": "missing"}]
    respx.post(f"{BASE_URL}/exports/").mock(return_value=httpx.Response(422, json=error_details))
    client = Client(api_key="test-key")
    with pytest.raises(ValidationError) as exc_info:
        client.exports.create()
    assert exc_info.value.status_code == 422
    assert exc_info.value.details == error_details
    client.close()


@respx.mock
def test_500_server_error():
    """500 response raises ServerError."""
    respx.get(f"{BASE_URL}/datasets/").mock(return_value=httpx.Response(500, json={"detail": "Internal server error."}))
    client = Client(api_key="test-key")
    with pytest.raises(ServerError) as exc_info:
        client.datasets.list()
    assert exc_info.value.status_code == 500
    client.close()


@respx.mock
def test_502_server_error():
    """502 Bad Gateway raises ServerError."""
    respx.get(f"{BASE_URL}/datasets/").mock(
        return_value=httpx.Response(502, content=b"Bad Gateway", headers={"Content-Type": "text/html"})
    )
    client = Client(api_key="test-key")
    with pytest.raises(ServerError) as exc_info:
        client.datasets.list()
    assert exc_info.value.status_code == 502
    client.close()


@respx.mock
def test_unknown_status_raises_avala_error():
    """Unrecognised status code (e.g. 409) raises base AvalaError."""
    respx.get(f"{BASE_URL}/datasets/").mock(return_value=httpx.Response(409, json={"detail": "Conflict."}))
    client = Client(api_key="test-key")
    with pytest.raises(AvalaError) as exc_info:
        client.datasets.list()
    assert exc_info.value.status_code == 409
    assert not isinstance(
        exc_info.value, (AuthenticationError, NotFoundError, RateLimitError, ValidationError, ServerError)
    )
    client.close()


@respx.mock
def test_429_without_retry_after():
    """429 without Retry-After header results in retry_after=None."""
    respx.get(f"{BASE_URL}/datasets/").mock(return_value=httpx.Response(429, json={"detail": "Too Many Requests."}))
    client = Client(api_key="test-key")
    with pytest.raises(RateLimitError) as exc_info:
        client.datasets.list()
    assert exc_info.value.retry_after is None
    client.close()


@respx.mock
def test_non_json_body_raises_error():
    """Non-JSON error body (e.g. plain HTML) still raises the correct error class."""
    respx.get(f"{BASE_URL}/datasets/").mock(
        return_value=httpx.Response(503, content=b"Service Unavailable", headers={"Content-Type": "text/plain"})
    )
    client = Client(api_key="test-key")
    with pytest.raises(ServerError) as exc_info:
        client.datasets.list()
    assert exc_info.value.status_code == 503
    client.close()
