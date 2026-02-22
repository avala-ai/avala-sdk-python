import httpx
import respx

from avala import Client
from avala.errors import AuthenticationError, NotFoundError, RateLimitError


@respx.mock
def test_authentication_error():
    import pytest

    respx.get("https://server.avala.ai/api/v1/datasets/").mock(
        return_value=httpx.Response(401, json={"detail": "Invalid API key."})
    )
    client = Client(api_key="bad-key")
    with pytest.raises(AuthenticationError, match="Invalid API key"):
        client.datasets.list()
    client.close()


@respx.mock
def test_not_found_error():
    import pytest

    respx.get("https://server.avala.ai/api/v1/datasets/nonexistent/").mock(
        return_value=httpx.Response(404, json={"detail": "Not found."})
    )
    client = Client(api_key="test-key")
    with pytest.raises(NotFoundError, match="Not found"):
        client.datasets.get("nonexistent")
    client.close()


@respx.mock
def test_rate_limit_error():
    import pytest

    respx.get("https://server.avala.ai/api/v1/datasets/").mock(
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
