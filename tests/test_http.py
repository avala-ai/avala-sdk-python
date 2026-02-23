"""Tests for SyncHTTPTransport."""

from __future__ import annotations

import pytest
import httpx
import respx

from avala._config import ClientConfig
from avala._http import SyncHTTPTransport, _extract_cursor
from avala.errors import (
    AuthenticationError,
    AvalaError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
)
from avala.types.dataset import Dataset

BASE_URL = "https://server.avala.ai/api/v1"


def make_transport(api_key: str = "test-key", base_url: str = BASE_URL) -> SyncHTTPTransport:
    config = ClientConfig.from_params(api_key=api_key, base_url=base_url)
    return SyncHTTPTransport(config)


class TestHeaders:
    @respx.mock
    def test_api_key_header_sent(self):
        """X-Avala-Api-Key header is included in every request."""
        route = respx.get(f"{BASE_URL}/datasets/").mock(
            return_value=httpx.Response(200, json={"results": [], "next": None, "previous": None})
        )
        transport = make_transport(api_key="my-key")
        transport.request_page("/datasets/", Dataset)
        assert route.called
        request = route.calls.last.request
        assert request.headers["X-Avala-Api-Key"] == "my-key"
        transport.close()

    @respx.mock
    def test_accept_json_header_sent(self):
        """Accept: application/json header is included."""
        route = respx.get(f"{BASE_URL}/datasets/").mock(
            return_value=httpx.Response(200, json={"results": [], "next": None, "previous": None})
        )
        transport = make_transport()
        transport.request_page("/datasets/", Dataset)
        request = route.calls.last.request
        assert request.headers["accept"] == "application/json"
        transport.close()


class TestHTTPMethods:
    @respx.mock
    def test_get_request(self):
        """GET request returns parsed JSON body."""
        uid = "ds-uid"
        respx.get(f"{BASE_URL}/datasets/{uid}/").mock(
            return_value=httpx.Response(200, json={"uid": uid, "name": "DS", "slug": "ds", "item_count": 0})
        )
        transport = make_transport()
        data = transport.request("GET", f"/datasets/{uid}/")
        assert data["uid"] == uid
        transport.close()

    @respx.mock
    def test_post_request(self):
        """POST request returns parsed JSON body."""
        respx.post(f"{BASE_URL}/exports/").mock(
            return_value=httpx.Response(201, json={"uid": "exp-uid", "status": "pending"})
        )
        transport = make_transport()
        data = transport.request("POST", "/exports/", json={"project": "proj-uid"})
        assert data["uid"] == "exp-uid"
        transport.close()

    @respx.mock
    def test_204_returns_none(self):
        """204 No Content response returns None."""
        respx.delete(f"{BASE_URL}/exports/exp-uid/").mock(return_value=httpx.Response(204))
        transport = make_transport()
        result = transport.request("DELETE", "/exports/exp-uid/")
        assert result is None
        transport.close()


class TestPagination:
    @respx.mock
    def test_request_page_returns_cursor_page(self):
        """request_page returns a CursorPage with parsed items."""
        respx.get(f"{BASE_URL}/datasets/").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [
                        {"uid": "ds-1", "name": "DS 1", "slug": "ds-1", "item_count": 10},
                        {"uid": "ds-2", "name": "DS 2", "slug": "ds-2", "item_count": 20},
                    ],
                    "next": f"{BASE_URL}/datasets/?cursor=next-cursor",
                    "previous": None,
                },
            )
        )
        transport = make_transport()
        page = transport.request_page("/datasets/", Dataset)
        assert len(page.items) == 2
        assert page.items[0].uid == "ds-1"
        assert page.has_more is True
        assert page.next_cursor == "next-cursor"
        assert page.previous_cursor is None
        transport.close()

    @respx.mock
    def test_request_page_empty_results(self):
        """request_page handles empty results list."""
        respx.get(f"{BASE_URL}/datasets/").mock(
            return_value=httpx.Response(200, json={"results": [], "next": None, "previous": None})
        )
        transport = make_transport()
        page = transport.request_page("/datasets/", Dataset)
        assert len(page.items) == 0
        assert page.has_more is False
        transport.close()


class TestErrorHandling:
    @respx.mock
    def test_400_raises_validation_error(self):
        respx.post(f"{BASE_URL}/exports/").mock(return_value=httpx.Response(400, json={"detail": "Bad request."}))
        transport = make_transport()
        with pytest.raises(ValidationError) as exc_info:
            transport.request("POST", "/exports/", json={})
        assert exc_info.value.status_code == 400
        transport.close()

    @respx.mock
    def test_401_raises_authentication_error(self):
        respx.get(f"{BASE_URL}/datasets/").mock(return_value=httpx.Response(401, json={"detail": "Invalid API key."}))
        transport = make_transport()
        with pytest.raises(AuthenticationError) as exc_info:
            transport.request_page("/datasets/", Dataset)
        assert exc_info.value.status_code == 401
        assert "Invalid API key" in str(exc_info.value)
        transport.close()

    @respx.mock
    def test_404_raises_not_found_error(self):
        respx.get(f"{BASE_URL}/datasets/missing/").mock(return_value=httpx.Response(404, json={"detail": "Not found."}))
        transport = make_transport()
        with pytest.raises(NotFoundError) as exc_info:
            transport.request("GET", "/datasets/missing/")
        assert exc_info.value.status_code == 404
        transport.close()

    @respx.mock
    def test_429_raises_rate_limit_error_with_retry_after(self):
        respx.get(f"{BASE_URL}/datasets/").mock(
            return_value=httpx.Response(
                429,
                json={"detail": "Rate limit exceeded."},
                headers={"Retry-After": "60"},
            )
        )
        transport = make_transport()
        with pytest.raises(RateLimitError) as exc_info:
            transport.request_page("/datasets/", Dataset)
        assert exc_info.value.status_code == 429
        assert exc_info.value.retry_after == 60.0
        transport.close()

    @respx.mock
    def test_429_raises_rate_limit_error_without_retry_after(self):
        respx.get(f"{BASE_URL}/datasets/").mock(
            return_value=httpx.Response(429, json={"detail": "Rate limit exceeded."})
        )
        transport = make_transport()
        with pytest.raises(RateLimitError) as exc_info:
            transport.request_page("/datasets/", Dataset)
        assert exc_info.value.retry_after is None
        transport.close()

    @respx.mock
    def test_500_raises_server_error(self):
        respx.get(f"{BASE_URL}/datasets/").mock(
            return_value=httpx.Response(500, json={"detail": "Internal server error."})
        )
        transport = make_transport()
        with pytest.raises(ServerError) as exc_info:
            transport.request_page("/datasets/", Dataset)
        assert exc_info.value.status_code == 500
        transport.close()

    @respx.mock
    def test_non_json_body_still_raises_error(self):
        """Non-JSON error responses still raise the correct error class."""
        respx.get(f"{BASE_URL}/datasets/").mock(
            return_value=httpx.Response(500, content=b"Internal Server Error", headers={"Content-Type": "text/plain"})
        )
        transport = make_transport()
        with pytest.raises(ServerError):
            transport.request_page("/datasets/", Dataset)
        transport.close()

    @respx.mock
    def test_unknown_4xx_raises_avala_error(self):
        """Unrecognised 4xx status raises base AvalaError."""
        respx.get(f"{BASE_URL}/datasets/").mock(return_value=httpx.Response(409, json={"detail": "Conflict."}))
        transport = make_transport()
        with pytest.raises(AvalaError) as exc_info:
            transport.request_page("/datasets/", Dataset)
        assert exc_info.value.status_code == 409
        transport.close()


class TestCursorExtraction:
    def test_extract_cursor_from_url(self):
        url = "https://server.avala.ai/api/v1/datasets/?cursor=abc123&limit=10"
        assert _extract_cursor(url) == "abc123"

    def test_extract_cursor_none_url(self):
        assert _extract_cursor(None) is None

    def test_extract_cursor_empty_string(self):
        assert _extract_cursor("") is None

    def test_extract_cursor_no_cursor_param(self):
        url = "https://server.avala.ai/api/v1/datasets/?limit=10"
        assert _extract_cursor(url) is None
