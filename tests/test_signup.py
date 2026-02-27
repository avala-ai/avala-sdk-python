"""Tests for standalone signup / async_signup functions."""

from __future__ import annotations

import pytest
import httpx
import respx

from avala.signup import signup, async_signup
from avala.types.account import SignupResponse, SignupUser
from avala.errors import RateLimitError, ValidationError

BASE_URL = "https://api.avala.ai/api/v1"

SIGNUP_RESPONSE = {
    "user": {
        "uid": "abc123",
        "username": "dev@acme.com",
        "email": "dev@acme.com",
        "first_name": "Jane",
        "last_name": "",
        "in_waitlist": True,
    },
    "api_key": "a1b2c3",
}


class TestSignup:
    @respx.mock
    def test_successful_signup_parses_response(self):
        """signup() returns a SignupResponse on 201."""
        respx.post(f"{BASE_URL}/signup/").mock(return_value=httpx.Response(201, json=SIGNUP_RESPONSE))
        result = signup(email="dev@acme.com", password="secret123")
        assert isinstance(result, SignupResponse)
        assert isinstance(result.user, SignupUser)
        assert result.user.uid == "abc123"
        assert result.user.username == "dev@acme.com"
        assert result.user.email == "dev@acme.com"
        assert result.user.first_name == "Jane"
        assert result.user.in_waitlist is True
        assert result.api_key == "a1b2c3"

    @respx.mock
    def test_signup_sends_all_optional_fields(self):
        """signup() sends first_name and last_name when provided."""
        route = respx.post(f"{BASE_URL}/signup/").mock(return_value=httpx.Response(201, json=SIGNUP_RESPONSE))
        signup(
            email="dev@acme.com",
            password="secret123",
            first_name="Jane",
            last_name="Doe",
        )
        assert route.called
        sent = route.calls.last.request
        import json

        body = json.loads(sent.content)
        assert body["first_name"] == "Jane"
        assert body["last_name"] == "Doe"

    @respx.mock
    def test_signup_duplicate_email_raises_validation_error(self):
        """signup() raises ValidationError on 400 duplicate email."""
        respx.post(f"{BASE_URL}/signup/").mock(
            return_value=httpx.Response(400, json={"detail": "Email already registered."})
        )
        with pytest.raises(ValidationError) as exc_info:
            signup(email="dev@acme.com", password="secret123")
        assert exc_info.value.status_code == 400

    @respx.mock
    def test_signup_rate_limit_raises_rate_limit_error(self):
        """signup() raises RateLimitError on 429."""
        respx.post(f"{BASE_URL}/signup/").mock(
            return_value=httpx.Response(
                429,
                json={"detail": "Too many requests."},
                headers={"Retry-After": "60"},
            )
        )
        with pytest.raises(RateLimitError) as exc_info:
            signup(email="dev@acme.com", password="secret123")
        assert exc_info.value.status_code == 429
        assert exc_info.value.retry_after == 60.0

    @respx.mock
    def test_signup_no_auth_header_sent(self):
        """signup() does NOT send X-Avala-Api-Key header."""
        route = respx.post(f"{BASE_URL}/signup/").mock(return_value=httpx.Response(201, json=SIGNUP_RESPONSE))
        signup(email="dev@acme.com", password="secret123")
        request = route.calls.last.request
        assert "x-avala-api-key" not in {k.lower() for k in request.headers.keys()}


class TestAsyncSignup:
    @respx.mock
    @pytest.mark.asyncio
    async def test_async_signup_parses_response(self):
        """async_signup() returns a SignupResponse on 201."""
        respx.post(f"{BASE_URL}/signup/").mock(return_value=httpx.Response(201, json=SIGNUP_RESPONSE))
        result = await async_signup(email="dev@acme.com", password="secret123")
        assert isinstance(result, SignupResponse)
        assert result.user.uid == "abc123"
        assert result.api_key == "a1b2c3"

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_signup_duplicate_email_raises_validation_error(self):
        """async_signup() raises ValidationError on 400."""
        respx.post(f"{BASE_URL}/signup/").mock(
            return_value=httpx.Response(400, json={"detail": "Email already registered."})
        )
        with pytest.raises(ValidationError):
            await async_signup(email="dev@acme.com", password="secret123")

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_signup_rate_limit_raises_rate_limit_error(self):
        """async_signup() raises RateLimitError on 429."""
        respx.post(f"{BASE_URL}/signup/").mock(
            return_value=httpx.Response(
                429,
                json={"detail": "Too many requests."},
                headers={"Retry-After": "30"},
            )
        )
        with pytest.raises(RateLimitError) as exc_info:
            await async_signup(email="dev@acme.com", password="secret123")
        assert exc_info.value.retry_after == 30.0

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_signup_no_auth_header_sent(self):
        """async_signup() does NOT send X-Avala-Api-Key header."""
        route = respx.post(f"{BASE_URL}/signup/").mock(return_value=httpx.Response(201, json=SIGNUP_RESPONSE))
        await async_signup(email="dev@acme.com", password="secret123")
        request = route.calls.last.request
        assert "x-avala-api-key" not in {k.lower() for k in request.headers.keys()}
