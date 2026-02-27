"""Standalone signup functions that do not require an API key."""

from __future__ import annotations

import os
from typing import Optional

import httpx

from avala._config import _normalize_base_url
from avala.errors import (
    AuthenticationError,
    AvalaError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
)
from avala.types.account import SignupResponse

_DEFAULT_BASE_URL = "https://api.avala.ai/api/v1"


def _raise_for_status(response: httpx.Response) -> None:
    if response.is_success:
        return

    body = None
    message = f"HTTP {response.status_code}"
    try:
        body = response.json()
        message = body.get("detail", message) if isinstance(body, dict) else message
    except Exception:
        pass

    status = response.status_code
    if status == 401:
        raise AuthenticationError(message, status, body)
    if status == 404:
        raise NotFoundError(message, status, body)
    if status == 429:
        retry_after = response.headers.get("Retry-After")
        raise RateLimitError(
            message,
            status,
            body,
            retry_after=float(retry_after) if retry_after else None,
        )
    if status in (400, 422):
        details = body if isinstance(body, list) else None
        raise ValidationError(message, status, body, details=details)
    if status >= 500:
        raise ServerError(message, status, body)
    raise AvalaError(message, status, body)


def signup(
    *,
    email: str,
    password: str,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 30.0,
) -> SignupResponse:
    """Create a new Avala account.

    This function does not require an API key. On success it returns a
    :class:`SignupResponse` containing the created user and their API key.

    Args:
        email: The user's email address.
        password: The desired password.
        first_name: Optional first name.
        last_name: Optional last name.
        base_url: Override the API base URL (defaults to ``https://api.avala.ai/api/v1``
            or the ``AVALA_BASE_URL`` environment variable).
        timeout: Request timeout in seconds (default 30).

    Returns:
        :class:`SignupResponse` with ``user`` and ``api_key`` fields.
    """
    raw_url: str = base_url or os.environ.get("AVALA_BASE_URL") or _DEFAULT_BASE_URL
    resolved_url = _normalize_base_url(raw_url)
    payload: dict[str, str] = {"email": email, "password": password}
    if first_name is not None:
        payload["first_name"] = first_name
    if last_name is not None:
        payload["last_name"] = last_name

    with httpx.Client(timeout=timeout) as client:
        response = client.post(f"{resolved_url}/signup/", json=payload)

    _raise_for_status(response)
    return SignupResponse.model_validate(response.json())


async def async_signup(
    *,
    email: str,
    password: str,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 30.0,
) -> SignupResponse:
    """Async variant of :func:`signup`.

    Create a new Avala account asynchronously.

    Args:
        email: The user's email address.
        password: The desired password.
        first_name: Optional first name.
        last_name: Optional last name.
        base_url: Override the API base URL (defaults to ``https://api.avala.ai/api/v1``
            or the ``AVALA_BASE_URL`` environment variable).
        timeout: Request timeout in seconds (default 30).

    Returns:
        :class:`SignupResponse` with ``user`` and ``api_key`` fields.
    """
    raw_url: str = base_url or os.environ.get("AVALA_BASE_URL") or _DEFAULT_BASE_URL
    resolved_url = _normalize_base_url(raw_url)
    payload: dict[str, str] = {"email": email, "password": password}
    if first_name is not None:
        payload["first_name"] = first_name
    if last_name is not None:
        payload["last_name"] = last_name

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{resolved_url}/signup/", json=payload)

    _raise_for_status(response)
    return SignupResponse.model_validate(response.json())
