"""Synchronous HTTP transport using httpx."""

from __future__ import annotations

from typing import Any, Type
from urllib.parse import parse_qs, urlparse

import httpx
from pydantic import BaseModel

from avala._config import ClientConfig
from avala._pagination import CursorPage
from avala.errors import (
    AuthenticationError,
    AvalaError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
)


class SyncHTTPTransport:
    def __init__(self, config: ClientConfig) -> None:
        self._config = config
        self._last_rate_limit: dict[str, str | None] = {
            "limit": None,
            "remaining": None,
            "reset": None,
        }
        self._client = httpx.Client(
            base_url=config.base_url,
            headers={
                "X-Avala-Api-Key": config.api_key,
                "Accept": "application/json",
            },
            timeout=config.timeout,
        )

    @property
    def last_rate_limit(self) -> dict[str, str | None]:
        return dict(self._last_rate_limit)

    def _extract_rate_limit_headers(self, response: httpx.Response) -> None:
        self._last_rate_limit = {
            "limit": response.headers.get("X-RateLimit-Limit"),
            "remaining": response.headers.get("X-RateLimit-Remaining"),
            "reset": response.headers.get("X-RateLimit-Reset"),
        }

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self._client.request(method, path, **kwargs)
        self._extract_rate_limit_headers(response)
        self._raise_for_status(response)
        if response.status_code == 204:
            return None
        return response.json()

    def request_page(
        self, path: str, model_cls: Type[BaseModel], params: dict[str, Any] | None = None
    ) -> CursorPage[Any]:
        response = self._client.get(path, params=params)
        self._extract_rate_limit_headers(response)
        self._raise_for_status(response)
        data = response.json()
        items = [model_cls.model_validate(item) for item in data.get("results", [])]
        next_cursor = _extract_cursor(data.get("next"))
        previous_cursor = _extract_cursor(data.get("previous"))
        return CursorPage(items=items, next_cursor=next_cursor, previous_cursor=previous_cursor)

    def close(self) -> None:
        self._client.close()

    def _raise_for_status(self, response: httpx.Response) -> None:
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


def _extract_cursor(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    cursors = qs.get("cursor", [])
    return cursors[0] if cursors else None
