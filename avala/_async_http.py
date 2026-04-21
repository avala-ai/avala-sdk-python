"""Asynchronous HTTP transport using httpx."""

from __future__ import annotations

from typing import Any, Type

import httpx
from pydantic import BaseModel

from avala._config import ClientConfig
from avala._http import _extract_cursor, _validate_path
from avala._pagination import CursorPage
from avala.errors import (
    AuthenticationError,
    AvalaError,
    ForbiddenError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ValidationError,
)


class AsyncHTTPTransport:
    def __init__(self, config: ClientConfig) -> None:
        self._config = config
        self._last_rate_limit: dict[str, str | None] = {
            "limit": None,
            "remaining": None,
            "reset": None,
        }
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            headers={
                "X-Avala-Api-Key": config.api_key,
                "Accept": "application/json",
            },
            timeout=config.timeout,
            # Explicit — httpx defaults to False, but a follow_redirects=True
            # regression would replay X-Avala-Api-Key on cross-host 3xx.
            follow_redirects=False,
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

    async def request(self, method: str, path: str, **kwargs: Any) -> Any:
        _validate_path(path)
        response = await self._client.request(method, path, **kwargs)
        self._extract_rate_limit_headers(response)
        self._raise_for_status(response)
        if response.status_code == 204:
            return None
        return response.json()

    async def request_page(
        self, path: str, model_cls: Type[BaseModel], params: dict[str, Any] | None = None
    ) -> CursorPage[Any]:
        _validate_path(path)
        response = await self._client.get(path, params=params)
        self._extract_rate_limit_headers(response)
        self._raise_for_status(response)
        data = response.json()
        items = [model_cls.model_validate(item) for item in data.get("results", [])]
        next_cursor = _extract_cursor(data.get("next"))
        previous_cursor = _extract_cursor(data.get("previous"))
        return CursorPage(items=items, next_cursor=next_cursor, previous_cursor=previous_cursor)

    async def request_list(
        self, path: str, model_cls: Type[BaseModel], params: dict[str, Any] | None = None
    ) -> list[Any]:
        """Fetch an endpoint that returns a plain JSON array (no pagination wrapper)."""
        _validate_path(path)
        response = await self._client.get(path, params=params)
        self._extract_rate_limit_headers(response)
        self._raise_for_status(response)
        data = response.json()
        return [model_cls.model_validate(item) for item in data]

    async def close(self) -> None:
        await self._client.aclose()

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
        if status == 403:
            raise ForbiddenError(message, status, body)
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
