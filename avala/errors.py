"""Error hierarchy for the Avala SDK."""

from __future__ import annotations

from typing import Any


class AvalaError(Exception):
    """Base exception for all Avala API errors."""

    def __init__(self, message: str, status_code: int | None = None, body: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.body = body


class AuthenticationError(AvalaError):
    """Raised on 401 responses."""


class NotFoundError(AvalaError):
    """Raised on 404 responses."""


class RateLimitError(AvalaError):
    """Raised on 429 responses."""

    def __init__(
        self,
        message: str,
        status_code: int = 429,
        body: Any = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, status_code, body)
        self.retry_after = retry_after


class ValidationError(AvalaError):
    """Raised on 400/422 responses."""

    def __init__(
        self,
        message: str,
        status_code: int = 400,
        body: Any = None,
        details: list[Any] | None = None,
    ) -> None:
        super().__init__(message, status_code, body)
        self.details = details or []


class ServerError(AvalaError):
    """Raised on 5xx responses."""
