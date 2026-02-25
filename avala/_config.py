"""Client configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse


def _is_truthy(value: str | None) -> bool:
    if not value:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_safe_localhost(hostname: str | None) -> bool:
    return hostname in {"localhost", "127.0.0.1", "::1"}


def _normalize_base_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Invalid base_url: expected a valid URL with scheme and host.")

    allow_insecure = _is_truthy(os.environ.get("AVALA_ALLOW_INSECURE_BASE_URL"))
    if parsed.scheme != "https":
        if not allow_insecure:
            raise ValueError(
                "Base URL must use HTTPS. Set AVALA_ALLOW_INSECURE_BASE_URL=true only for local development."
            )
        if parsed.scheme != "http":
            raise ValueError("With AVALA_ALLOW_INSECURE_BASE_URL=true, only http://localhost URLs are permitted.")
        if not _is_safe_localhost(parsed.hostname):
            raise ValueError("Non-HTTPS base URLs are allowed only for localhost addresses.")

    return base_url.rstrip("/")


@dataclass(frozen=True)
class ClientConfig:
    api_key: str
    base_url: str
    timeout: float
    max_retries: int

    @classmethod
    def from_params(
        cls,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> ClientConfig:
        resolved_key = api_key or os.environ.get("AVALA_API_KEY", "")
        if not resolved_key:
            raise ValueError("No API key provided. Pass api_key= or set the AVALA_API_KEY environment variable.")
        resolved_url = base_url or os.environ.get("AVALA_BASE_URL", "https://api.avala.ai/api/v1")
        resolved_url = _normalize_base_url(resolved_url)
        return cls(
            api_key=resolved_key,
            base_url=resolved_url,
            timeout=timeout,
            max_retries=max_retries,
        )
