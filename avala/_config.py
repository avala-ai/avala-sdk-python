"""Client configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


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
        resolved_url = base_url or os.environ.get("AVALA_BASE_URL", "https://server.avala.ai/api/v1")
        return cls(
            api_key=resolved_key,
            base_url=resolved_url.rstrip("/"),
            timeout=timeout,
            max_retries=max_retries,
        )
