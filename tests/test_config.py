"""Tests for ClientConfig.from_params()."""

from __future__ import annotations

import pytest

from avala._config import ClientConfig


def test_explicit_api_key():
    """Explicit api_key is stored on the config."""
    config = ClientConfig.from_params(api_key="explicit-key")
    assert config.api_key == "explicit-key"


def test_env_var_fallback(monkeypatch: pytest.MonkeyPatch):
    """Falls back to AVALA_API_KEY env var when api_key not passed."""
    monkeypatch.setenv("AVALA_API_KEY", "env-key")
    monkeypatch.delenv("AVALA_BASE_URL", raising=False)
    config = ClientConfig.from_params()
    assert config.api_key == "env-key"


def test_explicit_overrides_env_var(monkeypatch: pytest.MonkeyPatch):
    """Explicit api_key takes precedence over AVALA_API_KEY env var."""
    monkeypatch.setenv("AVALA_API_KEY", "env-key")
    config = ClientConfig.from_params(api_key="override-key")
    assert config.api_key == "override-key"


def test_missing_key_raises_value_error(monkeypatch: pytest.MonkeyPatch):
    """Raises ValueError when neither api_key param nor env var is set."""
    monkeypatch.delenv("AVALA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No API key"):
        ClientConfig.from_params()


def test_custom_base_url_trailing_slash_stripped():
    """Trailing slash is stripped from custom base_url."""
    config = ClientConfig.from_params(api_key="key", base_url="https://custom.example.com/api/v1/")
    assert config.base_url == "https://custom.example.com/api/v1"


def test_custom_base_url_without_trailing_slash():
    """Custom base_url without trailing slash is stored as-is."""
    config = ClientConfig.from_params(api_key="key", base_url="https://custom.example.com/api/v1")
    assert config.base_url == "https://custom.example.com/api/v1"


def test_base_url_from_env(monkeypatch: pytest.MonkeyPatch):
    """AVALA_BASE_URL env var is used when base_url not passed."""
    monkeypatch.setenv("AVALA_BASE_URL", "https://env-server.example.com/api/v1")
    config = ClientConfig.from_params(api_key="key")
    assert config.base_url == "https://env-server.example.com/api/v1"


def test_custom_timeout_and_retries():
    """Custom timeout and max_retries are stored correctly."""
    config = ClientConfig.from_params(api_key="key", timeout=60.0, max_retries=5)
    assert config.timeout == 60.0
    assert config.max_retries == 5


def test_default_timeout_and_retries():
    """Default timeout is 30.0 and default max_retries is 2."""
    config = ClientConfig.from_params(api_key="key")
    assert config.timeout == 30.0
    assert config.max_retries == 2
