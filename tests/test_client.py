import pytest

from avala import Client
from avala.errors import AvalaError


def test_client_requires_api_key():
    """Client raises ValueError when no API key is provided."""
    import os
    # Ensure env var is not set
    env_key = os.environ.pop("AVALA_API_KEY", None)
    try:
        with pytest.raises(ValueError, match="No API key"):
            Client()
    finally:
        if env_key is not None:
            os.environ["AVALA_API_KEY"] = env_key


def test_client_accepts_api_key():
    """Client can be created with an explicit API key."""
    client = Client(api_key="test-key")
    assert client.datasets is not None
    assert client.projects is not None
    assert client.exports is not None
    assert client.tasks is not None
    client.close()


def test_client_context_manager():
    """Client can be used as a context manager."""
    with Client(api_key="test-key") as client:
        assert client.datasets is not None
