import pytest

from avala import Client


def test_client_requires_api_key(monkeypatch: pytest.MonkeyPatch):
    """Client raises ValueError when no API key is provided."""
    monkeypatch.delenv("AVALA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No API key"):
        Client()


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
