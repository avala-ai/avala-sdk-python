import pytest


@pytest.fixture
def api_key():
    return "test-api-key-123"


@pytest.fixture
def base_url():
    return "https://server.avala.ai/api/v1"
