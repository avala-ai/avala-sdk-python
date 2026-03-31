"""Tests for Datasets.wait() and AsyncDatasets.wait()."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest
import respx

from avala import AsyncClient, Client

BASE_URL = "https://api.avala.ai/api/v1"

DATASET_CREATING = {"uid": "ds-001", "name": "Test", "slug": "test", "item_count": 0, "status": "creating"}
DATASET_CREATED = {"uid": "ds-001", "name": "Test", "slug": "test", "item_count": 42, "status": "created"}


@respx.mock
@patch("avala.resources.datasets.time.sleep", return_value=None)
def test_wait_immediate(mock_sleep):
    """wait() returns immediately when dataset already has the target status."""
    respx.get(f"{BASE_URL}/datasets/ds-001/").mock(return_value=httpx.Response(200, json=DATASET_CREATED))
    client = Client(api_key="test-key")
    dataset = client.datasets.wait("ds-001")
    assert dataset.status == "created"
    assert dataset.item_count == 42
    mock_sleep.assert_not_called()
    client.close()


@respx.mock
@patch("avala.resources.datasets.time.sleep", return_value=None)
def test_wait_polls_until_status_matches(mock_sleep):
    """wait() polls multiple times until the target status is reached."""
    respx.get(f"{BASE_URL}/datasets/ds-001/").mock(
        side_effect=[
            httpx.Response(200, json=DATASET_CREATING),
            httpx.Response(200, json=DATASET_CREATING),
            httpx.Response(200, json=DATASET_CREATED),
        ]
    )
    client = Client(api_key="test-key")
    dataset = client.datasets.wait("ds-001", interval=0.1)
    assert dataset.status == "created"
    assert mock_sleep.call_count == 2
    client.close()


@respx.mock
@patch("avala.resources.datasets.time.sleep", return_value=None)
@patch("avala.resources.datasets.time.monotonic")
def test_wait_timeout(mock_monotonic, mock_sleep):
    """wait() raises TimeoutError when the deadline is exceeded."""
    mock_monotonic.side_effect = [0.0, 1.0, 999.0]
    respx.get(f"{BASE_URL}/datasets/ds-001/").mock(
        side_effect=[
            httpx.Response(200, json=DATASET_CREATING),
            httpx.Response(200, json=DATASET_CREATING),
        ]
    )
    client = Client(api_key="test-key")
    with pytest.raises(TimeoutError, match="did not reach status"):
        client.datasets.wait("ds-001", timeout=10.0)
    client.close()


def test_wait_negative_timeout():
    """wait() raises ValueError for negative timeout."""
    client = Client(api_key="test-key")
    with pytest.raises(ValueError, match="timeout must be non-negative"):
        client.datasets.wait("ds-001", timeout=-1)
    client.close()


def test_wait_negative_interval():
    """wait() raises ValueError for negative interval."""
    client = Client(api_key="test-key")
    with pytest.raises(ValueError, match="interval must be non-negative"):
        client.datasets.wait("ds-001", interval=-1)
    client.close()


@respx.mock
@patch("avala.resources.datasets.time.sleep", return_value=None)
def test_wait_interval_clamped(mock_sleep):
    """wait() clamps interval to _MIN_INTERVAL (1.0)."""
    respx.get(f"{BASE_URL}/datasets/ds-001/").mock(
        side_effect=[
            httpx.Response(200, json=DATASET_CREATING),
            httpx.Response(200, json=DATASET_CREATED),
        ]
    )
    client = Client(api_key="test-key")
    client.datasets.wait("ds-001", interval=0.01)
    mock_sleep.assert_called_once_with(1.0)
    client.close()


@respx.mock
@patch("avala.resources.datasets.time.sleep", return_value=None)
def test_wait_custom_status(mock_sleep):
    """wait() supports custom target status."""
    respx.get(f"{BASE_URL}/datasets/ds-001/").mock(return_value=httpx.Response(200, json=DATASET_CREATING))
    client = Client(api_key="test-key")
    dataset = client.datasets.wait("ds-001", status="creating")
    assert dataset.status == "creating"
    mock_sleep.assert_not_called()
    client.close()


@respx.mock
@patch("avala.resources.datasets.time.sleep", return_value=None)
def test_wait_on_poll_callback(mock_sleep):
    """wait() calls _on_poll with the Dataset after each non-matching poll."""
    respx.get(f"{BASE_URL}/datasets/ds-001/").mock(
        side_effect=[
            httpx.Response(200, json=DATASET_CREATING),
            httpx.Response(200, json=DATASET_CREATED),
        ]
    )
    poll_datasets = []

    def on_poll(d):
        poll_datasets.append(d)

    client = Client(api_key="test-key")
    client.datasets.wait("ds-001", interval=0.1, _on_poll=on_poll)
    assert len(poll_datasets) == 1
    assert poll_datasets[0].status == "creating"
    client.close()


# --- Async tests ---


@respx.mock
async def test_async_wait_immediate():
    """AsyncDatasets.wait() returns immediately when status matches."""
    respx.get(f"{BASE_URL}/datasets/ds-001/").mock(return_value=httpx.Response(200, json=DATASET_CREATED))
    async with AsyncClient(api_key="test-key") as client:
        dataset = await client.datasets.wait("ds-001")
    assert dataset.status == "created"


@respx.mock
async def test_async_wait_polls():
    """AsyncDatasets.wait() polls until target status."""
    respx.get(f"{BASE_URL}/datasets/ds-001/").mock(
        side_effect=[
            httpx.Response(200, json=DATASET_CREATING),
            httpx.Response(200, json=DATASET_CREATED),
        ]
    )
    async with AsyncClient(api_key="test-key") as client:
        dataset = await client.datasets.wait("ds-001", interval=0.01)
    assert dataset.status == "created"


@respx.mock
async def test_async_wait_timeout():
    """AsyncDatasets.wait() raises TimeoutError on timeout."""
    respx.get(f"{BASE_URL}/datasets/ds-001/").mock(return_value=httpx.Response(200, json=DATASET_CREATING))
    async with AsyncClient(api_key="test-key") as client:
        with pytest.raises(TimeoutError, match="did not reach status"):
            await client.datasets.wait("ds-001", timeout=0.01, interval=0.01)
