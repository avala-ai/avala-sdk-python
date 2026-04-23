"""Tests for validation-friendly methods: get_frame, get_calibration, get_health.

These methods let an AI agent (or a human) verify a dataset programmatically
after ingest — e.g. "did all 569 frames land, do they have calibration?" —
without opening Mission Control.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from avala import AsyncClient, Client

BASE_URL = "https://api.avala.ai/api/v1"

SAMPLE_FRAME = {
    "frame_index": 0,
    "key": "frame-0.json",
    "model": "pinhole",
    "camera_model": "pinhole",
    "xi": None,
    "alpha": None,
    "device_position": {"x": 1.0, "y": 2.0, "z": 3.0},
    "device_heading": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    "images": [
        {
            "image_url": "s3://bucket/frame-0/cam_01.jpg",
            "position": {"x": 0.1, "y": 0.2, "z": 0.3},
            "heading": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            "width": 1920,
            "height": 1080,
            "fx": 824.74,
            "fy": 834.49,
            "cx": 960.0,
            "cy": 540.0,
            "model": "pinhole",
        },
        {
            "image_url": "s3://bucket/frame-0/cam_02.jpg",
            "position": {"x": 0.4, "y": 0.5, "z": 0.6},
            "heading": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            "width": 1920,
            "height": 1456,
            "fx": 744.78,
            "fy": 726.75,
            "cx": 960.0,
            "cy": 728.0,
            "model": "pinhole",
        },
    ],
}


def _sequence_payload(frames: list[dict]) -> dict:
    return {
        "uid": "55555555-5555-5555-5555-555555555555",
        "key": "full-scene-569",
        "status": "completed",
        "number_of_frames": len(frames),
        "frames": frames,
        "dataset_uid": "44444444-4444-4444-4444-444444444444",
        "allow_lidar_calibration": True,
        "lidar_calibration_enabled": True,
        "camera_calibration_enabled": True,
    }


HEALTH_PAYLOAD = {
    "dataset_uid": "44444444-4444-4444-4444-444444444444",
    "dataset_slug": "third-dimension-095940-full-scene",
    "dataset_status": "created",
    "item_count": 569,
    "sequence_count": 1,
    "total_frames": 569,
    "s3_prefix": "third-dimension/alf_data/third-dimension-095940-full-scene/full-scene-569",
    "gc_storage_prefix": None,
    "last_updated_at": "2026-04-21T15:33:02Z",
    "sequences": [
        {
            "uid": "55555555-5555-5555-5555-555555555555",
            "key": "full-scene-569",
            "status": "completed",
            "frame_count": 569,
            "has_lidar_calibration": True,
            "has_camera_calibration": True,
        }
    ],
    "ingest_ok": True,
    "issues": [],
}


# -- get_frame -------------------------------------------------------------


@respx.mock
def test_get_frame_returns_typed_metadata():
    frames = [SAMPLE_FRAME, {**SAMPLE_FRAME, "frame_index": 1, "key": "frame-1.json"}]
    respx.get(
        f"{BASE_URL}/datasets/thirddimension/third-dimension-095940-full-scene"
        "/sequences/55555555-5555-5555-5555-555555555555/"
    ).mock(return_value=httpx.Response(200, json=_sequence_payload(frames)))
    client = Client(api_key="test-key")
    frame = client.datasets.get_frame(
        "thirddimension",
        "third-dimension-095940-full-scene",
        "55555555-5555-5555-5555-555555555555",
        0,
    )
    assert frame.frame_index == 0
    assert frame.model == "pinhole"
    assert frame.device_position is not None
    assert frame.device_position.x == 1.0
    assert frame.images is not None
    assert len(frame.images) == 2
    assert frame.images[0].fx == 824.74
    assert frame.images[1].cy == 728.0
    assert frame.raw["key"] == "frame-0.json"
    client.close()


@respx.mock
def test_get_frame_out_of_range_raises_index_error():
    respx.get(
        f"{BASE_URL}/datasets/thirddimension/third-dimension-095940-full-scene"
        "/sequences/55555555-5555-5555-5555-555555555555/"
    ).mock(return_value=httpx.Response(200, json=_sequence_payload([SAMPLE_FRAME])))
    client = Client(api_key="test-key")
    with pytest.raises(IndexError, match="out of range"):
        client.datasets.get_frame(
            "thirddimension",
            "third-dimension-095940-full-scene",
            "55555555-5555-5555-5555-555555555555",
            99,
        )
    client.close()


# -- get_calibration -------------------------------------------------------


@respx.mock
def test_get_calibration_extracts_per_camera_rig_from_frame0():
    respx.get(
        f"{BASE_URL}/datasets/thirddimension/third-dimension-095940-full-scene"
        "/sequences/55555555-5555-5555-5555-555555555555/"
    ).mock(return_value=httpx.Response(200, json=_sequence_payload([SAMPLE_FRAME])))
    client = Client(api_key="test-key")
    calib = client.datasets.get_calibration(
        "thirddimension",
        "third-dimension-095940-full-scene",
        "55555555-5555-5555-5555-555555555555",
    )
    assert calib.sequence_uid == "55555555-5555-5555-5555-555555555555"
    assert len(calib.cameras) == 2
    assert calib.cameras[0].model == "pinhole"
    assert calib.cameras[0].fx == 824.74
    assert calib.cameras[1].cy == 728.0
    client.close()


@respx.mock
def test_get_calibration_empty_sequence_returns_empty_cameras():
    respx.get(
        f"{BASE_URL}/datasets/thirddimension/third-dimension-095940-full-scene"
        "/sequences/55555555-5555-5555-5555-555555555555/"
    ).mock(return_value=httpx.Response(200, json=_sequence_payload([])))
    client = Client(api_key="test-key")
    calib = client.datasets.get_calibration(
        "thirddimension",
        "third-dimension-095940-full-scene",
        "55555555-5555-5555-5555-555555555555",
    )
    assert calib.cameras == []
    client.close()


# -- get_health ------------------------------------------------------------


@respx.mock
def test_get_health_returns_typed_snapshot():
    respx.get(f"{BASE_URL}/datasets/thirddimension/third-dimension-095940-full-scene/health/").mock(
        return_value=httpx.Response(200, json=HEALTH_PAYLOAD)
    )
    client = Client(api_key="test-key")
    health = client.datasets.get_health("thirddimension", "third-dimension-095940-full-scene")
    assert health.total_frames == 569
    assert health.ingest_ok is True
    assert len(health.sequences) == 1
    assert health.sequences[0].frame_count == 569
    assert health.sequences[0].has_lidar_calibration is True
    assert health.issues == []
    client.close()


@respx.mock
def test_get_health_surfaces_issues():
    payload = {
        **HEALTH_PAYLOAD,
        "ingest_ok": False,
        "total_frames": 0,
        "issues": ["Dataset has zero items and zero sequences"],
        "sequences": [],
        "sequence_count": 0,
    }
    respx.get(f"{BASE_URL}/datasets/bad-org/empty-dataset/health/").mock(return_value=httpx.Response(200, json=payload))
    client = Client(api_key="test-key")
    health = client.datasets.get_health("bad-org", "empty-dataset")
    assert health.ingest_ok is False
    assert "zero items and zero sequences" in health.issues[0]
    client.close()


# -- Async parity ----------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_async_get_frame():
    respx.get(
        f"{BASE_URL}/datasets/thirddimension/third-dimension-095940-full-scene"
        "/sequences/55555555-5555-5555-5555-555555555555/"
    ).mock(return_value=httpx.Response(200, json=_sequence_payload([SAMPLE_FRAME])))
    async with AsyncClient(api_key="test-key") as client:
        frame = await client.datasets.get_frame(
            "thirddimension",
            "third-dimension-095940-full-scene",
            "55555555-5555-5555-5555-555555555555",
            0,
        )
    assert frame.model == "pinhole"
    assert frame.images is not None
    assert len(frame.images) == 2


@pytest.mark.asyncio
@respx.mock
async def test_async_get_health():
    respx.get(f"{BASE_URL}/datasets/thirddimension/third-dimension-095940-full-scene/health/").mock(
        return_value=httpx.Response(200, json=HEALTH_PAYLOAD)
    )
    async with AsyncClient(api_key="test-key") as client:
        health = await client.datasets.get_health("thirddimension", "third-dimension-095940-full-scene")
    assert health.total_frames == 569
    assert health.ingest_ok is True
