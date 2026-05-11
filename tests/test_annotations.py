"""Tests for the Annotations resource (SDK-level)."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from avala import Client

BASE_URL = "https://api.avala.ai/api/v1"
OWNER = "serve.robotics@avala.ai"
SLUG = "poc2"


def _frame_edit(item_uid: str, object_uuid: str, name: str = "vehicle") -> dict:
    return {
        "dataset_item_uid": item_uid,
        "sequence_uid": None,
        "action": "upsert",
        "duration": 0,
        "object_uuid": object_uuid,
        "object_type": "cuboid",
        "object_name": name,
        "object_name_plural": f"{name}s",
        "object_data": {
            "coordinate_system": "frame",
            "label": name,
            "object_id": object_uuid,
            "position": {"px": 0.0, "py": 0.0, "pz": 0.0},
            "dimensions": {"dx": 1.0, "dy": 1.0, "dz": 1.0},
            "rotation": {"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0},
        },
    }


@respx.mock
def test_bulk_edit_single_chunk():
    route = respx.post(f"{BASE_URL}/datasets/{OWNER}/{SLUG}/bulk-edition/").mock(
        return_value=httpx.Response(200, json={"updated": 2})
    )
    client = Client(api_key="test-key")
    edits = [_frame_edit("item-1", "obj-1"), _frame_edit("item-2", "obj-2")]
    results = list(client.annotations.bulk_edit(OWNER, SLUG, edits, chunk_size=10))
    assert len(results) == 1
    assert results[0][0] == 0
    assert results[0][1] == {"updated": 2}
    assert route.called
    sent = json.loads(route.calls[0].request.content)
    assert isinstance(sent, list)
    assert len(sent) == 2
    client.close()


@respx.mock
def test_bulk_edit_multi_chunk():
    route = respx.post(f"{BASE_URL}/datasets/{OWNER}/{SLUG}/bulk-edition/").mock(
        return_value=httpx.Response(200, json={"updated": 1})
    )
    client = Client(api_key="test-key")
    edits = [_frame_edit(f"item-{i}", f"obj-{i}") for i in range(7)]
    indices = [idx for idx, _resp in client.annotations.bulk_edit(OWNER, SLUG, edits, chunk_size=3)]
    assert indices == [0, 1, 2]
    assert route.call_count == 3
    chunks = [json.loads(c.request.content) for c in route.calls]
    assert [len(c) for c in chunks] == [3, 3, 1]
    client.close()


@respx.mock
def test_bulk_edit_surfaces_chunk_index_on_error():
    responses = [
        httpx.Response(200, json={}),
        httpx.Response(400, json={"detail": "bad payload"}),
    ]
    respx.post(f"{BASE_URL}/datasets/{OWNER}/{SLUG}/bulk-edition/").mock(side_effect=responses)
    client = Client(api_key="test-key")
    edits = [_frame_edit(f"item-{i}", f"obj-{i}") for i in range(4)]
    with pytest.raises(RuntimeError, match=r"chunk 1"):
        # Force iteration so the second chunk is posted.
        list(client.annotations.bulk_edit(OWNER, SLUG, edits, chunk_size=2))
    client.close()


def test_bulk_edit_rejects_zero_chunk_size():
    client = Client(api_key="test-key")
    with pytest.raises(ValueError, match="chunk_size"):
        list(client.annotations.bulk_edit(OWNER, SLUG, [_frame_edit("a", "b")], chunk_size=0))
    client.close()
