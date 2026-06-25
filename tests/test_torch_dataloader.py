from __future__ import annotations

import re

import httpx
import pytest
import respx
from avala import Client

pytest.importorskip("torch")

from avala.torch import AvalaDataset, AvalaIterableDataset  # noqa: E402

BASE_URL = "https://api.avala.ai/api/v1"
LIST_RE = re.escape(f"{BASE_URL}/datasets/o/s/items/") + r"(\?.*)?$"
ITEM_RE = re.escape(f"{BASE_URL}/datasets/o/s/items/") + r"[^/?]+/$"


def _item(uid: str, label: str) -> dict:
    return {
        "uid": uid,
        "key": f"{uid}.jpg",
        "url": f"https://cdn.example.com/{uid}.jpg",
        "annotations": {"label": label},
    }


def _list_handler(request: httpx.Request) -> httpx.Response:
    # Page 1 (no cursor) -> two items + a next cursor; page 2 -> one item, no next.
    if request.url.params.get("cursor") is None:
        return httpx.Response(
            200,
            json={
                "results": [_item("a", "cat"), _item("b", "dog")],
                "next": f"{BASE_URL}/datasets/o/s/items/?cursor=p2",
                "previous": None,
            },
        )
    return httpx.Response(
        200,
        json={"results": [_item("c", "fish")], "next": None, "previous": None},
    )


@respx.mock
def test_iterable_streams_all_items_without_export() -> None:
    respx.get(url__regex=LIST_RE).mock(side_effect=_list_handler)
    export_route = respx.post(f"{BASE_URL}/exports/").mock(return_value=httpx.Response(201, json={}))

    client = Client(api_key="test-key")
    ds = AvalaIterableDataset(client, "o", "s")
    samples = list(ds)

    assert [s["uid"] for s in samples] == ["a", "b", "c"]
    assert samples[0]["annotations"] == {"label": "cat"}
    # The whole point: no export was ever created.
    assert not export_route.called
    client.close()


@respx.mock
def test_iterable_applies_transform() -> None:
    respx.get(url__regex=LIST_RE).mock(side_effect=_list_handler)
    client = Client(api_key="test-key")
    ds = AvalaIterableDataset(client, "o", "s", transform=lambda s: s["uid"].upper())
    assert list(ds) == ["A", "B", "C"]
    client.close()


@respx.mock
def test_ddp_sharding_partitions_items_disjointly() -> None:
    respx.get(url__regex=LIST_RE).mock(side_effect=_list_handler)
    client = Client(api_key="test-key")

    rank0 = {s["uid"] for s in AvalaIterableDataset(client, "o", "s", rank=0, world_size=2)}
    rank1 = {s["uid"] for s in AvalaIterableDataset(client, "o", "s", rank=1, world_size=2)}

    assert rank0.isdisjoint(rank1)
    assert rank0 | rank1 == {"a", "b", "c"}
    client.close()


@respx.mock
def test_map_dataset_len_and_getitem() -> None:
    respx.get(url__regex=LIST_RE).mock(side_effect=_list_handler)
    respx.get(url__regex=ITEM_RE).mock(
        side_effect=lambda req: httpx.Response(200, json=_item(req.url.path.rstrip("/").split("/")[-1], "x"))
    )

    client = Client(api_key="test-key")
    ds = AvalaDataset(client, "o", "s")

    assert len(ds) == 3
    sample = ds[1]
    assert sample["uid"] == "b"
    assert sample["annotations"] == {"label": "x"}
    ds.close()
    client.close()
