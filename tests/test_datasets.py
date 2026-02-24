import httpx
import respx

from avala import Client


@respx.mock
def test_list_datasets():
    respx.get("https://api.avala.ai/api/v1/datasets/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "uid": "550e8400-e29b-41d4-a716-446655440000",
                        "name": "Test Dataset",
                        "slug": "test-dataset",
                        "item_count": 100,
                    }
                ],
                "next": None,
                "previous": None,
            },
        )
    )
    client = Client(api_key="test-key")
    page = client.datasets.list()
    assert len(page.items) == 1
    assert page.items[0].name == "Test Dataset"
    assert page.items[0].uid == "550e8400-e29b-41d4-a716-446655440000"
    assert page.has_more is False
    client.close()


@respx.mock
def test_get_dataset():
    uid = "550e8400-e29b-41d4-a716-446655440000"
    respx.get(f"https://api.avala.ai/api/v1/datasets/{uid}/").mock(
        return_value=httpx.Response(
            200,
            json={
                "uid": uid,
                "name": "Test Dataset",
                "slug": "test-dataset",
                "item_count": 100,
            },
        )
    )
    client = Client(api_key="test-key")
    dataset = client.datasets.get(uid)
    assert dataset.name == "Test Dataset"
    assert dataset.uid == uid
    client.close()


@respx.mock
def test_list_datasets_with_pagination():
    respx.get("https://api.avala.ai/api/v1/datasets/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [{"uid": "aaa", "name": "Dataset 1", "slug": "ds-1", "item_count": 10}],
                "next": "https://api.avala.ai/api/v1/datasets/?cursor=abc123",
                "previous": None,
            },
        )
    )
    client = Client(api_key="test-key")
    page = client.datasets.list()
    assert page.has_more is True
    assert page.next_cursor == "abc123"
    client.close()
