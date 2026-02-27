from __future__ import annotations

import httpx
import respx

from avala import Client

BASE_URL = "https://api.avala.ai/api/v1"


@respx.mock
def test_list_datasets():
    respx.get(f"{BASE_URL}/datasets/").mock(
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
def test_list_datasets_with_filters():
    """Datasets.list() sends filter query params to the API."""
    route = respx.get(f"{BASE_URL}/datasets/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "uid": "550e8400-e29b-41d4-a716-446655440000",
                        "name": "Highway MCAP",
                        "slug": "highway-mcap",
                        "item_count": 50,
                        "data_type": "mcap",
                    }
                ],
                "next": None,
                "previous": None,
            },
        )
    )
    client = Client(api_key="test-key")
    page = client.datasets.list(data_type="mcap", name="highway", status="created", visibility="private")
    assert len(page.items) == 1
    assert route.called
    request = route.calls[0].request
    assert request.url.params["data_type"] == "mcap"
    assert request.url.params["name"] == "highway"
    assert request.url.params["status"] == "created"
    assert request.url.params["visibility"] == "private"
    client.close()


@respx.mock
def test_get_dataset():
    uid = "550e8400-e29b-41d4-a716-446655440000"
    respx.get(f"{BASE_URL}/datasets/{uid}/").mock(
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
    respx.get(f"{BASE_URL}/datasets/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [{"uid": "aaa", "name": "Dataset 1", "slug": "ds-1", "item_count": 10}],
                "next": f"{BASE_URL}/datasets/?cursor=abc123",
                "previous": None,
            },
        )
    )
    client = Client(api_key="test-key")
    page = client.datasets.list()
    assert page.has_more is True
    assert page.next_cursor == "abc123"
    client.close()


@respx.mock
def test_create_dataset():
    """Datasets.create() sends a POST and returns a Dataset."""
    respx.post(f"{BASE_URL}/datasets/").mock(
        return_value=httpx.Response(
            201,
            json={
                "uid": "new-dataset-uid",
                "name": "New Dataset",
                "slug": "new-dataset",
                "item_count": 0,
                "data_type": "lidar",
            },
        )
    )
    client = Client(api_key="test-key")
    dataset = client.datasets.create(
        name="New Dataset",
        slug="new-dataset",
        data_type="lidar",
        is_sequence=True,
        visibility="private",
    )
    assert dataset.uid == "new-dataset-uid"
    assert dataset.name == "New Dataset"
    client.close()


@respx.mock
def test_create_dataset_with_provider_config():
    """Datasets.create() includes provider_config and owner_name in the payload."""
    route = respx.post(f"{BASE_URL}/datasets/").mock(
        return_value=httpx.Response(
            201,
            json={
                "uid": "s3-dataset-uid",
                "name": "S3 Dataset",
                "slug": "s3-dataset",
                "item_count": 0,
                "data_type": "image",
            },
        )
    )
    client = Client(api_key="test-key")
    dataset = client.datasets.create(
        name="S3 Dataset",
        slug="s3-dataset",
        data_type="image",
        provider_config={
            "provider": "aws_s3",
            "s3_bucket_name": "my-bucket",
            "s3_bucket_region": "us-east-1",
        },
        owner_name="my-org",
    )
    assert dataset.uid == "s3-dataset-uid"
    assert route.called
    # Verify request body
    import json

    request = route.calls[0].request
    body = json.loads(request.content)
    assert body["name"] == "S3 Dataset"
    assert body["data_type"] == "image"
    assert body["provider_config"]["provider"] == "aws_s3"
    assert body["owner_name"] == "my-org"
    client.close()


@respx.mock
def test_list_items():
    """Datasets.list_items() returns a CursorPage of DatasetItem objects."""
    owner = "test-org"
    slug = "test-dataset"
    respx.get(f"{BASE_URL}/datasets/{owner}/{slug}/items/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "uid": "item-001",
                        "key": "image_001.png",
                        "dataset": "ds-001",
                        "url": "https://example.com/image_001.png",
                    }
                ],
                "next": None,
                "previous": None,
            },
        )
    )
    client = Client(api_key="test-key")
    page = client.datasets.list_items(owner, slug)
    assert len(page.items) == 1
    assert page.items[0].uid == "item-001"
    assert page.items[0].key == "image_001.png"
    assert page.has_more is False
    client.close()


@respx.mock
def test_get_item():
    """Datasets.get_item() returns a single DatasetItem."""
    owner = "test-org"
    slug = "test-dataset"
    item_uid = "item-001"
    respx.get(f"{BASE_URL}/datasets/{owner}/{slug}/items/{item_uid}/").mock(
        return_value=httpx.Response(
            200,
            json={
                "uid": item_uid,
                "key": "image_001.png",
                "dataset": "ds-001",
                "url": "https://example.com/image_001.png",
                "video_thumbnail": "https://example.com/thumb.jpg",
            },
        )
    )
    client = Client(api_key="test-key")
    item = client.datasets.get_item(owner, slug, item_uid)
    assert item.uid == item_uid
    assert item.key == "image_001.png"
    assert item.video_thumbnail == "https://example.com/thumb.jpg"
    client.close()


@respx.mock
def test_list_sequences():
    """Datasets.list_sequences() returns a CursorPage of DatasetSequence objects."""
    owner = "test-org"
    slug = "test-dataset"
    respx.get(f"{BASE_URL}/datasets/{owner}/{slug}/sequences/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "uid": "seq-001",
                        "key": "sequence_001",
                        "custom_uuid": "custom-uuid-001",
                        "status": "new",
                        "featured_image": "https://example.com/thumb.jpg",
                        "number_of_frames": 120,
                    }
                ],
                "next": None,
                "previous": None,
            },
        )
    )
    client = Client(api_key="test-key")
    page = client.datasets.list_sequences(owner, slug)
    assert len(page.items) == 1
    assert page.items[0].uid == "seq-001"
    assert page.items[0].key == "sequence_001"
    assert page.items[0].status == "new"
    assert page.items[0].number_of_frames == 120
    assert page.has_more is False
    client.close()


@respx.mock
def test_get_sequence():
    """Datasets.get_sequence() uses /sequences/ path and returns DatasetSequence."""
    owner = "test-org"
    slug = "test-dataset"
    seq_uid = "seq-001"
    route = respx.get(f"{BASE_URL}/datasets/{owner}/{slug}/sequences/{seq_uid}/").mock(
        return_value=httpx.Response(
            200,
            json={
                "uid": seq_uid,
                "key": "sequence_001",
                "status": "new",
                "dataset_uid": "ds-001",
                "predefined_labels": [],
                "frames": [],
                "metrics": {},
            },
        )
    )
    client = Client(api_key="test-key")
    seq = client.datasets.get_sequence(owner, slug, seq_uid)
    assert seq.uid == seq_uid
    assert seq.key == "sequence_001"
    assert seq.dataset_uid == "ds-001"
    assert route.called
    client.close()
