from __future__ import annotations

import json

import httpx
import pytest
import respx
from avala import Client
from avala.errors import QuotaExceededError

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
def test_get_dataset_by_slug():
    owner = "serve.robotics@avala.ai"
    slug = "poc2"
    respx.get(f"{BASE_URL}/datasets/{owner}/{slug}/").mock(
        return_value=httpx.Response(
            200,
            json={
                "uid": "ds-uid",
                "name": "POC2",
                "slug": slug,
                "data_type": "lidar",
                "is_sequence": True,
                "predefined_labels": [
                    {
                        "name": "car",
                        "label_code": 1,
                        "locked": False,
                        "is_countable": True,
                    },
                ],
            },
        )
    )
    client = Client(api_key="test-key")
    dataset = client.datasets.get_by_slug(owner, slug)
    assert dataset.slug == slug
    assert dataset.data_type == "lidar"
    assert dataset.is_sequence is True
    assert dataset.predefined_labels == [
        {"name": "car", "label_code": 1, "locked": False, "is_countable": True},
    ]
    client.close()


@respx.mock
def test_list_datasets_with_pagination():
    respx.get(f"{BASE_URL}/datasets/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "uid": "aaa",
                        "name": "Dataset 1",
                        "slug": "ds-1",
                        "item_count": 10,
                    }
                ],
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
    route = respx.post(f"{BASE_URL}/datasets/").mock(
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
        visibility="private",
    )
    assert dataset.uid == "new-dataset-uid"
    assert dataset.name == "New Dataset"
    body = json.loads(route.calls[0].request.content)
    assert "is_sequence" not in body
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
    request = route.calls[0].request
    body = json.loads(request.content)
    assert body["name"] == "S3 Dataset"
    assert body["data_type"] == "image"
    assert body["provider_config"]["provider"] == "aws_s3"
    assert body["owner_name"] == "my-org"
    client.close()


@respx.mock
def test_create_dataset_with_organization_id():
    """Datasets.create() includes organization_id and other optional fields in the payload."""
    route = respx.post(f"{BASE_URL}/datasets/").mock(
        return_value=httpx.Response(
            201,
            json={
                "uid": "org-dataset-uid",
                "name": "Org Dataset",
                "slug": "org-dataset",
                "item_count": 0,
                "data_type": "lidar",
            },
        )
    )
    client = Client(api_key="test-key")
    dataset = client.datasets.create(
        name="Org Dataset",
        slug="org-dataset",
        data_type="lidar",
        organization_id=265,
        gpu_texture_format="ktx",
        industry=265,
        license=67,
        metadata={"key": "value"},
    )
    assert dataset.uid == "org-dataset-uid"
    assert route.called
    request = route.calls[0].request
    body = json.loads(request.content)
    assert body["organization_id"] == 265
    assert body["gpu_texture_format"] == "ktx"
    assert body["industry"] == 265
    assert body["license"] == 67
    assert body["metadata"] == {"key": "value"}
    assert "is_sequence" not in body
    client.close()


@respx.mock
def test_create_dataset_with_organization_uid():
    """Datasets.create() includes organization_uid when supplied."""
    route = respx.post(f"{BASE_URL}/datasets/").mock(
        return_value=httpx.Response(
            201,
            json={
                "uid": "org-dataset-uid",
                "name": "Org Dataset",
                "slug": "org-dataset",
                "item_count": 0,
                "data_type": "image",
            },
        )
    )
    client = Client(api_key="test-key")
    dataset = client.datasets.create(
        name="Org Dataset",
        slug="org-dataset",
        data_type="image",
        organization_uid="ef08fe22-7131-4bca-b0c3-c544b49072ad",
    )
    assert dataset.uid == "org-dataset-uid"
    assert route.called
    request = route.calls[0].request
    body = json.loads(request.content)
    assert body["organization_uid"] == "ef08fe22-7131-4bca-b0c3-c544b49072ad"
    client.close()


@respx.mock
def test_create_manual_upload_url():
    """Datasets.create_manual_upload_url() sends the local-upload presign payload."""
    route = respx.post(f"{BASE_URL}/datasets/manual-upload/file-upload-url/").mock(
        return_value=httpx.Response(
            200,
            json={
                "method": "POST",
                "url": "https://s3.example/upload",
                "fields": {"key": "user/dataset/frame.jpg"},
            },
        )
    )
    client = Client(api_key="test-key")
    upload_info = client.datasets.create_manual_upload_url(
        dataset_name="New Dataset",
        file_path_in_dataset="frame.jpg",
        content_length=1234,
    )
    assert upload_info["url"] == "https://s3.example/upload"
    body = json.loads(route.calls[0].request.content)
    assert body == {
        "dataset_name": "New Dataset",
        "file_path_in_dataset": "frame.jpg",
        "content_length": 1234,
    }
    client.close()


@respx.mock
def test_create_from_manual_upload():
    """Datasets.create_from_manual_upload() creates a dataset from uploaded local files."""
    route = respx.post(f"{BASE_URL}/datasets/manual-upload/").mock(
        return_value=httpx.Response(
            201,
            json={
                "uid": "manual-dataset-uid",
                "name": "Manual Dataset",
                "slug": "manual-dataset",
                "item_count": 0,
                "data_type": "image",
            },
        )
    )
    client = Client(api_key="test-key")
    dataset = client.datasets.create_from_manual_upload(
        name="Manual Dataset",
        slug="manual-dataset",
        data_type="image",
        industry=10,
        license=20,
    )
    assert dataset.uid == "manual-dataset-uid"
    body = json.loads(route.calls[0].request.content)
    assert body["name"] == "Manual Dataset"
    assert body["industry"] == 10
    assert body["license"] == 20
    assert "provider_config" not in body
    assert "is_sequence" not in body
    # Absent unless asked for, so a personal upload keeps its existing payload.
    assert "organization_uid" not in body
    client.close()


@respx.mock
def test_manual_upload_is_org_scoped_on_both_calls():
    """The presign and the create call must carry the SAME organization_uid.

    The server derives the S3 key prefix from org context on both. Presign
    under the user and create under the org (or vice versa) and the dataset's
    provider_config points at a prefix with no objects in it — the upload
    "succeeds" and the dataset lists zero items. This is the exact trap that
    forced a bespoke upload driver for the Delovantage ingest.
    """
    org = "e23266f5-18c2-4bda-8bcc-cc2a84dbd52e"
    presign = respx.post(f"{BASE_URL}/datasets/manual-upload/file-upload-url/").mock(
        return_value=httpx.Response(200, json={"method": "POST", "url": "https://s3.example/u", "fields": {}})
    )
    create = respx.post(f"{BASE_URL}/datasets/manual-upload/").mock(
        return_value=httpx.Response(
            201,
            json={"uid": "u", "name": "N", "slug": "n", "item_count": 0, "data_type": "mcap"},
        )
    )

    client = Client(api_key="test-key")
    client.datasets.create_manual_upload_url(
        dataset_name="N",
        file_path_in_dataset="a.mcap",
        content_length=1,
        organization_uid=org,
    )
    client.datasets.create_from_manual_upload(
        name="N",
        slug="n",
        data_type="mcap",
        organization_uid=org,
    )
    client.close()

    assert json.loads(presign.calls[0].request.content)["organization_uid"] == org
    assert json.loads(create.calls[0].request.content)["organization_uid"] == org


@respx.mock
def test_manual_upload_quota():
    """Datasets.manual_upload_quota() reads the owner's storage meter."""
    route = respx.get(f"{BASE_URL}/datasets/manual-upload/quota/").mock(
        return_value=httpx.Response(200, json={"used": 40 * 1024**3, "limit": 100 * 1024**3})
    )
    client = Client(api_key="test-key")
    quota = client.datasets.manual_upload_quota(organization_uid="org-uid")
    client.close()

    assert quota.used == 40 * 1024**3
    assert quota.remaining == 60 * 1024**3
    assert route.calls[0].request.url.params["organization_uid"] == "org-uid"


def test_upload_quota_remaining_never_negative():
    """A reconcile can land ``used`` above ``limit`` — the cap is enforced at
    presign, not applied retroactively. ``remaining`` must not go negative and
    imply headroom that doesn't exist."""
    from avala.types.manual_upload import UploadQuota

    assert UploadQuota(used=120, limit=100).remaining == 0


@respx.mock
def test_presign_quota_rejection_raises_quota_exceeded(tmp_path):
    """A 413 from presign carries the numbers the caller needs to act on."""
    (tmp_path / "a.mcap").write_bytes(b"x" * 16)
    respx.post(f"{BASE_URL}/datasets/manual-upload/file-upload-url/").mock(
        return_value=httpx.Response(413, json={"detail": "Storage quota exceeded", "limit": 100, "used": 99})
    )

    client = Client(api_key="test-key")
    with pytest.raises(QuotaExceededError) as excinfo:
        client.datasets.upload_files(dataset_name="X", files=[(str(tmp_path / "a.mcap"), "a.mcap")], workers=1)
    client.close()

    assert excinfo.value.limit == 100
    assert excinfo.value.used == 99


@respx.mock
def test_create_from_local_preflights_quota(tmp_path):
    """An upload that cannot possibly fit is refused before any bytes move."""
    (tmp_path / "a.mcap").write_bytes(b"x" * 4096)
    quota = respx.get(f"{BASE_URL}/datasets/manual-upload/quota/").mock(
        return_value=httpx.Response(200, json={"used": 90, "limit": 100})
    )
    presign = respx.post(f"{BASE_URL}/datasets/manual-upload/file-upload-url/").mock(
        return_value=httpx.Response(200, json={"url": "https://s3.us-east-1.amazonaws.com/u", "fields": {}})
    )

    client = Client(api_key="test-key")
    with pytest.raises(QuotaExceededError):
        client.datasets.create_from_local(source=str(tmp_path), name="N", slug="n", data_type="mcap")
    client.close()

    assert quota.called
    assert not presign.called  # nothing was uploaded


@respx.mock
def test_create_from_local_proceeds_when_quota_is_unreadable(tmp_path):
    """The meter needs the ``datasets.read`` scope. A write-only key legitimately
    lacks it, and that must not block an upload the server would accept — the
    presign's 413 stays the authority."""
    (tmp_path / "a.mcap").write_bytes(b"x" * 16)
    respx.get(f"{BASE_URL}/datasets/manual-upload/quota/").mock(
        return_value=httpx.Response(403, json={"detail": "missing scope datasets.read"})
    )
    respx.post(f"{BASE_URL}/datasets/manual-upload/file-upload-url/").mock(
        return_value=httpx.Response(200, json={"url": "https://s3.us-east-1.amazonaws.com/u", "fields": {}})
    )
    respx.post("https://s3.us-east-1.amazonaws.com/u").mock(return_value=httpx.Response(204))
    respx.post(f"{BASE_URL}/datasets/manual-upload/").mock(
        return_value=httpx.Response(
            201, json={"uid": "u", "name": "N", "slug": "n", "item_count": 1, "data_type": "mcap"}
        )
    )

    client = Client(api_key="test-key")
    dataset = client.datasets.create_from_local(source=str(tmp_path), name="N", slug="n", data_type="mcap")
    client.close()

    assert dataset.uid == "u"


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
