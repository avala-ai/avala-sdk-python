import httpx
import respx

from avala import Client


@respx.mock
def test_list_storage_configs():
    respx.get("https://server.avala.ai/api/v1/storage-configs/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "uid": "550e8400-e29b-41d4-a716-446655440000",
                        "name": "My S3 Bucket",
                        "provider": "aws_s3",
                        "s3_bucket_name": "my-bucket",
                        "s3_bucket_region": "us-east-1",
                        "s3_bucket_prefix": "",
                        "s3_is_accelerated": False,
                        "gc_storage_bucket_name": "",
                        "gc_storage_prefix": "",
                        "is_verified": True,
                        "last_verified_at": "2026-01-01T00:00:00Z",
                        "created_at": "2026-01-01T00:00:00Z",
                        "updated_at": "2026-01-02T00:00:00Z",
                    }
                ],
                "next": None,
                "previous": None,
            },
        )
    )
    client = Client(api_key="test-key")
    page = client.storage_configs.list()
    assert len(page.items) == 1
    assert page.items[0].name == "My S3 Bucket"
    assert page.items[0].provider == "aws_s3"
    assert page.items[0].is_verified is True
    assert page.has_more is False
    client.close()


@respx.mock
def test_create_storage_config():
    respx.post("https://server.avala.ai/api/v1/storage-configs/").mock(
        return_value=httpx.Response(
            201,
            json={
                "uid": "550e8400-e29b-41d4-a716-446655440000",
                "name": "My S3 Bucket",
                "provider": "aws_s3",
                "s3_bucket_name": "my-bucket",
                "s3_bucket_region": "us-east-1",
                "s3_bucket_prefix": "",
                "s3_is_accelerated": False,
                "gc_storage_bucket_name": "",
                "gc_storage_prefix": "",
                "is_verified": True,
                "last_verified_at": "2026-01-01T00:00:00Z",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            },
        )
    )
    client = Client(api_key="test-key")
    config = client.storage_configs.create(
        name="My S3 Bucket",
        provider="aws_s3",
        s3_bucket_name="my-bucket",
        s3_bucket_region="us-east-1",
        s3_access_key_id="AKIAEXAMPLE",
        s3_secret_access_key="secret",
    )
    assert config.name == "My S3 Bucket"
    assert config.provider == "aws_s3"
    assert config.uid == "550e8400-e29b-41d4-a716-446655440000"
    client.close()


@respx.mock
def test_test_storage_config():
    uid = "550e8400-e29b-41d4-a716-446655440000"
    respx.post(f"https://server.avala.ai/api/v1/storage-configs/{uid}/test/").mock(
        return_value=httpx.Response(200, json={"verified": True})
    )
    client = Client(api_key="test-key")
    result = client.storage_configs.test(uid)
    assert result["verified"] is True
    client.close()


@respx.mock
def test_delete_storage_config():
    uid = "550e8400-e29b-41d4-a716-446655440000"
    respx.delete(f"https://server.avala.ai/api/v1/storage-configs/{uid}/").mock(
        return_value=httpx.Response(204)
    )
    client = Client(api_key="test-key")
    client.storage_configs.delete(uid)
    client.close()


@respx.mock
def test_rate_limit_headers():
    respx.get("https://server.avala.ai/api/v1/storage-configs/").mock(
        return_value=httpx.Response(
            200,
            headers={
                "X-RateLimit-Limit": "100",
                "X-RateLimit-Remaining": "99",
                "X-RateLimit-Reset": "1700000000",
            },
            json={"results": [], "next": None, "previous": None},
        )
    )
    client = Client(api_key="test-key")
    client.storage_configs.list()
    info = client.rate_limit_info
    assert info["limit"] == "100"
    assert info["remaining"] == "99"
    assert info["reset"] == "1700000000"
    client.close()
