from __future__ import annotations

import json

import httpx
import pytest
import respx
from avala import Client
from avala.importers import available_importers, import_cloud, import_dataset
from avala.importers.cloud import parse_cloud_uri

BASE_URL = "https://api.avala.ai/api/v1"
DATASETS_URL = f"{BASE_URL}/datasets/"


# ── URI parsing ──
def test_parse_s3_uri():
    assert parse_cloud_uri("s3://my-bucket/data/run01/") == ("aws_s3", "my-bucket", "data/run01/")


def test_parse_gs_and_gcs_uri():
    assert parse_cloud_uri("gs://b/p") == ("gc_storage", "b", "p")
    assert parse_cloud_uri("gcs://b/p") == ("gc_storage", "b", "p")


def test_parse_uri_no_prefix():
    assert parse_cloud_uri("s3://only-bucket") == ("aws_s3", "only-bucket", "")


def test_parse_uri_rejects_unknown_scheme():
    with pytest.raises(ValueError, match="unsupported cloud URI"):
        parse_cloud_uri("http://example.com/x")


def test_parse_uri_rejects_missing_bucket():
    with pytest.raises(ValueError, match="missing bucket"):
        parse_cloud_uri("s3:///prefix")


# ── registry / validation ──
def test_cloud_registered():
    assert "cloud" in available_importers()


def test_invalid_data_type_rejected():
    with pytest.raises(ValueError, match="invalid data_type"):
        import_cloud(Client(api_key="k"), uri="s3://b/p", name="n", slug="s", data_type="pointcloud")


def test_s3_requires_region():
    with pytest.raises(ValueError, match="region is required"):
        import_cloud(
            Client(api_key="k"),
            uri="s3://b/p",
            name="n",
            slug="s",
            data_type="image",
            access_key_id="a",
            secret_access_key="b",
        )


def test_s3_requires_credentials():
    with pytest.raises(ValueError, match="provide S3 credentials"):
        import_cloud(Client(api_key="k"), uri="s3://b/p", name="n", slug="s", data_type="image", region="us-west-2")


def test_gcs_requires_auth_json():
    with pytest.raises(ValueError, match="gcs_auth_json"):
        import_cloud(Client(api_key="k"), uri="gs://b/p", name="n", slug="s", data_type="image")


def test_keyless_role_arn_requires_organization():
    # the server resolves the cross-account external id from the org, so keyless needs one
    with pytest.raises(ValueError, match="require an owning organization"):
        import_cloud(
            Client(api_key="k"),
            uri="s3://b/p",
            name="n",
            slug="s",
            data_type="image",
            region="us-west-2",
            role_arn="arn:aws:iam::123456789012:role/AvalaRead",
        )


def test_gcs_rejects_missing_key_file():
    # a typo'd path is neither inline JSON nor an existing file -> fail locally, not send the path
    with pytest.raises(ValueError, match="neither inline JSON nor a path"):
        import_cloud(
            Client(api_key="k"),
            uri="gs://b/p",
            name="n",
            slug="s",
            data_type="image",
            gcs_auth_json="./does-not-exist.json",
        )


# ── provider_config wiring (respx captures the create body) ──
def _mock_create(data_type="image"):
    return respx.post(DATASETS_URL).mock(
        return_value=httpx.Response(
            201, json={"uid": "d1", "name": "N", "slug": "s", "data_type": data_type, "item_count": 0}
        )
    )


def _sent_provider_config(route):
    body = json.loads(route.calls.last.request.content)
    return body["provider_config"]


@respx.mock
def test_s3_access_key_provider_config():
    route = _mock_create()
    client = Client(api_key="test-key")
    ds = import_cloud(
        client,
        uri="s3://my-bucket/run01/",
        name="N",
        slug="s",
        data_type="image",
        region="us-west-2",
        access_key_id="AKIA...",
        secret_access_key="secret",
        accelerated=True,
        included_extensions=["webp", "png"],
        ignored_paths="*/tmp/*",
    )
    client.close()
    pc = _sent_provider_config(route)
    assert pc["provider"] == "aws_s3"
    assert pc["s3_bucket_name"] == "my-bucket"
    assert pc["s3_bucket_prefix"] == "run01/"
    assert pc["s3_bucket_region"] == "us-west-2"
    assert pc["s3_auth_method"] == "access_key"
    assert pc["s3_access_key_id"] == "AKIA..."
    assert pc["s3_secret_access_key"] == "secret"
    assert pc["s3_is_accelerated"] is True
    assert pc["included_extensions"] == "webp,png"
    assert pc["ignored_paths"] == "*/tmp/*"
    assert ds.uid == "d1"


@respx.mock
def test_include_extensions_csv_string_is_normalized():
    # whitespace-padded CSV from the CLI must be stripped, or the server filter misses objects
    route = _mock_create()
    client = Client(api_key="test-key")
    import_cloud(
        client,
        uri="s3://b/p",
        name="N",
        slug="s",
        data_type="image",
        region="us-east-1",
        access_key_id="a",
        secret_access_key="b",
        included_extensions="jpg, png ,webp",
    )
    client.close()
    assert _sent_provider_config(route)["included_extensions"] == "jpg,png,webp"


@respx.mock
def test_s3_iam_role_provider_config():
    route = _mock_create()
    client = Client(api_key="test-key")
    import_cloud(
        client,
        uri="s3://b/p",
        name="N",
        slug="s",
        data_type="image",
        region="eu-central-1",
        role_arn="arn:aws:iam::123456789012:role/AvalaRead",
        organization_uid="org_123",
    )
    client.close()
    pc = _sent_provider_config(route)
    assert pc["s3_auth_method"] == "iam_role"
    assert pc["s3_role_arn"] == "arn:aws:iam::123456789012:role/AvalaRead"
    assert "s3_access_key_id" not in pc  # keyless: no static creds sent


@respx.mock
def test_gcs_provider_config_inline_json():
    route = _mock_create()
    sa = '{"type": "service_account", "project_id": "x"}'
    client = Client(api_key="test-key")
    import_cloud(client, uri="gs://gbucket/pre/", name="N", slug="s", data_type="image", gcs_auth_json=sa)
    client.close()
    pc = _sent_provider_config(route)
    assert pc["provider"] == "gc_storage"
    assert pc["gc_storage_bucket_name"] == "gbucket"
    assert pc["gc_storage_prefix"] == "pre/"
    assert pc["gc_storage_auth_json_content"] == sa


@respx.mock
def test_gcs_auth_json_from_file(tmp_path):
    key = tmp_path / "sa.json"
    key.write_text('{"type": "service_account"}', encoding="utf-8")
    route = _mock_create()
    client = Client(api_key="test-key")
    import_cloud(client, uri="gs://b/p", name="N", slug="s", data_type="image", gcs_auth_json=str(key))
    client.close()
    assert _sent_provider_config(route)["gc_storage_auth_json_content"] == '{"type": "service_account"}'


@respx.mock
def test_import_dataset_dispatches_to_cloud():
    _mock_create()
    client = Client(api_key="test-key")
    ds = import_dataset(
        "cloud",
        client,
        uri="s3://b/p",
        name="N",
        slug="s",
        data_type="image",
        region="us-east-1",
        access_key_id="a",
        secret_access_key="b",
    )
    client.close()
    assert ds.uid == "d1"


@respx.mock
def test_cloud_wait_polls_until_created():
    _mock_create()
    respx.get(f"{DATASETS_URL}d1/").mock(
        return_value=httpx.Response(
            200,
            json={"uid": "d1", "name": "N", "slug": "s", "data_type": "image", "status": "created", "item_count": 7},
        )
    )
    client = Client(api_key="test-key")
    ds = import_cloud(
        client,
        uri="s3://b/p",
        name="N",
        slug="s",
        data_type="image",
        region="us-east-1",
        access_key_id="a",
        secret_access_key="b",
        wait=True,
    )
    client.close()
    assert ds.item_count == 7
