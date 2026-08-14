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


# ── storage configs ──
STORAGE_CONFIG_UID = "sc-1234"
STORAGE_CONFIG_URL = f"{BASE_URL}/storage-configs/{STORAGE_CONFIG_UID}/"


def _mock_storage_config(**overrides):
    # The lookup is scoped to the target org, which means resolving its slug
    # first — see _apply_storage_config for why an unscoped fetch is unsafe.
    respx.get(f"{BASE_URL}/organizations/").mock(
        return_value=httpx.Response(
            200, json={"results": [{"uid": "org-1", "name": "Org One", "slug": "org-one"}], "next": None}
        )
    )
    payload = {
        "uid": STORAGE_CONFIG_UID,
        "name": "Partner bucket",
        "provider": "aws_s3",
        "s3_bucket_name": "partner-bucket",
        "s3_bucket_region": "ap-south-1",
        "s3_bucket_prefix": "deliveries",
        "s3_is_accelerated": False,
        "s3_auth_method": "iam_role",
        "is_verified": True,
    }
    payload.update(overrides)
    return respx.get(STORAGE_CONFIG_URL).mock(return_value=httpx.Response(200, json=payload))


@respx.mock
def test_storage_config_supplies_bucket_region_and_prefix():
    """The point of a saved config: stop retyping the bucket and region on every
    import. Retyping them is how a dataset ends up pointed at the wrong region
    and fails to index with no obvious cause."""
    _mock_storage_config()
    route = _mock_create()

    client = Client(api_key="test-key")
    import_cloud(
        client,
        name="N",
        slug="s",
        data_type="image",
        storage_config_uid=STORAGE_CONFIG_UID,
        role_arn="arn:aws:iam::1:role/r",
        organization_uid="org-1",
    )
    client.close()

    config = _sent_provider_config(route)
    assert config["s3_bucket_name"] == "partner-bucket"
    assert config["s3_bucket_region"] == "ap-south-1"
    assert config["s3_bucket_prefix"] == "deliveries"


@respx.mock
def test_storage_config_uri_selects_a_narrower_prefix():
    """One config commonly backs many datasets under different prefixes."""
    _mock_storage_config()
    route = _mock_create()

    client = Client(api_key="test-key")
    import_cloud(
        client,
        uri="s3://partner-bucket/deliveries/2026-08",
        name="N",
        slug="s",
        data_type="image",
        storage_config_uid=STORAGE_CONFIG_UID,
        role_arn="arn:aws:iam::1:role/r",
        organization_uid="org-1",
    )
    client.close()

    assert _sent_provider_config(route)["s3_bucket_prefix"] == "deliveries/2026-08"


@respx.mock
def test_storage_config_rejects_a_uri_for_a_different_bucket():
    """A URI naming another bucket is a mistake, not an override — silently
    letting it win would import from a bucket the config was never verified
    against."""
    _mock_storage_config()
    _mock_create()

    client = Client(api_key="test-key")
    with pytest.raises(ValueError, match="points at bucket"):
        import_cloud(
            client,
            uri="s3://someone-elses-bucket/data",
            name="N",
            slug="s",
            data_type="image",
            storage_config_uid=STORAGE_CONFIG_UID,
            role_arn="arn:aws:iam::1:role/r",
            organization_uid="org-1",
        )
    client.close()


@respx.mock
def test_unverified_storage_config_is_refused():
    _mock_storage_config(is_verified=False)

    client = Client(api_key="test-key")
    with pytest.raises(ValueError, match="not been verified"):
        import_cloud(
            client,
            name="N",
            slug="s",
            data_type="image",
            storage_config_uid=STORAGE_CONFIG_UID,
            role_arn="arn:aws:iam::1:role/r",
            organization_uid="org-1",
        )
    client.close()


@respx.mock
def test_storage_config_without_credentials_explains_why():
    """The server returns no credentials and no role ARN, so a config alone can
    never be enough. The error has to say that, or it reads as a bug."""
    _mock_storage_config()

    client = Client(api_key="test-key")
    with pytest.raises(ValueError, match="not credentials"):
        import_cloud(
            client,
            name="N",
            slug="s",
            data_type="image",
            storage_config_uid=STORAGE_CONFIG_UID,
            organization_uid="org-1",
        )
    client.close()


def test_cloud_import_requires_a_uri_or_a_storage_config():
    client = Client(api_key="test-key")
    with pytest.raises(ValueError, match="storage_config_uid"):
        import_cloud(client, name="N", slug="s", data_type="image")
    client.close()


@respx.mock
def test_storage_config_rejects_a_uri_outside_the_configured_prefix():
    """Same-bucket is not the boundary a config represents. A config verified
    for `deliveries/` says nothing about `finance-exports/`, and honouring the
    latter would import from a location whose access was never checked."""
    _mock_storage_config()
    _mock_create()

    client = Client(api_key="test-key")
    with pytest.raises(ValueError, match="outside storage config"):
        import_cloud(
            client,
            uri="s3://partner-bucket/finance-exports",
            name="N",
            slug="s",
            data_type="image",
            storage_config_uid=STORAGE_CONFIG_UID,
            role_arn="arn:aws:iam::1:role/r",
            organization_uid="org-1",
        )
    client.close()


@respx.mock
def test_storage_config_lookup_uses_the_retrieve_endpoint():
    """Pins the SDK against the route that actually exists.

    `StorageConfigs.get()` issues `GET /storage-configs/<uid>/`. That URL mapped
    only DELETE until this change, so every `--storage-config` import died with
    405 while the unit tests happily mocked a GET the server never served — a
    test validating a fiction. The server now maps `get: retrieve`; this asserts
    the SDK calls exactly that.
    """
    route = _mock_storage_config()
    _mock_create()

    client = Client(api_key="test-key")
    import_cloud(
        client,
        name="N",
        slug="s",
        data_type="image",
        storage_config_uid=STORAGE_CONFIG_UID,
        role_arn="arn:aws:iam::1:role/r",
        organization_uid="org-1",
    )
    client.close()

    assert route.call_count == 1
    assert route.calls[0].request.method == "GET"
    assert route.calls[0].request.url.path == f"/api/v1/storage-configs/{STORAGE_CONFIG_UID}/"


@respx.mock
def test_storage_config_uri_with_a_stray_leading_slash_is_refused():
    """`s3://bucket//deliveries/run` asks for the key prefix `/deliveries/run`,
    which is NOT inside a config rooted at `deliveries` — leading slashes are
    significant in an S3 key.

    This used to normalize the slash away and import anyway. That is a guess:
    the SDK cannot tell a typo from a literal `/deliveries/` namespace, and
    guessing wrong indexes objects the config never verified. Rejecting names
    both strings and lets the caller decide."""
    _mock_storage_config()

    client = Client(api_key="test-key")
    with pytest.raises(ValueError, match="outside storage config"):
        import_cloud(
            client,
            uri="s3://partner-bucket//deliveries/run",
            name="N",
            slug="s",
            data_type="image",
            storage_config_uid=STORAGE_CONFIG_UID,
            role_arn="arn:aws:iam::1:role/r",
            organization_uid="org-1",
        )
    client.close()


@respx.mock
def test_storage_config_lookup_is_scoped_to_the_target_organization():
    """A config is org-owned but the read model exposes no owner, so an unscoped
    fetch would let a caller in orgs A and B register a B-owned dataset over A's
    bucket. The scope is sent to the server, which enforces it."""
    respx.get(f"{BASE_URL}/organizations/").mock(
        return_value=httpx.Response(
            200, json={"results": [{"uid": "org-1", "name": "Org One", "slug": "org-one"}], "next": None}
        )
    )
    route = respx.get(STORAGE_CONFIG_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "uid": STORAGE_CONFIG_UID,
                "name": "Partner bucket",
                "provider": "aws_s3",
                "s3_bucket_name": "partner-bucket",
                "s3_bucket_region": "ap-south-1",
                "s3_bucket_prefix": "deliveries",
                "s3_is_accelerated": False,
                "is_verified": True,
            },
        )
    )
    _mock_create()

    client = Client(api_key="test-key")
    import_cloud(
        client,
        name="N",
        slug="s",
        data_type="image",
        storage_config_uid=STORAGE_CONFIG_UID,
        role_arn="arn:aws:iam::1:role/r",
        organization_uid="org-1",
    )
    client.close()

    assert route.calls[0].request.url.params["organization"] == "org-one"


@respx.mock
def test_unresolvable_organization_is_refused():
    """If the caller is not a member of the target org, there is no slug to scope
    the lookup with — proceeding unscoped is exactly the cross-tenant case."""
    respx.get(f"{BASE_URL}/organizations/").mock(return_value=httpx.Response(200, json={"results": [], "next": None}))

    client = Client(api_key="test-key")
    with pytest.raises(ValueError, match="could not resolve organization"):
        import_cloud(
            client,
            name="N",
            slug="s",
            data_type="image",
            storage_config_uid=STORAGE_CONFIG_UID,
            role_arn="arn:aws:iam::1:role/r",
            organization_uid="org-not-mine",
        )
    client.close()


@respx.mock
def test_storage_config_uri_preserves_a_trailing_delimiter():
    """S3's `Prefix` filter is lexical: `deliveries/run` also matches
    `deliveries/run-old/…`. Stripping a meaningful trailing slash silently
    widens the directory the customer selected."""
    _mock_storage_config()
    route = _mock_create()

    client = Client(api_key="test-key")
    import_cloud(
        client,
        uri="s3://partner-bucket/deliveries/run/",
        name="N",
        slug="s",
        data_type="image",
        storage_config_uid=STORAGE_CONFIG_UID,
        role_arn="arn:aws:iam::1:role/r",
        organization_uid="org-1",
    )
    client.close()

    assert _sent_provider_config(route)["s3_bucket_prefix"] == "deliveries/run/"


@respx.mock
def test_a_leading_slash_config_keeps_its_namespace():
    """`/finance/x` and `finance/x` are different S3 objects, so a config rooted
    at `/finance/` must stay in that namespace when a URI narrows it.

    Normalizing the slash away (which the first fix for the double-slash case
    did) is silently destructive: the boundary check compares stripped values, so
    it passes, and the dataset then registers against plain `finance/` — omitting
    every object the customer meant and indexing unrelated ones that happen to
    sort nearby."""
    _mock_storage_config(s3_bucket_prefix="/finance/")
    route = _mock_create()

    client = Client(api_key="test-key")
    import_cloud(
        client,
        uri="s3://partner-bucket//finance/run",
        name="N",
        slug="s",
        data_type="image",
        storage_config_uid=STORAGE_CONFIG_UID,
        role_arn="arn:aws:iam::1:role/r",
        organization_uid="org-1",
    )
    client.close()

    assert _sent_provider_config(route)["s3_bucket_prefix"] == "/finance/run"


@respx.mock
def test_a_slash_only_config_still_pins_its_namespace():
    """A config rooted at `/` pins the leading-slash namespace, so a URI of
    `s3://bucket/finance` — key prefix `finance`, no leading slash — is outside
    it and must be refused rather than quietly admitted.

    `/` strips to `""`, so any comparison that strips first sees no constraint
    at all and lets everything through."""
    _mock_storage_config(s3_bucket_prefix="/")

    client = Client(api_key="test-key")
    with pytest.raises(ValueError, match="outside storage config"):
        import_cloud(
            client,
            uri="s3://partner-bucket/finance",
            name="N",
            slug="s",
            data_type="image",
            storage_config_uid=STORAGE_CONFIG_UID,
            role_arn="arn:aws:iam::1:role/r",
            organization_uid="org-1",
        )
    client.close()


@respx.mock
def test_repeated_delimiters_in_a_config_prefix_are_significant():
    """`finance//` and `finance/` are different S3 namespaces. A URI of
    `finance/run` is inside the latter but not the former, and collapsing the
    repeat would let it register outside the verified prefix."""
    _mock_storage_config(s3_bucket_prefix="finance//")

    client = Client(api_key="test-key")
    with pytest.raises(ValueError, match="outside storage config"):
        import_cloud(
            client,
            uri="s3://partner-bucket/finance/run",
            name="N",
            slug="s",
            data_type="image",
            storage_config_uid=STORAGE_CONFIG_UID,
            role_arn="arn:aws:iam::1:role/r",
            organization_uid="org-1",
        )
    client.close()


@respx.mock
def test_storage_config_with_organization_id_is_refused():
    """The config lookup is scoped by slug and there is no id->slug route, so an
    id-only caller would fall through to an unscoped fetch — the same
    cross-organization exposure, reached by the other parameter."""
    client = Client(api_key="test-key")
    with pytest.raises(ValueError, match="requires organization_uid"):
        import_cloud(
            client,
            name="N",
            slug="s",
            data_type="image",
            storage_config_uid=STORAGE_CONFIG_UID,
            access_key_id="a",
            secret_access_key="b",
            organization_id=7,
        )
    client.close()


def test_gcs_auth_error_never_echoes_the_credential():
    """The value is a service-account private key or a path, and this error
    reaches terminal and CI logs. A BOM-prefixed or truncated blob fails the
    inline-JSON test and must not be printed back."""
    from avala.importers.cloud import _read_gcs_auth

    secret = '﻿{"type":"service_account","private_key":"-----BEGIN PRIVATE KEY-----SUPERSECRET"}'
    with pytest.raises(ValueError) as excinfo:
        _read_gcs_auth(secret)

    assert "SUPERSECRET" not in str(excinfo.value)
    assert "private_key" not in str(excinfo.value)


@respx.mock
def test_uri_restating_the_config_prefix_keeps_its_delimiter():
    """`finance` is not inside `finance/` — it is strictly BROADER, since a
    lexical S3 prefix of `finance` also matches `finance-old/…`. So a URI that
    drops the config's trailing delimiter widens past what was verified and is
    refused.

    This previously silently upgraded `finance` to `finance/`. The outcome was
    safe, but it was still the SDK rewriting the customer's prefix on a guess,
    and the same rewriting logic is what let three other delimiter shapes
    through. Rejecting is one rule instead of a table of special cases."""
    _mock_storage_config(s3_bucket_prefix="finance/")

    client = Client(api_key="test-key")
    with pytest.raises(ValueError, match="outside storage config"):
        import_cloud(
            client,
            uri="s3://partner-bucket/finance",
            name="N",
            slug="s",
            data_type="image",
            storage_config_uid=STORAGE_CONFIG_UID,
            role_arn="arn:aws:iam::1:role/r",
            organization_uid="org-1",
        )
    client.close()


@respx.mock
def test_a_saved_config_requires_the_target_organization():
    """Storage configs are always organization-owned — the server resolves one
    unconditionally on create and 403s a caller with no membership, so a
    personal config does not exist.

    Omitting `organization_uid` therefore did two wrong things at once: the
    config lookup went unscoped across every membership, and `datasets.create`
    received no organization, quietly producing a personally-owned dataset over
    an organization's bucket — invisible to the colleagues it was for, and
    still owned by the user after they leave.

    Note the pre-existing keyless guard covers only the `role_arn` case; a saved
    config that authenticates by access key reached neither check."""
    client = Client(api_key="test-key")
    with pytest.raises(ValueError, match="organization_uid is required"):
        import_cloud(
            client,
            uri="s3://partner-bucket/deliveries/run",
            name="N",
            slug="s",
            data_type="image",
            storage_config_uid=STORAGE_CONFIG_UID,
        )
    client.close()


@respx.mock
def test_a_413_without_quota_fields_is_not_a_quota_error():
    """A reverse proxy — or any endpoint that simply got too large a request —
    can return 413. Reporting that as QuotaExceededError makes callers tell the
    user to free storage or ask for a bigger cap, which is misdirection."""
    from avala.errors import AvalaError, QuotaExceededError

    respx.get(f"{BASE_URL}/datasets/").mock(return_value=httpx.Response(413, text="Payload Too Large"))

    client = Client(api_key="test-key")
    with pytest.raises(AvalaError) as excinfo:
        client.datasets.list()
    client.close()

    assert not isinstance(excinfo.value, QuotaExceededError)
    assert excinfo.value.status_code == 413


@respx.mock
def test_a_quota_shaped_413_still_raises_quota_exceeded():
    from avala.errors import QuotaExceededError

    respx.get(f"{BASE_URL}/datasets/").mock(
        return_value=httpx.Response(413, json={"detail": "over cap", "limit": 100, "used": 120})
    )

    client = Client(api_key="test-key")
    with pytest.raises(QuotaExceededError) as excinfo:
        client.datasets.list()
    client.close()

    assert excinfo.value.limit == 100 and excinfo.value.used == 120
