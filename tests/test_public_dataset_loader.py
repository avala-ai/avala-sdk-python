"""Contract and security tests for the credential-free public dataset loader."""

from __future__ import annotations

import asyncio
import copy
import gzip
import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

import httpx
import pytest
import respx

import avala
import avala._async_resolver as async_resolver_module
import avala._resolver as resolver_module
import avala.datasets as datasets_module
from avala._resolver import is_provider_hostname_allowed, join_api_path, parse_dataset_reference
from avala.errors import (
    DatasetDownloadError,
    DatasetIntegrityError,
    DatasetReferenceError,
    DatasetResolverError,
    DatasetRevisionWithdrawnError,
    MutableDatasetAliasWarning,
    UnsupportedDatasetModeError,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = json.loads((REPOSITORY_ROOT / "contracts/dataset_resolver_v1_fixtures.json").read_text())
BASE_URL = "https://api.avala.ai/api/v1"
RESOLVE_PATH = "/resolve/acme/navigation@main/"
RESOLVE_URL = f"{BASE_URL}{RESOLVE_PATH}"
OBJECTS_PATH = FIXTURES["resolve_response"]["body"]["manifest"]["objects_path"]
OBJECTS_URL = f"{BASE_URL}{OBJECTS_PATH}"
ACCESS_PATH = FIXTURES["object_page_response"]["body"]["results"][0]["access_path"]
ACCESS_URL = f"{BASE_URL}{ACCESS_PATH}"
DOWNLOAD_URL = FIXTURES["access_grant_response"]["body"]["url"]
FIXTURE_SERVER_DATE = datetime(2026, 8, 23, 18, 0, tzinfo=timezone.utc)
MAX_JSON_SAFE_INTEGER = 9_007_199_254_740_991


@pytest.fixture(autouse=True)
def _freeze_grant_receipt_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(resolver_module, "_utc_now", lambda: FIXTURE_SERVER_DATE)


def _resolve_response() -> httpx.Response:
    return httpx.Response(200, json=copy.deepcopy(FIXTURES["resolve_response"]["body"]))


def _page_response(body: dict[str, Any] | None = None) -> httpx.Response:
    return httpx.Response(
        200,
        json=copy.deepcopy(body or FIXTURES["object_page_response"]["body"]),
    )


def _grant_response(body: dict[str, Any] | None = None) -> httpx.Response:
    return httpx.Response(
        200,
        headers=FIXTURES["access_grant_response"]["headers"],
        json=copy.deepcopy(body or FIXTURES["access_grant_response"]["body"]),
    )


def _manifest_object(index: int, *, role: str = "episode", size_bytes: int = 0) -> dict[str, Any]:
    document = copy.deepcopy(FIXTURES["object_page_response"]["body"]["results"][0])
    document.update(
        {
            "object_uid": f"00000000-0000-0000-0000-{index:012d}",
            "logical_path": f"{role}/{index:04d}.bin",
            "role": role,
            "ordinal": index,
            "size_bytes": size_bytes,
            "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        }
    )
    document["access_path"] = f"{OBJECTS_PATH}{document['object_uid']}/access/"
    return document


def _sdk_traceback_locals(error: BaseException) -> str:
    captured: list[str] = []
    traceback = error.__traceback__
    while traceback is not None:
        module_name = traceback.tb_frame.f_globals.get("__name__", "")
        if isinstance(module_name, str) and module_name.startswith("avala"):
            captured.append(repr(traceback.tb_frame.f_locals))
        traceback = traceback.tb_next
    return "\n".join(captured)


class _PartialSyncGrantStream(httpx.SyncByteStream):
    def __iter__(self) -> Iterator[bytes]:
        yield f'{{"url":"{DOWNLOAD_URL}"'.encode()
        raise httpx.ReadError("synthetic partial grant failure")


class _PartialAsyncGrantStream(httpx.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield f'{{"url":"{DOWNLOAD_URL}"'.encode()
        raise httpx.ReadError("synthetic partial grant failure")


class _PartialCancelledAsyncGrantStream(httpx.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield f'{{"url":"{DOWNLOAD_URL}"'.encode()
        raise asyncio.CancelledError


@pytest.mark.parametrize("vector", FIXTURES["path_join_vectors"], ids=lambda vector: vector["name"])
def test_api_root_relative_path_join_vectors(vector: dict[str, Any]) -> None:
    assert join_api_path(vector["api_root"], vector["api_relative_path"]) == vector["expected_url"]


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "/resolve/uid/%5c..%5cadmin/",
        "/resolve/uid/%255c..%255cadmin/",
    ],
)
def test_api_root_relative_path_rejects_encoded_backslashes_and_double_encoding(unsafe_path: str) -> None:
    with pytest.raises(DatasetIntegrityError, match="invalid_resolver_path"):
        join_api_path(BASE_URL, unsafe_path)


@pytest.mark.parametrize("vector", FIXTURES["provider_host_vectors"])
def test_provider_hostname_allowlist_vectors(vector: dict[str, Any]) -> None:
    assert (
        is_provider_hostname_allowed(
            vector["provider"],
            vector["hostname"],
            api_root_hostname=vector.get("api_root_hostname"),
        )
        is vector["allowed"]
    )


@pytest.mark.parametrize(
    ("provider", "hostname"),
    [
        ("aws_s3", "s3-accelerate.amazonaws.com"),
        ("aws_s3", "fixture-bucket.s3.us-gov-west-1.amazonaws.com"),
        ("aws_s3", "dotted.bucket.s3.us-west-2.amazonaws.com"),
        ("gcs", "two.labels.storage.googleapis.com"),
    ],
)
def test_provider_hostname_allowlist_rejects_unsupported_service_shapes(provider: str, hostname: str) -> None:
    assert not is_provider_hostname_allowed(provider, hostname)


def test_provider_hostname_allowlist_accepts_path_style_s3_dualstack() -> None:
    assert is_provider_hostname_allowed("aws_s3", "s3.dualstack.us-west-2.amazonaws.com")


@pytest.mark.parametrize("vector", FIXTURES["load_reference_vectors"])
def test_load_reference_vectors(vector: dict[str, Any]) -> None:
    if not vector["valid"]:
        with pytest.raises(DatasetReferenceError, match=vector["error"]):
            parse_dataset_reference(vector["reference"], vector["revision"])
        return

    parsed = parse_dataset_reference(vector["reference"], vector["revision"])
    assert parsed.requested_reference.endswith(f"@{vector['selector']}")
    assert parsed.path.startswith("/resolve/")


def test_streaming_false_fails_before_opening_a_transport() -> None:
    with pytest.raises(UnsupportedDatasetModeError, match="streaming=False"):
        avala.load("acme/navigation", streaming=False)


@respx.mock
def test_canonical_reference_uses_the_exact_uid_digest_route() -> None:
    canonical = FIXTURES["resolve_response"]["body"]["canonical_reference"]
    canonical_body = copy.deepcopy(FIXTURES["resolve_response"]["body"])
    canonical_body["requested_reference"] = canonical
    canonical_path = canonical.replace("avala://datasets/", "/resolve/uid/").replace("@", "/") + "/"
    route = respx.get(f"{BASE_URL}{canonical_path}").mock(return_value=httpx.Response(200, json=canonical_body))

    with avala.load(canonical) as dataset:
        assert dataset.canonical_reference == canonical

    assert route.call_count == 1


@respx.mock
def test_friendly_alias_warns_before_returning_the_canonical_identity() -> None:
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())

    with pytest.warns(MutableDatasetAliasWarning, match="aliases are mutable"):
        with avala.load("acme/navigation") as dataset:
            assert dataset.canonical_reference == FIXTURES["resolve_response"]["body"]["canonical_reference"]


@respx.mock
def test_sync_dataset_walk_retries_429_after_server_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(resolver_module.time, "sleep", sleeps.append)
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response())
    grant_route = respx.get(ACCESS_URL).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "7"}, json={"detail": "throttled"}),
            _grant_response(),
        ]
    )
    respx.get(DOWNLOAD_URL).mock(return_value=httpx.Response(200, content=b"test"))

    with pytest.warns(MutableDatasetAliasWarning):
        with avala.load("acme/navigation") as dataset:
            assert dataset.episodes[0].read() == b"test"

    assert grant_route.call_count == 2
    assert sleeps == [7.0]


@respx.mock
@pytest.mark.asyncio
async def test_async_dataset_walk_retries_429_after_server_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(async_resolver_module.asyncio, "sleep", record_sleep)
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response())
    grant_route = respx.get(ACCESS_URL).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "7"}, json={"detail": "throttled"}),
            _grant_response(),
        ]
    )
    respx.get(DOWNLOAD_URL).mock(return_value=httpx.Response(200, content=b"test"))

    with pytest.warns(MutableDatasetAliasWarning):
        async with await avala.async_load("acme/navigation") as dataset:
            episode = await dataset.episodes[0]
            assert await episode.read() == b"test"

    assert grant_route.call_count == 2
    assert sleeps == [7.0]


@respx.mock
def test_friendly_full_digest_is_exact_and_does_not_warn() -> None:
    digest = FIXTURES["resolve_response"]["body"]["revision_sha256"]
    exact_path = f"/resolve/acme/navigation@{digest}/"
    exact_body = copy.deepcopy(FIXTURES["resolve_response"]["body"])
    exact_body["requested_reference"] = f"acme/navigation@{digest}"
    respx.get(f"{BASE_URL}{exact_path}").mock(return_value=httpx.Response(200, json=exact_body))

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with avala.load(f"acme/navigation@{digest}"):
            pass

    assert not [warning for warning in captured if issubclass(warning.category, MutableDatasetAliasWarning)]


@respx.mock
def test_manifest_object_count_is_bounded_during_resolve() -> None:
    oversized = copy.deepcopy(FIXTURES["resolve_response"]["body"])
    oversized["manifest"]["object_count"] = 100_001
    respx.get(RESOLVE_URL).mock(return_value=httpx.Response(200, json=oversized))

    with pytest.raises(DatasetIntegrityError, match="invalid_resolver_response"):
        avala.load("acme/navigation")


@respx.mock
def test_manifest_total_size_is_bounded_to_json_safe_integers() -> None:
    oversized = copy.deepcopy(FIXTURES["resolve_response"]["body"])
    oversized["manifest"]["total_size_bytes"] = MAX_JSON_SAFE_INTEGER + 1
    respx.get(RESOLVE_URL).mock(return_value=httpx.Response(200, json=oversized))

    with pytest.raises(DatasetIntegrityError, match="invalid_resolver_response"):
        avala.load("acme/navigation")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", ""),
        ("name", "x" * 1025),
        ("url", ""),
        ("url", "http://example.test/insecure-terms"),
        ("url", "javascript:alert(1)"),
        ("url", "https://"),
        ("url", " https://example.test/space-wrapped-terms "),
        ("url", "https://user@example.test/credential-bearing-terms"),
        ("url", "https://user:password@example.test/credential-bearing-terms"),
    ],
)
@respx.mock
def test_resolver_rejects_malformed_rights_metadata(field: str, value: str) -> None:
    malformed = copy.deepcopy(FIXTURES["resolve_response"]["body"])
    malformed["rights"][field] = value
    respx.get(RESOLVE_URL).mock(return_value=httpx.Response(200, json=malformed))

    with pytest.raises(DatasetIntegrityError, match="invalid_resolver_response"):
        avala.load("acme/navigation")


@respx.mock
def test_resolver_rejects_undeclared_rights_metadata() -> None:
    malformed = copy.deepcopy(FIXTURES["resolve_response"]["body"])
    malformed["rights"]["download_url"] = DOWNLOAD_URL
    respx.get(RESOLVE_URL).mock(return_value=httpx.Response(200, json=malformed))

    with pytest.raises(DatasetIntegrityError, match="invalid_resolver_response"):
        avala.load("acme/navigation")


@respx.mock
def test_origin_names_keep_their_distinct_canonical_grammar() -> None:
    resolved = copy.deepcopy(FIXTURES["resolve_response"]["body"])
    resolved["manifest"]["origins"][0]["name"] = "primary.origin-1"
    page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    page["results"][0]["origin"] = "primary.origin-1"
    respx.get(RESOLVE_URL).mock(return_value=httpx.Response(200, json=resolved))
    respx.get(OBJECTS_URL).mock(return_value=_page_response(page))

    with avala.load("acme/navigation") as dataset:
        assert dataset.episodes[0].origin == "primary.origin-1"


@respx.mock
def test_manifest_roles_use_the_v1_role_grammar() -> None:
    invalid_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    invalid_page["results"][0]["role"] = "episode.v2"
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response(invalid_page))

    with avala.load("acme/navigation") as dataset:
        with pytest.raises(ValueError, match="canonical manifest role"):
            dataset.for_role("episode.v2")
        with pytest.raises(DatasetIntegrityError, match="invalid_object_page"):
            _ = dataset.episodes[0]


@respx.mock
@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../escape",
        "/absolute",
        "episodes/",
        "episodes//0001.mcap",
        "episodes/./0001.mcap",
        "episodes/../secret.mcap",
        "episodes\\0001.mcap",
        "episodes/%2F/0001.mcap",
        "metadata/e\u0301.json",
        "metadata/\x00.json",
        "x" * 1025,
    ],
)
def test_manifest_logical_paths_must_be_canonical_posix_relative(unsafe_path: str) -> None:
    invalid_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    invalid_page["results"][0]["logical_path"] = unsafe_path
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response(invalid_page))

    with avala.load("acme/navigation") as dataset:
        with pytest.raises(DatasetIntegrityError, match="invalid_object_page"):
            _ = dataset.episodes[0]


@respx.mock
@pytest.mark.parametrize("field", ["ordinal", "episode_ordinal", "size_bytes"])
def test_manifest_object_integers_are_bounded_to_json_safe_values(field: str) -> None:
    invalid_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    invalid_page["results"][0][field] = MAX_JSON_SAFE_INTEGER + 1
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response(invalid_page))

    with avala.load("acme/navigation") as dataset:
        with pytest.raises(DatasetIntegrityError, match="invalid_object_page"):
            _ = dataset.episodes[0]


@respx.mock
def test_episode_objects_cannot_reference_an_episode_ordinal() -> None:
    invalid_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    invalid_page["results"][0]["episode_ordinal"] = 0
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response(invalid_page))

    with avala.load("acme/navigation") as dataset:
        with pytest.raises(DatasetIntegrityError, match="invalid_object_page"):
            _ = dataset.episodes[0]


@respx.mock
def test_manifest_objects_require_an_explicit_nullable_episode_ordinal() -> None:
    invalid_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    invalid_page["results"][0].pop("episode_ordinal")
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response(invalid_page))

    with avala.load("acme/navigation") as dataset:
        with pytest.raises(DatasetIntegrityError, match="invalid_object_page"):
            _ = dataset.episodes[0]


@respx.mock
def test_manifest_object_pages_cannot_exceed_the_server_page_limit() -> None:
    oversized_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    oversized_page["results"] = [_manifest_object(index) for index in range(101)]
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response(oversized_page))

    with avala.load("acme/navigation") as dataset:
        with pytest.raises(DatasetIntegrityError, match="invalid_object_page"):
            list(dataset.objects)


@respx.mock
def test_sync_loader_resolves_lazily_and_downloads_verified_raw_bytes_without_credentials() -> None:
    resolve_route = respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    page_route = respx.get(OBJECTS_URL).mock(return_value=_page_response())
    grant_route = respx.get(ACCESS_URL).mock(return_value=_grant_response())
    download_route = respx.get(DOWNLOAD_URL).mock(return_value=httpx.Response(200, content=b"test"))

    with avala.load("acme/navigation") as dataset:
        assert dataset.canonical_reference == FIXTURES["resolve_response"]["body"]["canonical_reference"]
        assert dataset.manifest_sha256 == FIXTURES["resolve_response"]["body"]["manifest"]["sha256"]
        assert dataset.object_count == 2
        assert resolve_route.call_count == 1
        assert page_route.call_count == 0
        assert grant_route.call_count == 0

        episode = dataset.episodes[0]
        assert episode.logical_path == "episodes/0001.mcap"
        assert page_route.calls.last.request.url.params["role"] == "episode"
        assert page_route.calls.last.request.url.params["ordinal"] == "0"
        assert episode.read() == b"test"

        assert DOWNLOAD_URL not in repr(dataset)
        assert DOWNLOAD_URL not in repr(episode)

    for route in (resolve_route, page_route, grant_route):
        request = route.calls.last.request
        assert "X-Avala-Api-Key" not in request.headers
        assert "Authorization" not in request.headers
        assert "Cookie" not in request.headers
    download_request = download_route.calls.last.request
    assert download_request.headers["Accept-Encoding"] == "identity"
    assert "X-Avala-Api-Key" not in download_request.headers
    assert "Authorization" not in download_request.headers
    assert "Cookie" not in download_request.headers


@respx.mock
def test_gcs_download_requests_raw_content_encoded_bytes() -> None:
    raw_gzip = gzip.compress(b"test", mtime=0)
    raw_sha256 = hashlib.sha256(raw_gzip).hexdigest()
    resolved = copy.deepcopy(FIXTURES["resolve_response"]["body"])
    resolved["manifest"]["total_size_bytes"] = len(raw_gzip)
    page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    page["results"][0]["size_bytes"] = len(raw_gzip)
    page["results"][0]["sha256"] = raw_sha256
    grant = copy.deepcopy(FIXTURES["access_grant_response"]["body"])
    grant.update(
        {
            "provider": "gcs",
            "url": "https://storage.googleapis.com/fixture-bucket/episodes/0001.mcap",
            "size_bytes": len(raw_gzip),
            "sha256": raw_sha256,
        }
    )
    respx.get(RESOLVE_URL).mock(return_value=httpx.Response(200, json=resolved))
    respx.get(OBJECTS_URL).mock(return_value=_page_response(page))
    respx.get(ACCESS_URL).mock(return_value=_grant_response(grant))
    download_route = respx.get(grant["url"]).mock(
        return_value=httpx.Response(200, headers={"Content-Encoding": "gzip"}, content=raw_gzip)
    )

    with pytest.warns(MutableDatasetAliasWarning):
        with avala.load("acme/navigation") as dataset:
            assert dataset.episodes[0].read() == raw_gzip

    assert download_route.calls.last.request.headers["Accept-Encoding"] == "gzip"


@respx.mock
def test_verified_open_rolls_large_content_out_of_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(resolver_module, "_DOWNLOAD_SPOOL_MEMORY_BYTES", 1)
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response())
    respx.get(ACCESS_URL).mock(return_value=_grant_response())
    respx.get(DOWNLOAD_URL).mock(return_value=httpx.Response(200, content=b"test"))

    with pytest.warns(MutableDatasetAliasWarning):
        with avala.load("acme/navigation") as dataset:
            episode = dataset.episodes[0]
            with episode.open() as content:
                assert getattr(content, "_rolled", False) is True
                assert content.read() == b"test"


@respx.mock
def test_path_style_s3_dualstack_grant_downloads() -> None:
    grant = copy.deepcopy(FIXTURES["access_grant_response"]["body"])
    grant["url"] = "https://s3.dualstack.us-west-2.amazonaws.com/fixture.bucket/episodes/0001.mcap?fixture=1"
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response())
    respx.get(ACCESS_URL).mock(return_value=_grant_response(grant))
    download_route = respx.get(grant["url"]).mock(return_value=httpx.Response(200, content=b"test"))

    with avala.load("acme/navigation") as dataset:
        assert dataset.episodes[0].read() == b"test"

    assert download_route.call_count == 1


@respx.mock
def test_sync_spool_is_allocated_before_requesting_a_signed_grant(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_spool_allocation() -> Any:
        raise MemoryError("synthetic spool allocation failure")

    monkeypatch.setattr(datasets_module, "_new_download_spool", fail_spool_allocation)
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response())
    grant_route = respx.get(ACCESS_URL).mock(return_value=_grant_response())

    with avala.load("acme/navigation") as dataset:
        with pytest.raises(MemoryError, match="spool allocation"):
            dataset.episodes[0].open()

    assert grant_route.call_count == 0


@respx.mock
@pytest.mark.asyncio
async def test_async_spool_is_allocated_before_requesting_a_signed_grant(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_spool_allocation() -> Any:
        raise MemoryError("synthetic spool allocation failure")

    monkeypatch.setattr(datasets_module, "_new_async_download_spool", fail_spool_allocation)
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response())
    grant_route = respx.get(ACCESS_URL).mock(return_value=_grant_response())

    async with await avala.async_load("acme/navigation") as dataset:
        episode = await dataset.episodes[0]
        with pytest.raises(MemoryError, match="spool allocation"):
            await episode.open()

    assert grant_route.call_count == 0


@respx.mock
def test_read_rejects_objects_above_the_default_memory_budget_before_granting_access() -> None:
    oversized_bytes = 64 * 1024 * 1024 + 1
    resolved = copy.deepcopy(FIXTURES["resolve_response"]["body"])
    resolved["manifest"]["total_size_bytes"] = oversized_bytes
    page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    page["results"][0]["size_bytes"] = oversized_bytes
    respx.get(RESOLVE_URL).mock(return_value=httpx.Response(200, json=resolved))
    respx.get(OBJECTS_URL).mock(return_value=_page_response(page))
    grant_route = respx.get(ACCESS_URL).mock(return_value=_grant_response())

    with pytest.warns(MutableDatasetAliasWarning):
        with avala.load("acme/navigation") as dataset:
            with pytest.raises(UnsupportedDatasetModeError, match=r"consume open\(\) instead"):
                dataset.episodes[0].read()

    assert grant_route.call_count == 0


@respx.mock
@pytest.mark.asyncio
async def test_async_loader_supports_awaitable_ordinal_lookup_and_verified_read() -> None:
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response())
    respx.get(ACCESS_URL).mock(return_value=_grant_response())
    download_route = respx.get(DOWNLOAD_URL).mock(return_value=httpx.Response(200, content=b"test"))

    async with await avala.async_load("acme/navigation") as dataset:
        episode = await dataset.episodes[0]
        assert episode.object_uid == FIXTURES["object_page_response"]["body"]["results"][0]["object_uid"]
        assert await episode.read() == b"test"
        assert DOWNLOAD_URL not in repr(dataset)
        assert DOWNLOAD_URL not in repr(episode)

    assert download_route.calls.last.request.headers["Accept-Encoding"] == "identity"
    assert "Authorization" not in download_route.calls.last.request.headers


@respx.mock
@pytest.mark.asyncio
async def test_async_gcs_download_requests_raw_content_encoded_bytes() -> None:
    raw_gzip = gzip.compress(b"test", mtime=0)
    raw_sha256 = hashlib.sha256(raw_gzip).hexdigest()
    resolved = copy.deepcopy(FIXTURES["resolve_response"]["body"])
    resolved["manifest"]["total_size_bytes"] = len(raw_gzip)
    page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    page["results"][0]["size_bytes"] = len(raw_gzip)
    page["results"][0]["sha256"] = raw_sha256
    grant = copy.deepcopy(FIXTURES["access_grant_response"]["body"])
    grant.update(
        {
            "provider": "gcs",
            "url": "https://storage.googleapis.com/fixture-bucket/episodes/0001.mcap",
            "size_bytes": len(raw_gzip),
            "sha256": raw_sha256,
        }
    )
    respx.get(RESOLVE_URL).mock(return_value=httpx.Response(200, json=resolved))
    respx.get(OBJECTS_URL).mock(return_value=_page_response(page))
    respx.get(ACCESS_URL).mock(return_value=_grant_response(grant))
    download_route = respx.get(grant["url"]).mock(
        return_value=httpx.Response(200, headers={"Content-Encoding": "gzip"}, content=raw_gzip)
    )

    with pytest.warns(MutableDatasetAliasWarning):
        async with await avala.async_load("acme/navigation") as dataset:
            episode = await dataset.episodes[0]
            content = await episode.open()
            with content:
                assert content.read() == raw_gzip

    assert download_route.calls.last.request.headers["Accept-Encoding"] == "gzip"


@respx.mock
@pytest.mark.asyncio
async def test_async_disk_backed_spool_writes_run_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(async_resolver_module, "_DOWNLOAD_SPOOL_MEMORY_BYTES", 1)
    thread_calls: list[str] = []

    async def run_in_thread(function: Any, *args: Any, **kwargs: Any) -> Any:
        thread_calls.append(function.__name__)
        return function(*args, **kwargs)

    monkeypatch.setattr(async_resolver_module.asyncio, "to_thread", run_in_thread)
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response())
    respx.get(ACCESS_URL).mock(return_value=_grant_response())
    respx.get(DOWNLOAD_URL).mock(return_value=httpx.Response(200, content=b"test"))

    async with await avala.async_load("acme/navigation") as dataset:
        episode = await dataset.episodes[0]
        content = await episode.open()
        with content:
            assert getattr(content, "_rolled", False) is True
            assert content.read() == b"test"

    assert thread_calls == ["write"]


@respx.mock
@pytest.mark.asyncio
async def test_async_disk_backed_read_runs_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(async_resolver_module, "_DOWNLOAD_SPOOL_MEMORY_BYTES", 1)
    thread_calls: list[str] = []

    async def run_in_thread(function: Any, *args: Any, **kwargs: Any) -> Any:
        thread_calls.append(function.__name__)
        return function(*args, **kwargs)

    monkeypatch.setattr(async_resolver_module.asyncio, "to_thread", run_in_thread)
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response())
    respx.get(ACCESS_URL).mock(return_value=_grant_response())
    respx.get(DOWNLOAD_URL).mock(return_value=httpx.Response(200, content=b"test"))

    async with await avala.async_load("acme/navigation") as dataset:
        episode = await dataset.episodes[0]
        assert await episode.read() == b"test"

    assert thread_calls == ["write", "read"]


@respx.mock
@pytest.mark.asyncio
async def test_async_httpx_download_error_has_no_signed_url_context_or_traceback_local() -> None:
    def fail_download(request: httpx.Request) -> None:
        raise httpx.ConnectError("synthetic async provider failure", request=request)

    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response())
    respx.get(ACCESS_URL).mock(return_value=_grant_response())
    respx.get(DOWNLOAD_URL).mock(side_effect=fail_download)

    async with await avala.async_load("acme/navigation") as dataset:
        episode = await dataset.episodes[0]
        with pytest.raises(DatasetDownloadError) as exc_info:
            await episode.read()

    assert exc_info.value.__context__ is None
    traceback_locals = _sdk_traceback_locals(exc_info.value)
    assert DOWNLOAD_URL not in traceback_locals
    assert "DatasetAccessGrant(" not in traceback_locals


@respx.mock
@pytest.mark.asyncio
async def test_async_download_cancellation_has_no_signed_url_context_or_traceback_local() -> None:
    def cancel_download(request: httpx.Request) -> None:
        del request
        raise asyncio.CancelledError

    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response())
    respx.get(ACCESS_URL).mock(return_value=_grant_response())
    respx.get(DOWNLOAD_URL).mock(side_effect=cancel_download)

    async with await avala.async_load("acme/navigation") as dataset:
        episode = await dataset.episodes[0]
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await episode.read()

    assert exc_info.value.__context__ is None
    traceback_locals = _sdk_traceback_locals(exc_info.value)
    assert DOWNLOAD_URL not in traceback_locals
    assert "DatasetAccessGrant(" not in traceback_locals


@respx.mock
def test_object_iteration_uses_opaque_cursors_and_checks_manifest_count() -> None:
    first_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    first_page["next_cursor"] = "signed-opaque-cursor"
    second_object = copy.deepcopy(first_page["results"][0])
    second_object.update(
        {
            "object_uid": "9f56110b-f3cc-4264-a8e8-7a2a85253166",
            "logical_path": "metadata/dataset.json",
            "role": "metadata",
            "size_bytes": 0,
            "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        }
    )
    second_object["access_path"] = f"{OBJECTS_PATH}{second_object['object_uid']}/access/"
    second_page = copy.deepcopy(first_page)
    second_page["next_cursor"] = None
    second_page["previous_cursor"] = "signed-previous-cursor"
    second_page["results"] = [second_object]

    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    page_route = respx.get(OBJECTS_URL).mock(side_effect=[_page_response(first_page), _page_response(second_page)])

    with avala.load("acme/navigation") as dataset:
        assert [item.role for item in dataset.objects] == ["episode", "metadata"]

    assert page_route.call_count == 2
    assert page_route.calls[1].request.url.params["cursor"] == "signed-opaque-cursor"


@respx.mock
def test_filtered_iteration_rejects_a_backward_cursor_on_the_initial_page() -> None:
    truncated_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    truncated_page["previous_cursor"] = "unexpected-backward-cursor"
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response(truncated_page))

    with avala.load("acme/navigation") as dataset:
        with pytest.raises(DatasetIntegrityError, match="object_page_identity_mismatch"):
            list(dataset.episodes)


@respx.mock
@pytest.mark.asyncio
async def test_async_filtered_iteration_rejects_a_backward_cursor_on_the_initial_page() -> None:
    truncated_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    truncated_page["previous_cursor"] = "unexpected-backward-cursor"
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response(truncated_page))

    async with await avala.async_load("acme/navigation") as dataset:
        with pytest.raises(DatasetIntegrityError, match="object_page_identity_mismatch"):
            _ = [item async for item in dataset.episodes]


@respx.mock
@pytest.mark.parametrize("invalid_continuation", ["empty", "missing_previous"])
def test_filtered_continuation_pages_must_prove_their_position(invalid_continuation: str) -> None:
    first_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    first_page["next_cursor"] = "signed-next-cursor"
    second_page = copy.deepcopy(first_page)
    second_page["next_cursor"] = None
    second_page["previous_cursor"] = "signed-previous-cursor"
    second_page["results"] = [_manifest_object(1)]
    if invalid_continuation == "empty":
        second_page["results"] = []
    else:
        second_page["previous_cursor"] = None
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(side_effect=[_page_response(first_page), _page_response(second_page)])

    with avala.load("acme/navigation") as dataset:
        with pytest.raises(DatasetIntegrityError, match="object_page_identity_mismatch"):
            list(dataset.episodes)


@respx.mock
@pytest.mark.asyncio
async def test_async_continuation_page_requires_previous_cursor() -> None:
    first_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    first_page["next_cursor"] = "signed-next-cursor"
    second_page = copy.deepcopy(first_page)
    second_page["next_cursor"] = None
    second_page["previous_cursor"] = None
    second_page["results"] = [_manifest_object(1)]
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(side_effect=[_page_response(first_page), _page_response(second_page)])

    async with await avala.async_load("acme/navigation") as dataset:
        with pytest.raises(DatasetIntegrityError, match="object_page_identity_mismatch"):
            _ = [item async for item in dataset.episodes]


@respx.mock
def test_filtered_object_iteration_cannot_exceed_manifest_count() -> None:
    oversized_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    oversized_page["results"] = [_manifest_object(index) for index in range(3)]
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response(oversized_page))

    with avala.load("acme/navigation") as dataset:
        with pytest.raises(DatasetIntegrityError, match="manifest_object_count_mismatch"):
            list(dataset.episodes)


@respx.mock
@pytest.mark.asyncio
async def test_async_filtered_object_iteration_cannot_exceed_manifest_count() -> None:
    oversized_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    oversized_page["results"] = [_manifest_object(index) for index in range(3)]
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response(oversized_page))

    async with await avala.async_load("acme/navigation") as dataset:
        with pytest.raises(DatasetIntegrityError, match="manifest_object_count_mismatch"):
            _ = [item async for item in dataset.episodes]


@respx.mock
def test_complete_unfiltered_iteration_verifies_manifest_total_size() -> None:
    resolved = copy.deepcopy(FIXTURES["resolve_response"]["body"])
    resolved["manifest"]["object_count"] = 1
    resolved["manifest"]["total_size_bytes"] = 5
    respx.get(RESOLVE_URL).mock(return_value=httpx.Response(200, json=resolved))
    respx.get(OBJECTS_URL).mock(return_value=_page_response())

    with avala.load("acme/navigation") as dataset:
        with pytest.raises(DatasetIntegrityError, match="manifest_total_size_mismatch"):
            list(dataset.objects)


@respx.mock
def test_complete_unfiltered_iteration_rejects_dangling_episode_references() -> None:
    page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    metadata = _manifest_object(0, role="metadata")
    metadata["episode_ordinal"] = 999
    page["results"].append(metadata)
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response(page))

    with avala.load("acme/navigation") as dataset:
        with pytest.raises(DatasetIntegrityError, match="dangling_episode_reference"):
            list(dataset.objects)


@respx.mock
@pytest.mark.asyncio
async def test_async_complete_iteration_rejects_dangling_episode_references() -> None:
    page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    metadata = _manifest_object(0, role="metadata")
    metadata["episode_ordinal"] = 999
    page["results"].append(metadata)
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response(page))

    async with await avala.async_load("acme/navigation") as dataset:
        with pytest.raises(DatasetIntegrityError, match="dangling_episode_reference"):
            _ = [item async for item in dataset.objects]


@respx.mock
@pytest.mark.parametrize("access_mode", ["filtered", "ordinal"])
def test_role_specific_access_rejects_dangling_episode_references(access_mode: str) -> None:
    support_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    support = _manifest_object(0, role="calibration")
    support["episode_ordinal"] = 999
    support_page["results"] = [support]
    empty_episode_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    empty_episode_page["results"] = []
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    object_route = respx.get(OBJECTS_URL).mock(
        side_effect=[_page_response(support_page), _page_response(empty_episode_page)]
    )

    with avala.load("acme/navigation") as dataset:
        calibration = dataset.for_role("calibration")
        with pytest.raises(DatasetIntegrityError, match="dangling_episode_reference"):
            if access_mode == "filtered":
                list(calibration)
            else:
                calibration[0]

    assert object_route.call_count == 2


@respx.mock
@pytest.mark.parametrize("access_mode", ["filtered", "ordinal"])
@pytest.mark.asyncio
async def test_async_role_specific_access_rejects_dangling_episode_references(access_mode: str) -> None:
    support_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    support = _manifest_object(0, role="calibration")
    support["episode_ordinal"] = 999
    support_page["results"] = [support]
    empty_episode_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    empty_episode_page["results"] = []
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    object_route = respx.get(OBJECTS_URL).mock(
        side_effect=[_page_response(support_page), _page_response(empty_episode_page)]
    )

    async with await avala.async_load("acme/navigation") as dataset:
        calibration = dataset.for_role("calibration")
        with pytest.raises(DatasetIntegrityError, match="dangling_episode_reference"):
            if access_mode == "filtered":
                _ = [item async for item in calibration]
            else:
                await calibration[0]

    assert object_route.call_count == 2


@respx.mock
def test_sync_dataset_views_share_manifest_evidence_bounds() -> None:
    resolved = copy.deepcopy(FIXTURES["resolve_response"]["body"])
    resolved["manifest"]["object_count"] = 1
    calibration_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    calibration_page["results"] = [_manifest_object(0, role="calibration")]
    respx.get(RESOLVE_URL).mock(return_value=httpx.Response(200, json=resolved))
    respx.get(OBJECTS_URL).mock(side_effect=[_page_response(), _page_response(calibration_page)])

    with avala.load("acme/navigation") as dataset:
        _ = dataset.episodes[0]
        with pytest.raises(DatasetIntegrityError, match="manifest_object_count_mismatch"):
            _ = dataset.for_role("calibration")[0]


@respx.mock
@pytest.mark.asyncio
async def test_async_dataset_views_share_manifest_evidence_bounds() -> None:
    resolved = copy.deepcopy(FIXTURES["resolve_response"]["body"])
    resolved["manifest"]["object_count"] = 1
    calibration_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    calibration_page["results"] = [_manifest_object(0, role="calibration")]
    respx.get(RESOLVE_URL).mock(return_value=httpx.Response(200, json=resolved))
    respx.get(OBJECTS_URL).mock(side_effect=[_page_response(), _page_response(calibration_page)])

    async with await avala.async_load("acme/navigation") as dataset:
        _ = await dataset.episodes[0]
        with pytest.raises(DatasetIntegrityError, match="manifest_object_count_mismatch"):
            _ = await dataset.for_role("calibration")[0]


@respx.mock
@pytest.mark.parametrize(
    ("second_extension", "conflicts"),
    [
        ({"beta": 2, "alpha": 1}, False),
        ({"alpha": 1, "beta": 3}, True),
    ],
)
def test_sync_dataset_views_share_extension_evidence(
    second_extension: dict[str, int],
    conflicts: bool,
) -> None:
    first_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    first_page["results"][0]["future_extension"] = {"alpha": 1, "beta": 2}
    second_page = copy.deepcopy(first_page)
    second_page["results"][0]["future_extension"] = second_extension
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(side_effect=[_page_response(first_page), _page_response(second_page)])

    with avala.load("acme/navigation") as dataset:
        assert dataset.episodes[0].extra_fields["future_extension"] == {"alpha": 1, "beta": 2}
        if conflicts:
            with pytest.raises(DatasetIntegrityError, match="duplicate_manifest_object"):
                _ = dataset.for_role("episode")[0]
        else:
            assert dataset.for_role("episode")[0].extra_fields["future_extension"] == second_extension


@respx.mock
@pytest.mark.parametrize(
    ("second_extension", "conflicts"),
    [
        ({"beta": 2, "alpha": 1}, False),
        ({"alpha": 1, "beta": 3}, True),
    ],
)
@pytest.mark.asyncio
async def test_async_dataset_views_share_extension_evidence(
    second_extension: dict[str, int],
    conflicts: bool,
) -> None:
    first_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    first_page["results"][0]["future_extension"] = {"alpha": 1, "beta": 2}
    second_page = copy.deepcopy(first_page)
    second_page["results"][0]["future_extension"] = second_extension
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(side_effect=[_page_response(first_page), _page_response(second_page)])

    async with await avala.async_load("acme/navigation") as dataset:
        first = await dataset.episodes[0]
        assert first.extra_fields["future_extension"] == {"alpha": 1, "beta": 2}
        if conflicts:
            with pytest.raises(DatasetIntegrityError, match="duplicate_manifest_object"):
                _ = await dataset.for_role("episode")[0]
        else:
            second = await dataset.for_role("episode")[0]
            assert second.extra_fields["future_extension"] == second_extension


def test_manifest_object_extension_evidence_is_fixed_size() -> None:
    document = resolver_module.ManifestObjectDocument.model_validate(
        {
            **_manifest_object(0),
            "future_extension": {"large_value": "x" * 100_000},
        }
    )

    evidence = datasets_module._manifest_object_evidence(document)

    assert len(evidence[-1]) == 64
    assert set(evidence[-1]) <= set("0123456789abcdef")


@respx.mock
def test_sync_role_views_share_episode_reference_cache() -> None:
    resolved = copy.deepcopy(FIXTURES["resolve_response"]["body"])
    resolved["manifest"]["object_count"] = 3
    first_support_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    first_support = _manifest_object(0, role="calibration")
    first_support["episode_ordinal"] = 0
    first_support_page["results"] = [first_support]
    second_support_page = copy.deepcopy(first_support_page)
    second_support = _manifest_object(1, role="calibration")
    second_support["episode_ordinal"] = 0
    second_support_page["results"] = [second_support]
    respx.get(RESOLVE_URL).mock(return_value=httpx.Response(200, json=resolved))
    object_route = respx.get(OBJECTS_URL).mock(
        side_effect=[_page_response(first_support_page), _page_response(), _page_response(second_support_page)]
    )

    with avala.load("acme/navigation") as dataset:
        _ = dataset.for_role("calibration")[0]
        _ = dataset.for_role("calibration")[1]

    assert object_route.call_count == 3


@respx.mock
@pytest.mark.asyncio
async def test_async_role_views_share_episode_reference_cache() -> None:
    resolved = copy.deepcopy(FIXTURES["resolve_response"]["body"])
    resolved["manifest"]["object_count"] = 3
    first_support_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    first_support = _manifest_object(0, role="calibration")
    first_support["episode_ordinal"] = 0
    first_support_page["results"] = [first_support]
    second_support_page = copy.deepcopy(first_support_page)
    second_support = _manifest_object(1, role="calibration")
    second_support["episode_ordinal"] = 0
    second_support_page["results"] = [second_support]
    respx.get(RESOLVE_URL).mock(return_value=httpx.Response(200, json=resolved))
    object_route = respx.get(OBJECTS_URL).mock(
        side_effect=[_page_response(first_support_page), _page_response(), _page_response(second_support_page)]
    )

    async with await avala.async_load("acme/navigation") as dataset:
        _ = await dataset.for_role("calibration")[0]
        _ = await dataset.for_role("calibration")[1]

    assert object_route.call_count == 3


@respx.mock
@pytest.mark.parametrize("access_mode", ["filtered", "ordinal"])
@pytest.mark.parametrize(
    ("overflow", "reason"),
    [
        ("count", "manifest_object_count_mismatch"),
        ("size", "manifest_total_size_mismatch"),
    ],
)
def test_role_specific_access_counts_referenced_episodes_against_manifest_bounds(
    access_mode: str,
    overflow: str,
    reason: str,
) -> None:
    resolved = copy.deepcopy(FIXTURES["resolve_response"]["body"])
    support_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    support = _manifest_object(0, role="calibration", size_bytes=1 if overflow == "size" else 0)
    support["episode_ordinal"] = 0
    support_page["results"] = [support]
    if overflow == "count":
        resolved["manifest"]["object_count"] = 1
    respx.get(RESOLVE_URL).mock(return_value=httpx.Response(200, json=resolved))
    object_route = respx.get(OBJECTS_URL).mock(side_effect=[_page_response(support_page), _page_response()])

    with avala.load("acme/navigation") as dataset:
        calibration = dataset.for_role("calibration")
        with pytest.raises(DatasetIntegrityError, match=reason):
            if access_mode == "filtered":
                list(calibration)
            else:
                calibration[0]

    assert object_route.call_count == 2


@respx.mock
@pytest.mark.parametrize("access_mode", ["filtered", "ordinal"])
@pytest.mark.parametrize(
    ("overflow", "reason"),
    [
        ("count", "manifest_object_count_mismatch"),
        ("size", "manifest_total_size_mismatch"),
    ],
)
@pytest.mark.asyncio
async def test_async_role_specific_access_counts_referenced_episodes_against_manifest_bounds(
    access_mode: str,
    overflow: str,
    reason: str,
) -> None:
    resolved = copy.deepcopy(FIXTURES["resolve_response"]["body"])
    support_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    support = _manifest_object(0, role="calibration", size_bytes=1 if overflow == "size" else 0)
    support["episode_ordinal"] = 0
    support_page["results"] = [support]
    if overflow == "count":
        resolved["manifest"]["object_count"] = 1
    respx.get(RESOLVE_URL).mock(return_value=httpx.Response(200, json=resolved))
    object_route = respx.get(OBJECTS_URL).mock(side_effect=[_page_response(support_page), _page_response()])

    async with await avala.async_load("acme/navigation") as dataset:
        calibration = dataset.for_role("calibration")
        with pytest.raises(DatasetIntegrityError, match=reason):
            if access_mode == "filtered":
                _ = [item async for item in calibration]
            else:
                await calibration[0]

    assert object_route.call_count == 2


@respx.mock
def test_exact_episode_batch_counts_every_result_against_collection_bounds() -> None:
    resolved = copy.deepcopy(FIXTURES["resolve_response"]["body"])
    resolved["manifest"]["object_count"] = 3
    resolved["manifest"]["total_size_bytes"] = 0
    support_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    support_objects = [_manifest_object(index + 100, role="calibration") for index in range(2)]
    for index, support in enumerate(support_objects):
        support["episode_ordinal"] = index
    support_page["results"] = support_objects
    episode_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    episode_page["results"] = [_manifest_object(index) for index in range(2)]
    respx.get(RESOLVE_URL).mock(return_value=httpx.Response(200, json=resolved))
    object_route = respx.get(OBJECTS_URL).mock(side_effect=[_page_response(support_page), _page_response(episode_page)])

    with avala.load("acme/navigation") as dataset:
        with pytest.raises(DatasetIntegrityError, match="manifest_object_count_mismatch"):
            list(dataset.for_role("calibration"))

    assert object_route.call_count == 2


@respx.mock
@pytest.mark.asyncio
async def test_async_exact_episode_batch_counts_every_result_against_collection_bounds() -> None:
    resolved = copy.deepcopy(FIXTURES["resolve_response"]["body"])
    resolved["manifest"]["object_count"] = 3
    resolved["manifest"]["total_size_bytes"] = 0
    support_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    support_objects = [_manifest_object(index + 100, role="calibration") for index in range(2)]
    for index, support in enumerate(support_objects):
        support["episode_ordinal"] = index
    support_page["results"] = support_objects
    episode_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    episode_page["results"] = [_manifest_object(index) for index in range(2)]
    respx.get(RESOLVE_URL).mock(return_value=httpx.Response(200, json=resolved))
    object_route = respx.get(OBJECTS_URL).mock(side_effect=[_page_response(support_page), _page_response(episode_page)])

    async with await avala.async_load("acme/navigation") as dataset:
        with pytest.raises(DatasetIntegrityError, match="manifest_object_count_mismatch"):
            _ = [item async for item in dataset.for_role("calibration")]

    assert object_route.call_count == 2


@respx.mock
def test_role_specific_access_counts_a_shared_episode_once() -> None:
    resolved = copy.deepcopy(FIXTURES["resolve_response"]["body"])
    resolved["manifest"]["object_count"] = 3
    support_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    support_objects = [_manifest_object(index, role="calibration") for index in range(2)]
    for support in support_objects:
        support["episode_ordinal"] = 0
    support_page["results"] = support_objects
    respx.get(RESOLVE_URL).mock(return_value=httpx.Response(200, json=resolved))
    object_route = respx.get(OBJECTS_URL).mock(side_effect=[_page_response(support_page), _page_response()])

    with avala.load("acme/navigation") as dataset:
        assert len(list(dataset.for_role("calibration"))) == 2

    assert object_route.call_count == 2


@respx.mock
@pytest.mark.asyncio
async def test_async_role_specific_access_counts_a_shared_episode_once() -> None:
    resolved = copy.deepcopy(FIXTURES["resolve_response"]["body"])
    resolved["manifest"]["object_count"] = 3
    support_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    support_objects = [_manifest_object(index, role="calibration") for index in range(2)]
    for support in support_objects:
        support["episode_ordinal"] = 0
    support_page["results"] = support_objects
    respx.get(RESOLVE_URL).mock(return_value=httpx.Response(200, json=resolved))
    object_route = respx.get(OBJECTS_URL).mock(side_effect=[_page_response(support_page), _page_response()])

    async with await avala.async_load("acme/navigation") as dataset:
        assert len([item async for item in dataset.for_role("calibration")]) == 2

    assert object_route.call_count == 2


@respx.mock
def test_role_specific_access_batches_episode_reference_validation() -> None:
    resolved = copy.deepcopy(FIXTURES["resolve_response"]["body"])
    resolved["manifest"]["object_count"] = 62
    resolved["manifest"]["total_size_bytes"] = 0
    support_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    support_objects = [_manifest_object(index + 31, role="calibration") for index in range(31)]
    for index, support in enumerate(support_objects):
        support["episode_ordinal"] = index
    support_page["results"] = support_objects
    episode_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    episode_page["results"] = [_manifest_object(index, size_bytes=0) for index in range(31)]
    respx.get(RESOLVE_URL).mock(return_value=httpx.Response(200, json=resolved))
    object_route = respx.get(OBJECTS_URL).mock(side_effect=[_page_response(support_page), _page_response(episode_page)])

    with avala.load("acme/navigation") as dataset:
        assert len(list(dataset.for_role("calibration"))) == 31

    assert object_route.call_count == 2
    assert object_route.calls[1].request.url.params["role"] == "episode"
    assert "ordinal" not in object_route.calls[1].request.url.params
    assert object_route.calls[1].request.url.params.get_list("ordinals") == [str(index) for index in range(31)]


@respx.mock
@pytest.mark.asyncio
async def test_async_role_specific_access_batches_episode_reference_validation() -> None:
    resolved = copy.deepcopy(FIXTURES["resolve_response"]["body"])
    resolved["manifest"]["object_count"] = 62
    resolved["manifest"]["total_size_bytes"] = 0
    support_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    support_objects = [_manifest_object(index + 31, role="calibration") for index in range(31)]
    for index, support in enumerate(support_objects):
        support["episode_ordinal"] = index + 2_800
    support_page["results"] = support_objects
    episode_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    episode_page["results"] = [_manifest_object(index + 2_800, size_bytes=0) for index in range(31)]
    respx.get(RESOLVE_URL).mock(return_value=httpx.Response(200, json=resolved))
    object_route = respx.get(OBJECTS_URL).mock(side_effect=[_page_response(support_page), _page_response(episode_page)])

    async with await avala.async_load("acme/navigation") as dataset:
        assert len([item async for item in dataset.for_role("calibration")]) == 31

    assert object_route.call_count == 2
    assert object_route.calls[1].request.url.params["role"] == "episode"
    assert "ordinal" not in object_route.calls[1].request.url.params
    assert object_route.calls[1].request.url.params.get_list("ordinals") == [
        str(index) for index in range(2_800, 2_831)
    ]


@respx.mock
def test_episode_reference_lookup_is_constant_request_count_for_far_ordinals() -> None:
    resolved = copy.deepcopy(FIXTURES["resolve_response"]["body"])
    resolved["manifest"]["object_count"] = 2
    resolved["manifest"]["total_size_bytes"] = 0
    support_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    support = _manifest_object(0, role="calibration")
    support["episode_ordinal"] = 2_800
    support_page["results"] = [support]
    episode_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    episode_page["results"] = [_manifest_object(2_800, size_bytes=0)]
    respx.get(RESOLVE_URL).mock(return_value=httpx.Response(200, json=resolved))
    object_route = respx.get(OBJECTS_URL).mock(
        side_effect=[
            _page_response(support_page),
            _page_response(episode_page),
        ]
    )

    with avala.load("acme/navigation") as dataset:
        assert len(list(dataset.for_role("calibration"))) == 1

    assert object_route.call_count == 2
    assert object_route.calls[1].request.url.params.get_list("ordinals") == ["2800"]


@respx.mock
def test_exact_episode_batch_rejects_unrequested_results() -> None:
    resolved = copy.deepcopy(FIXTURES["resolve_response"]["body"])
    resolved["manifest"]["object_count"] = 3
    resolved["manifest"]["total_size_bytes"] = 0
    support_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    support = _manifest_object(100, role="calibration")
    support["episode_ordinal"] = 0
    support_page["results"] = [support]
    episode_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    episode_page["results"] = [_manifest_object(0), _manifest_object(1)]
    respx.get(RESOLVE_URL).mock(return_value=httpx.Response(200, json=resolved))
    respx.get(OBJECTS_URL).mock(side_effect=[_page_response(support_page), _page_response(episode_page)])

    with avala.load("acme/navigation") as dataset:
        with pytest.raises(DatasetIntegrityError, match="object_page_identity_mismatch"):
            list(dataset.for_role("calibration"))


@respx.mock
def test_manifest_iteration_rejects_noncanonical_object_order_across_pages() -> None:
    first_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    metadata = _manifest_object(0, role="metadata")
    first_page["next_cursor"] = "signed-next-cursor"
    first_page["results"] = [metadata]
    second_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    second_page["previous_cursor"] = "signed-previous-cursor"
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(side_effect=[_page_response(first_page), _page_response(second_page)])

    with avala.load("acme/navigation") as dataset:
        with pytest.raises(DatasetIntegrityError, match="invalid_manifest_order"):
            list(dataset.objects)


@respx.mock
@pytest.mark.asyncio
async def test_async_manifest_iteration_rejects_noncanonical_object_order() -> None:
    page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    metadata = _manifest_object(0, role="metadata")
    page["results"] = [metadata, page["results"][0]]
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response(page))

    async with await avala.async_load("acme/navigation") as dataset:
        with pytest.raises(DatasetIntegrityError, match="invalid_manifest_order"):
            _ = [item async for item in dataset.objects]


@respx.mock
def test_ordinal_lookup_rejects_paginated_responses() -> None:
    paginated = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    paginated["next_cursor"] = "unexpected-cursor"
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response(paginated))

    with avala.load("acme/navigation") as dataset:
        with pytest.raises(DatasetIntegrityError, match="object_page_identity_mismatch"):
            _ = dataset.episodes[0]


@respx.mock
def test_sync_ordinal_lookup_rejects_values_above_the_json_safe_bound() -> None:
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    object_route = respx.get(OBJECTS_URL).mock(return_value=_page_response())

    with avala.load("acme/navigation") as dataset:
        with pytest.raises(IndexError, match="JSON-safe"):
            _ = dataset.episodes[MAX_JSON_SAFE_INTEGER + 1]

    assert object_route.call_count == 0


@respx.mock
@pytest.mark.asyncio
async def test_async_ordinal_lookup_rejects_values_above_the_json_safe_bound() -> None:
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    object_route = respx.get(OBJECTS_URL).mock(return_value=_page_response())

    async with await avala.async_load("acme/navigation") as dataset:
        with pytest.raises(IndexError, match="JSON-safe"):
            _ = await dataset.episodes[MAX_JSON_SAFE_INTEGER + 1]

    assert object_route.call_count == 0


@respx.mock
def test_ordinal_lookup_rejects_an_object_when_manifest_count_is_zero() -> None:
    resolved = copy.deepcopy(FIXTURES["resolve_response"]["body"])
    resolved["manifest"]["object_count"] = 0
    resolved["manifest"]["total_size_bytes"] = 0
    respx.get(RESOLVE_URL).mock(return_value=httpx.Response(200, json=resolved))
    respx.get(OBJECTS_URL).mock(return_value=_page_response())

    with avala.load("acme/navigation") as dataset:
        with pytest.raises(DatasetIntegrityError, match="manifest_object_count_mismatch"):
            _ = dataset.episodes[0]


@respx.mock
@pytest.mark.asyncio
async def test_async_ordinal_lookup_rejects_an_object_larger_than_manifest_total() -> None:
    resolved = copy.deepcopy(FIXTURES["resolve_response"]["body"])
    resolved["manifest"]["total_size_bytes"] = 3
    respx.get(RESOLVE_URL).mock(return_value=httpx.Response(200, json=resolved))
    respx.get(OBJECTS_URL).mock(return_value=_page_response())

    async with await avala.async_load("acme/navigation") as dataset:
        with pytest.raises(DatasetIntegrityError, match="manifest_total_size_mismatch"):
            _ = await dataset.episodes[0]


@respx.mock
def test_object_iteration_rejects_duplicate_manifest_identities_across_pages() -> None:
    first_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    first_page["next_cursor"] = "signed-next-cursor"
    duplicate = copy.deepcopy(first_page["results"][0])
    duplicate["object_uid"] = "00000000-0000-0000-0000-000000000001"
    duplicate["access_path"] = f"{OBJECTS_PATH}{duplicate['object_uid']}/access/"
    second_page = copy.deepcopy(first_page)
    second_page["next_cursor"] = None
    second_page["previous_cursor"] = "signed-previous-cursor"
    second_page["results"] = [duplicate]

    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(side_effect=[_page_response(first_page), _page_response(second_page)])

    with avala.load("acme/navigation") as dataset:
        with pytest.raises(DatasetIntegrityError, match="duplicate_manifest_object"):
            list(dataset.objects)


@respx.mock
def test_short_access_grant_is_discarded_and_refreshed_once() -> None:
    short_grant = copy.deepcopy(FIXTURES["access_grant_response"]["body"])
    short_grant["expires_at"] = "2026-08-23T18:00:04Z"
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response())
    grant_route = respx.get(ACCESS_URL).mock(side_effect=[_grant_response(short_grant), _grant_response()])
    respx.get(DOWNLOAD_URL).mock(return_value=httpx.Response(200, content=b"test"))

    with avala.load("acme/navigation") as dataset:
        assert dataset.episodes[0].read() == b"test"

    assert grant_route.call_count == 2


@respx.mock
def test_elapsed_time_after_grant_receipt_reduces_usable_lifetime(monkeypatch: pytest.MonkeyPatch) -> None:
    monotonic_times = iter([100.0, 396.0, 400.0, 400.0])
    monkeypatch.setattr(resolver_module, "_monotonic_time", lambda: next(monotonic_times))
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response())
    grant_route = respx.get(ACCESS_URL).mock(side_effect=[_grant_response(), _grant_response()])
    respx.get(DOWNLOAD_URL).mock(return_value=httpx.Response(200, content=b"test"))

    with avala.load("acme/navigation") as dataset:
        assert dataset.episodes[0].read() == b"test"

    assert grant_route.call_count == 2


@respx.mock
def test_grant_at_server_expiry_is_refreshed_without_a_clock_skew_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_times = iter([datetime(2026, 8, 23, 18, 5, tzinfo=timezone.utc), FIXTURE_SERVER_DATE])
    monkeypatch.setattr(resolver_module, "_utc_now", lambda: next(receipt_times))
    monkeypatch.setattr(resolver_module, "_monotonic_time", lambda: 100.0)
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response())
    grant_route = respx.get(ACCESS_URL).mock(side_effect=[_grant_response(), _grant_response()])
    respx.get(DOWNLOAD_URL).mock(return_value=httpx.Response(200, content=b"test"))

    with avala.load("acme/navigation") as dataset:
        assert dataset.episodes[0].read() == b"test"

    assert grant_route.call_count == 2


@respx.mock
def test_grant_date_rejects_more_than_thirty_seconds_of_negative_clock_skew(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        resolver_module,
        "_utc_now",
        lambda: datetime(2026, 8, 23, 17, 59, 29, tzinfo=timezone.utc),
    )
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response())
    respx.get(ACCESS_URL).mock(return_value=_grant_response())

    with avala.load("acme/navigation") as dataset:
        with pytest.raises(DatasetIntegrityError, match="invalid_grant_expiry"):
            dataset.episodes[0].read()


@respx.mock
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_uid", "00000000-0000-0000-0000-000000000000"),
        ("revision_sha256", "0" * 64),
        ("manifest_sha256", "1" * 64),
        ("object_uid", "00000000-0000-0000-0000-000000000000"),
        ("size_bytes", 5),
        ("sha256", "2" * 64),
    ],
)
def test_access_grant_repeated_identities_must_match_manifest_object(field: str, value: Any) -> None:
    mismatched_grant = copy.deepcopy(FIXTURES["access_grant_response"]["body"])
    mismatched_grant[field] = value
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response())
    respx.get(ACCESS_URL).mock(return_value=_grant_response(mismatched_grant))

    with avala.load("acme/navigation") as dataset:
        with pytest.raises(DatasetIntegrityError, match="access_grant_identity_mismatch") as exc_info:
            dataset.episodes[0].read()

    assert DOWNLOAD_URL not in _sdk_traceback_locals(exc_info.value)


@respx.mock
def test_access_grant_expiry_cannot_exceed_300_seconds_after_server_date() -> None:
    long_grant = copy.deepcopy(FIXTURES["access_grant_response"]["body"])
    long_grant["expires_at"] = "2026-08-23T18:05:01Z"
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response())
    respx.get(ACCESS_URL).mock(return_value=_grant_response(long_grant))

    with avala.load("acme/navigation") as dataset:
        with pytest.raises(DatasetIntegrityError, match="invalid_grant_expiry"):
            dataset.episodes[0].read()


@respx.mock
@pytest.mark.parametrize(
    "unsafe_url",
    [
        "http://fixture-bucket.s3.us-west-2.amazonaws.com/object?secret=do-not-log",
        "https://user@fixture-bucket.s3.us-west-2.amazonaws.com/object?secret=do-not-log",
        "https://fixture-bucket.s3.us-west-2.amazonaws.com:443/object?secret=do-not-log",
        "https://fixture-bucket.s3.us-west-2.amazonaws.com/object?secret=do-not-log#fragment",
        "https://fixture-bucket.s3.us-west-2.amazonaws.com.evil.test/object?secret=do-not-log",
        "https://fixture-bucket.s3.us-west-2.amazonaws.com/object?secret=do-not-log\t",
    ],
)
def test_unsafe_grant_url_is_rejected_without_leaking_the_secret_url(unsafe_url: str) -> None:
    unsafe_grant = copy.deepcopy(FIXTURES["access_grant_response"]["body"])
    unsafe_grant["url"] = unsafe_url
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response())
    respx.get(ACCESS_URL).mock(return_value=_grant_response(unsafe_grant))

    with avala.load("acme/navigation") as dataset:
        with pytest.raises(DatasetIntegrityError) as exc_info:
            dataset.episodes[0].read()

    assert unsafe_url not in str(exc_info.value)
    assert unsafe_url not in repr(exc_info.value)
    assert unsafe_url not in _sdk_traceback_locals(exc_info.value)
    assert "do-not-log" not in str(exc_info.value)


@respx.mock
@pytest.mark.parametrize(
    "unsafe_path",
    [
        "/api/v1/resolve/download/../admin",
        "/api/v1/resolve/download/%2e%2e/admin",
        "/api/v1/resolve/download/%5c..%5cadmin",
        "/api/v1/resolve/download/%255c..%255cadmin",
        "/api/v1/resolve/download//evil",
    ],
)
def test_avala_proxy_download_path_is_validated_after_normalization(unsafe_path: str) -> None:
    unsafe_grant = copy.deepcopy(FIXTURES["access_grant_response"]["body"])
    unsafe_grant["provider"] = "avala_proxy"
    unsafe_grant["url"] = f"https://api.avala.ai{unsafe_path}?secret=do-not-log"
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response())
    respx.get(ACCESS_URL).mock(return_value=_grant_response(unsafe_grant))

    with avala.load("acme/navigation") as dataset:
        with pytest.raises(DatasetIntegrityError, match="untrusted_download_path") as exc_info:
            dataset.episodes[0].read()

    assert "do-not-log" not in _sdk_traceback_locals(exc_info.value)


@respx.mock
@pytest.mark.parametrize(
    ("download_response", "reason"),
    [
        (httpx.Response(302, headers={"Location": "https://evil.test/stolen"}), "redirect_rejected"),
        (httpx.Response(200, content=b"fail"), "sha256_mismatch"),
    ],
)
def test_download_failures_are_typed_and_url_free(download_response: httpx.Response, reason: str) -> None:
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response())
    respx.get(ACCESS_URL).mock(return_value=_grant_response())
    respx.get(DOWNLOAD_URL).mock(return_value=download_response)

    with avala.load("acme/navigation") as dataset:
        with pytest.raises(DatasetDownloadError) as exc_info:
            dataset.episodes[0].read()

    assert exc_info.value.reason == reason
    assert DOWNLOAD_URL not in str(exc_info.value)
    assert DOWNLOAD_URL not in repr(exc_info.value)
    traceback_locals = _sdk_traceback_locals(exc_info.value)
    assert DOWNLOAD_URL not in traceback_locals
    assert "DatasetAccessGrant(" not in traceback_locals
    assert "fixture=1" not in str(exc_info.value)


@respx.mock
def test_httpx_download_error_is_not_retained_as_a_signed_url_context() -> None:
    def fail_download(request: httpx.Request) -> None:
        raise httpx.ConnectError("synthetic provider failure", request=request)

    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response())
    respx.get(ACCESS_URL).mock(return_value=_grant_response())
    respx.get(DOWNLOAD_URL).mock(side_effect=fail_download)

    with avala.load("acme/navigation") as dataset:
        with pytest.raises(DatasetDownloadError) as exc_info:
            dataset.episodes[0].read()

    assert exc_info.value.reason == "transport_error"
    assert exc_info.value.__context__ is None
    assert DOWNLOAD_URL not in str(exc_info.value)
    traceback_locals = _sdk_traceback_locals(exc_info.value)
    assert DOWNLOAD_URL not in traceback_locals
    assert "DatasetAccessGrant(" not in traceback_locals


@respx.mock
def test_partial_grant_failure_is_detached_from_signed_url_context() -> None:
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response())
    respx.get(ACCESS_URL).mock(return_value=httpx.Response(200, stream=_PartialSyncGrantStream()))

    with avala.load("acme/navigation") as dataset:
        with pytest.raises(DatasetResolverError, match="transport_error") as exc_info:
            dataset.episodes[0].read()

    assert exc_info.value.__context__ is None
    assert DOWNLOAD_URL not in _sdk_traceback_locals(exc_info.value)


@respx.mock
@pytest.mark.asyncio
async def test_async_partial_grant_failure_is_detached_from_signed_url_context() -> None:
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response())
    respx.get(ACCESS_URL).mock(return_value=httpx.Response(200, stream=_PartialAsyncGrantStream()))

    async with await avala.async_load("acme/navigation") as dataset:
        episode = await dataset.episodes[0]
        with pytest.raises(DatasetResolverError, match="transport_error") as exc_info:
            await episode.read()

    assert exc_info.value.__context__ is None
    assert DOWNLOAD_URL not in _sdk_traceback_locals(exc_info.value)


@respx.mock
@pytest.mark.asyncio
async def test_async_partial_grant_cancellation_is_detached_from_signed_url_context() -> None:
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response())
    respx.get(ACCESS_URL).mock(return_value=httpx.Response(200, stream=_PartialCancelledAsyncGrantStream()))

    async with await avala.async_load("acme/navigation") as dataset:
        episode = await dataset.episodes[0]
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await episode.read()

    assert exc_info.value.__context__ is None
    assert DOWNLOAD_URL not in _sdk_traceback_locals(exc_info.value)


@respx.mock
def test_malformed_successful_grant_response_is_scrubbed_before_raise() -> None:
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response())
    respx.get(ACCESS_URL).mock(
        return_value=httpx.Response(200, json=[copy.deepcopy(FIXTURES["access_grant_response"]["body"])])
    )

    with avala.load("acme/navigation") as dataset:
        with pytest.raises(DatasetIntegrityError, match="invalid_resolver_response") as exc_info:
            dataset.episodes[0].read()

    assert DOWNLOAD_URL not in _sdk_traceback_locals(exc_info.value)


@respx.mock
@pytest.mark.asyncio
async def test_async_malformed_successful_grant_response_is_scrubbed_before_raise() -> None:
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response())
    respx.get(ACCESS_URL).mock(
        return_value=httpx.Response(200, json=[copy.deepcopy(FIXTURES["access_grant_response"]["body"])])
    )

    async with await avala.async_load("acme/navigation") as dataset:
        episode = await dataset.episodes[0]
        with pytest.raises(DatasetIntegrityError, match="invalid_resolver_response") as exc_info:
            await episode.read()

    assert DOWNLOAD_URL not in _sdk_traceback_locals(exc_info.value)


@respx.mock
def test_unexpected_grant_json_decoder_failure_is_detached_and_scrubbed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_json = httpx.Response.json

    def fail_grant_json(response: httpx.Response, *args: Any, **kwargs: Any) -> Any:
        if str(response.request.url) == ACCESS_URL:
            raise RecursionError(DOWNLOAD_URL)
        return original_json(response, *args, **kwargs)

    monkeypatch.setattr(httpx.Response, "json", fail_grant_json)
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response())
    respx.get(ACCESS_URL).mock(return_value=_grant_response())

    with avala.load("acme/navigation") as dataset:
        with pytest.raises(DatasetIntegrityError, match="invalid_resolver_response") as exc_info:
            dataset.episodes[0].read()

    assert exc_info.value.__context__ is None
    assert DOWNLOAD_URL not in _sdk_traceback_locals(exc_info.value)


@respx.mock
@pytest.mark.asyncio
async def test_async_unexpected_grant_json_decoder_failure_is_detached_and_scrubbed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_json = httpx.Response.json

    def fail_grant_json(response: httpx.Response, *args: Any, **kwargs: Any) -> Any:
        if str(response.request.url) == ACCESS_URL:
            raise RecursionError(DOWNLOAD_URL)
        return original_json(response, *args, **kwargs)

    monkeypatch.setattr(httpx.Response, "json", fail_grant_json)
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response())
    respx.get(ACCESS_URL).mock(return_value=_grant_response())

    async with await avala.async_load("acme/navigation") as dataset:
        episode = await dataset.episodes[0]
        with pytest.raises(DatasetIntegrityError, match="invalid_resolver_response") as exc_info:
            await episode.read()

    assert exc_info.value.__context__ is None
    assert DOWNLOAD_URL not in _sdk_traceback_locals(exc_info.value)


@respx.mock
def test_unexpected_grant_parser_failure_is_scrubbed_before_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_provider_validation(url: str, provider: str, base_url: str) -> None:
        raise RuntimeError("synthetic parser failure")

    monkeypatch.setattr(resolver_module, "_validate_provider_url", fail_provider_validation)
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response())
    respx.get(ACCESS_URL).mock(return_value=_grant_response())

    with avala.load("acme/navigation") as dataset:
        with pytest.raises(DatasetIntegrityError, match="invalid_access_grant") as exc_info:
            dataset.episodes[0].read()

    assert exc_info.value.__context__ is None
    assert DOWNLOAD_URL not in _sdk_traceback_locals(exc_info.value)


@respx.mock
@pytest.mark.asyncio
async def test_async_unexpected_grant_parser_failure_is_scrubbed_before_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_provider_validation(url: str, provider: str, base_url: str) -> None:
        raise RuntimeError("synthetic parser failure")

    monkeypatch.setattr(resolver_module, "_validate_provider_url", fail_provider_validation)
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response())
    respx.get(ACCESS_URL).mock(return_value=_grant_response())

    async with await avala.async_load("acme/navigation") as dataset:
        episode = await dataset.episodes[0]
        with pytest.raises(DatasetIntegrityError, match="invalid_access_grant") as exc_info:
            await episode.read()

    assert exc_info.value.__context__ is None
    assert DOWNLOAD_URL not in _sdk_traceback_locals(exc_info.value)


@respx.mock
def test_rejected_grant_response_is_scrubbed_before_raise() -> None:
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response())
    respx.get(ACCESS_URL).mock(
        return_value=httpx.Response(503, json=copy.deepcopy(FIXTURES["access_grant_response"]["body"]))
    )

    with avala.load("acme/navigation") as dataset:
        with pytest.raises(DatasetResolverError, match="service_unavailable") as exc_info:
            dataset.episodes[0].read()

    assert DOWNLOAD_URL not in _sdk_traceback_locals(exc_info.value)


@respx.mock
def test_grant_style_410_response_is_scrubbed_before_invalid_withdrawal_raise() -> None:
    respx.get(RESOLVE_URL).mock(
        return_value=httpx.Response(410, json=copy.deepcopy(FIXTURES["access_grant_response"]["body"]))
    )

    with pytest.raises(DatasetIntegrityError, match="invalid_withdrawal") as exc_info:
        avala.load("acme/navigation")

    assert DOWNLOAD_URL not in _sdk_traceback_locals(exc_info.value)


@respx.mock
@pytest.mark.asyncio
async def test_async_rejected_grant_response_is_scrubbed_before_raise() -> None:
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response())
    respx.get(ACCESS_URL).mock(
        return_value=httpx.Response(503, json=copy.deepcopy(FIXTURES["access_grant_response"]["body"]))
    )

    async with await avala.async_load("acme/navigation") as dataset:
        episode = await dataset.episodes[0]
        with pytest.raises(DatasetResolverError, match="service_unavailable") as exc_info:
            await episode.read()

    assert DOWNLOAD_URL not in _sdk_traceback_locals(exc_info.value)


@respx.mock
def test_resolver_rejects_cross_identity_metadata() -> None:
    mismatched = copy.deepcopy(FIXTURES["resolve_response"]["body"])
    mismatched["manifest"]["objects_path"] = "/resolve/uid/00000000-0000-0000-0000-000000000000/bad/objects/"
    respx.get(RESOLVE_URL).mock(return_value=httpx.Response(200, json=mismatched))

    with pytest.raises(DatasetIntegrityError, match="resolver_identity_mismatch"):
        avala.load("acme/navigation")


@respx.mock
@pytest.mark.parametrize(
    "display_reference",
    [
        f"other/navigation@{FIXTURES['resolve_response']['body']['revision_sha256']}",
        f"acme/navigation@{'0' * 64}",
        FIXTURES["resolve_response"]["body"]["canonical_reference"],
    ],
)
def test_resolver_requires_a_matching_digest_pinned_friendly_display_reference(display_reference: str) -> None:
    mismatched = copy.deepcopy(FIXTURES["resolve_response"]["body"])
    mismatched["display_reference"] = display_reference
    respx.get(RESOLVE_URL).mock(return_value=httpx.Response(200, json=mismatched))

    with pytest.raises(DatasetIntegrityError, match="resolver_identity_mismatch"):
        avala.load("acme/navigation")


@respx.mock
def test_object_page_must_repeat_the_resolved_manifest_identity() -> None:
    mismatched_page = copy.deepcopy(FIXTURES["object_page_response"]["body"])
    mismatched_page["manifest_sha256"] = "0" * 64
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response(mismatched_page))

    with avala.load("acme/navigation") as dataset:
        with pytest.raises(DatasetIntegrityError, match="object_page_identity_mismatch"):
            _ = dataset.episodes[0]


@respx.mock
def test_withdrawn_revision_raises_typed_safe_error() -> None:
    canonical = FIXTURES["withdrawn_response"]["body"]["canonical_reference"]
    canonical_url = f"{BASE_URL}/resolve/uid/{FIXTURES['withdrawn_response']['body']['dataset_uid']}/{FIXTURES['withdrawn_response']['body']['revision_sha256']}/"
    respx.get(canonical_url).mock(
        return_value=httpx.Response(
            FIXTURES["withdrawn_response"]["http_status"],
            json=FIXTURES["withdrawn_response"]["body"],
        )
    )

    with pytest.raises(DatasetRevisionWithdrawnError) as exc_info:
        avala.load(canonical)

    assert exc_info.value.canonical_reference == canonical
    assert exc_info.value.status_code == 410


@respx.mock
@pytest.mark.parametrize("mismatch_kind", ["requested_identity", "canonical_self_identity"])
def test_withdrawn_tombstone_must_match_the_exact_request(mismatch_kind: str) -> None:
    withdrawn = copy.deepcopy(FIXTURES["withdrawn_response"]["body"])
    requested_canonical = withdrawn["canonical_reference"]
    canonical_url = f"{BASE_URL}/resolve/uid/{withdrawn['dataset_uid']}/{withdrawn['revision_sha256']}/"
    if mismatch_kind == "requested_identity":
        withdrawn["dataset_uid"] = "00000000-0000-0000-0000-000000000000"
        withdrawn["revision_sha256"] = "0" * 64
        withdrawn["canonical_reference"] = f"avala://datasets/{withdrawn['dataset_uid']}@{withdrawn['revision_sha256']}"
    else:
        withdrawn["canonical_reference"] = f"avala://datasets/{withdrawn['dataset_uid']}@{'0' * 64}"
    respx.get(canonical_url).mock(return_value=httpx.Response(410, json=withdrawn))

    with pytest.raises(DatasetIntegrityError, match="withdrawal_identity_mismatch"):
        avala.load(requested_canonical)


@respx.mock
def test_alias_request_never_accepts_a_withdrawn_tombstone() -> None:
    respx.get(RESOLVE_URL).mock(
        return_value=httpx.Response(410, json=copy.deepcopy(FIXTURES["withdrawn_response"]["body"]))
    )

    with pytest.warns(MutableDatasetAliasWarning):
        with pytest.raises(DatasetIntegrityError, match="withdrawal_identity_mismatch"):
            avala.load("acme/navigation")


@respx.mock
@pytest.mark.asyncio
async def test_async_withdrawn_tombstone_must_match_the_exact_request() -> None:
    withdrawn = copy.deepcopy(FIXTURES["withdrawn_response"]["body"])
    requested_canonical = withdrawn["canonical_reference"]
    canonical_url = f"{BASE_URL}/resolve/uid/{withdrawn['dataset_uid']}/{withdrawn['revision_sha256']}/"
    withdrawn["dataset_uid"] = "00000000-0000-0000-0000-000000000000"
    withdrawn["canonical_reference"] = f"avala://datasets/{withdrawn['dataset_uid']}@{withdrawn['revision_sha256']}"
    respx.get(canonical_url).mock(return_value=httpx.Response(410, json=withdrawn))

    with pytest.raises(DatasetIntegrityError, match="withdrawal_identity_mismatch"):
        await avala.async_load(requested_canonical)


@pytest.mark.asyncio
async def test_async_load_closes_transport_when_resolution_is_cancelled(monkeypatch: pytest.MonkeyPatch) -> None:
    class CancellingTransport:
        def __init__(self, base_url: str) -> None:
            self.base_url = base_url
            self.closed = False

        async def resolve(self, parsed_reference: Any) -> None:
            del parsed_reference
            raise asyncio.CancelledError

        async def close(self) -> None:
            self.closed = True

    transport = CancellingTransport(BASE_URL)
    monkeypatch.setattr(datasets_module, "AsyncDatasetResolverTransport", lambda base_url: transport)

    with pytest.raises(asyncio.CancelledError):
        await avala.async_load(FIXTURES["resolve_response"]["body"]["canonical_reference"])

    assert transport.closed is True
