"""The negotiated edge transport preserves anonymous authority and file integrity."""

from __future__ import annotations

import copy

import httpx
import pytest
import respx

import avala
from avala import _resolver
from avala.errors import DatasetDownloadError, DatasetIntegrityError
from tests.test_public_dataset_loader import (
    ACCESS_PATH,
    BASE_URL,
    FIXTURE_SERVER_DATE,
    FIXTURES,
    OBJECTS_URL,
    RESOLVE_URL,
    _grant_response,
    _page_response,
    _resolve_response,
    _sdk_traceback_locals,
)

BODY = FIXTURES["access_grant_response"]["body"]
EVIDENCE = "e" * 64
EDGE_PATH = (
    f"/v1/datasets/{BODY['dataset_uid']}/{BODY['revision_sha256']}/{BODY['manifest_sha256']}"
    f"/objects/{BODY['object_uid']}/{EVIDENCE}"
)
EDGE_URL = f"https://data.avala.ai{EDGE_PATH}?grant=synthetic%3Asigned%3Acapability"
GRANT_URL = f"{BASE_URL}{ACCESS_PATH}?transport=avala-edge-v1"


@pytest.fixture(autouse=True)
def freeze_grant_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_resolver, "_utc_now", lambda: FIXTURE_SERVER_DATE)


def setup_resolver(url: str = EDGE_URL, provider: str = "avala_edge") -> respx.Route:
    grant = copy.deepcopy(BODY)
    grant.update(provider=provider, url=url)
    respx.get(RESOLVE_URL).mock(return_value=_resolve_response())
    respx.get(OBJECTS_URL).mock(return_value=_page_response())
    return respx.post(GRANT_URL).mock(return_value=_grant_response(grant))


async def read_object(async_mode: bool) -> bytes:
    if async_mode:
        async with await avala.async_load("acme/navigation") as dataset:
            episode = await dataset.episodes[0]
            return await episode.read()
    with avala.load("acme/navigation") as dataset:
        return dataset.episodes[0].read()


@respx.mock
@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_negotiates_edge_and_verifies_anonymous_bytes(async_mode: bool) -> None:
    grant = setup_resolver()
    download = respx.get(EDGE_URL).mock(return_value=httpx.Response(200, content=b"test"))
    assert await read_object(async_mode) == b"test"
    assert grant.call_count == download.call_count == 1
    assert dict(grant.calls[0].request.url.params) == {"transport": "avala-edge-v1"}
    headers = download.calls[0].request.headers
    assert "authorization" not in headers and "cookie" not in headers
    assert headers["accept-encoding"] == "identity"


@respx.mock
@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("token", ["x", "x" * 4096, "%3A" * 4096])
async def test_edge_token_bounds_apply_to_the_decoded_capability(async_mode: bool, token: str) -> None:
    url = f"https://data.avala.ai{EDGE_PATH}?grant={token}"
    setup_resolver(url)
    respx.get(url).mock(return_value=httpx.Response(200, content=b"test"))
    assert await read_object(async_mode) == b"test"


@respx.mock
@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("provider", ["aws_s3", "gcs", "cloudflare_r2"])
async def test_server_can_return_legacy_direct_transport_without_another_grant_request(
    async_mode: bool, provider: str
) -> None:
    urls = {
        "aws_s3": BODY["url"],
        "gcs": "https://storage.googleapis.com/fixture/key?signed=fixture",
        "cloudflare_r2": f"https://{'a' * 32}.eu.r2.cloudflarestorage.com/fixture/key?signed=fixture",
    }
    grant = setup_resolver(urls[provider], provider)
    respx.get(urls[provider]).mock(return_value=httpx.Response(200, content=b"test"))
    assert await read_object(async_mode) == b"test"
    assert grant.call_count == 1


@respx.mock
@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize(
    "url",
    [
        EDGE_URL.replace("data.avala.ai", "data.avala.ai.evil.example"),
        EDGE_URL.replace("data.avala.ai", "api.avala.ai"),
        EDGE_URL.replace("https://", "http://"),
        EDGE_URL.replace("https://", "https://user@"),
        EDGE_URL.replace("data.avala.ai", "data.avala.ai:443"),
        EDGE_URL.replace(str(BODY["dataset_uid"]), "00000000-0000-4000-8000-000000000001"),
        EDGE_URL.replace(str(BODY["revision_sha256"]), "0" * 64),
        EDGE_URL.replace(str(BODY["manifest_sha256"]), "0" * 64),
        EDGE_URL.replace(str(BODY["object_uid"]), "00000000-0000-4000-8000-000000000002"),
        EDGE_URL.replace(EVIDENCE, "E" * 64),
        EDGE_URL.replace(EVIDENCE, "e" * 63),
        EDGE_URL.replace("/objects/", "/%6fbjects/"),
        EDGE_URL.replace("/objects/", "/discard/../objects/"),
        EDGE_URL.replace("/objects/", "//objects/"),
        EDGE_URL.replace("?grant=", "/?grant="),
        EDGE_URL.split("?")[0],
        EDGE_URL.replace("grant=", "%67rant="),
        EDGE_URL.replace("grant=", "signature="),
        EDGE_URL.split("?")[0] + "?grant=",
        EDGE_URL.split("?")[0] + "?grant=" + "x" * 4097,
        EDGE_URL.split("?")[0] + "?grant=" + "%3A" * 4097,
        EDGE_URL + "%00",
        EDGE_URL + "%FF",
        EDGE_URL + "%253A",
        EDGE_URL + "+suffix",
        EDGE_URL + "&grant=other",
        EDGE_URL + "&url=https://evil.example",
        EDGE_URL + "#fragment",
        EDGE_URL + "#",
        EDGE_URL + "\n",
    ],
)
async def test_rejects_untrusted_edge_identity_without_downloading_or_exposing_token(
    async_mode: bool, url: str
) -> None:
    grant = setup_resolver(url)
    with pytest.raises(DatasetIntegrityError) as error:
        await read_object(async_mode)
    assert grant.call_count == 1
    assert len(respx.calls) == 3
    assert "capability" not in str(error.value)
    assert "capability" not in _sdk_traceback_locals(error.value)
    assert error.value.__context__ is None


@respx.mock
@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("status", [301, 302, 304, 401, 403, 404, 410, 429, 500, 503, 206])
async def test_edge_failure_never_falls_back_to_a_direct_grant(async_mode: bool, status: int) -> None:
    grant = setup_resolver()
    respx.get(EDGE_URL).mock(return_value=httpx.Response(status, content=b"test", headers={"Location": BODY["url"]}))
    with pytest.raises(DatasetDownloadError) as error:
        await read_object(async_mode)
    assert grant.call_count == 1
    assert len(respx.calls) == 4
    assert error.value.provider == "avala_edge"
    assert EDGE_URL not in str(error.value)
    assert "capability" not in _sdk_traceback_locals(error.value)
    assert error.value.__context__ is None


@respx.mock
@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize(
    "content,reason", [(b"nope", "sha256_mismatch"), (b"tes", "size_mismatch"), (b"tests", "size_mismatch")]
)
async def test_edge_bytes_still_require_full_manifest_integrity(async_mode: bool, content: bytes, reason: str) -> None:
    setup_resolver()
    respx.get(EDGE_URL).mock(return_value=httpx.Response(200, content=content))
    with pytest.raises(DatasetDownloadError) as error:
        await read_object(async_mode)
    assert error.value.reason == reason


@respx.mock
@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("provider", ["cloudflare_r2", "avala_proxy"])
async def test_edge_url_cannot_be_labeled_as_a_direct_r2_or_api_proxy_grant(async_mode: bool, provider: str) -> None:
    setup_resolver(provider=provider)
    with pytest.raises(DatasetIntegrityError):
        await read_object(async_mode)
    assert len(respx.calls) == 3
