"""Development serving authority is bound to the exact development API."""

from __future__ import annotations

import copy

import httpx
import pytest
import respx

import avala
from avala import _resolver
from avala.errors import DatasetIntegrityError
from tests.test_public_dataset_loader import (
    ACCESS_PATH,
    FIXTURE_SERVER_DATE,
    OBJECTS_PATH,
    RESOLVE_PATH,
    _grant_response,
    _page_response,
    _resolve_response,
    _sdk_traceback_locals,
)
from tests.test_public_edge_loader import BODY, EDGE_URL

DEV_API = "https://server.dev.alala.ai/api/v1"
DEV_EDGE = EDGE_URL.replace("data.avala.ai", "data-development.avala.ai")


@pytest.fixture(autouse=True)
def freeze_grant_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_resolver, "_utc_now", lambda: FIXTURE_SERVER_DATE)


def setup_resolver(base: str, url: str, provider: str = "avala_edge") -> respx.Route:
    grant = copy.deepcopy(BODY)
    grant.update(provider=provider, url=url)
    respx.get(f"{base}{RESOLVE_PATH}").mock(return_value=_resolve_response())
    respx.get(f"{base}{OBJECTS_PATH}").mock(return_value=_page_response())
    return respx.post(f"{base}{ACCESS_PATH}?transport=avala-edge-v1").mock(return_value=_grant_response(grant))


async def read_object(base: str, async_mode: bool) -> bytes:
    if async_mode:
        async with await avala.async_load("acme/navigation", base_url=base) as dataset:
            episode = await dataset.episodes[0]
            return await episode.read()
    with avala.load("acme/navigation", base_url=base) as dataset:
        return dataset.episodes[0].read()


@respx.mock
@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize(
    "base,url",
    [
        (DEV_API, DEV_EDGE),
        ("https://api.avala.ai/api/v1", EDGE_URL),
        ("https://server.avala.ai/api/v1", EDGE_URL),
    ],
)
async def test_exact_environment_pair_downloads_verified_anonymous_bytes(base: str, url: str, async_mode: bool) -> None:
    grant = setup_resolver(base, url)
    download = respx.get(url).mock(return_value=httpx.Response(200, content=b"test"))
    assert await read_object(base, async_mode) == b"test"
    assert grant.call_count == download.call_count == 1
    assert "authorization" not in download.calls[0].request.headers
    assert "cookie" not in download.calls[0].request.headers


@respx.mock
@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize(
    "base,url",
    [
        (DEV_API, EDGE_URL),
        ("https://api.avala.ai/api/v1", DEV_EDGE),
        ("https://server.avala.ai/api/v1", DEV_EDGE),
        ("https://api.example.test/api/v1", EDGE_URL),
        ("https://api.example.test/api/v1", DEV_EDGE),
        ("https://server.dev.alala.ai.evil.example/api/v1", DEV_EDGE),
        ("https://server.dev.alala.ai:443/api/v1", DEV_EDGE),
        ("https://server.dev.alala.ai/api/v10", DEV_EDGE),
        ("https://server.dev.alala.ai/alternate/api/v1", DEV_EDGE),
        (DEV_API, DEV_EDGE.replace("data-development.avala.ai", "data-development.avala.ai.evil.example")),
    ],
)
async def test_mixed_environment_refuses_before_downloading_and_redacts_capability(
    base: str, url: str, async_mode: bool
) -> None:
    setup_resolver(base, url)
    with pytest.raises(DatasetIntegrityError) as error:
        await read_object(base, async_mode)
    assert len(respx.calls) == 3
    assert "capability" not in _sdk_traceback_locals(error.value)
    assert error.value.__context__ is None


@respx.mock
@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("base", [DEV_API, "https://api.example.test/api/v1"])
async def test_existing_direct_grants_remain_usable_with_custom_and_development_apis(
    base: str, async_mode: bool
) -> None:
    setup_resolver(base, BODY["url"], provider=BODY["provider"])
    respx.get(BODY["url"]).mock(return_value=httpx.Response(200, content=b"test"))
    assert await read_object(base, async_mode) == b"test"
