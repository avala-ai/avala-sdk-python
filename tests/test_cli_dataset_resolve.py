"""The CLI hands off immutable public metadata without opening object access."""

from __future__ import annotations

import pytest

pytest.importorskip("click", reason="CLI dependencies not installed (pip install avala[cli])")

import copy  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402
from unittest.mock import Mock  # noqa: E402

import httpx  # noqa: E402
import respx  # noqa: E402
import avala.cli as cli  # noqa: E402
from avala._resolver import SyncDatasetResolverTransport  # noqa: E402
from click.testing import CliRunner  # noqa: E402

FIXTURES = json.loads((Path(__file__).resolve().parent / "fixtures/dataset_resolver_v1_fixtures.json").read_text())
BASE_URL = "https://api.avala.ai/api/v1"
SIGNED_URL = FIXTURES["access_grant_response"]["body"]["url"]


def _runner() -> CliRunner:
    # Click 8.1 mixes stderr into stdout by default; 8.2 always separates it.
    # Exercise the same machine-readable stdout contract on both versions.
    if "mix_stderr" in inspect.signature(CliRunner).parameters:
        return CliRunner(mix_stderr=False)
    return CliRunner()


@pytest.fixture(autouse=True)
def _isolate_cli_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AVALA_API_KEY", raising=False)
    monkeypatch.delenv("AVALA_BASE_URL", raising=False)


def _body(requested_reference: str = "acme/navigation@main") -> dict[str, Any]:
    body = copy.deepcopy(FIXTURES["resolve_response"]["body"])
    body["requested_reference"] = requested_reference
    return body


def _expected(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "canonical_reference": body["canonical_reference"],
        "requested_reference": body["requested_reference"],
        "display_reference": body["display_reference"],
        "dataset_uid": body["dataset_uid"],
        "revision_uid": body["revision_uid"],
        "revision_sha256": body["revision_sha256"],
        "manifest_sha256": body["manifest"]["sha256"],
        "object_count": body["manifest"]["object_count"],
        "total_size_bytes": body["manifest"]["total_size_bytes"],
        "rights": body["rights"],
    }


@pytest.mark.parametrize("credential_source", ["absent", "environment", "flag"])
@respx.mock
def test_resolve_is_anonymous_and_metadata_only(
    credential_source: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = Mock(wraps=cli.Client)
    monkeypatch.setattr(cli, "Client", client)
    monkeypatch.setenv("AVALA_API_KEY", "stale-fixture-key" if credential_source == "environment" else "")
    monkeypatch.setenv("AVALA_BASE_URL", BASE_URL)
    monkeypatch.setenv("HTTPS_PROXY", "unsupported://proxy.invalid")
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "missing-cert.pem"))
    body = _body()
    route = respx.get(f"{BASE_URL}/resolve/acme/navigation@main/").mock(return_value=httpx.Response(200, json=body))
    args = ["--api-key", "invalid-credential-\N{SNOWMAN}"] if credential_source == "flag" else []

    result = _runner().invoke(cli.main, [*args, "--output", "json", "datasets", "resolve", "acme/navigation"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == _expected(body)
    assert result.stderr == ""
    client.assert_not_called()
    assert len(respx.calls) == route.call_count == 1
    request = route.calls.last.request
    assert request.method == "GET"
    assert not {"authorization", "x-avala-api-key", "cookie"} & set(request.headers)
    assert SIGNED_URL not in result.output


@pytest.mark.parametrize("selector", ["main", "release-1", "digest", "canonical"])
@pytest.mark.parametrize("revision_option", [False, True])
@respx.mock
def test_resolve_reference_forms(selector: str, revision_option: bool) -> None:
    digest = _body()["revision_sha256"]
    canonical = _body()["canonical_reference"]
    if selector == "canonical":
        reference = canonical
        requested = canonical
        path = canonical.replace("avala://datasets/", "/resolve/uid/").replace("@", "/") + "/"
    else:
        selected = digest if selector == "digest" else selector
        reference = "acme/navigation" if revision_option else f"acme/navigation@{selected}"
        requested = f"acme/navigation@{selected}"
        path = f"/resolve/{requested}/"
    args = ["--revision", "main" if selector == "canonical" else selected] if revision_option else []
    if selector == "canonical" and revision_option:
        result = _runner().invoke(cli.main, ["datasets", "resolve", reference, *args])
        assert result.exit_code != 0
        assert "canonical_reference_with_revision" in result.stderr
        assert not respx.calls
        return
    body = _body(requested)
    respx.get(f"{BASE_URL}{path}").mock(return_value=httpx.Response(200, json=body))

    result = _runner().invoke(cli.main, ["-o", "json", "datasets", "resolve", reference, *args])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == _expected(body)
    assert len(respx.calls) == 1


@pytest.mark.parametrize("use_flag", [False, True])
@respx.mock
def test_resolve_honors_base_url_and_flag_precedence(use_flag: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AVALA_BASE_URL", "https://environment.example.test/platform/api/v1/")
    base_url = (
        "https://explicit.example.test/custom/api/v1"
        if use_flag
        else "https://environment.example.test/platform/api/v1"
    )
    route = respx.get(f"{base_url}/resolve/acme/navigation@main/").mock(return_value=httpx.Response(200, json=_body()))
    args = ["--base-url", base_url + "/"] if use_flag else []

    result = _runner().invoke(cli.main, [*args, "-o", "json", "datasets", "resolve", "acme/navigation"])

    assert result.exit_code == 0, result.output
    assert route.call_count == len(respx.calls) == 1


@respx.mock
def test_resolve_table_and_context_close(monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[SyncDatasetResolverTransport] = []
    original_close = SyncDatasetResolverTransport.close

    def close(transport: SyncDatasetResolverTransport) -> None:
        original_close(transport)
        closed.append(transport)

    monkeypatch.setattr(SyncDatasetResolverTransport, "close", close)
    respx.get(f"{BASE_URL}/resolve/acme/navigation@main/").mock(return_value=httpx.Response(200, json=_body()))

    result = _runner().invoke(cli.main, ["datasets", "resolve", "acme/navigation"])

    assert result.exit_code == 0, result.output
    assert "Canonical reference" in result.output
    assert "4.0 B" in result.output
    assert "open_download" in result.output
    assert len(closed) == 1
    assert closed[0]._client.is_closed
    assert len(respx.calls) == 1


@respx.mock
def test_resolve_omits_response_extensions() -> None:
    body = _body()
    body["download_url"] = SIGNED_URL
    body["manifest"]["download_url"] = SIGNED_URL
    body["manifest"]["origins"][0]["credential"] = SIGNED_URL
    respx.get(f"{BASE_URL}/resolve/acme/navigation@main/").mock(return_value=httpx.Response(200, json=body))

    result = _runner().invoke(cli.main, ["-o", "json", "datasets", "resolve", "acme/navigation"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == _expected(body)
    assert SIGNED_URL not in result.output
    assert "download_url" not in result.output


@pytest.mark.parametrize("failure", ["not_found", "withdrawn", "identity", "rights", "transport"])
@respx.mock
def test_resolve_errors_are_safe_and_have_no_json_payload(failure: str) -> None:
    body = _body()
    reference = body["canonical_reference"]
    path = reference.replace("avala://datasets/", "/resolve/uid/").replace("@", "/") + "/"
    route = respx.get(f"{BASE_URL}{path}")
    if failure == "not_found":
        route.mock(return_value=httpx.Response(404, json={"detail": SIGNED_URL}))
    elif failure == "withdrawn":
        withdrawn = copy.deepcopy(FIXTURES["withdrawn_response"]["body"])
        withdrawn["download_url"] = SIGNED_URL
        route.mock(return_value=httpx.Response(410, json=withdrawn))
    elif failure == "transport":
        route.mock(side_effect=httpx.ConnectError(SIGNED_URL))
    else:
        body["requested_reference"] = reference
        if failure == "rights":
            body["rights"]["download_url"] = SIGNED_URL
        else:
            body["revision_sha256"] = "0" * 64
        route.mock(return_value=httpx.Response(200, json=body))

    result = _runner().invoke(cli.main, ["-o", "json", "datasets", "resolve", reference])

    assert result.exit_code != 0
    assert result.stdout == ""
    assert "Dataset resolver failed" in result.stderr
    assert SIGNED_URL not in result.stdout + result.stderr
    assert len(result.stderr) < 200
    assert len(respx.calls) == 1


@pytest.mark.parametrize("args", [["bad reference"], ["acme/navigation@main", "--revision", "other"]])
@respx.mock
def test_resolve_rejects_invalid_reference_before_network(args: list[str]) -> None:
    result = _runner().invoke(cli.main, ["-o", "json", "datasets", "resolve", *args])
    assert result.exit_code != 0
    assert result.stdout == ""
    assert not respx.calls


@respx.mock
def test_authenticated_dataset_sibling_keeps_client_and_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Mock(wraps=cli.Client)
    monkeypatch.setattr(cli, "Client", client)
    route = respx.get(f"{BASE_URL}/datasets/").mock(
        return_value=httpx.Response(200, json={"results": [], "next": None, "previous": None})
    )

    result = _runner().invoke(cli.main, ["--api-key", "fixture-key", "datasets", "list"])

    assert result.exit_code == 0, result.output
    client.assert_called_once_with(api_key="fixture-key")
    assert route.calls.last.request.headers["X-Avala-Api-Key"] == "fixture-key"


def test_resolve_help_does_not_initialize_authenticated_client(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Mock(wraps=cli.Client)
    monkeypatch.setattr(cli, "Client", client)
    result = _runner().invoke(cli.main, ["datasets", "resolve", "--help"])
    assert result.exit_code == 0
    assert "--revision" in result.output
    client.assert_not_called()
