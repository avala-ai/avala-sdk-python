"""Tests for the ``avala view`` local MCAP viewer command."""

from __future__ import annotations

import pytest

pytest.importorskip("click", reason="CLI dependencies not installed (pip install avala[cli])")

import threading  # noqa: E402
import urllib.request  # noqa: E402
from pathlib import Path  # noqa: E402
from urllib.error import HTTPError  # noqa: E402
from urllib.parse import parse_qs, urlparse  # noqa: E402

from avala.cli import main  # noqa: E402
from avala.cli.view import build_viewer_url, serve_file  # noqa: E402
from click.testing import CliRunner  # noqa: E402

# A minimal MCAP-magic-prefixed blob. We only ever validate the leading magic
# and stream raw bytes, so a well-formed MCAP body is unnecessary here.
_MCAP_MAGIC = b"\x89MCAP0\r\n"
_BODY = _MCAP_MAGIC + bytes(range(256)) * 8  # 8 bytes magic + 2048 bytes


def _write_mcap(tmp_path: Path, name: str = "sample.mcap") -> Path:
    path = tmp_path / name
    path.write_bytes(_BODY)
    return path


def _get(url: str, headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req) as resp:  # noqa: S310 (loopback test server)
            return resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read()
    except HTTPError as exc:
        return exc.code, {k.lower(): v for k, v in exc.headers.items()}, exc.read()


_VIEWER_ORIGIN = "https://avala.ai"


class _RunningServer:
    """Context manager that runs ``serve_file`` in a background thread."""

    def __init__(self, path: Path, allowed_origin: str | None = _VIEWER_ORIGIN) -> None:
        self._server = serve_file(path, port=0, allowed_origin=allowed_origin)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> "_RunningServer":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def url(self, name: str) -> str:
        return f"http://127.0.0.1:{self.port}/{name}"


# --- pure helpers ---------------------------------------------------------


def test_build_viewer_url_composes_route_and_params():
    url = build_viewer_url("https://avala.ai", 8731)
    parsed = urlparse(url)
    assert parsed.scheme == "https"
    assert parsed.netloc == "avala.ai"
    assert parsed.path == "/mcap"
    qs = parse_qs(parsed.query)
    # Served under an opaque path — never the real local basename.
    assert qs["url"] == ["http://127.0.0.1:8731/recording.mcap"]
    # Funnel attribution rides along.
    assert qs["utm_source"] == ["cli"]
    assert qs["utm_medium"] == ["avala_view"]
    assert qs["utm_campaign"] == ["local_viewer"]


def test_build_viewer_url_omits_local_filename():
    # Privacy: the real basename (which may carry customer/route identifiers)
    # must not appear anywhere in the hosted URL sent to avala.ai.
    url = build_viewer_url("https://avala.ai", 8731)
    assert "drive" not in url
    assert "file=" not in url


def test_origin_of_normalizes_default_ports():
    from avala.cli.view import _origin_of

    # Default ports are dropped and host lowercased, matching browser Origin.
    assert _origin_of("https://Viewer.Example:443/mcap") == "https://viewer.example"
    assert _origin_of("http://host:80") == "http://host"
    # Non-default ports are preserved.
    assert _origin_of("https://host:8443") == "https://host:8443"
    # IPv6 literals keep their brackets (matching the browser Origin header).
    assert _origin_of("https://[::1]:8443") == "https://[::1]:8443"
    assert _origin_of("https://[::1]") == "https://[::1]"
    # Unparseable URLs and out-of-range/non-numeric ports yield "" (rejected
    # by the command, not a traceback).
    assert _origin_of("not-a-url") == ""
    assert _origin_of("https://host:99999") == ""
    assert _origin_of("https://host:notaport") == ""


def test_build_viewer_url_trims_trailing_slash():
    url = build_viewer_url("https://staging.example.com/", 9000)
    parsed = urlparse(url)
    assert parsed.netloc == "staging.example.com"
    qs = parse_qs(parsed.query)
    assert qs["url"] == ["http://127.0.0.1:9000/recording.mcap"]


# --- range server ---------------------------------------------------------


def test_server_full_get_returns_whole_file(tmp_path):
    path = _write_mcap(tmp_path)
    with _RunningServer(path) as srv:
        status, headers, body = _get(srv.url(path.name))
    assert status == 200
    assert body == _BODY
    assert headers["accept-ranges"] == "bytes"
    assert headers["content-length"] == str(len(_BODY))
    # An originless (curl) request needs no CORS grant.
    assert "access-control-allow-origin" not in headers


def test_server_honors_byte_range(tmp_path):
    path = _write_mcap(tmp_path)
    with _RunningServer(path) as srv:
        status, headers, body = _get(srv.url(path.name), {"Range": "bytes=8-15"})
    assert status == 206
    assert body == _BODY[8:16]
    assert headers["content-range"] == f"bytes 8-15/{len(_BODY)}"
    assert headers["content-length"] == "8"


def test_server_honors_suffix_range(tmp_path):
    path = _write_mcap(tmp_path)
    with _RunningServer(path) as srv:
        status, headers, body = _get(srv.url(path.name), {"Range": "bytes=-16"})
    assert status == 206
    assert body == _BODY[-16:]
    assert headers["content-range"] == f"bytes {len(_BODY) - 16}-{len(_BODY) - 1}/{len(_BODY)}"


def test_server_open_ended_range(tmp_path):
    path = _write_mcap(tmp_path)
    with _RunningServer(path) as srv:
        status, _headers, body = _get(srv.url(path.name), {"Range": "bytes=2040-"})
    assert status == 206
    assert body == _BODY[2040:]


def test_server_unsatisfiable_range_returns_416(tmp_path):
    path = _write_mcap(tmp_path)
    with _RunningServer(path) as srv:
        status, headers, _body = _get(srv.url(path.name), {"Range": f"bytes={len(_BODY) + 10}-"})
    assert status == 416
    assert headers["content-range"] == f"bytes */{len(_BODY)}"


def test_server_head_reports_size_without_body(tmp_path):
    path = _write_mcap(tmp_path)
    with _RunningServer(path) as srv:
        req = urllib.request.Request(srv.url(path.name), method="HEAD")
        with urllib.request.urlopen(req) as resp:  # noqa: S310
            assert resp.status == 200
            assert resp.headers["Content-Length"] == str(len(_BODY))
            assert resp.headers["Accept-Ranges"] == "bytes"
            assert resp.read() == b""


def test_server_options_preflight_allows_cors_and_range(tmp_path):
    path = _write_mcap(tmp_path)
    with _RunningServer(path) as srv:
        req = urllib.request.Request(srv.url(path.name), method="OPTIONS", headers={"Origin": _VIEWER_ORIGIN})
        with urllib.request.urlopen(req) as resp:  # noqa: S310
            assert resp.status == 204
            assert resp.headers["Access-Control-Allow-Origin"] == _VIEWER_ORIGIN
            assert "Range" in resp.headers["Access-Control-Allow-Headers"]
            assert "GET" in resp.headers["Access-Control-Allow-Methods"]


def test_server_grants_private_network_access_preflight(tmp_path):
    # An HTTPS page fetching loopback triggers a PNA preflight; the response
    # must echo the allowed origin and grant Access-Control-Allow-Private-Network.
    path = _write_mcap(tmp_path)
    with _RunningServer(path) as srv:
        req = urllib.request.Request(
            srv.url(path.name),
            method="OPTIONS",
            headers={
                "Origin": _VIEWER_ORIGIN,
                "Access-Control-Request-Private-Network": "true",
            },
        )
        with urllib.request.urlopen(req) as resp:  # noqa: S310
            assert resp.status == 204
            assert resp.headers["Access-Control-Allow-Origin"] == _VIEWER_ORIGIN
            assert resp.headers["Access-Control-Allow-Private-Network"] == "true"


def test_server_echoes_origin_on_get(tmp_path):
    path = _write_mcap(tmp_path)
    with _RunningServer(path) as srv:
        _status, headers, _body = _get(srv.url(path.name), {"Origin": _VIEWER_ORIGIN})
    assert headers["access-control-allow-origin"] == _VIEWER_ORIGIN
    assert "origin" in headers.get("vary", "").lower()


def test_server_denies_foreign_origin(tmp_path):
    # A site other than the configured viewer gets no CORS/PNA grant even if it
    # guesses the port — the browser then blocks it from reading the file.
    path = _write_mcap(tmp_path)
    with _RunningServer(path, allowed_origin=_VIEWER_ORIGIN) as srv:
        req = urllib.request.Request(
            srv.url(path.name),
            method="OPTIONS",
            headers={
                "Origin": "https://evil.example.com",
                "Access-Control-Request-Private-Network": "true",
            },
        )
        with urllib.request.urlopen(req) as resp:  # noqa: S310
            assert resp.status == 204
            assert "Access-Control-Allow-Origin" not in resp.headers
            assert "Access-Control-Allow-Private-Network" not in resp.headers


def test_server_serves_only_its_file_regardless_of_path(tmp_path):
    # No path-traversal surface: any request path returns the one served file.
    path = _write_mcap(tmp_path)
    with _RunningServer(path) as srv:
        status, _headers, body = _get(f"http://127.0.0.1:{srv.port}/../../etc/passwd")
    assert status == 200
    assert body == _BODY


@pytest.mark.skipif(not hasattr(__import__("os"), "symlink"), reason="requires symlink support")
def test_server_reads_held_fd_after_symlink_swap(tmp_path):
    # The server opens the file once and reads from that descriptor, so
    # repointing the symlink it was launched with cannot redirect requests to
    # a different file.
    import os

    real_a = _write_mcap(tmp_path, "a.mcap")
    real_b = tmp_path / "b.mcap"
    real_b.write_bytes(_MCAP_MAGIC + b"\x99" * 4096)  # different contents
    link = tmp_path / "link.mcap"
    try:
        os.symlink(real_a, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlink not permitted here")

    with _RunningServer(link) as srv:
        link.unlink()
        os.symlink(real_b, link)  # swap the name to a different inode
        status, _headers, body = _get(srv.url("link.mcap"))
    assert status == 200
    assert body == _BODY  # still the original file's bytes, not b.mcap's


# --- CLI command ----------------------------------------------------------


def test_view_rejects_non_mcap_file(tmp_path):
    bogus = tmp_path / "notes.txt"
    bogus.write_bytes(b"hello world, not an mcap")
    result = CliRunner().invoke(main, ["view", str(bogus), "--no-browser"])
    assert result.exit_code != 0
    assert "does not look like an MCAP file" in result.output


def test_view_rejects_partial_magic(tmp_path):
    # Full 8-byte magic is required — the 5-byte prefix alone is not enough.
    near = tmp_path / "near.mcap"
    near.write_bytes(b"\x89MCAP" + b"XYZ" + b"\x00" * 32)
    result = CliRunner().invoke(main, ["view", str(near), "--no-browser"])
    assert result.exit_code != 0
    assert "does not look like an MCAP file" in result.output


def test_view_rejects_bad_viewer_url(tmp_path, monkeypatch):
    path = _write_mcap(tmp_path)
    _patch_nonblocking_serve(monkeypatch)
    result = CliRunner().invoke(main, ["view", str(path), "--no-browser", "--viewer-url", "not-a-url"])
    assert result.exit_code != 0
    assert "--viewer-url" in result.output


def test_view_missing_file_errors():
    result = CliRunner().invoke(main, ["view", "/no/such/file.mcap", "--no-browser"])
    assert result.exit_code != 0


# The command and its module share the name ``view`` (same collision pattern as
# ``status.py``), so ``avala.cli.view`` resolves to the *command*. Reach the
# real module through importlib to monkeypatch its ``serve_file`` global.
def _view_module():
    import importlib

    return importlib.import_module("avala.cli.view")


def _patch_nonblocking_serve(monkeypatch):
    """Patch ``serve_file`` so the command never blocks in ``serve_forever``."""
    view_mod = _view_module()
    real_serve_file = view_mod.serve_file

    def _fast_serve(file_path, *, port=0, host="127.0.0.1", allowed_origin=None):
        server = real_serve_file(file_path, port=port, host=host, allowed_origin=allowed_origin)
        # Return immediately from the run loop, and make the matching shutdown
        # a no-op — an unstarted BaseServer.shutdown() would otherwise block
        # forever waiting on an event serve_forever never sets.
        server.serve_forever = lambda *a, **k: None  # type: ignore[method-assign]
        server.shutdown = lambda *a, **k: None  # type: ignore[method-assign]
        return server

    monkeypatch.setattr(view_mod, "serve_file", _fast_serve)


def test_view_no_browser_prints_viewer_url(tmp_path, monkeypatch):
    path = _write_mcap(tmp_path)
    _patch_nonblocking_serve(monkeypatch)

    result = CliRunner().invoke(main, ["view", str(path), "--no-browser", "--viewer-url", "https://avala.ai"])
    assert result.exit_code == 0, result.output
    assert "/mcap?" in result.output
    assert "utm_source=cli" in result.output
    assert "serving" in result.output


def test_view_force_serves_non_mcap(tmp_path, monkeypatch):
    bogus = tmp_path / "raw.bin"
    bogus.write_bytes(b"not mcap at all")
    _patch_nonblocking_serve(monkeypatch)

    result = CliRunner().invoke(main, ["view", str(bogus), "--no-browser", "--force"])
    assert result.exit_code == 0, result.output
    assert "/mcap?" in result.output
