"""``avala view`` — open a local MCAP file in the Mission Control viewer.

This is the local-viewer funnel wedge: a robotics engineer with no Avala
account can visualize a ``.mcap`` recording straight from the terminal. The
command starts a tiny localhost HTTP server that streams the file over HTTP
range requests and points a browser at the hosted, no-account Mission Control
MCAP viewer (``/mcap``). The file never leaves the machine — the viewer reads
it back over ``127.0.0.1`` — and no API key is required.

Security notes:

* The server binds to loopback only and serves exactly one pre-opened file. It
  never maps a request path onto the filesystem, so there is no path-traversal
  surface (CWE-22).
* Cross-origin reads from the hosted viewer are allowed via permissive CORS.
  Because only a single, user-chosen local file is exposed on an ephemeral
  loopback port, ``Access-Control-Allow-Origin: *`` is acceptable here.
"""

from __future__ import annotations

import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast
from urllib.parse import urlencode, urlparse

import click

# Every MCAP file begins with this exact 8-byte magic.
_MCAP_MAGIC = b"\x89MCAP0\r\n"

# Default host for the Mission Control web app that serves the ``/mcap`` viewer.
# Overridable so self-hosted / staging deployments and local MC builds work.
DEFAULT_VIEWER_URL = "https://avala.ai"

# Attribution params so the web app can credit sessions to the CLI funnel.
_FUNNEL_PARAMS = {
    "utm_source": "cli",
    "utm_medium": "avala_view",
    "utm_campaign": "local_viewer",
}


_DEFAULT_PORTS = {"http": 80, "https": 443}


def _origin_of(url: str) -> str:
    """Return the normalized ``scheme://host[:port]`` origin of *url*.

    Empty string when *url* has no scheme/host (or an out-of-range port). The
    port is dropped when it's the scheme default and the host is lowercased,
    matching how browsers serialize the ``Origin`` header — otherwise
    ``https://host:443`` would never equal the browser's ``https://host``.
    """
    parsed = urlparse(url)
    try:
        port = parsed.port
    except ValueError:
        # Out-of-range / non-numeric port (e.g. ``:99999``) — treat as invalid.
        return ""
    if not parsed.scheme or not parsed.hostname:
        return ""
    scheme = parsed.scheme.lower()
    host = parsed.hostname.lower()
    if ":" in host:
        # IPv6 literal — browsers keep the brackets in the Origin header.
        host = f"[{host}]"
    if port is None or port == _DEFAULT_PORTS.get(scheme):
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"


# Opaque path the loopback file is served under. The local server ignores the
# request path (it always serves the one file), so this reveals nothing — unlike
# the real basename, which can carry customer/route/project identifiers.
_SERVED_PATH = "recording.mcap"


def build_viewer_url(viewer_base: str, port: int, *, host: str = "127.0.0.1") -> str:
    """Compose the hosted-viewer URL that auto-loads the locally served file.

    The Mission Control ``/mcap`` route reads ``?url=`` (the file to fetch); the
    ``utm_*`` params ride along for funnel attribution and are ignored by the
    route parser. The local basename is deliberately kept out of the query so it
    never reaches the hosted app / edge logs — only the loopback URL is sent.
    """
    local_url = f"http://{host}:{port}/{_SERVED_PATH}"
    params = {"url": local_url, **_FUNNEL_PARAMS}
    return f"{viewer_base.rstrip('/')}/mcap?{urlencode(params)}"


class _McapFileServer(ThreadingHTTPServer):
    """A loopback HTTP server that streams exactly one MCAP file.

    ``allowed_origin`` scopes cross-origin (CORS + Private Network Access) grants
    to a single web origin — the hosted viewer. ``None`` means "no browser is
    expected" and disables CORS entirely; ``"*"`` allows any origin (test-only).
    """

    daemon_threads = True

    def __init__(self, address: tuple[str, int], file_path: Path, allowed_origin: str | None) -> None:
        super().__init__(address, _McapRangeHandler)
        self.file_name = file_path.name
        self.allowed_origin = allowed_origin
        # Open the file once and serve every request from this descriptor for the
        # server's lifetime. Reading through the fd (never re-opening by path)
        # means a later symlink/path swap cannot redirect requests to a different
        # file, and keeps size/etag consistent with the bytes actually served.
        # O_BINARY is a no-op on POSIX but stops Windows from translating \r\n.
        self._fd = os.open(str(file_path), os.O_RDONLY | getattr(os, "O_BINARY", 0))
        stat = os.fstat(self._fd)
        self.file_size = stat.st_size
        self.etag = f'"{stat.st_size:x}-{int(stat.st_mtime):x}"'
        self._read_lock = threading.Lock()

    def read_at(self, offset: int, length: int) -> bytes:
        """Thread-safe positional read from the held descriptor."""
        if hasattr(os, "pread"):
            return os.pread(self._fd, length, offset)
        # Windows lacks pread: serialize seek+read across worker threads.
        with self._read_lock:
            os.lseek(self._fd, offset, os.SEEK_SET)
            return os.read(self._fd, length)

    def server_close(self) -> None:
        try:
            super().server_close()
        finally:
            try:
                os.close(self._fd)
            except OSError:
                pass


class _McapRangeHandler(BaseHTTPRequestHandler):
    """Serves the server's single file, honoring HTTP range requests + CORS."""

    protocol_version = "HTTP/1.1"

    # -- helpers -----------------------------------------------------------

    @property
    def _server(self) -> _McapFileServer:
        return cast(_McapFileServer, self.server)

    def _allowed_cors_origin(self) -> str | None:
        """Return the value to echo in ``Access-Control-Allow-Origin``, or None.

        Only the configured viewer origin is honored — reflecting an arbitrary
        ``Origin`` would let any HTTPS page that guesses the ephemeral port read
        the local recording once it clears the PNA preflight.
        """
        request_origin: str | None = self.headers.get("Origin")
        allowed = self._server.allowed_origin
        if not request_origin or allowed is None:
            return None
        if allowed == "*" or request_origin == allowed:
            return request_origin
        return None

    def _send_cors_headers(self) -> None:
        origin = self._allowed_cors_origin()
        if origin is None:
            return
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Range")
        self.send_header("Access-Control-Expose-Headers", "Content-Range, Content-Length, Accept-Ranges, ETag")

    def _parse_range(self, header: str, size: int) -> tuple[int, int] | None:
        """Parse a single ``bytes=`` range into an inclusive ``(start, end)``.

        Returns None when the header is malformed or unsatisfiable. Only the
        first range of a (rare) multi-range request is honored.
        """
        if not header.startswith("bytes="):
            return None
        spec = header[len("bytes=") :].split(",", 1)[0].strip()
        start_s, sep, end_s = spec.partition("-")
        if not sep:
            return None
        try:
            if not start_s:
                # Suffix range: ``bytes=-N`` -> last N bytes.
                n = int(end_s)
                if n <= 0:
                    return None
                start = max(0, size - n)
                end = size - 1
            else:
                start = int(start_s)
                end = int(end_s) if end_s else size - 1
        except ValueError:
            return None
        end = min(end, size - 1)
        if start < 0 or end < 0 or start > end or start >= size:
            return None
        return start, end

    # -- verb handlers -----------------------------------------------------

    def do_OPTIONS(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler dispatch)
        self.send_response(204)
        self._send_cors_headers()
        # Private Network Access: an HTTPS page fetching a loopback address makes
        # Chromium send a preflight with ``Access-Control-Request-Private-Network``.
        # Grant it only for the allowed viewer origin (see _allowed_cors_origin),
        # never to an arbitrary site that guessed the port.
        pna_requested = self.headers.get("Access-Control-Request-Private-Network") == "true"
        if pna_requested and self._allowed_cors_origin() is not None:
            self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_HEAD(self) -> None:  # noqa: N802
        server = self._server
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(server.file_size))
        self.send_header("ETag", server.etag)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        server = self._server
        size = server.file_size
        range_header = self.headers.get("Range")

        if range_header:
            parsed = self._parse_range(range_header, size)
            if parsed is None:
                self.send_response(416)
                self._send_cors_headers()
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            start, end = parsed
            length = end - start + 1
            self.send_response(206)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(length))
            self.send_header("ETag", server.etag)
            self.end_headers()
            self._stream(start, length)
            return

        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(size))
        self.send_header("ETag", server.etag)
        self.end_headers()
        self._stream(0, size)

    def _stream(self, start: int, length: int) -> None:
        """Write *length* bytes from *start* to the client in bounded chunks."""
        chunk_size = 1 << 20  # 1 MiB
        remaining = length
        offset = start
        while remaining > 0:
            chunk = self._server.read_at(offset, min(chunk_size, remaining))
            if not chunk:
                break
            try:
                self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                # Viewer aborted this range (normal during seeking) — stop.
                return
            offset += len(chunk)
            remaining -= len(chunk)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 (match base signature)
        """Silence the default per-request stderr logging."""
        del format, args


def serve_file(
    file_path: Path,
    *,
    port: int = 0,
    host: str = "127.0.0.1",
    allowed_origin: str | None = None,
) -> _McapFileServer:
    """Create (but do not run) a loopback server for *file_path*.

    Pass ``port=0`` to bind an ephemeral port; read the chosen port back from
    ``server.server_address[1]``. ``allowed_origin`` restricts CORS / Private
    Network Access grants to a single web origin (the hosted viewer). The caller
    owns ``serve_forever`` / shutdown.
    """
    return _McapFileServer((host, port), file_path, allowed_origin)


@click.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--port", type=int, default=0, help="Local port to serve on (default: an ephemeral free port).")
@click.option(
    "--viewer-url",
    envvar="AVALA_VIEWER_URL",
    default=DEFAULT_VIEWER_URL,
    help="Base URL of the Mission Control web app (or set AVALA_VIEWER_URL).",
)
@click.option("--no-browser", is_flag=True, default=False, help="Print the viewer URL instead of opening a browser.")
@click.option("--force", is_flag=True, default=False, help="Serve the file even if it doesn't look like MCAP.")
def view(file: Path, port: int, viewer_url: str, no_browser: bool, force: bool) -> None:
    """Open a local MCAP FILE in the Mission Control viewer — no account required.

    Starts a loopback web server that streams FILE over HTTP range requests and
    opens the hosted, no-login MCAP viewer pointed at it. The file stays on your
    machine; press Ctrl+C to stop serving.
    """
    # Only the hosted viewer's own origin may read the file cross-origin.
    allowed_origin = _origin_of(viewer_url)
    if not allowed_origin:
        raise click.ClickException(f"--viewer-url must be an absolute URL with a scheme and host; got: {viewer_url}")

    server = serve_file(file, port=port, allowed_origin=allowed_origin)
    # Validate the magic through the server's own descriptor, not a fresh open of
    # the path — that's the exact fd we'll serve, so a symlink swap between check
    # and serve can't slip a different file past validation.
    if not force and server.read_at(0, len(_MCAP_MAGIC)) != _MCAP_MAGIC:
        server.server_close()
        raise click.ClickException(
            f"{file} does not look like an MCAP file (missing \\x89MCAP magic). Pass --force to serve it anyway."
        )
    bound_host, bound_port = str(server.server_address[0]), int(server.server_address[1])
    url = build_viewer_url(viewer_url, bound_port, host=bound_host)

    click.echo(f"→ serving {click.style(file.name, bold=True)} on http://{bound_host}:{bound_port} (local only)")
    click.echo(f"→ opening {click.style(url, fg='cyan')}")
    if no_browser:
        click.echo("  (--no-browser: open the URL above manually)")
    else:
        # webbrowser can print noise to stdout on some Linux setups; hush it.
        opened = False
        try:
            _devnull = os.open(os.devnull, os.O_RDWR)
            saved_out, saved_err = os.dup(1), os.dup(2)
            os.dup2(_devnull, 1)
            os.dup2(_devnull, 2)
            try:
                opened = webbrowser.open(url)
            finally:
                os.dup2(saved_out, 1)
                os.dup2(saved_err, 2)
                for _fd in (_devnull, saved_out, saved_err):
                    os.close(_fd)
        except OSError:
            opened = False
        if not opened:
            click.echo("  (couldn't launch a browser — open the URL above manually)")

    click.echo("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("\nStopped.")
    finally:
        server.shutdown()
        server.server_close()
