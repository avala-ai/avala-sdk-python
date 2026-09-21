"""Guarded Fleet upload consumer for the current unpaginated server protocol."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlsplit

import httpx

from avala._fleet_uploads import (
    SourceFile,
    SourceInventory,
    UploadBinding,
    _failure,
    _open_regular_file,
    fleet_checkpoint,
    upload_binding,
)
from avala._uploads import validate_presigned_url
from avala.errors import NotFoundError, UploadStateError
from avala.types.fleet_upload import (
    UploadProgress,
    UploadSession,
    UploadStatusResponse,
    UploadUrlEntry,
    UploadUrlsResponse,
)

if TYPE_CHECKING:
    from avala.resources.fleet.uploads import FleetUploadManager

_QUIET_HTTP: ContextVar[bool] = ContextVar("fleet_private_http", default=False)
_COUNTS = {"total_files", "total_bytes", "confirmed_files", "confirmed_bytes"}
_STATUS_FIELDS = _COUNTS | {"session_uid", "status", "pending_paths"}
_STATES = {"initiated", "uploading", "completing", "completed", "abandoned"}
_CHUNK_SIZE = 1024 * 1024
_REGION = r"[a-z]{2}(?:-[a-z0-9]+)+-\d+"
_PATH_STYLE_HOST = re.compile(
    rf"(?:s3(?:-external-1)?|s3(?:-fips)?(?:\.dualstack)?\.{_REGION}|s3-{_REGION})\.amazonaws\.com\Z"
)


class _PrivateHTTPLogs(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not _QUIET_HTTP.get()


# HTTPX logs signed request URLs; HTTPcore debug records can include headers.
# The filter applies only inside this consumer's synchronous transfer context,
# including worker threads, and does not change other callers' logger levels.
for _name in ("httpx", "httpcore.connection", "httpcore.http11", "httpcore.http2", "httpcore.proxy", "httpcore.socks"):
    logging.getLogger(_name).addFilter(_PrivateHTTPLogs())


def _assert_source(manager: FleetUploadManager, inventory: SourceInventory) -> None:
    if manager.collect_files(inventory.source) != inventory:
        raise _failure("source changed after upload initialization")


def _check_counts(data: dict[str, Any], inventory: SourceInventory, confirmed: set[str] | None = None) -> None:
    if not _COUNTS <= data.keys() or any(type(data[key]) is not int or data[key] < 0 for key in _COUNTS):
        raise _failure("server counters are missing or invalid")
    if data["total_files"] != len(inventory.files) or data["total_bytes"] != inventory.total_bytes:
        raise _failure("server totals differ from the retained manifest")
    if data["confirmed_files"] > len(inventory.files) or data["confirmed_bytes"] > inventory.total_bytes:
        raise _failure("server confirmation counters exceed the manifest")
    if confirmed is not None and (
        data["confirmed_files"] != len(confirmed)
        or data["confirmed_bytes"] != sum(file.size_bytes for file in inventory.files if file.path in confirmed)
    ):
        raise _failure("server confirmation counters disagree with file evidence")


def _reconcile(status: UploadStatusResponse, session_uid: str, inventory: SourceInventory) -> set[str]:
    if (
        not _STATUS_FIELDS <= status.model_fields_set
        or status.session_uid != session_uid
        or status.status not in _STATES
    ):
        raise _failure("server status is incomplete or names a different session")
    pending = status.pending_paths
    if len(pending) >= 1000:
        raise _failure("server pending inventory reached its 1000-path cap; safe resume requires complete pagination")
    paths = {file.path for file in inventory.files}
    if len(set(pending)) != len(pending) or not set(pending) <= paths:
        raise _failure("server pending inventory contains duplicate or unknown paths")
    confirmed = paths - set(pending)
    _check_counts(status.model_dump(), inventory, confirmed)
    if status.status == "abandoned" or (status.status in {"completing", "completed"} and pending):
        raise _failure("server session cannot accept upload work")
    return confirmed


def _grants(response: UploadUrlsResponse, files: list[SourceFile], prefix: str) -> dict[str, UploadUrlEntry]:
    if "urls" not in response.model_fields_set or len(response.urls) != len(files):
        raise _failure("server did not return the complete presign batch")
    paths = {file.path for file in files}
    result: dict[str, UploadUrlEntry] = {}
    for grant in response.urls:
        if not {"path", "put_url", "s3_key", "headers"} <= grant.model_fields_set:
            raise _failure("presign entry is incomplete")
        if grant.path not in paths or grant.path in result or grant.s3_key != prefix + grant.path:
            raise _failure("presign entries do not match the requested files")
        validate_presigned_url(grant.put_url)
        url = urlsplit(grant.put_url)
        raw_path = url.path
        effective_path = httpx.URL(grant.put_url).raw_path.split(b"?", 1)[0].decode("ascii")
        expected_path = "/" + quote(grant.s3_key, safe="/")
        path_style = bool(url.hostname and _PATH_STYLE_HOST.fullmatch(url.hostname))
        bucket_path = raw_path[: -len(expected_path)] if raw_path.endswith(expected_path) else ""
        if raw_path != effective_path or not (
            (not path_style and raw_path == expected_path)
            or (path_style and re.fullmatch(r"/[A-Za-z0-9._-]+", bucket_path))
        ):
            raise _failure("presign URL does not name its declared object key")
        headers = {key.lower(): value for key, value in grant.headers.items()}
        if (
            len(headers) != len(grant.headers)
            or any(
                key not in {"content-type", "cache-control", "content-encoding"} or "\r" in value or "\n" in value
                for key, value in headers.items()
            )
            or headers.get("content-type", "application/octet-stream") != "application/octet-stream"
            or (headers.get("content-encoding", "gzip") != "gzip")
            or headers.get("cache-control", "no-cache") not in {"no-cache", "private, immutable, max-age=31536000"}
        ):
            raise _failure("presign headers are not supported by the Fleet protocol")
        result[grant.path] = grant
    return result


def _put_file(source: Path, file: SourceFile, grant: UploadUrlEntry) -> str:
    token = _QUIET_HTTP.set(True)
    try:
        for attempt in range(3):
            try:
                with _open_regular_file(source / file.path) as handle:
                    before = hashlib.sha256()
                    while chunk := handle.read(_CHUNK_SIZE):
                        before.update(chunk)
                    if before.hexdigest() != file.sha256:
                        raise _failure("source content changed before transfer")
                    handle.seek(0)
                    digest = hashlib.sha256()
                    consumed = 0
                    exhausted = False

                    def chunks() -> Iterator[bytes]:
                        nonlocal consumed, exhausted
                        while chunk := handle.read(_CHUNK_SIZE):
                            consumed += len(chunk)
                            digest.update(chunk)
                            yield chunk
                        exhausted = True

                    headers = dict(grant.headers)
                    if not any(key.lower() == "content-type" for key in headers):
                        headers["Content-Type"] = "application/octet-stream"
                    headers["Content-Length"] = str(file.size_bytes)
                    response = httpx.put(
                        grant.put_url, content=chunks(), headers=headers, timeout=300.0, follow_redirects=False
                    )
                    response.raise_for_status()
                    if not exhausted or consumed != file.size_bytes or digest.hexdigest() != file.sha256:
                        raise _failure("transferred source bytes differ from the retained manifest")
                    etag: str = response.headers.get("ETag", "").strip('"')
                    if not etag or len(etag) > 128 or etag != etag.strip() or any(ord(char) < 32 for char in etag):
                        raise _failure("storage returned no usable upload receipt")
                return etag
            except (httpx.TransportError, httpx.HTTPStatusError) as error:
                retryable = isinstance(error, httpx.TransportError) or error.response.status_code in {
                    429,
                    500,
                    502,
                    503,
                    504,
                }
                if not retryable or attempt == 2:
                    break
                time.sleep(2**attempt)
        raise _failure("file transfer failed; no successful receipt was confirmed")
    finally:
        _QUIET_HTTP.reset(token)


def _result(status: UploadStatusResponse) -> UploadSession:
    return UploadSession(uid=status.session_uid, **status.model_dump(exclude={"session_uid", "pending_paths"}))


class _UploadObserver:
    """Reuse one source inventory only for read-only status observations.

    Intermediate polls do not establish current source integrity. A terminal
    caller must request verification before announcing readiness. Every poll
    rereads the retained receipt and checks the live destination and credentials.
    """

    def __init__(
        self,
        manager: FleetUploadManager,
        recording_uid: str,
        session_uid: str,
        source_dir: str | Path,
        storage_config_uid: str | None,
        state_dir: Path,
    ) -> None:
        self._manager = manager
        self._inventory = manager.collect_files(source_dir)
        self._recording_uid = recording_uid
        self._session_uid = session_uid
        self._storage_config_uid = storage_config_uid
        self._state_dir = state_dir
        self._binding = self._current_binding()

    def _current_binding(self) -> UploadBinding:
        return upload_binding(
            self._inventory,
            base_url=self._manager._transport.base_url,
            api_key=self._manager._transport.api_key,
            recording_uid=self._recording_uid,
            storage_config_uid=self._storage_config_uid,
        )

    def _assert_binding(self) -> None:
        if self._current_binding() != self._binding:
            raise _failure("polling destination or credentials changed")

    def poll(self, *, verify_source: bool = True) -> UploadStatusResponse:
        """Observe status, optionally verifying source before terminal success."""
        token = _QUIET_HTTP.set(True)
        try:
            return self._poll(verify_source=verify_source)
        except UploadStateError:
            raise
        except Exception:
            raise _failure("retained upload status could not be verified") from None
        finally:
            _QUIET_HTTP.reset(token)

    def _poll(self, *, verify_source: bool) -> UploadStatusResponse:
        self._assert_binding()
        with fleet_checkpoint(self._state_dir, self._binding) as checkpoint:
            state = checkpoint.read()
            if (
                state is None
                or state.session_uid != self._session_uid
                or state.phase not in {"finalizing", "completed"}
            ):
                raise _failure("polling requires the exact retained finalization receipt")
            status = self._manager.get_upload_status(self._binding.recording_uid)
            _reconcile(status, self._session_uid, self._inventory)
            if status.status not in {"completing", "completed"} or (
                state.phase == "completed" and status.status != "completed"
            ):
                raise _failure("retained finalization has no matching server outcome")
            self._assert_binding()
            if verify_source and status.status == "completed":
                _assert_source(self._manager, self._inventory)
                self._assert_binding()
                if state.phase == "finalizing" and status.status == "completed":
                    checkpoint.write("completed", self._session_uid)
            return status


def run_upload(
    manager: FleetUploadManager,
    recording_uid: str,
    source_dir: str | Path,
    storage_config_uid: str | None,
    max_workers: int,
    on_progress: Callable[[UploadProgress], None] | None,
    state_dir: Path,
) -> UploadSession:
    if type(max_workers) is not int or max_workers <= 0:
        raise _failure("max_workers must be a positive integer")
    inventory = manager.collect_files(source_dir)
    binding = upload_binding(
        inventory,
        base_url=manager._transport.base_url,
        api_key=manager._transport.api_key,
        recording_uid=recording_uid,
        storage_config_uid=storage_config_uid,
    )
    recording_uid = binding.recording_uid
    with fleet_checkpoint(state_dir, binding) as checkpoint:
        state = checkpoint.read()
        if state is not None and state.phase == "initializing":
            raise _failure("previous initialization has no safely retained server response")
        confirmed: set[str] = set()
        if state is None:
            try:
                manager.get_upload_status(recording_uid)
            except NotFoundError as error:
                if error.body != {"detail": "No upload session found for this recording."}:
                    raise _failure("server did not establish absence of an earlier session") from None
            else:
                raise _failure("server already has a session without matching local evidence")
            _assert_source(manager, inventory)
            checkpoint.write("initializing")
            initialized = manager.init_upload(
                recording_uid, inventory.manifest(), storage_config_uid=storage_config_uid
            )
            if (
                not {"uid", "total_files", "total_bytes", "s3_prefix", "status"} <= initialized.model_fields_set
                or initialized.total_files != len(inventory.files)
                or initialized.total_bytes != inventory.total_bytes
                or initialized.status != "initiated"
                or not isinstance(initialized.s3_prefix, str)
                or not initialized.s3_prefix.endswith("/")
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", initialized.uid)
            ):
                raise _failure("initialization response did not establish a matching session")
            session_uid = initialized.uid
            prefix = initialized.s3_prefix
            checkpoint.write("active", session_uid, s3_prefix=prefix)
        else:
            assert state.session_uid is not None
            session_uid = state.session_uid
            assert state.s3_prefix is not None
            prefix = state.s3_prefix
            status = manager.get_upload_status(recording_uid)
            confirmed = _reconcile(status, session_uid, inventory)
            if state.phase == "confirming":
                if not set(state.confirmation_paths) <= confirmed:
                    raise _failure(
                        "previous confirmation may still execute; pending status cannot authorize repeating it"
                    )
                checkpoint.write("active", session_uid)
                state = checkpoint.read()
                assert state is not None
            if state.phase in {"finalizing", "completed"} or status.status in {"completing", "completed"}:
                if status.status not in {"completing", "completed"} or (
                    state.phase == "completed" and status.status != "completed"
                ):
                    raise _failure("retained finalization has no matching server outcome")
                _assert_source(manager, inventory)
                if state.phase == "active":
                    checkpoint.write("finalizing", session_uid)
                if status.status == "completed" and state.phase != "completed":
                    checkpoint.write("completed", session_uid)
                return _result(status)

        remaining = [file for file in inventory.files if file.path not in confirmed]
        uploaded_bytes = 0
        for start in range(0, len(remaining), 100):
            files = remaining[start : start + 100]
            grants = _grants(
                manager.get_upload_urls(recording_uid, session_uid, [file.path for file in files]), files, prefix
            )
            accepted: list[dict[str, Any]] = []
            failed = 0
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_put_file, inventory.source, file, grants[file.path]): file for file in files}
                for future in as_completed(futures):
                    file = futures[future]
                    try:
                        etag = future.result()
                    except Exception:
                        failed += 1
                        continue
                    accepted.append({"path": file.path, "etag": etag, "size_bytes": file.size_bytes})
                    uploaded_bytes += file.size_bytes
            # Check this accepted batch again before confirming it. Re-reading
            # the entire recording per 100-file batch would be quadratic.
            for file in files:
                if not any(entry["path"] == file.path for entry in accepted):
                    continue
                digest = hashlib.sha256()
                with _open_regular_file(inventory.source / file.path) as handle:
                    while chunk := handle.read(_CHUNK_SIZE):
                        digest.update(chunk)
                if digest.hexdigest() != file.sha256:
                    raise _failure("source changed before file confirmation")
            if accepted:
                expected = confirmed | {entry["path"] for entry in accepted}
                checkpoint.write(
                    "confirming", session_uid, confirmation_paths=tuple(sorted(entry["path"] for entry in accepted))
                )
                receipt = manager.confirm_upload(recording_uid, session_uid, accepted)
                if not isinstance(receipt, dict) or receipt.get("session_uid") != session_uid:
                    raise _failure("confirmation did not acknowledge the retained session")
                _check_counts(receipt, inventory, expected)
                checkpoint.write("active", session_uid)
                confirmed = expected
            if on_progress:
                on_progress(
                    UploadProgress(
                        total_files=len(inventory.files),
                        uploaded_files=len(confirmed),
                        total_bytes=inventory.total_bytes,
                        uploaded_bytes=uploaded_bytes,
                        failed_files=failed,
                    )
                )
            if failed:
                raise _failure("one or more files failed; confirmed progress remains resumable")

        status = manager.get_upload_status(recording_uid)
        confirmed = _reconcile(status, session_uid, inventory)
        if len(confirmed) != len(inventory.files):
            raise _failure("server still has pending files; upload cannot finalize")
        _assert_source(manager, inventory)
        checkpoint.write("finalizing", session_uid)
        if status.status not in {"completing", "completed"}:
            acknowledgement = manager.finalize_upload(recording_uid, session_uid)
            if not isinstance(acknowledgement, dict) or not isinstance(acknowledgement.get("detail"), str):
                raise _failure("finalization acknowledgement is unavailable")
            if acknowledgement.get("session_uid", session_uid) != session_uid:
                raise _failure("finalization acknowledgement names another session")
            status = manager.get_upload_status(recording_uid)
            _reconcile(status, session_uid, inventory)
        if status.status not in {"completing", "completed"}:
            raise _failure("finalization has no confirmed server outcome")
        _assert_source(manager, inventory)
        if status.status == "completed":
            checkpoint.write("completed", session_uid)
        return _result(status)
