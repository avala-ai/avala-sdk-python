"""Fleet recording upload manager — resumable, per-file upload via presigned URLs."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from avala.resources._base import BaseSyncResource
from avala.types.fleet_upload import (
    UploadProgress,
    UploadSession,
    UploadStatusResponse,
    UploadUrlEntry,
    UploadUrlsResponse,
)

logger = logging.getLogger(__name__)

_STATE_DIR = Path.home() / ".avala" / "uploads"
_URL_BATCH_SIZE = 100
_CONFIRM_BATCH_SIZE = 500
_MAX_RETRIES = 3

# Allow-list of host suffixes that may receive file contents via presigned
# PUT URLs. Prevents a compromised/misbehaving control-plane response (or
# MITM) from redirecting local file bytes to an attacker-controlled host.
# Extend cautiously — each entry is a trust boundary for raw file data.
_PRESIGNED_URL_HOST_SUFFIXES = (
    ".amazonaws.com",  # S3 (any region) — e.g. s3.us-west-2.amazonaws.com
    ".storage.googleapis.com",  # GCS presigned URLs
    ".blob.core.windows.net",  # Azure Blob SAS URLs
)


def _validate_presigned_put_url(url: str) -> None:
    """Ensure a server-provided upload URL targets a known cloud-storage host.

    The SDK trusts the server to mint presigned URLs, but a hijacked
    control-plane response would otherwise exfiltrate raw file bytes to any
    host. This enforces `https://` and a host-suffix allow-list.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"Upload URL must use HTTPS, got scheme '{parsed.scheme}'.")
    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("Upload URL has no host.")
    if not any(host == suffix.lstrip(".") or host.endswith(suffix) for suffix in _PRESIGNED_URL_HOST_SUFFIXES):
        raise ValueError(
            f"Upload URL host '{host}' is not in the presigned-URL allow-list. "
            "Expected S3 (*.amazonaws.com), GCS (*.storage.googleapis.com), or Azure Blob "
            "(*.blob.core.windows.net)."
        )


class FleetUploadManager(BaseSyncResource):
    """Resumable upload manager for fleet recordings.

    Coordinates per-file uploads via server-generated presigned S3 PUT URLs.
    Tracks state locally for crash recovery and remotely for device replacement.
    """

    # -- High-level API --

    def upload_recording(
        self,
        recording_uid: str,
        source_dir: str | Path,
        *,
        storage_config_uid: str | None = None,
        max_workers: int = 4,
        on_progress: Callable[[UploadProgress], None] | None = None,
    ) -> UploadSession:
        """Upload a local directory to a fleet recording via resumable presigned URLs.

        Args:
            recording_uid: The recording UID to upload into.
            source_dir: Local directory containing files to upload.
            storage_config_uid: Storage config UID for S3 target (optional).
            max_workers: Parallel upload threads (default 4).
            on_progress: Callback invoked after each confirmation batch.

        Returns:
            The final UploadSession state.
        """
        source_path = Path(source_dir).resolve()
        if not source_path.is_dir():
            raise ValueError(f"Source directory does not exist: {source_dir}")

        # Build file manifest
        manifest: list[dict[str, Any]] = []
        for root, _, files in os.walk(source_path):
            for fname in files:
                local_path = Path(root) / fname
                relative = local_path.relative_to(source_path).as_posix()
                manifest.append({"path": relative, "size_bytes": local_path.stat().st_size})

        if not manifest:
            raise ValueError(f"No files found in {source_dir}")

        # Try to resume from local state
        state = self._load_local_state(recording_uid)
        session_uid: str | None = state.get("session_uid") if state else None
        confirmed_set: set[str] = set(state.get("confirmed", [])) if state else set()

        # Reconcile with server if resuming
        if session_uid:
            try:
                server_status = self.get_upload_status(recording_uid)
                confirmed_set.update(
                    p
                    for p in (manifest_entry["path"] for manifest_entry in manifest)
                    if p not in server_status.pending_paths
                )
                session_uid = server_status.session_uid
            except Exception:
                session_uid = None
                confirmed_set = set()

        # Init new session if needed
        if not session_uid:
            session = self.init_upload(recording_uid, manifest, storage_config_uid=storage_config_uid)
            session_uid = session.uid
            confirmed_set = set()

        # Save initial state
        self._save_local_state(recording_uid, session_uid, source_path, confirmed_set, len(manifest))

        # Filter to remaining files
        remaining = [m for m in manifest if m["path"] not in confirmed_set]

        # Upload in batches
        uploaded_bytes = 0
        failed_count = 0

        for batch_start in range(0, len(remaining), _URL_BATCH_SIZE):
            batch = remaining[batch_start : batch_start + _URL_BATCH_SIZE]
            batch_paths = [f["path"] for f in batch]

            # Get presigned URLs
            urls_response = self.get_upload_urls(recording_uid, session_uid, batch_paths)
            url_map = {u.path: u for u in urls_response.urls}

            # Upload in parallel
            batch_confirmed: list[dict[str, Any]] = []

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {}
                for file_info in batch:
                    path = file_info["path"]
                    url_entry = url_map.get(path)
                    if not url_entry:
                        failed_count += 1
                        logger.warning("No presigned URL returned for %s, skipping", path)
                        continue
                    local_path = source_path / path
                    futures[pool.submit(self._upload_file, local_path, url_entry)] = file_info

                for future in as_completed(futures):
                    file_info = futures[future]
                    try:
                        etag = future.result()
                        batch_confirmed.append(
                            {
                                "path": file_info["path"],
                                "etag": etag,
                                "size_bytes": file_info["size_bytes"],
                            }
                        )
                        uploaded_bytes += file_info["size_bytes"]
                    except Exception as exc:
                        failed_count += 1
                        logger.warning("Failed to upload %s: %s", file_info["path"], exc)

            # Confirm batch with server
            if batch_confirmed:
                for confirm_start in range(0, len(batch_confirmed), _CONFIRM_BATCH_SIZE):
                    confirm_batch = batch_confirmed[confirm_start : confirm_start + _CONFIRM_BATCH_SIZE]
                    self.confirm_upload(recording_uid, session_uid, confirm_batch)

                confirmed_set.update(f["path"] for f in batch_confirmed)
                self._save_local_state(recording_uid, session_uid, source_path, confirmed_set, len(manifest))

            # Progress callback
            if on_progress:
                on_progress(
                    UploadProgress(
                        total_files=len(manifest),
                        uploaded_files=len(confirmed_set),
                        total_bytes=sum(m["size_bytes"] for m in manifest),
                        uploaded_bytes=uploaded_bytes,
                        failed_files=failed_count,
                    )
                )

        # Finalize if all confirmed
        if failed_count == 0:
            self.finalize_upload(recording_uid, session_uid)
            self._cleanup_local_state(recording_uid)

        # Return final session state
        status = self.get_upload_status(recording_uid)
        return UploadSession(
            uid=session_uid,
            total_files=status.total_files,
            total_bytes=status.total_bytes,
            confirmed_files=status.confirmed_files,
            confirmed_bytes=status.confirmed_bytes,
            s3_prefix=None,
            status=status.status,
        )

    # -- Low-level API (thin wrappers around server endpoints) --

    def init_upload(
        self,
        recording_uid: str,
        files: list[dict[str, Any]],
        *,
        storage_config_uid: str | None = None,
    ) -> UploadSession:
        """Initialize an upload session."""
        payload: dict[str, Any] = {"files": files}
        if storage_config_uid:
            payload["storage_config_uid"] = storage_config_uid
        data = self._transport.request("POST", f"/fleet/recordings/{recording_uid}/upload/init/", json=payload)
        return UploadSession.model_validate(data)

    def get_upload_urls(
        self, recording_uid: str, session_uid: str, file_paths: list[str], *, ttl_seconds: int = 3600
    ) -> UploadUrlsResponse:
        """Get presigned PUT URLs for a batch of files."""
        payload = {"session_uid": session_uid, "file_paths": file_paths, "ttl_seconds": ttl_seconds}
        data = self._transport.request("POST", f"/fleet/recordings/{recording_uid}/upload/urls/", json=payload)
        return UploadUrlsResponse.model_validate(data)

    def confirm_upload(self, recording_uid: str, session_uid: str, files: list[dict[str, Any]]) -> dict[str, Any]:
        """Confirm files have been uploaded."""
        payload = {"session_uid": session_uid, "files": files}
        data = self._transport.request("POST", f"/fleet/recordings/{recording_uid}/upload/confirm/", json=payload)
        return data  # type: ignore[no-any-return]

    def finalize_upload(self, recording_uid: str, session_uid: str) -> dict[str, Any]:
        """Finalize the upload session, triggering server-side processing."""
        payload = {"session_uid": session_uid}
        data = self._transport.request("POST", f"/fleet/recordings/{recording_uid}/upload/finalize/", json=payload)
        return data  # type: ignore[no-any-return]

    def get_upload_status(self, recording_uid: str) -> UploadStatusResponse:
        """Get current upload progress."""
        data = self._transport.request("GET", f"/fleet/recordings/{recording_uid}/upload/status/")
        return UploadStatusResponse.model_validate(data)

    # -- Internal helpers --

    @staticmethod
    def _upload_file(local_path: Path, url_entry: UploadUrlEntry) -> str:
        """Upload a single file to S3 via presigned PUT URL. Returns the ETag."""
        import httpx

        _validate_presigned_put_url(url_entry.put_url)

        headers = dict(url_entry.headers)
        file_size = local_path.stat().st_size
        headers["Content-Length"] = str(file_size)

        for attempt in range(_MAX_RETRIES):
            try:
                with open(local_path, "rb") as f:
                    resp = httpx.put(url_entry.put_url, content=f, headers=headers, timeout=300.0)
                resp.raise_for_status()
                etag: str = resp.headers.get("ETag", "")
                return etag.strip('"')
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                if attempt == _MAX_RETRIES - 1:
                    raise
                wait = 2**attempt
                logger.debug("Retry %d for %s after %s (wait %ds)", attempt + 1, local_path.name, exc, wait)
                time.sleep(wait)

        raise RuntimeError(f"Upload failed after {_MAX_RETRIES} retries: {local_path}")

    @staticmethod
    def _load_local_state(recording_uid: str) -> dict[str, Any] | None:
        """Load local upload state from disk."""
        state_file = _STATE_DIR / f"{recording_uid}.json"
        if not state_file.exists():
            return None
        try:
            return json.loads(state_file.read_text())  # type: ignore[no-any-return]
        except (json.JSONDecodeError, OSError):
            return None

    @staticmethod
    def _save_local_state(
        recording_uid: str,
        session_uid: str,
        source_dir: str | Path,
        confirmed: set[str],
        total_files: int,
    ) -> None:
        """Save local upload state to disk for crash recovery."""
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        state = {
            "session_uid": session_uid,
            "recording_uid": recording_uid,
            "source_dir": str(source_dir),
            "confirmed": sorted(confirmed),
            "total_files": total_files,
        }
        state_file = _STATE_DIR / f"{recording_uid}.json"
        tmp_file = state_file.with_suffix(".tmp")
        tmp_file.write_text(json.dumps(state, indent=2))
        tmp_file.replace(state_file)  # atomic on POSIX

    @staticmethod
    def _cleanup_local_state(recording_uid: str) -> None:
        """Remove local state file after successful upload."""
        state_file = _STATE_DIR / f"{recording_uid}.json"
        state_file.unlink(missing_ok=True)
