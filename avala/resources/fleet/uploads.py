"""Fleet recording uploads with retained source and session evidence."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from avala._fleet_uploads import SourceInventory, _failure, collect_source
from avala._uploads import STATE_DIR, validate_presigned_url
from avala.errors import UploadStateError
from avala.resources._base import BaseSyncResource
from avala.types.fleet_upload import UploadProgress, UploadSession, UploadStatusResponse, UploadUrlsResponse
from avala.types.fleet_managed_upload import ManagedFleetUploadStatus

if TYPE_CHECKING:
    from avala.resources.fleet._upload_guard import _UploadObserver

_STATE_DIR = STATE_DIR
_validate_presigned_put_url = validate_presigned_url


class FleetUploadManager(BaseSyncResource):
    """Upload a bound source without replacing ambiguous server work."""

    def collect_files(self, source_dir: str | Path) -> SourceInventory:
        """Return the same safe local inventory used by upload and CLI preview."""
        return collect_source(source_dir, state_dir=_STATE_DIR)

    def upload_managed_recording(
        self,
        recording_uid: str,
        source_dir: str | Path,
        *,
        max_workers: int = 2,
        wait_timeout: float = 0,
    ) -> ManagedFleetUploadStatus:
        """Explicitly upload/resume managed MCAP originals for an enrolled organization.

        Returns the exact pending, failed or published finalization snapshot.
        ``wait_timeout`` bounds optional publication polling in seconds; zero
        returns after the first observation. Publication does not imply that
        the Fleet recording is READY. Retain the receipt and original files.
        """
        from avala.resources.fleet._managed_consumer import run_managed_upload
        from avala.resources.fleet._upload_guard import _QUIET_HTTP

        token = _QUIET_HTTP.set(True)
        message = str(
            _failure("managed upload could not establish exact evidence; retain the session and original files")
        )
        try:
            return run_managed_upload(
                self,
                recording_uid,
                source_dir,
                max_workers=max_workers,
                wait_timeout=wait_timeout,
                state_dir=_STATE_DIR,
            )
        except UploadStateError as error:
            message = str(error)
        except Exception:
            pass
        finally:
            _QUIET_HTTP.reset(token)
        # A retained internal traceback can expose signed grants in frame locals.
        # Only safe diagnostic text crosses the public boundary.
        raise UploadStateError(message)

    def upload_recording(
        self,
        recording_uid: str,
        source_dir: str | Path,
        *,
        storage_config_uid: str | None = None,
        max_workers: int = 4,
        on_progress: Callable[[UploadProgress], None] | None = None,
    ) -> UploadSession:
        """Upload or resume an exact source; retain receipts through completion.

        An ambiguous initialization, changed inputs or incomplete server evidence
        raises UploadStateError. A completing result means processing is queued,
        not that the recording is ready. Local receipts are deliberately retained.
        """
        from avala.resources.fleet._upload_guard import _QUIET_HTTP, run_upload

        token = _QUIET_HTTP.set(True)
        try:
            return run_upload(self, recording_uid, source_dir, storage_config_uid, max_workers, on_progress, _STATE_DIR)
        except UploadStateError:
            raise
        except Exception:
            raise _failure("upload request failed; the retained receipt must be reconciled before retry") from None
        finally:
            _QUIET_HTTP.reset(token)

    def _observe_upload(
        self, recording_uid: str, session_uid: str, source_dir: str | Path, *, storage_config_uid: str | None = None
    ) -> UploadStatusResponse:
        """Observe retained finalization without any remote upload mutations."""
        return self._upload_observer(
            recording_uid, session_uid, source_dir, storage_config_uid=storage_config_uid
        ).poll()

    def _upload_observer(
        self, recording_uid: str, session_uid: str, source_dir: str | Path, *, storage_config_uid: str | None = None
    ) -> _UploadObserver:
        """Bind a read-only wait to one validated inventory and destination."""
        from avala.resources.fleet._upload_guard import _QUIET_HTTP, _UploadObserver

        token = _QUIET_HTTP.set(True)
        try:
            return _UploadObserver(self, recording_uid, session_uid, source_dir, storage_config_uid, _STATE_DIR)
        except UploadStateError:
            raise
        except Exception:
            raise _failure("retained upload status could not be verified") from None
        finally:
            _QUIET_HTTP.reset(token)

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
        return UploadSession.model_validate(data, strict=True)

    def get_upload_urls(
        self, recording_uid: str, session_uid: str, file_paths: list[str], *, ttl_seconds: int = 3600
    ) -> UploadUrlsResponse:
        """Get presigned PUT URLs for a batch of files."""
        payload = {"session_uid": session_uid, "file_paths": file_paths, "ttl_seconds": ttl_seconds}
        data = self._transport.request("POST", f"/fleet/recordings/{recording_uid}/upload/urls/", json=payload)
        return UploadUrlsResponse.model_validate(data, strict=True)

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
        return UploadStatusResponse.model_validate(data, strict=True)
