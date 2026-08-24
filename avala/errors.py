"""Error hierarchy for the Avala SDK."""

from __future__ import annotations

from typing import Any


class AvalaError(Exception):
    """Base exception for all Avala API errors."""

    def __init__(self, message: str, status_code: int | None = None, body: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.body = body


class AuthenticationError(AvalaError):
    """Raised on 401 responses."""


class ForbiddenError(AvalaError):
    """Raised on 403 responses (insufficient permissions, wrong scope, or plan level)."""


class NotFoundError(AvalaError):
    """Raised on 404 responses."""


class RateLimitError(AvalaError):
    """Raised on 429 responses."""

    def __init__(
        self,
        message: str,
        status_code: int = 429,
        body: Any = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, status_code, body)
        self.retry_after = retry_after


class ValidationError(AvalaError):
    """Raised on 400/422 responses."""

    def __init__(
        self,
        message: str,
        status_code: int = 400,
        body: Any = None,
        details: list[Any] | None = None,
    ) -> None:
        super().__init__(message, status_code, body)
        self.details = details or []


class QuotaExceededError(AvalaError):
    """Raised on 413 responses from the manual-upload presign endpoint.

    The server rejects a presign whose ``content_length`` would push the owner
    (user or organization) past their storage cap, and reports both numbers.
    They are surfaced here because the generic ``AvalaError`` path loses them:
    callers need ``limit``/``used`` to tell "delete something" from "ask for a
    bigger cap". Both are bytes, and either may be ``None`` if the server
    response did not carry the structured body.
    """

    def __init__(
        self,
        message: str,
        status_code: int = 413,
        body: Any = None,
        limit: int | None = None,
        used: int | None = None,
    ) -> None:
        super().__init__(message, status_code, body)
        self.limit = limit
        self.used = used


class UploadStateError(AvalaError):
    """Raised when persisted resume state exists but cannot be trusted."""


class ServerError(AvalaError):
    """Raised on 5xx responses."""


class DatasetReferenceError(ValueError):
    """Raised when a public dataset reference is ambiguous or malformed."""


class UnsupportedDatasetModeError(ValueError):
    """Raised when a loader mode is not implemented by the public v1 contract."""


class MutableDatasetAliasWarning(UserWarning):
    """Warns that a friendly alias is mutable and unsuitable for reproducible inputs."""


class DatasetResolverError(AvalaError):
    """Raised when the anonymous dataset resolver cannot complete safely."""

    def __init__(self, reason: str, status_code: int | None = None) -> None:
        super().__init__(f"Dataset resolver failed ({reason}).", status_code)
        self.reason = reason


class DatasetRevisionWithdrawnError(DatasetResolverError):
    """Raised when an exact immutable revision has been withdrawn."""

    def __init__(self, canonical_reference: str, dataset_uid: str, revision_sha256: str) -> None:
        super().__init__("revision_withdrawn", 410)
        self.canonical_reference = canonical_reference
        self.dataset_uid = dataset_uid
        self.revision_sha256 = revision_sha256


class DatasetIntegrityError(DatasetResolverError):
    """Raised when resolver metadata or downloaded bytes violate their identities."""


class DatasetDownloadError(AvalaError):
    """URL-free failure raised by an isolated public-object download."""

    def __init__(self, provider: str, object_uid: str, reason: str) -> None:
        super().__init__(
            f"Dataset object download failed (provider={provider}, object_uid={object_uid}, reason={reason})."
        )
        self.provider = provider
        self.object_uid = object_uid
        self.reason = reason
