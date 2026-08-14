"""Pydantic models for Avala-managed ("manual") dataset uploads."""

from __future__ import annotations

from pydantic import BaseModel


class UploadQuota(BaseModel):
    """Storage quota for one upload owner (a user, or an organization).

    ``used`` is the owner's settled bytes plus every open presign reservation,
    so an upload that is still in flight already counts. Both values are bytes.
    """

    used: int
    limit: int

    @property
    def remaining(self) -> int:
        """Bytes still available. Never negative — a reconcile can land ``used``
        above ``limit`` (the cap is enforced at presign, not retroactively)."""
        return max(0, self.limit - self.used)


class AllowedMimes(BaseModel):
    """Advisory list of MIME types the manual-upload endpoint indexes.

    The server treats anything outside the list as ``application/octet-stream``
    rather than rejecting it, so this is a hint for pickers, not a validator.
    """

    mime_types: list[str] = []
    video_prefix: str = ""
