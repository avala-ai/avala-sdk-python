"""Exact managed Fleet responses; publication and recording readiness are separate."""

from __future__ import annotations

from typing import List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROTOCOL = "fleet-managed-mcap-v1"
PART_SIZE = 64 * 1024**2


def canonical_uid(value: str) -> str:
    if not isinstance(value, str) or str(UUID(value)) != value:
        raise ValueError("An exact canonical UUID is required")
    return value


class ManagedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class ManagedGeneration(ManagedModel):
    generation_uid: str
    number: int = Field(ge=1)
    state: Literal["initializing", "active", "abandoned", "completing", "completed"]

    _uid = field_validator("generation_uid")(canonical_uid)


class ManagedOriginal(ManagedModel):
    file_uid: str
    path: str
    size_bytes: int = Field(gt=0, le=8 * 1024**3)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_type: Literal["application/octet-stream"]
    content_encoding: Literal[""]
    part_size: int = Field(ge=PART_SIZE, le=PART_SIZE)
    part_count: int = Field(ge=1, le=128)
    verified: bool
    generation: Optional[ManagedGeneration]

    _uid = field_validator("file_uid")(canonical_uid)

    @model_validator(mode="after")
    def geometry(self) -> ManagedOriginal:
        if self.part_count != (self.size_bytes + PART_SIZE - 1) // PART_SIZE:
            raise ValueError("Original part geometry differs from its size")
        if self.verified and (self.generation is None or self.generation.state != "completed"):
            raise ValueError("Verified original requires a completed generation")
        return self


class ManagedFinalization(ManagedModel):
    uid: str
    state: Literal["pending", "running", "retry_wait", "succeeded", "failed"]
    stage: Literal["complete", "verify", "attach", "prepare", "publish"]
    attempts: int = Field(ge=0)
    failures: int = Field(ge=0, le=5)
    code: Literal[
        "pending",
        "running",
        "capacity_pending",
        "worker_disabled",
        "deadline_exceeded",
        "lease_lost",
        "source_unavailable",
        "attempts_exhausted",
        "published",
        "content_mismatch",
        "invalid_mcap",
    ]
    available_at: str
    publication_uid: Optional[str]
    dataset_uid: Optional[str]

    _uid = field_validator("uid")(canonical_uid)

    @model_validator(mode="after")
    def outcome(self) -> ManagedFinalization:
        from datetime import datetime

        if datetime.fromisoformat(self.available_at.replace("Z", "+00:00")).utcoffset() is None:
            raise ValueError("Finalization time requires a timezone")
        if self.state == "succeeded":
            if self.stage != "publish" or self.code != "published" or self.failures != 0:
                raise ValueError("Publication result is inconsistent")
            canonical_uid(self.publication_uid or "")
            canonical_uid(self.dataset_uid or "")
        elif self.publication_uid is not None or self.dataset_uid is not None:
            raise ValueError("Unfinished work cannot claim publication")
        return self


class ManagedFleetUploadStatus(ManagedModel):
    """A retained upload snapshot, including pending or terminal publication work."""

    protocol: Literal["fleet-managed-mcap-v1"]
    session_uid: str
    recording_uid: str
    status: Literal["initiated", "uploading", "completing", "completed"]
    total_files: int = Field(ge=1, le=64)
    total_bytes: int = Field(gt=0, le=64 * 1024**3)
    confirmed_files: int = Field(ge=0, le=64)
    confirmed_bytes: int = Field(ge=0, le=64 * 1024**3)
    files: List[ManagedOriginal] = Field(min_length=1, max_length=64)
    finalization: Optional[ManagedFinalization]

    _uids = field_validator("session_uid", "recording_uid")(canonical_uid)

    @model_validator(mode="after")
    def totals(self) -> ManagedFleetUploadStatus:
        if (
            len(self.files) != self.total_files
            or len({file.file_uid for file in self.files}) != len(self.files)
            or len({file.path for file in self.files}) != len(self.files)
            or sum(file.size_bytes for file in self.files) != self.total_bytes
            or sum(file.verified for file in self.files) != self.confirmed_files
            or sum(file.size_bytes for file in self.files if file.verified) != self.confirmed_bytes
        ):
            raise ValueError("Managed upload counters disagree with exact originals")
        if self.status == "completed" and self.confirmed_files != self.total_files:
            raise ValueError("Completed session contains unverified originals")
        if self.finalization is not None and self.finalization.state == "succeeded" and self.status != "completed":
            raise ValueError("Publication requires completed session evidence")
        return self
