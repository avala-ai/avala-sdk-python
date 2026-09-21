"""Versioned managed evidence uses the same discoverable Fleet recording lock."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Literal, Optional
from uuid import uuid4

from pydantic import Field, field_validator

from avala._fleet_uploads import (
    FleetCheckpoint,
    UploadBinding,
    _digest,
    _failure,
    _open_regular_file,
    _unique_fields,
    _write_private_receipt,
)
from avala.types.fleet_managed_upload import ManagedModel, canonical_uid

_MAX_BYTES = 1024 * 1024


class OriginalReceipt(ManagedModel):
    file_uid: str
    initializing: bool = False
    generation_uid: Optional[str] = None
    number: Optional[int] = None
    transport_pin: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    verified: bool = False

    _uid = field_validator("file_uid")(canonical_uid)


class ManagedReceipt(ManagedModel):
    version: Literal[3] = 3
    protocol: Literal["fleet-managed-mcap-v1"] = "fleet-managed-mcap-v1"
    binding: str
    evidence: Dict[str, Any]
    session_uid: str
    originals: Dict[str, OriginalReceipt] = Field(default_factory=dict)
    finalizing: bool = False
    finalization_uid: Optional[str] = None
    publication_uid: Optional[str] = None
    dataset_uid: Optional[str] = None

    _uid = field_validator("session_uid")(canonical_uid)


class ReceiptStore:
    def __init__(self, locked: FleetCheckpoint, binding: UploadBinding) -> None:
        self.locked, self.binding = locked, binding
        self.value: ManagedReceipt
        if not os.path.lexists(locked.path):
            self.value = ManagedReceipt(binding=binding.digest, evidence=binding.document(), session_uid=str(uuid4()))
            self.save()
            return
        try:
            with _open_regular_file(locked.path) as handle:
                encoded = handle.read(_MAX_BYTES + 1)
            if len(encoded) > _MAX_BYTES:
                raise ValueError
            data = json.loads(encoded, object_pairs_hook=_unique_fields)
            if (
                not isinstance(data, dict)
                or set(data) != set(ManagedReceipt.model_fields)
                or type(data["version"]) is not int
            ):
                raise ValueError
            if not isinstance(data["originals"], dict) or any(
                not isinstance(row, dict) or set(row) != set(OriginalReceipt.model_fields)
                for row in data["originals"].values()
            ):
                raise ValueError
            self.value = ManagedReceipt.model_validate(data)
            self._validate()
        except Exception:
            raise _failure("managed checkpoint is corrupt or belongs to a different protocol or source") from None

    def _validate(self) -> None:
        receipt = self.value
        if (
            receipt.binding != self.binding.digest
            or receipt.evidence != self.binding.document()
            or _digest(receipt.evidence) != receipt.binding
            or set(receipt.originals) not in (set(), {file.path for file in self.binding.inventory.files})
            or len({row.file_uid for row in receipt.originals.values()}) != len(receipt.originals)
        ):
            raise ValueError("Managed receipt identity is inconsistent")
        for row in receipt.originals.values():
            if row.generation_uid is not None:
                canonical_uid(row.generation_uid)
                if not row.initializing or row.number != 1:
                    raise ValueError("Only the original initialized generation is retained")
            elif row.number is not None or row.transport_pin is not None or row.verified:
                raise ValueError("Transport evidence requires a generation")
        if receipt.finalizing and (
            not receipt.originals or any(row.generation_uid is None for row in receipt.originals.values())
        ):
            raise ValueError("Finalization requires every exact generation")
        if receipt.finalization_uid is not None:
            canonical_uid(receipt.finalization_uid)
            if not receipt.finalizing:
                raise ValueError("Finalization requires a retained request")
        if (receipt.publication_uid is None) != (receipt.dataset_uid is None):
            raise ValueError("Publication requires both result identities")
        if receipt.publication_uid is not None:
            canonical_uid(receipt.publication_uid)
            canonical_uid(receipt.dataset_uid or "")
            if receipt.finalization_uid is None:
                raise ValueError("Publication requires an exact finalization")

    def save(self) -> None:
        if self.locked._closed:
            raise _failure("checkpoint ownership has ended")
        self._validate()
        _write_private_receipt(self.locked.path, self.value.model_dump(), maximum=_MAX_BYTES)

    def update_original(self, path: str, **changes: object) -> None:
        originals = dict(self.value.originals)
        originals[path] = originals[path].model_copy(update=changes)
        self.value = self.value.model_copy(update={"originals": originals})
        self.save()
