"""Explicit managed Fleet transfers with durable, immutable recovery identity."""

from __future__ import annotations

import hashlib
import math
import time
import unicodedata
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
from avala._uploads import is_retryable, retry_delay
from avala.resources.fleet._managed_protocol import ValidatedGrant, validate_grant
from avala.resources.fleet._managed_receipt import OriginalReceipt, ReceiptStore
from avala.resources.fleet._upload_guard import _QUIET_HTTP
from avala.types.fleet_managed_upload import (
    PART_SIZE,
    PROTOCOL,
    ManagedFinalization,
    ManagedFleetUploadStatus,
    ManagedGeneration,
    canonical_uid,
)

if TYPE_CHECKING:
    from avala.resources.fleet.uploads import FleetUploadManager

_TERMINAL_CODES = {"content_mismatch", "invalid_mcap"}


def _exact(data: Any, fields: set[str]) -> dict[str, Any]:
    if not isinstance(data, dict) or set(data) != fields:
        raise _failure("managed response has missing or unsupported fields")
    return data


def _put(grant: ValidatedGrant, part: bytes) -> None:
    token = _QUIET_HTTP.set(True)
    succeeded = False
    try:
        response = httpx.put(grant.url, headers=grant.headers, content=part, timeout=60, follow_redirects=False)
        response.raise_for_status()
        succeeded = response.status_code == 200
    except Exception:
        # A lost response may have stored the part. The next invocation observes
        # this exact generation before issuing another capability.
        pass
    finally:
        _QUIET_HTTP.reset(token)
    if not succeeded:
        raise _failure("part transfer is uncertain; resume the retained generation")


class _ManagedUpload:
    def __init__(
        self,
        manager: FleetUploadManager,
        inventory: SourceInventory,
        binding: UploadBinding,
        receipt: ReceiptStore,
        workers: int,
    ) -> None:
        self.manager, self.inventory, self.binding, self.receipt, self.workers = (
            manager,
            inventory,
            binding,
            receipt,
            workers,
        )
        self.endpoint = f"/fleet/recordings/{binding.recording_uid}/managed-upload"
        self.identity = {"protocol": PROTOCOL, "session_uid": receipt.value.session_uid}

    def authority(self) -> None:
        if (
            upload_binding(
                self.inventory,
                base_url=self.manager._transport.base_url,
                api_key=self.manager._transport.api_key,
                recording_uid=self.binding.recording_uid,
                storage_config_uid=None,
            )
            != self.binding
        ):
            raise _failure("managed upload authority changed")

    def source(self) -> None:
        self.authority()
        if self.manager.collect_files(self.inventory.source) != self.inventory:
            raise _failure("managed source differs from the retained manifest")
        self.authority()

    def request(
        self,
        action: str,
        *,
        data: dict[str, Any] | None = None,
        get: bool = False,
        retry: bool = False,
        timeout: float = 30,
    ) -> Any:
        payload = self.identity | (data or {})
        for attempt in range(3):
            self.authority()
            try:
                response = self.manager._transport.request(
                    "GET" if get else "POST",
                    f"{self.endpoint}/{action}/",
                    **({"params": payload} if get else {"json": payload}),
                    timeout=timeout,
                )
            except Exception as error:
                if not retry or attempt == 2 or not is_retryable(error):
                    break
                time.sleep(retry_delay(attempt, error))
                continue
            self.authority()
            return response
        raise _failure("managed request is uncertain; retain the session and original files")

    def observe(self, data: Any = None, *, timeout: float = 30) -> ManagedFleetUploadStatus:
        status = ManagedFleetUploadStatus.model_validate(
            self.request("status", get=True, timeout=timeout) if data is None else data
        )
        self.authority()
        if status.session_uid != self.receipt.value.session_uid or status.recording_uid != self.binding.recording_uid:
            raise _failure("managed status names a different recording or session")
        files = {file.path: file for file in self.inventory.files}
        if {row.path for row in status.files} != set(files):
            raise _failure("managed status differs from the complete local manifest")
        originals = dict(self.receipt.value.originals)
        for row in status.files:
            file = files[row.path]
            if row.size_bytes != file.size_bytes or row.content_sha256 != file.sha256:
                raise _failure("managed status differs from the admitted content")
            previous = originals.get(row.path)
            if previous is None:
                if row.generation is not None or row.verified:
                    raise _failure("unrequested multipart work cannot be adopted")
                originals[row.path] = OriginalReceipt(file_uid=row.file_uid)
                continue
            if row.file_uid != previous.file_uid or (previous.verified and not row.verified):
                raise _failure("managed original identity or verification regressed")
            generation = row.generation
            if generation is None:
                if previous.generation_uid is not None:
                    raise _failure("retained multipart generation disappeared")
            elif (
                not previous.initializing
                or generation.number != 1
                or (previous.generation_uid is not None and previous.generation_uid != generation.generation_uid)
            ):
                raise _failure("multipart generation was substituted")
            else:
                originals[row.path] = previous.model_copy(
                    update={
                        "generation_uid": generation.generation_uid,
                        "number": generation.number,
                        "verified": row.verified,
                    }
                )
        self.receipt.value = self.receipt.value.model_copy(update={"originals": originals})
        outcome = status.finalization
        previous_uid = self.receipt.value.finalization_uid
        if outcome is not None:
            self.retain_finalization(outcome)
        elif previous_uid is not None:
            raise _failure("retained finalization disappeared")
        self.receipt.save()
        return status

    def retain_finalization(self, outcome: ManagedFinalization) -> None:
        value = self.receipt.value
        if not value.finalizing or (value.finalization_uid is not None and value.finalization_uid != outcome.uid):
            raise _failure("finalization differs from the retained request")
        if value.publication_uid is not None and (
            outcome.state != "succeeded"
            or outcome.publication_uid != value.publication_uid
            or outcome.dataset_uid != value.dataset_uid
        ):
            raise _failure("published result identity changed")
        self.receipt.value = value.model_copy(
            update={
                "finalization_uid": outcome.uid,
                "publication_uid": outcome.publication_uid,
                "dataset_uid": outcome.dataset_uid,
            }
        )

    def admit(self) -> ManagedFleetUploadStatus:
        if self.receipt.value.originals:
            return self.observe()
        self.source()
        # A client-generated UUID makes retrying an uncertain admission safe.
        return self.observe(
            self.request(
                "admit",
                data={
                    "files": [
                        {"path": file.path, "size_bytes": file.size_bytes, "content_sha256": file.sha256}
                        for file in self.inventory.files
                    ]
                },
                retry=True,
            )
        )

    def initialize(self, file: SourceFile) -> None:
        original = self.receipt.value.originals[file.path]
        if original.initializing:
            # Admission/status recovery already inspected this exact session.
            # None/initializing/abandoned cannot authorize a successor attempt.
            return
        self.receipt.update_original(file.path, initializing=True)
        raw = self.request("multipart/init", data={"file_uid": original.file_uid})
        raw = _exact(
            raw,
            {
                "protocol",
                "session_uid",
                "file_uid",
                "generation_uid",
                "number",
                "state",
                "part_size",
                "part_count",
                "initialization_deadline",
            },
        )
        self.envelope(raw, {"file_uid": original.file_uid})
        generation = ManagedGeneration.model_validate({key: raw[key] for key in ("generation_uid", "number", "state")})
        if (
            generation.number != 1
            or generation.state != "active"
            or type(raw["part_size"]) is not int
            or raw["part_size"] != PART_SIZE
            or type(raw["part_count"]) is not int
            or raw["part_count"] != (file.size_bytes + PART_SIZE - 1) // PART_SIZE
            or not isinstance(raw["initialization_deadline"], str)
            or datetime.fromisoformat(raw["initialization_deadline"].replace("Z", "+00:00")).utcoffset() is None
        ):
            raise _failure("multipart initialization is outside the negotiated profile")
        self.receipt.update_original(file.path, generation_uid=generation.generation_uid, number=1)

    def envelope(self, raw: dict[str, Any], extra: dict[str, Any]) -> None:
        if any(raw.get(key) != value for key, value in (self.identity | extra).items()):
            raise _failure("managed response names different upload work")

    def progress(self, file: SourceFile) -> set[int]:
        original = self.receipt.value.originals[file.path]
        extra = {"file_uid": original.file_uid, "generation_uid": original.generation_uid}
        after, present = 0, set()
        count = (file.size_bytes + PART_SIZE - 1) // PART_SIZE
        for _ in range(2):
            raw = _exact(
                self.request("multipart/progress", data=extra | {"after": after}, get=True, retry=True),
                {"protocol", "session_uid", "file_uid", "generation_uid", "parts", "next_cursor"},
            )
            self.envelope(raw, extra)
            parts = raw["parts"]
            if not isinstance(parts, list) or len(parts) > 100:
                raise _failure("multipart inventory is unbounded")
            last = after
            for part in parts:
                part = _exact(part, {"part_number", "size"})
                number, size = part["part_number"], part["size"]
                if (
                    type(number) is not int
                    or not last < number <= count
                    or type(size) is not int
                    or size != min(PART_SIZE, file.size_bytes - (number - 1) * PART_SIZE)
                ):
                    raise _failure("multipart inventory geometry is inconsistent")
                last = number
                present.add(number)
            cursor = raw["next_cursor"]
            if cursor is None:
                return present
            if type(cursor) is not int or not parts or cursor != last or cursor >= count:
                raise _failure("multipart inventory cursor cannot advance safely")
            after = cursor
        raise _failure("multipart inventory exceeds the negotiated page budget")

    def transfer(self, file: SourceFile) -> None:
        self.initialize(file)
        status = self.observe()
        row = next(row for row in status.files if row.path == file.path)
        if row.verified or (row.generation is not None and row.generation.state == "completed"):
            return
        if row.generation is None or row.generation.state not in {"active", "completing"}:
            raise _failure("initialization has no safely retained active generation")
        present = self.progress(file)
        original = self.receipt.value.originals[file.path]
        extra = {"file_uid": original.file_uid, "generation_uid": original.generation_uid}
        digest = hashlib.sha256()
        with (
            _open_regular_file(self.inventory.source / file.path) as handle,
            ThreadPoolExecutor(max_workers=self.workers) as pool,
        ):
            pending: list[Future[None]] = []
            for number in range(1, row.part_count + 1):
                self.authority()
                if len(pending) == self.workers:
                    for future in pending:
                        future.result()
                    pending.clear()
                part = handle.read(min(PART_SIZE, file.size_bytes - (number - 1) * PART_SIZE))
                expected = min(PART_SIZE, file.size_bytes - (number - 1) * PART_SIZE)
                if len(part) != expected:
                    raise _failure("original changed during multipart transfer")
                digest.update(part)
                if number in present:
                    continue
                raw = _exact(
                    self.request("multipart/parts", data=extra | {"numbers": [number]}, retry=True),
                    {"protocol", "session_uid", "file_uid", "generation_uid", "parts"},
                )
                self.envelope(raw, extra)
                if not isinstance(raw["parts"], list) or len(raw["parts"]) != 1:
                    raise _failure("part capability response differs from its request")
                grant = validate_grant(
                    raw["parts"][0],
                    session_uid=self.identity["session_uid"],
                    file=file,
                    number=number,
                    pin=self.receipt.value.originals[file.path].transport_pin,
                )
                self.receipt.update_original(file.path, transport_pin=grant.pin)
                self.authority()
                pending.append(pool.submit(_put, grant, part))
            if handle.read(1) or digest.hexdigest() != file.sha256:
                raise _failure("transferred source differs from its immutable manifest")
            for future in pending:
                future.result()

    def finalize(self) -> ManagedFleetUploadStatus:
        self.source()
        status = self.observe()
        outcome = status.finalization
        if outcome is None or (outcome.state == "failed" and outcome.code not in _TERMINAL_CODES):
            self.receipt.value = self.receipt.value.model_copy(update={"finalizing": True})
            self.receipt.save()
            files = sorted(
                [
                    {"file_uid": row.file_uid, "generation_uid": row.generation_uid}
                    for row in self.receipt.value.originals.values()
                ],
                key=lambda row: row["file_uid"] or "",
            )
            raw = _exact(
                self.request("finalize", data={"files": files}, retry=True), {"protocol", "session_uid", "finalization"}
            )
            self.envelope(raw, {})
            self.retain_finalization(ManagedFinalization.model_validate(raw["finalization"]))
            self.receipt.save()
        return self.observe()


def validate_managed_options(max_workers: int, wait_timeout: float) -> None:
    """Apply the same local bounds before CLI preview and managed transfer."""
    if type(max_workers) is not int or not 1 <= max_workers <= 4:
        raise _failure("managed upload concurrency must be between one and four")
    if type(wait_timeout) not in (int, float) or not math.isfinite(wait_timeout) or not 0 <= wait_timeout <= 3600:
        raise _failure("publication wait timeout must be finite and between zero and 3600 seconds")


def validate_managed_inventory(inventory: SourceInventory) -> None:
    """Reject unsupported originals without admitting or locking an upload."""
    if (
        not 1 <= len(inventory.files) <= 64
        or inventory.total_bytes > 64 * 1024**3
        or any(
            not file.path.lower().endswith(".mcap")
            or not 0 < file.size_bytes <= 8 * 1024**3
            or file.path != unicodedata.normalize("NFC", file.path)
            or file.path.startswith("__derived__/")
            or any(ord(char) < 32 or ord(char) == 127 for char in file.path)
            for file in inventory.files
        )
    ):
        raise _failure("managed upload requires one to 64 uncompressed MCAP originals within its size profile")


def run_managed_upload(
    manager: FleetUploadManager,
    recording_uid: str,
    source_dir: str | Path,
    *,
    max_workers: int,
    wait_timeout: float,
    state_dir: Path,
) -> ManagedFleetUploadStatus:
    validate_managed_options(max_workers, wait_timeout)
    inventory = manager.collect_files(source_dir)
    validate_managed_inventory(inventory)
    binding = upload_binding(
        inventory,
        base_url=manager._transport.base_url,
        api_key=manager._transport.api_key,
        recording_uid=canonical_uid(recording_uid),
        storage_config_uid=None,
    )
    with fleet_checkpoint(state_dir, binding) as locked:
        receipt = ReceiptStore(locked, binding)
        upload = _ManagedUpload(manager, inventory, binding, receipt, max_workers)
        status = upload.admit()
        outcome = status.finalization
        terminal = outcome is not None and (outcome.state == "succeeded" or outcome.code in _TERMINAL_CODES)
        if not terminal:
            for file in inventory.files:
                upload.transfer(file)
        status = upload.finalize()
        deadline = time.monotonic() + wait_timeout
        while status.finalization is not None and status.finalization.state not in {"failed", "succeeded"}:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(2, remaining))
            remaining = deadline - time.monotonic()
            if remaining > 0:
                status = upload.observe(timeout=min(30, remaining))
        upload.source()
        return status
