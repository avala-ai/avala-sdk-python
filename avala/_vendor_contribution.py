"""Bind a named vendor source to one stable recording across transfer attempts."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable
from tempfile import TemporaryFile
from typing import Any, TypeVar
from uuid import UUID

import httpx

from avala._managed_upload import transfer_managed_upload
from avala._uploads import MAX_RETRIES, is_retryable, sleep_backoff

_T = TypeVar("_T")


def retry_upload_operation(operation: Callable[[], _T]) -> _T:
    for attempt in range(MAX_RETRIES - 1):
        try:
            return operation()
        except Exception as exc:
            if not is_retryable(exc):
                raise
            sleep_backoff(attempt, exc)
    return operation()


def _stamp(info: os.stat_result) -> tuple[int, ...]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def contribute_file(
    path: str,
    relative: str,
    destination: UUID,
    run_uid: str,
    request: Callable[..., Any],
    timeout: httpx.Timeout,
) -> tuple[str, bool, bool]:
    # Registration binds the path to this digest permanently. Stage one private
    # file so every PUT and resumed part reads exactly the bytes that were hashed.
    with TemporaryFile() as snapshot:
        initial = os.lstat(path)
        if not stat.S_ISREG(initial.st_mode):
            raise ValueError("Video contribution requires a regular file.")
        digest = hashlib.sha256()
        with open(path, "rb") as original:
            if _stamp(initial) != _stamp(os.fstat(original.fileno())):
                raise ValueError("Recording changed before hashing.")
            for chunk in iter(lambda: original.read(1024 * 1024), b""):
                if snapshot.tell() + len(chunk) > initial.st_size:
                    raise ValueError("Recording grew while hashing.")
                digest.update(chunk)
                snapshot.write(chunk)
        if snapshot.tell() != initial.st_size or _stamp(initial) != _stamp(os.lstat(path)):
            raise ValueError("Recording changed while hashing.")
        snapshot.flush()
        payload = {"source_path": relative, "size": initial.st_size, "sha256": digest.hexdigest(), "run_uid": run_uid}

        def register_and_transfer() -> tuple[str, bool, bool]:
            descriptor = request("POST", f"/vendor-intakes/{destination}/sources/", json=payload)
            run = str(UUID(descriptor["run_uid"]))
            receipt: dict[str, Any] = {}
            transfer_managed_upload(
                snapshot,
                descriptor,
                destination_uid=destination,
                request=request,
                timeout=timeout,
                cancelled=lambda: False,
                on_complete=receipt.update,
            )
            return run, descriptor["complete"], receipt.get("sha256_verified") is True

        return retry_upload_operation(register_and_transfer)
