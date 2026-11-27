"""Bounded-memory transfers for API-issued managed upload sessions."""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from typing import Any, BinaryIO, ContextManager
from uuid import UUID

import httpx

from avala._uploads import validate_presigned_url

_MIN_PART_SIZE = 5 * 1024 * 1024
_MAX_PART_SIZE = 128 * 1024 * 1024


def transfer_managed_upload(
    path: str | os.PathLike[str] | BinaryIO,
    descriptor: dict[str, Any],
    *,
    request: Callable[..., Any],
    timeout: httpx.Timeout,
    cancelled: Callable[[], bool],
    destination_uid: UUID | None = None,
    on_sent: Callable[[int], None] | None = None,
    on_complete: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Transfer a session, verify completion, and return newly transferred bytes.

    The server owns part state. Retrying the same batch/file presign resumes
    its existing session; no signed URLs or credentials are saved to disk.
    """
    if cancelled():
        raise InterruptedError("Managed upload cancelled before sending bytes.")
    upload_uid = str(UUID(str(descriptor["upload_uid"])))
    endpoint = f"/datasets/manual-upload/uploads/{upload_uid}"
    if destination_uid is not None:
        endpoint = f"/vendor-intakes/{destination_uid}/uploads/{upload_uid}"

    def complete() -> None:
        receipt = request("POST", f"{endpoint}/complete/", json={})
        if on_complete is not None:
            on_complete(receipt)

    if descriptor.get("complete") is True:
        complete()
        return 0

    def open_content() -> ContextManager[BinaryIO]:
        return open(path, "rb") if isinstance(path, (str, os.PathLike)) else nullcontext(path)

    if isinstance(path, (str, os.PathLike)):
        size = os.path.getsize(path)
    else:
        path.seek(0)
        size = os.fstat(path.fileno()).st_size
    method = descriptor.get("method")
    transferred_bytes = 0
    if method == "PUT":
        validate_presigned_url(descriptor["url"])
        transferred_bytes = size
        with open_content() as content:
            headers = {**descriptor.get("headers", {}), "Content-Length": str(size)}
            response = httpx.put(descriptor["url"], headers=headers, content=content, timeout=timeout)
            if response.status_code == 412:
                transferred_bytes = 0
            else:
                response.raise_for_status()
        if on_sent is not None and transferred_bytes:
            on_sent(transferred_bytes)
    elif method == "MULTIPART":
        part_size = descriptor["part_size"]
        count = descriptor["part_count"]
        if (
            type(part_size) is not int
            or not _MIN_PART_SIZE <= part_size <= _MAX_PART_SIZE
            or type(count) is not int
            or not 1 <= count <= 10000
            or count != (size + part_size - 1) // part_size
        ):
            raise ValueError("Invalid managed upload multipart geometry.")
        uploaded = {part["part_number"]: part for part in descriptor.get("uploaded_parts", [])}

        def upload_part(number: int, part: bytes) -> None:
            signed = request("POST", f"{endpoint}/parts/", json={"part_numbers": [number]})["parts"]
            if len(signed) != 1 or signed[0].get("part_number") != number:
                raise ValueError("Part signing response does not match the requested part.")
            target = signed[0]
            validate_presigned_url(target["url"])
            response = httpx.put(target["url"], headers=target.get("headers", {}), content=part, timeout=timeout)
            response.raise_for_status()
            if on_sent is not None:
                on_sent(len(part))

        with open_content() as content, ThreadPoolExecutor(max_workers=2) as executor:
            pending: list[Future[None]] = []
            for number in range(1, count + 1):
                if cancelled():
                    raise InterruptedError("Managed upload cancelled; its session remains resumable.")
                expected = min(part_size, size - (number - 1) * part_size)
                existing = uploaded.get(number)
                if existing is not None and existing.get("size") == expected:
                    content.seek(expected, os.SEEK_CUR)
                    continue
                part = content.read(expected)
                if len(part) != expected:
                    raise ValueError("Local file changed while reading an upload part.")
                pending.append(executor.submit(upload_part, number, part))
                transferred_bytes += len(part)
                if len(pending) == 2:
                    for future in pending:
                        future.result()
                    pending.clear()
            for future in pending:
                future.result()
    else:
        raise ValueError("Unsupported managed upload method.")
    if cancelled():
        raise InterruptedError("Managed upload cancelled before completion; resume the existing session.")
    complete()
    return transferred_bytes
