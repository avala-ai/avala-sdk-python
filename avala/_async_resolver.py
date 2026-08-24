"""Asynchronous credential-free transport for public dataset revisions."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any, BinaryIO, Mapping, cast

import httpx

from avala._config import DEFAULT_BASE_URL
from avala._resolver import (
    _MAX_JSON_SAFE_INTEGER,
    _MAX_RATE_LIMIT_RETRIES,
    _REQUEST_TIMEOUT_SECONDS,
    _DOWNLOAD_SPOOL_MEMORY_BYTES,
    ManifestObjectDocument,
    ManifestObjectPageDocument,
    ParsedDatasetReference,
    ResolvedRevisionDocument,
    DatasetAccessGrant,
    _download_headers,
    _decode_resolver_json,
    _new_download_spool as _new_sync_download_spool,
    _normalize_public_base_url,
    _parse_model,
    _try_parse_access_grant,
    _try_parse_withdrawn_revision,
    _validate_object_page,
    _validate_resolved_revision,
    _rate_limit_retry_delay,
    join_api_path,
)
from avala.errors import (
    DatasetDownloadError,
    DatasetIntegrityError,
    DatasetResolverError,
    DatasetRevisionWithdrawnError,
)


class AsyncDatasetResolverTransport:
    """Async anonymous-only resolver client with no credential state."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL) -> None:
        self.base_url = _normalize_public_base_url(base_url)
        self._client = httpx.AsyncClient(
            headers={"Accept": "application/json"},
            timeout=_REQUEST_TIMEOUT_SECONDS,
            follow_redirects=False,
            trust_env=False,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def _request(
        self,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        expected_dataset_uid: str | None = None,
        expected_revision_sha256: str | None = None,
    ) -> tuple[httpx.Response, Mapping[str, Any]]:
        url = join_api_path(self.base_url, path)
        for rate_limit_attempt in range(_MAX_RATE_LIMIT_RETRIES + 1):
            response: httpx.Response | None = None
            cancelled = False
            try:
                response = await self._client.get(url, params=params)
            except httpx.HTTPError:
                pass
            except asyncio.CancelledError:
                cancelled = True
            if cancelled:
                # A partially received grant may already contain its signed URL.
                # Re-raise only after the transport traceback has been released.
                raise asyncio.CancelledError from None
            if response is None:
                raise DatasetResolverError("transport_error") from None
            data, unexpected_decode_error = _decode_resolver_json(response)
            if unexpected_decode_error:
                del data
                del response
                raise DatasetIntegrityError("invalid_resolver_response") from None
            if response.status_code == 429 and rate_limit_attempt < _MAX_RATE_LIMIT_RETRIES:
                delay = _rate_limit_retry_delay(response.headers.get("Retry-After"), rate_limit_attempt)
                del data
                del response
                await asyncio.sleep(delay)
                continue
            if response.status_code == 410:
                if not isinstance(data, dict):
                    del data
                    del response
                    raise DatasetResolverError("invalid_withdrawal", 410)
                withdrawn, error_reason = _try_parse_withdrawn_revision(
                    data,
                    expected_dataset_uid=expected_dataset_uid,
                    expected_revision_sha256=expected_revision_sha256,
                )
                del data
                del response
                if withdrawn is None:
                    raise DatasetIntegrityError(error_reason or "invalid_withdrawal")
                canonical_reference = withdrawn.canonical_reference
                dataset_uid = withdrawn.dataset_uid
                revision_sha256 = withdrawn.revision_sha256
                del withdrawn
                raise DatasetRevisionWithdrawnError(
                    canonical_reference,
                    dataset_uid,
                    revision_sha256,
                )
            if not response.is_success:
                status_code = response.status_code
                if status_code == 404:
                    reason = "not_found"
                elif status_code == 429:
                    reason = "rate_limited"
                elif status_code >= 500:
                    reason = "service_unavailable"
                else:
                    reason = "request_rejected"
                del data
                del response
                raise DatasetResolverError(reason, status_code)
            if not isinstance(data, dict):
                del data
                del response
                raise DatasetIntegrityError("invalid_resolver_response")
            return response, cast(Mapping[str, Any], data)
        raise AssertionError("unreachable")

    async def resolve(self, parsed_reference: ParsedDatasetReference) -> ResolvedRevisionDocument:
        _, data = await self._request(
            parsed_reference.path,
            expected_dataset_uid=parsed_reference.exact_dataset_uid,
            expected_revision_sha256=parsed_reference.exact_revision_sha256,
        )
        document = cast(
            ResolvedRevisionDocument,
            _parse_model(ResolvedRevisionDocument, data, "invalid_resolver_response"),
        )
        return _validate_resolved_revision(document, parsed_reference)

    async def list_objects(
        self,
        revision: ResolvedRevisionDocument,
        *,
        role: str | None = None,
        ordinal: int | None = None,
        ordinals: tuple[int, ...] | None = None,
        cursor: str | None = None,
        limit: int = 100,
    ) -> ManifestObjectPageDocument:
        if ordinals is not None and (
            role is None
            or ordinal is not None
            or cursor is not None
            or not 1 <= len(ordinals) <= 100
            or len(ordinals) != len(set(ordinals))
            or any(type(value) is not int or not 0 <= value <= _MAX_JSON_SAFE_INTEGER for value in ordinals)
        ):
            raise ValueError("ordinals must contain 1-100 unique canonical integers for one role")
        params: dict[str, Any] = {} if ordinals is not None else {"limit": limit}
        if role is not None:
            params["role"] = role
        if ordinal is not None:
            params["ordinal"] = ordinal
        if ordinals is not None:
            params["ordinals"] = ordinals
        if cursor is not None:
            params["cursor"] = cursor
        _, data = await self._request(
            revision.manifest.objects_path,
            params=params,
            expected_dataset_uid=revision.dataset_uid,
            expected_revision_sha256=revision.revision_sha256,
        )
        page = cast(
            ManifestObjectPageDocument,
            _parse_model(ManifestObjectPageDocument, data, "invalid_object_page"),
        )
        return _validate_object_page(
            page,
            revision,
            role=role,
            ordinal=ordinal,
            ordinals=ordinals,
            cursor=cursor,
        )

    async def issue_access_grant(
        self,
        revision: ResolvedRevisionDocument,
        manifest_object: ManifestObjectDocument,
    ) -> DatasetAccessGrant:
        response, data = await self._request(
            manifest_object.access_path,
            expected_dataset_uid=revision.dataset_uid,
            expected_revision_sha256=revision.revision_sha256,
        )
        grant, error_reason = _try_parse_access_grant(
            data,
            date_header=response.headers.get("Date"),
            revision=revision,
            manifest_object=manifest_object,
            base_url=self.base_url,
        )
        del data
        del response
        if grant is None:
            raise DatasetIntegrityError(error_reason or "invalid_access_grant")
        return grant


def _new_download_spool() -> BinaryIO:
    """Allocate using the async downloader's independently configurable memory threshold."""

    return _new_sync_download_spool(max_size=_DOWNLOAD_SPOOL_MEMORY_BYTES)


async def _write_download_chunk(content: BinaryIO, chunk: bytes) -> None:
    """Keep rollover and all later disk-backed spool writes off the event loop."""

    will_use_disk = bool(getattr(content, "_rolled", False)) or (
        content.tell() + len(chunk) > _DOWNLOAD_SPOOL_MEMORY_BYTES
    )
    if will_use_disk:
        await asyncio.to_thread(content.write, chunk)
    else:
        content.write(chunk)


async def download_object_file(content: BinaryIO, grant: DatasetAccessGrant) -> BinaryIO:
    """Download into a verified seekable spool without retaining the object in RAM."""

    provider = grant.provider
    object_uid = grant.object_uid
    expected_size_bytes = grant.size_bytes
    expected_sha256 = grant.sha256
    download_url = grant.url
    del grant
    digest = hashlib.sha256()
    bytes_written = 0
    chunk = b""
    failure_reason: str | None = None
    cancelled = False
    client: httpx.AsyncClient | None = None
    response: httpx.Response | None = None
    try:
        client = httpx.AsyncClient(
            headers=_download_headers(provider),
            timeout=_REQUEST_TIMEOUT_SECONDS,
            follow_redirects=False,
            trust_env=False,
        )
        async with client:
            async with client.stream("GET", download_url) as response:
                if response.status_code != 200:
                    failure_reason = "redirect_rejected" if 300 <= response.status_code < 400 else "provider_response"
                else:
                    async for chunk in response.aiter_raw():
                        await _write_download_chunk(content, chunk)
                        bytes_written += len(chunk)
                        digest.update(chunk)
                        if bytes_written > expected_size_bytes:
                            failure_reason = "size_mismatch"
                            break
    except httpx.HTTPError:
        failure_reason = "transport_error"
    except OSError:
        failure_reason = "local_storage_error"
    except asyncio.CancelledError:
        cancelled = True
    finally:
        download_url = ""
        client = None
        response = None
        chunk = b""
    if cancelled:
        content.close()
        raise asyncio.CancelledError from None
    if failure_reason is not None:
        content.close()
        raise DatasetDownloadError(provider, object_uid, failure_reason)
    if bytes_written != expected_size_bytes:
        content.close()
        raise DatasetDownloadError(provider, object_uid, "size_mismatch")
    if digest.hexdigest() != expected_sha256:
        content.close()
        raise DatasetDownloadError(provider, object_uid, "sha256_mismatch")
    content.seek(0)
    return content


async def download_object_bytes(content: BinaryIO, grant: DatasetAccessGrant) -> bytes:
    """Download through the verified spool and return its bytes."""

    try:
        content = await download_object_file(content, grant)
    finally:
        del grant
    with content:
        if bool(getattr(content, "_rolled", False)):
            return await asyncio.to_thread(content.read)
        return content.read()


__all__ = ["AsyncDatasetResolverTransport", "download_object_bytes", "download_object_file"]
