"""Lazy, credential-free interfaces for immutable public dataset revisions."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import warnings
from collections.abc import AsyncIterator, Coroutine, Iterator, Mapping
from typing import Any, BinaryIO, Optional

from avala._async_resolver import AsyncDatasetResolverTransport
from avala._async_resolver import _new_download_spool as _new_async_download_spool
from avala._async_resolver import download_object_bytes as async_download_object_bytes
from avala._async_resolver import download_object_file as async_download_object_file
from avala._config import DEFAULT_BASE_URL
from avala._resolver import (
    _MAX_JSON_SAFE_INTEGER,
    _MIN_GRANT_REMAINING_SECONDS,
    _ROLE_PATTERN,
    _new_download_spool,
    DatasetAccessGrant,
    ManifestObjectDocument,
    ManifestObjectPageDocument,
    ResolvedRevisionDocument,
    SyncDatasetResolverTransport,
    download_object_file,
    parse_dataset_reference,
)
from avala.errors import DatasetIntegrityError, MutableDatasetAliasWarning, UnsupportedDatasetModeError

_DEFAULT_IN_MEMORY_READ_LIMIT_BYTES = 64 * 1024 * 1024
_ManifestObjectEvidence = tuple[str, str, str, int, Optional[int], str, str, int, str, str, str]


def _warn_if_mutable_alias(exact_revision_sha256: str | None) -> None:
    if exact_revision_sha256 is None:
        warnings.warn(
            "Dataset aliases are mutable; persist the resolved canonical_reference for reproducible inputs.",
            MutableDatasetAliasWarning,
            stacklevel=3,
        )


def _validate_ordinal_manifest_bounds(
    document: ManifestObjectDocument,
    revision: ResolvedRevisionDocument,
) -> None:
    if revision.manifest.object_count < 1:
        raise DatasetIntegrityError("manifest_object_count_mismatch")
    if document.size_bytes > revision.manifest.total_size_bytes:
        raise DatasetIntegrityError("manifest_total_size_mismatch")


def _manifest_object_order_key(document: ManifestObjectDocument) -> tuple[str, int, str]:
    return (document.role, document.ordinal, document.logical_path)


def _record_episode_grouping(
    document: ManifestObjectDocument,
    episode_ordinals: set[int],
    referenced_episode_ordinals: set[int],
) -> None:
    if document.role == "episode":
        episode_ordinals.add(document.ordinal)
    elif document.episode_ordinal is not None:
        referenced_episode_ordinals.add(document.episode_ordinal)


def _validate_episode_grouping(
    episode_ordinals: set[int],
    referenced_episode_ordinals: set[int],
) -> None:
    if not referenced_episode_ordinals.issubset(episode_ordinals):
        raise DatasetIntegrityError("dangling_episode_reference")


def _manifest_object_evidence(document: ManifestObjectDocument) -> _ManifestObjectEvidence:
    extra_fields_digest = hashlib.sha256()
    encoder = json.JSONEncoder(ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    for chunk in encoder.iterencode(document.model_extra or {}):
        extra_fields_digest.update(chunk.encode("utf-8"))
    return (
        document.object_uid,
        document.logical_path,
        document.role,
        document.ordinal,
        document.episode_ordinal,
        document.content_type,
        document.origin,
        document.size_bytes,
        document.sha256,
        document.access_path,
        extra_fields_digest.hexdigest(),
    )


def _validate_direct_lookup_ordinal(ordinal: int) -> None:
    if type(ordinal) is not int or not 0 <= ordinal <= _MAX_JSON_SAFE_INTEGER:
        raise IndexError("manifest object ordinal must be a JSON-safe non-negative integer")


class _ObservedManifestBounds:
    """Enforce global manifest bounds over unique objects observed by one view."""

    def __init__(self, revision: ResolvedRevisionDocument) -> None:
        self._revision = revision
        self._documents_by_uid: dict[str, _ManifestObjectEvidence] = {}
        self._role_ordinals: set[tuple[str, int]] = set()
        self._logical_paths: set[str] = set()
        self._total_size_bytes = 0

    def include(self, document: ManifestObjectDocument) -> None:
        self.include_evidence(_manifest_object_evidence(document))

    def include_evidence(self, evidence: _ManifestObjectEvidence) -> None:
        object_uid, logical_path, role, ordinal, _, _, _, size_bytes, _, _, _ = evidence
        existing = self._documents_by_uid.get(object_uid)
        if existing is not None:
            if existing != evidence:
                raise DatasetIntegrityError("duplicate_manifest_object")
            return
        role_ordinal = (role, ordinal)
        if role_ordinal in self._role_ordinals or logical_path in self._logical_paths:
            raise DatasetIntegrityError("duplicate_manifest_object")
        if len(self._documents_by_uid) + 1 > self._revision.manifest.object_count:
            raise DatasetIntegrityError("manifest_object_count_mismatch")
        total_size_bytes = self._total_size_bytes + size_bytes
        if total_size_bytes > self._revision.manifest.total_size_bytes:
            raise DatasetIntegrityError("manifest_total_size_mismatch")
        self._documents_by_uid[object_uid] = evidence
        self._role_ordinals.add(role_ordinal)
        self._logical_paths.add(logical_path)
        self._total_size_bytes = total_size_bytes


class _EpisodeReferenceIndex:
    """Cache exact episode batches while sharing collection-wide evidence bounds."""

    def __init__(self) -> None:
        self._evidence_by_ordinal: dict[int, _ManifestObjectEvidence] = {}

    def missing_ordinals(self, ordinals: set[int]) -> tuple[int, ...]:
        return tuple(sorted(ordinals.difference(self._evidence_by_ordinal)))

    def include_batch(
        self,
        page: ManifestObjectPageDocument,
        *,
        requested_ordinals: tuple[int, ...],
        observed_bounds: _ObservedManifestBounds,
    ) -> None:
        returned_ordinals: set[int] = set()
        returned_evidence: list[tuple[int, _ManifestObjectEvidence]] = []
        previous_order_key: tuple[str, int, str] | None = None
        for document in page.results:
            order_key = _manifest_object_order_key(document)
            if document.ordinal in returned_ordinals:
                raise DatasetIntegrityError("duplicate_manifest_object")
            if previous_order_key is not None and order_key <= previous_order_key:
                raise DatasetIntegrityError("invalid_manifest_order")
            evidence = _manifest_object_evidence(document)
            existing = self._evidence_by_ordinal.get(document.ordinal)
            if existing is not None and existing != evidence:
                raise DatasetIntegrityError("duplicate_manifest_object")
            returned_evidence.append((document.ordinal, evidence))
            returned_ordinals.add(document.ordinal)
            previous_order_key = order_key
        if returned_ordinals != set(requested_ordinals):
            raise DatasetIntegrityError("dangling_episode_reference")
        for ordinal, evidence in returned_evidence:
            observed_bounds.include_evidence(evidence)
            self._evidence_by_ordinal[ordinal] = evidence


class ResolvedDatasetObject:
    """One immutable manifest object whose bytes are granted only when opened."""

    def __init__(
        self,
        document: ManifestObjectDocument,
        revision: ResolvedRevisionDocument,
        transport: SyncDatasetResolverTransport,
    ) -> None:
        self._document = document
        self._revision = revision
        self._transport = transport

    @property
    def object_uid(self) -> str:
        return self._document.object_uid

    @property
    def logical_path(self) -> str:
        return self._document.logical_path

    @property
    def role(self) -> str:
        return self._document.role

    @property
    def ordinal(self) -> int:
        return self._document.ordinal

    @property
    def episode_ordinal(self) -> int | None:
        return self._document.episode_ordinal

    @property
    def content_type(self) -> str:
        return self._document.content_type

    @property
    def origin(self) -> str:
        return self._document.origin

    @property
    def size_bytes(self) -> int:
        return self._document.size_bytes

    @property
    def sha256(self) -> str:
        return self._document.sha256

    @property
    def extra_fields(self) -> Mapping[str, Any]:
        return dict(self._document.model_extra or {})

    def _live_grant(self) -> DatasetAccessGrant:
        grant = self._transport.issue_access_grant(self._revision, self._document, explicit_download=True)
        if grant.remaining_seconds < _MIN_GRANT_REMAINING_SECONDS:
            del grant
            grant = self._transport.issue_access_grant(self._revision, self._document, explicit_download=True)
        if grant.remaining_seconds < _MIN_GRANT_REMAINING_SECONDS:
            del grant
            raise DatasetIntegrityError("grant_expiry_too_short")
        return grant

    def open(self) -> BinaryIO:
        """Return a verified seekable spool; close it after consuming the object."""

        content = _new_download_spool()
        try:
            return download_object_file(content, self._live_grant())
        except BaseException:
            content.close()
            raise

    def read(self, *, max_bytes: int = _DEFAULT_IN_MEMORY_READ_LIMIT_BYTES) -> bytes:
        """Return verified bytes when the object fits the caller's memory budget."""

        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("max_bytes must be a non-negative integer")
        if self.size_bytes > max_bytes:
            raise UnsupportedDatasetModeError("object exceeds read() memory limit; consume open() instead")
        with self.open() as content:
            return content.read()

    def __repr__(self) -> str:
        return (
            "ResolvedDatasetObject("
            f"object_uid={self.object_uid!r}, role={self.role!r}, ordinal={self.ordinal}, "
            f"logical_path={self.logical_path!r})"
        )


class DatasetObjectCollection:
    """Lazy cursor-paginated view over immutable manifest objects."""

    def __init__(
        self,
        revision: ResolvedRevisionDocument,
        transport: SyncDatasetResolverTransport,
        *,
        role: str | None,
        episode_reference_index: _EpisodeReferenceIndex,
        observed_manifest_bounds: _ObservedManifestBounds,
    ) -> None:
        if role is not None and re.fullmatch(_ROLE_PATTERN, role) is None:
            raise ValueError("role must be a canonical manifest role")
        self._revision = revision
        self._transport = transport
        self._role = role
        self._episode_reference_index = episode_reference_index
        self._observed_manifest_bounds = observed_manifest_bounds

    def _validate_episode_references(self, documents: list[ManifestObjectDocument]) -> None:
        referenced_ordinals = {
            document.episode_ordinal for document in documents if document.episode_ordinal is not None
        }
        missing_ordinals = self._episode_reference_index.missing_ordinals(referenced_ordinals)
        if not missing_ordinals:
            return
        page = self._transport.list_objects(
            self._revision,
            role="episode",
            ordinals=missing_ordinals,
        )
        self._episode_reference_index.include_batch(
            page,
            requested_ordinals=missing_ordinals,
            observed_bounds=self._observed_manifest_bounds,
        )

    def __iter__(self) -> Iterator[ResolvedDatasetObject]:
        cursor = None
        seen_cursors: set[str] = set()
        seen_object_uids: set[str] = set()
        seen_role_ordinals: set[tuple[str, int]] = set()
        seen_logical_paths: set[str] = set()
        episode_ordinals: set[int] = set()
        referenced_episode_ordinals: set[int] = set()
        previous_order_key: tuple[str, int, str] | None = None
        yielded = 0
        yielded_size_bytes = 0
        while True:
            page = self._transport.list_objects(self._revision, role=self._role, cursor=cursor)
            resolved_page: list[ResolvedDatasetObject] = []
            for document in page.results:
                role_ordinal = (document.role, document.ordinal)
                if (
                    document.object_uid in seen_object_uids
                    or role_ordinal in seen_role_ordinals
                    or document.logical_path in seen_logical_paths
                ):
                    raise DatasetIntegrityError("duplicate_manifest_object")
                order_key = _manifest_object_order_key(document)
                if previous_order_key is not None and order_key <= previous_order_key:
                    raise DatasetIntegrityError("invalid_manifest_order")
                previous_order_key = order_key
                seen_object_uids.add(document.object_uid)
                seen_role_ordinals.add(role_ordinal)
                seen_logical_paths.add(document.logical_path)
                _record_episode_grouping(document, episode_ordinals, referenced_episode_ordinals)
                yielded += 1
                yielded_size_bytes += document.size_bytes
                if yielded > self._revision.manifest.object_count:
                    raise DatasetIntegrityError("manifest_object_count_mismatch")
                if yielded_size_bytes > self._revision.manifest.total_size_bytes:
                    raise DatasetIntegrityError("manifest_total_size_mismatch")
                self._observed_manifest_bounds.include(document)
                resolved_page.append(ResolvedDatasetObject(document, self._revision, self._transport))
            if self._role is not None:
                self._validate_episode_references(page.results)
            yield from resolved_page
            next_cursor = page.next_cursor
            if next_cursor is None:
                break
            if next_cursor in seen_cursors or not page.results:
                raise DatasetIntegrityError("invalid_cursor_sequence")
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        if self._role is None and yielded != self._revision.manifest.object_count:
            raise DatasetIntegrityError("manifest_object_count_mismatch")
        if self._role is None and yielded_size_bytes != self._revision.manifest.total_size_bytes:
            raise DatasetIntegrityError("manifest_total_size_mismatch")
        if self._role is None:
            _validate_episode_grouping(episode_ordinals, referenced_episode_ordinals)

    def __getitem__(self, ordinal: int) -> ResolvedDatasetObject:
        if self._role is None:
            raise TypeError("ordinal lookup requires a role-specific collection")
        _validate_direct_lookup_ordinal(ordinal)
        page = self._transport.list_objects(self._revision, role=self._role, ordinal=ordinal)
        if not page.results:
            raise IndexError(f"No {self._role!r} object exists at ordinal {ordinal}.")
        document = page.results[0]
        _validate_ordinal_manifest_bounds(document, self._revision)
        self._observed_manifest_bounds.include(document)
        self._validate_episode_references([document])
        return ResolvedDatasetObject(document, self._revision, self._transport)


class ResolvedDataset:
    """Resolved immutable dataset revision with lazy manifest-object access."""

    def __init__(self, revision: ResolvedRevisionDocument, transport: SyncDatasetResolverTransport) -> None:
        self._revision = revision
        self._transport = transport
        self._episode_reference_index = _EpisodeReferenceIndex()
        self._observed_manifest_bounds = _ObservedManifestBounds(revision)
        self.objects = self._collection(role=None)
        self.episodes = self._collection(role="episode")

    def _collection(self, *, role: str | None) -> DatasetObjectCollection:
        return DatasetObjectCollection(
            self._revision,
            self._transport,
            role=role,
            episode_reference_index=self._episode_reference_index,
            observed_manifest_bounds=self._observed_manifest_bounds,
        )

    @property
    def canonical_reference(self) -> str:
        return self._revision.canonical_reference

    @property
    def requested_reference(self) -> str:
        return self._revision.requested_reference

    @property
    def display_reference(self) -> str:
        return self._revision.display_reference

    @property
    def dataset_uid(self) -> str:
        return self._revision.dataset_uid

    @property
    def revision_uid(self) -> str:
        return self._revision.revision_uid

    @property
    def revision_sha256(self) -> str:
        return self._revision.revision_sha256

    @property
    def manifest_sha256(self) -> str:
        return self._revision.manifest.sha256

    @property
    def object_count(self) -> int:
        return self._revision.manifest.object_count

    @property
    def total_size_bytes(self) -> int:
        return self._revision.manifest.total_size_bytes

    @property
    def rights(self) -> Mapping[str, Any]:
        return self._revision.rights.model_dump()

    def for_role(self, role: str) -> DatasetObjectCollection:
        return self._collection(role=role)

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> ResolvedDataset:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"ResolvedDataset(canonical_reference={self.canonical_reference!r}, object_count={self.object_count})"


class AsyncResolvedDatasetObject:
    """Async counterpart of :class:`ResolvedDatasetObject`."""

    def __init__(
        self,
        document: ManifestObjectDocument,
        revision: ResolvedRevisionDocument,
        transport: AsyncDatasetResolverTransport,
    ) -> None:
        self._document = document
        self._revision = revision
        self._transport = transport

    @property
    def object_uid(self) -> str:
        return self._document.object_uid

    @property
    def logical_path(self) -> str:
        return self._document.logical_path

    @property
    def role(self) -> str:
        return self._document.role

    @property
    def ordinal(self) -> int:
        return self._document.ordinal

    @property
    def episode_ordinal(self) -> int | None:
        return self._document.episode_ordinal

    @property
    def content_type(self) -> str:
        return self._document.content_type

    @property
    def origin(self) -> str:
        return self._document.origin

    @property
    def size_bytes(self) -> int:
        return self._document.size_bytes

    @property
    def sha256(self) -> str:
        return self._document.sha256

    @property
    def extra_fields(self) -> Mapping[str, Any]:
        return dict(self._document.model_extra or {})

    async def _live_grant(self) -> DatasetAccessGrant:
        grant = await self._transport.issue_access_grant(self._revision, self._document, explicit_download=True)
        if grant.remaining_seconds < _MIN_GRANT_REMAINING_SECONDS:
            del grant
            grant = await self._transport.issue_access_grant(self._revision, self._document, explicit_download=True)
        if grant.remaining_seconds < _MIN_GRANT_REMAINING_SECONDS:
            del grant
            raise DatasetIntegrityError("grant_expiry_too_short")
        return grant

    async def open(self) -> BinaryIO:
        """Return a verified seekable spool; close it after consuming the object."""

        content = _new_async_download_spool()
        try:
            return await async_download_object_file(content, await self._live_grant())
        except BaseException:
            content.close()
            raise

    async def read(self, *, max_bytes: int = _DEFAULT_IN_MEMORY_READ_LIMIT_BYTES) -> bytes:
        """Return verified bytes when the object fits the caller's memory budget."""

        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("max_bytes must be a non-negative integer")
        if self.size_bytes > max_bytes:
            raise UnsupportedDatasetModeError("object exceeds read() memory limit; consume open() instead")
        content = _new_async_download_spool()
        try:
            return await async_download_object_bytes(content, await self._live_grant())
        except BaseException:
            content.close()
            raise

    def __repr__(self) -> str:
        return (
            "AsyncResolvedDatasetObject("
            f"object_uid={self.object_uid!r}, role={self.role!r}, ordinal={self.ordinal}, "
            f"logical_path={self.logical_path!r})"
        )


class AsyncDatasetObjectCollection:
    """Async lazy cursor-paginated view over manifest objects."""

    def __init__(
        self,
        revision: ResolvedRevisionDocument,
        transport: AsyncDatasetResolverTransport,
        *,
        role: str | None,
        episode_reference_index: _EpisodeReferenceIndex,
        episode_reference_lock: asyncio.Lock,
        observed_manifest_bounds: _ObservedManifestBounds,
    ) -> None:
        if role is not None and re.fullmatch(_ROLE_PATTERN, role) is None:
            raise ValueError("role must be a canonical manifest role")
        self._revision = revision
        self._transport = transport
        self._role = role
        self._episode_reference_index = episode_reference_index
        self._episode_reference_lock = episode_reference_lock
        self._observed_manifest_bounds = observed_manifest_bounds

    async def _include_observed_documents(self, documents: list[ManifestObjectDocument]) -> None:
        async with self._episode_reference_lock:
            for document in documents:
                self._observed_manifest_bounds.include(document)

    async def _validate_episode_references(self, documents: list[ManifestObjectDocument]) -> None:
        referenced_ordinals = {
            document.episode_ordinal for document in documents if document.episode_ordinal is not None
        }
        if not referenced_ordinals:
            return
        async with self._episode_reference_lock:
            missing_ordinals = self._episode_reference_index.missing_ordinals(referenced_ordinals)
            if not missing_ordinals:
                return
            page = await self._transport.list_objects(
                self._revision,
                role="episode",
                ordinals=missing_ordinals,
            )
            self._episode_reference_index.include_batch(
                page,
                requested_ordinals=missing_ordinals,
                observed_bounds=self._observed_manifest_bounds,
            )

    async def _iterate(self) -> AsyncIterator[AsyncResolvedDatasetObject]:
        cursor = None
        seen_cursors: set[str] = set()
        seen_object_uids: set[str] = set()
        seen_role_ordinals: set[tuple[str, int]] = set()
        seen_logical_paths: set[str] = set()
        episode_ordinals: set[int] = set()
        referenced_episode_ordinals: set[int] = set()
        previous_order_key: tuple[str, int, str] | None = None
        yielded = 0
        yielded_size_bytes = 0
        while True:
            page = await self._transport.list_objects(self._revision, role=self._role, cursor=cursor)
            resolved_page: list[AsyncResolvedDatasetObject] = []
            for document in page.results:
                role_ordinal = (document.role, document.ordinal)
                if (
                    document.object_uid in seen_object_uids
                    or role_ordinal in seen_role_ordinals
                    or document.logical_path in seen_logical_paths
                ):
                    raise DatasetIntegrityError("duplicate_manifest_object")
                order_key = _manifest_object_order_key(document)
                if previous_order_key is not None and order_key <= previous_order_key:
                    raise DatasetIntegrityError("invalid_manifest_order")
                previous_order_key = order_key
                seen_object_uids.add(document.object_uid)
                seen_role_ordinals.add(role_ordinal)
                seen_logical_paths.add(document.logical_path)
                _record_episode_grouping(document, episode_ordinals, referenced_episode_ordinals)
                yielded += 1
                yielded_size_bytes += document.size_bytes
                if yielded > self._revision.manifest.object_count:
                    raise DatasetIntegrityError("manifest_object_count_mismatch")
                if yielded_size_bytes > self._revision.manifest.total_size_bytes:
                    raise DatasetIntegrityError("manifest_total_size_mismatch")
                resolved_page.append(AsyncResolvedDatasetObject(document, self._revision, self._transport))
            await self._include_observed_documents(page.results)
            if self._role is not None:
                await self._validate_episode_references(page.results)
            for resolved_object in resolved_page:
                yield resolved_object
            next_cursor = page.next_cursor
            if next_cursor is None:
                break
            if next_cursor in seen_cursors or not page.results:
                raise DatasetIntegrityError("invalid_cursor_sequence")
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        if self._role is None and yielded != self._revision.manifest.object_count:
            raise DatasetIntegrityError("manifest_object_count_mismatch")
        if self._role is None and yielded_size_bytes != self._revision.manifest.total_size_bytes:
            raise DatasetIntegrityError("manifest_total_size_mismatch")
        if self._role is None:
            _validate_episode_grouping(episode_ordinals, referenced_episode_ordinals)

    def __aiter__(self) -> AsyncIterator[AsyncResolvedDatasetObject]:
        return self._iterate()

    async def get(self, ordinal: int) -> AsyncResolvedDatasetObject:
        if self._role is None:
            raise TypeError("ordinal lookup requires a role-specific collection")
        _validate_direct_lookup_ordinal(ordinal)
        page = await self._transport.list_objects(self._revision, role=self._role, ordinal=ordinal)
        if not page.results:
            raise IndexError(f"No {self._role!r} object exists at ordinal {ordinal}.")
        document = page.results[0]
        _validate_ordinal_manifest_bounds(document, self._revision)
        await self._include_observed_documents([document])
        await self._validate_episode_references([document])
        return AsyncResolvedDatasetObject(document, self._revision, self._transport)

    def __getitem__(self, ordinal: int) -> Coroutine[Any, Any, AsyncResolvedDatasetObject]:
        return self.get(ordinal)


class AsyncResolvedDataset:
    """Async resolved immutable dataset revision."""

    def __init__(self, revision: ResolvedRevisionDocument, transport: AsyncDatasetResolverTransport) -> None:
        self._revision = revision
        self._transport = transport
        self._episode_reference_index = _EpisodeReferenceIndex()
        self._episode_reference_lock = asyncio.Lock()
        self._observed_manifest_bounds = _ObservedManifestBounds(revision)
        self.objects = self._collection(role=None)
        self.episodes = self._collection(role="episode")

    def _collection(self, *, role: str | None) -> AsyncDatasetObjectCollection:
        return AsyncDatasetObjectCollection(
            self._revision,
            self._transport,
            role=role,
            episode_reference_index=self._episode_reference_index,
            episode_reference_lock=self._episode_reference_lock,
            observed_manifest_bounds=self._observed_manifest_bounds,
        )

    @property
    def canonical_reference(self) -> str:
        return self._revision.canonical_reference

    @property
    def requested_reference(self) -> str:
        return self._revision.requested_reference

    @property
    def display_reference(self) -> str:
        return self._revision.display_reference

    @property
    def dataset_uid(self) -> str:
        return self._revision.dataset_uid

    @property
    def revision_uid(self) -> str:
        return self._revision.revision_uid

    @property
    def revision_sha256(self) -> str:
        return self._revision.revision_sha256

    @property
    def manifest_sha256(self) -> str:
        return self._revision.manifest.sha256

    @property
    def object_count(self) -> int:
        return self._revision.manifest.object_count

    @property
    def total_size_bytes(self) -> int:
        return self._revision.manifest.total_size_bytes

    @property
    def rights(self) -> Mapping[str, Any]:
        return self._revision.rights.model_dump()

    def for_role(self, role: str) -> AsyncDatasetObjectCollection:
        return self._collection(role=role)

    async def close(self) -> None:
        await self._transport.close()

    async def __aenter__(self) -> AsyncResolvedDataset:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self.close()

    def __repr__(self) -> str:
        return (
            f"AsyncResolvedDataset(canonical_reference={self.canonical_reference!r}, object_count={self.object_count})"
        )


def load(
    reference: str,
    *,
    revision: str | None = None,
    streaming: bool = True,
    base_url: str = DEFAULT_BASE_URL,
) -> ResolvedDataset:
    """Resolve a public immutable dataset without accepting or sending credentials."""

    if streaming is not True:
        raise UnsupportedDatasetModeError("streaming=False is not supported by the public v1 loader")
    parsed_reference = parse_dataset_reference(reference, revision)
    _warn_if_mutable_alias(parsed_reference.exact_revision_sha256)
    transport = SyncDatasetResolverTransport(base_url)
    try:
        resolved_revision = transport.resolve(parsed_reference)
    except Exception:
        transport.close()
        raise
    return ResolvedDataset(resolved_revision, transport)


async def async_load(
    reference: str,
    *,
    revision: str | None = None,
    streaming: bool = True,
    base_url: str = DEFAULT_BASE_URL,
) -> AsyncResolvedDataset:
    """Asynchronously resolve a public immutable dataset without credentials."""

    if streaming is not True:
        raise UnsupportedDatasetModeError("streaming=False is not supported by the public v1 loader")
    parsed_reference = parse_dataset_reference(reference, revision)
    _warn_if_mutable_alias(parsed_reference.exact_revision_sha256)
    transport = AsyncDatasetResolverTransport(base_url)
    try:
        resolved_revision = await transport.resolve(parsed_reference)
    except BaseException:
        await transport.close()
        raise
    return AsyncResolvedDataset(resolved_revision, transport)


__all__ = [
    "AsyncDatasetObjectCollection",
    "AsyncResolvedDataset",
    "AsyncResolvedDatasetObject",
    "DatasetObjectCollection",
    "ResolvedDataset",
    "ResolvedDatasetObject",
    "async_load",
    "load",
]
