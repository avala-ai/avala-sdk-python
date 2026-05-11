"""Annotations resource — bulk-edit cuboids and other 3D annotations."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Iterator
from typing import Any

from avala.resources._base import BaseAsyncResource, BaseSyncResource

DEFAULT_CHUNK_SIZE = 500


def _bulk_edit_path(owner: str, slug: str) -> str:
    return f"/datasets/{owner}/{slug}/bulk-edition/"


def _chunk(edits: Iterable[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    if size <= 0:
        raise ValueError("chunk_size must be positive")
    buffer: list[dict[str, Any]] = []
    for edit in edits:
        buffer.append(edit)
        if len(buffer) >= size:
            yield buffer
            buffer = []
    if buffer:
        yield buffer


class Annotations(BaseSyncResource):
    def bulk_edit(
        self,
        owner: str,
        slug: str,
        edits: Iterable[dict[str, Any]],
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> Iterator[tuple[int, Any]]:
        """POST cuboid / point-cloud-polyline edits to a dataset in chunks.

        Yields ``(chunk_index, response_body)`` for each successfully posted
        chunk. Raises on the first chunk that errors; the chunk index is
        included in the exception so the caller can resume.

        Edits follow the ``DatasetObjectEditionSerializer`` shape: each entry
        is a dict with ``action``, ``object_uuid``, ``object_type``, exactly
        one of ``dataset_item_uid`` / ``sequence_uid``, and (for upserts)
        ``object_data``. See the server's ``bulk_edition`` endpoint for the
        full contract.
        """
        path = _bulk_edit_path(owner, slug)
        for index, batch in enumerate(_chunk(edits, chunk_size)):
            try:
                response = self._transport.request("POST", path, json=batch)
            except Exception as exc:
                raise RuntimeError(f"bulk_edit chunk {index} failed: {exc}") from exc
            yield index, response


class AsyncAnnotations(BaseAsyncResource):
    async def bulk_edit(
        self,
        owner: str,
        slug: str,
        edits: Iterable[dict[str, Any]],
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> AsyncIterator[tuple[int, Any]]:
        """Async counterpart of :meth:`Annotations.bulk_edit`.

        Returns an async iterator. Use::

            async for index, response in await client.annotations.bulk_edit(...):
                ...
        """
        path = _bulk_edit_path(owner, slug)

        async def _iter() -> AsyncIterator[tuple[int, Any]]:
            for index, batch in enumerate(_chunk(edits, chunk_size)):
                try:
                    response = await self._transport.request("POST", path, json=batch)
                except Exception as exc:
                    raise RuntimeError(f"bulk_edit chunk {index} failed: {exc}") from exc
                yield index, response

        return _iter()
