"""PyTorch data loading for Avala datasets.

Stream annotated frames straight from Avala into training — no ``exports.create``,
no archive download, no ETL. The datasets here page lazily over the dataset's items
endpoint (which already returns presigned media URLs and inline annotations), so a
training run reads exactly the frames it needs.

Requires the optional ``torch`` extra::

    pip install "avala[torch]"

Example::

    from avala import Client
    from avala.torch import AvalaIterableDataset
    from torch.utils.data import DataLoader

    client = Client()
    ds = AvalaIterableDataset(client, "my-org", "my-dataset", decode_images=True)
    loader = DataLoader(ds, batch_size=8, num_workers=4)  # workers shard automatically
    for batch in loader:
        ...
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional

try:
    import torch
    from torch.utils.data import Dataset, IterableDataset
except ModuleNotFoundError as exc:  # pragma: no cover - exercised via install extra
    raise ModuleNotFoundError('avala.torch requires PyTorch. Install it with: pip install "avala[torch]"') from exc

import httpx

from avala.types.dataset import DatasetItem

if TYPE_CHECKING:
    from avala._client import Client

__all__ = ["AvalaDataset", "AvalaIterableDataset"]

Sample = Dict[str, Any]
Transform = Callable[[Sample], Any]

_DEFAULT_PAGE_SIZE = 100
_MEDIA_TIMEOUT = 30.0


def _decode_image(content: bytes) -> Any:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via install extra
        raise ModuleNotFoundError(
            'Decoding images requires Pillow. Install it with: pip install "avala[torch]"'
        ) from exc
    return Image.open(io.BytesIO(content)).convert("RGB")


def _item_to_sample(
    item: DatasetItem,
    *,
    media_client: Optional[httpx.Client],
    decode_images: bool,
) -> Sample:
    """Map a :class:`DatasetItem` to a plain sample dict.

    Annotations are already inline on the item, so no per-frame round trip is needed.
    When ``decode_images`` is set, the presigned ``url`` is fetched and decoded.
    """
    sample: Sample = {
        "uid": item.uid,
        "key": item.key,
        "url": item.url,
        "annotations": item.annotations,
        "export_snippet": item.export_snippet,
        "metadata": item.metadata,
    }
    if decode_images and item.url and media_client is not None:
        resp = media_client.get(item.url)
        resp.raise_for_status()
        sample["image"] = _decode_image(resp.content)
    return sample


def _iter_item_pages(client: "Client", owner: str, slug: str, page_size: int) -> Iterator[DatasetItem]:
    """Lazily walk every item in a dataset via cursor pagination."""
    cursor: Optional[str] = None
    while True:
        page = client.datasets.list_items(owner, slug, limit=page_size, cursor=cursor)
        for item in page.items:
            yield item
        if not page.has_more:
            return
        cursor = page.next_cursor


def _resolve_distributed(rank: Optional[int], world_size: Optional[int]) -> tuple[int, int]:
    """Resolve (rank, world_size), preferring explicit args, then torch.distributed."""
    if rank is not None and world_size is not None:
        return rank, world_size
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
    except Exception:  # pragma: no cover - defensive
        pass
    return 0, 1


class AvalaIterableDataset(IterableDataset):  # type: ignore[type-arg]
    """Streaming, no-export dataset over an Avala dataset's items.

    Pages lazily over the items endpoint and shards work across DataLoader workers
    and (optionally) Distributed Data Parallel ranks, so each replica/worker reads a
    disjoint slice without materializing the whole dataset.

    Args:
        client: An Avala :class:`~avala._client.Client`.
        owner: Dataset owner (org slug).
        slug: Dataset slug.
        transform: Optional callable applied to each sample dict before it is yielded.
        decode_images: Fetch each item's presigned URL and decode it to a PIL image
            under the ``"image"`` key. Off by default so non-image datasets and
            URL-only pipelines stay cheap.
        page_size: Items fetched per API page.
        rank / world_size: Override DDP sharding; by default these are read from
            ``torch.distributed`` when initialized.
    """

    def __init__(
        self,
        client: "Client",
        owner: str,
        slug: str,
        *,
        transform: Optional[Transform] = None,
        decode_images: bool = False,
        page_size: int = _DEFAULT_PAGE_SIZE,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.client = client
        self.owner = owner
        self.slug = slug
        self.transform = transform
        self.decode_images = decode_images
        self.page_size = page_size
        self._rank = rank
        self._world_size = world_size

    def _shard(self) -> tuple[int, int]:
        """Compute (shard_id, num_shards) combining DDP rank and worker id."""
        rank, world_size = _resolve_distributed(self._rank, self._world_size)
        worker = torch.utils.data.get_worker_info()
        num_workers = worker.num_workers if worker is not None else 1
        worker_id = worker.id if worker is not None else 0
        num_shards = world_size * num_workers
        shard_id = rank * num_workers + worker_id
        return shard_id, num_shards

    def __iter__(self) -> Iterator[Any]:
        shard_id, num_shards = self._shard()
        media_client = httpx.Client(timeout=_MEDIA_TIMEOUT) if self.decode_images else None
        try:
            for idx, item in enumerate(_iter_item_pages(self.client, self.owner, self.slug, self.page_size)):
                if idx % num_shards != shard_id:
                    continue
                sample = _item_to_sample(item, media_client=media_client, decode_images=self.decode_images)
                yield self.transform(sample) if self.transform else sample
        finally:
            if media_client is not None:
                media_client.close()


class AvalaDataset(Dataset):  # type: ignore[type-arg]
    """Map-style, no-export dataset over an Avala dataset's items.

    Materializes the list of item UIDs up front (one cursor walk) so it supports
    ``len()`` and random access; each ``__getitem__`` re-fetches the item, which
    re-signs media URLs. Prefer :class:`AvalaIterableDataset` for large datasets and
    streaming throughput; use this when you need shuffling or indexed access.
    """

    def __init__(
        self,
        client: "Client",
        owner: str,
        slug: str,
        *,
        transform: Optional[Transform] = None,
        decode_images: bool = False,
        page_size: int = _DEFAULT_PAGE_SIZE,
    ) -> None:
        self.client = client
        self.owner = owner
        self.slug = slug
        self.transform = transform
        self.decode_images = decode_images
        self._uids: List[str] = [item.uid for item in _iter_item_pages(client, owner, slug, page_size)]
        self._media_client = httpx.Client(timeout=_MEDIA_TIMEOUT) if decode_images else None

    def __len__(self) -> int:
        return len(self._uids)

    def __getitem__(self, index: int) -> Any:
        item = self.client.datasets.get_item(self.owner, self.slug, self._uids[index])
        sample = _item_to_sample(item, media_client=self._media_client, decode_images=self.decode_images)
        return self.transform(sample) if self.transform else sample

    def close(self) -> None:
        if self._media_client is not None:
            self._media_client.close()
            self._media_client = None
