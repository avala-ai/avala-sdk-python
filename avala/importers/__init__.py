"""Import datasets into Avala Mission Control from external sources.

A small, extensible adapter framework: each source ("folder", and — coming next —
"lerobot", "coco", "cloud") is a function registered in a registry and dispatched by
:func:`import_dataset`. Adapters convert/stage an external source into Avala-ingestible
files and create a Mission Control dataset via the reusable
``client.datasets.create_from_local`` / ``upload_files`` backbone.

Example::

    from avala import Client
    from avala.importers import import_folder

    ds = import_folder(Client(), source="./frames", name="My Drive", slug="my-drive")
    print(ds.uid, ds.data_type)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from avala._client import Client
    from avala.types.dataset import Dataset

__all__ = [
    "import_dataset",
    "import_folder",
    "import_lerobot",
    "import_cloud",
    "import_ros_bag",
    "register_importer",
    "available_importers",
    "detect_data_type",
]

# File suffixes the Avala ingest pipeline indexes, per data_type. These MUST match the
# extensions the platform actually admits: a dataset created from files the ingester
# rejects finalizes empty/unusable. Keep this list conservative — only extensions Avala
# indexes belong here.
#
# Suffixes (not just final extensions) so multi-part forms like ``.alp.gz`` and
# ``.compressed.ply`` are matched. ``.ply`` is treated as a SPLAT.
_INDEXED_SUFFIXES: Dict[str, Tuple[str, ...]] = {
    "image": (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"),
    "video": (".mp4", ".webm", ".mkv", ".mov", ".episode"),
    "lidar": (".alp", ".alp.gz"),
    "mcap": (".mcap",),
    "splat": (".ply", ".compressed.ply", ".splat", ".sog", ".spz", ".ksplat"),
}

# Reverse index (suffix -> data_type) for auto-detection. Longest suffixes first
# so ``.compressed.ply`` resolves before ``.ply`` and ``.alp.gz`` before any
# single-part match.
_SUFFIX_TO_TYPE = sorted(
    ((suffix, dtype) for dtype, suffixes in _INDEXED_SUFFIXES.items() for suffix in suffixes),
    key=lambda pair: len(pair[0]),
    reverse=True,
)

Importer = Callable[..., "Dataset"]
_REGISTRY: Dict[str, Importer] = {}


def register_importer(name: str, fn: Importer) -> None:
    """Register a source adapter under ``name`` (e.g. 'folder', 'lerobot')."""
    _REGISTRY[name] = fn


def available_importers() -> List[str]:
    """Return the registered importer source names, sorted."""
    return sorted(_REGISTRY)


def import_dataset(source_type: str, client: "Client", **kwargs: Any) -> "Dataset":
    """Dispatch to a registered importer by ``source_type``."""
    if source_type not in _REGISTRY:
        raise ValueError(f"unknown importer {source_type!r}; available: {available_importers()}")
    return _REGISTRY[source_type](client, **kwargs)


def _detect_one(relative: str) -> Optional[str]:
    """Return the data_type for a single relative path, or ``None`` if unindexed."""
    lower = relative.lower()
    for suffix, dtype in _SUFFIX_TO_TYPE:
        if lower.endswith(suffix):
            return dtype
    return None


def detect_data_type(files: List[tuple[str, str]]) -> str:
    """Infer the Avala ``data_type`` from a set of ``(local_path, relative)`` files.

    Only counts files whose suffix the server actually indexes. Returns the single
    data_type if all recognized files agree; raises ``ValueError`` if no files are
    recognized or the set is mixed (pass ``data_type`` explicitly).
    """
    found = {dtype for _local, relative in files if (dtype := _detect_one(relative)) is not None}
    if not found:
        raise ValueError(
            "could not infer data_type from file extensions; pass data_type explicitly "
            "(one of: image, video, lidar, mcap, splat)"
        )
    if len(found) > 1:
        raise ValueError(f"mixed data types detected ({sorted(found)}); pass data_type explicitly to disambiguate")
    return next(iter(found))


def _assert_server_indexable(files: List[tuple[str, str]], data_type: str) -> None:
    """Guard against the empty-dataset trap: refuse to import when none of the files
    carry a suffix Avala indexes for ``data_type``.

    Without this, an explicit (or mis-detected) ``data_type`` whose files the ingester
    rejects would upload everything and then finalize an empty, unusable dataset.
    """
    admitted = _INDEXED_SUFFIXES.get(data_type)
    if admitted is None:
        raise ValueError(f"unsupported data_type {data_type!r}; expected one of {sorted(_INDEXED_SUFFIXES)}")
    if not any(relative.lower().endswith(admitted) for _local, relative in files):
        raise ValueError(
            f"none of the files are indexable as data_type={data_type!r}; the server only "
            f"ingests {', '.join(admitted)} for this type. Convert the files or pass the correct data_type."
        )


def import_folder(
    client: "Client",
    *,
    source: str,
    name: str,
    slug: str,
    data_type: Optional[str] = None,
    visibility: str = "private",
    owner_name: Optional[str] = None,
    industry: Optional[int] = None,
    license: Optional[int] = None,
    workers: int = 8,
    on_progress: "Optional[Callable[[str, int], None]]" = None,
    wait: bool = False,
    wait_timeout: float = 3600.0,
) -> "Dataset":
    """Create a Mission Control dataset from a local file or directory.

    The universal importer: uploads any local media (images / video / LiDAR / MCAP /
    splat) and creates the dataset. ``data_type`` is inferred from file extensions
    when omitted.
    """
    from avala.resources.datasets import gather_local_files

    files = gather_local_files(source)
    if not files:
        raise ValueError(f"no files found in {source}")
    resolved_type = data_type or detect_data_type(files)
    _assert_server_indexable(files, resolved_type)
    return client.datasets.create_from_local(
        source=source,
        name=name,
        slug=slug,
        data_type=resolved_type,
        visibility=visibility,
        owner_name=owner_name,
        industry=industry,
        license=license,
        workers=workers,
        on_progress=on_progress,
        wait=wait,
        wait_timeout=wait_timeout,
    )


register_importer("folder", import_folder)

# Register additional source adapters. Kept at the end so ``register_importer`` is
# defined first. These only import their heavy deps (lerobot / rosbags / mcap / PIL)
# lazily inside the importer (cloud is dependency-light), so importing them here stays cheap.
from avala.importers.cloud import import_cloud  # noqa: E402,F401
from avala.importers.lerobot import import_lerobot  # noqa: E402,F401
from avala.importers.rosbag import import_ros_bag  # noqa: E402,F401
