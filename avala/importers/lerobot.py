"""Import a LeRobot dataset (local or Hugging Face Hub) into Avala as an MCAP dataset.

Each LeRobot *episode* becomes one ``.mcap`` file, which Avala ingests as one MCAP
episode (one ``.mcap`` = one ``DatasetItem`` = one ``McapEpisode``). Inside each file:

* **Camera streams** are written as ``foxglove.CompressedImage`` (protobuf) — the format
  the Avala server indexer and the Mission Control MCAP viewer both understand, so the
  frames render as image panels.
* **Proprioception** (``observation.state``, ``action``) is written as
  ``google.protobuf.Struct`` ``{"data": [floats]}`` messages on ``/observation/state`` and
  ``/action``.
* **Every other non-image per-frame column** — numeric or categorical, e.g. rewards,
  ``episode_uuid``, ``failure_type`` — is carried on its own topic (``a.b`` -> ``/a/b``),
  and a ``control_source`` column (``policy``/``teleop``/``intervention``/``hold``, often
  stored as ``annotation.<vendor>.control_source``) on the stable topic
  ``/lerobot/control_source``. The per-frame task text goes to ``/lerobot/task``. Nothing
  is dropped. Full table: :mod:`avala.converters.lerobot_v3.mcap`.
* An ``avala.lerobot`` MCAP metadata record keeps the original feature specs and the
  feature->topic map, so the episode can be exported back to LeRobot unchanged.

  NOTE: Mission Control's embedded MCAP viewer renders images, point clouds and logs, but
  does not yet *chart* scalar time-series. State/action are therefore preserved and
  raw-viewable, but not plotted. That's a viewer feature, not an import limitation.

Two interchangeable readers (``backend=``):

* ``"lerobot"`` — the ``lerobot`` library (``pip install 'avala[lerobot]'``; Python 3.12+,
  pulls in torch), verified against ``lerobot`` 0.5.x;
* ``"core"`` — the torch-free :mod:`avala.converters.lerobot_v3` reader
  (``pip install 'avala[lerobot-core-video]'``), LeRobot v3 datasets only. A ``repo_id``
  without ``root`` is downloaded with ``huggingface_hub`` when it is installed.

``"auto"`` (the default) uses the lerobot library when it is importable and the core
reader otherwise. Both produce the same MCAP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence

from avala.importers import register_importer

if TYPE_CHECKING:
    from avala._client import Client
    from avala.types.dataset import Dataset

__all__ = ["import_lerobot", "write_episode_mcap"]


def write_episode_mcap(out_path: str, frames: Iterable[Dict[str, Any]], **kwargs: Any) -> int:
    """Write writer frames to an MCAP file; see :func:`avala.converters.lerobot_v3.mcap.write_episode_mcap`.

    Kept here (it is the historical import location); the implementation is
    dependency-light (mcap + foxglove protobuf + pillow, no lerobot/torch).
    """
    from avala.converters.lerobot_v3.mcap import write_episode_mcap as _write

    return _write(out_path, frames, **kwargs)


def _to_hwc_uint8(value: Any) -> Any:
    """Normalize a LeRobot image (CHW or HWC, float[0,1] or uint8) to HWC uint8."""
    from avala.converters.lerobot_v3._values import to_hwc_uint8

    return to_hwc_uint8(value)


def _to_list(value: Any) -> List[float]:
    """Flatten a tensor / array / scalar of numbers to a flat list of floats."""
    from avala.converters.lerobot_v3._values import flat_numbers

    return [float(x) for x in flat_numbers(value)]


def _episode_samples(dataset: Any) -> Iterator[tuple]:
    """Yield ``(episode_index, samples_iter)`` groups over all loaded frames.

    Frames within a LeRobot episode are contiguous and ordered, so grouping the linear
    ``dataset[i]`` stream by each sample's ``episode_index`` recovers per-episode bounds
    without relying on ``episode_data_index`` (removed in lerobot 0.5) or the
    absolute/relative index remapping that subset loading applies.
    """
    import itertools

    def _samples() -> Iterator[tuple]:
        for i in range(len(dataset)):
            sample = dataset[i]
            ep = sample.get("episode_index", 0)
            ep = int(ep.item() if hasattr(ep, "item") else ep)
            yield ep, sample

    for ep, group in itertools.groupby(_samples(), key=lambda pair: pair[0]):
        yield ep, (sample for _ep, sample in group)


class _LibrarySource:
    """Frames via the ``lerobot`` library."""

    def __init__(self, rid: str, root: Optional[str]) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        self._rid = rid
        self._root = root
        meta = LeRobotDatasetMetadata(rid, root=root)
        self.camera_keys: List[str] = list(meta.camera_keys)
        self.features: Dict[str, Any] = dict(meta.features)
        self.fps = float(meta.fps)
        self.total_episodes = int(meta.total_episodes)
        self.robot_type: Optional[str] = getattr(meta, "robot_type", None)
        info = getattr(meta, "info", None)
        self.codebase_version: Optional[str] = info.get("codebase_version") if isinstance(info, dict) else None

    def episode_samples(self, selected: Sequence[int], keys: Sequence[str]) -> Iterator[tuple]:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        # Pass ``episodes=`` so LeRobot only downloads/loads the requested episodes.
        dataset = LeRobotDataset(self._rid, root=self._root, episodes=list(selected))
        return _episode_samples(dataset)


class _CoreSource:
    """Frames via the torch-free :mod:`avala.converters.lerobot_v3` reader (v3 layout only)."""

    def __init__(self, repo_id: Optional[str], root: Optional[str]) -> None:
        from avala.converters.lerobot_v3.reader import LeRobotV3Dataset

        self._ds = LeRobotV3Dataset(root if root else _download_snapshot(repo_id or ""))
        self.camera_keys = self._ds.camera_keys
        self.features: Dict[str, Any] = self._ds.features
        self.fps = self._ds.fps
        self.total_episodes = self._ds.total_episodes
        self.robot_type = self._ds.robot_type
        self.codebase_version: Optional[str] = str(self._ds.info.get("codebase_version"))

    def episode_samples(self, selected: Sequence[int], keys: Sequence[str]) -> Iterator[tuple]:
        for ep in selected:
            yield ep, self._ds.iter_frames(ep, keys=list(keys))


def _download_snapshot(repo_id: str) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "importing a Hub repo_id without the lerobot library needs huggingface_hub "
            "(pip install huggingface_hub), or pass root= to a local copy of the dataset"
        ) from exc
    return str(snapshot_download(repo_id, repo_type="dataset"))


_BACKENDS = ("auto", "lerobot", "core")


def _open_source(repo_id: Optional[str], root: Optional[str], rid: str, backend: str) -> Any:
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}")
    if backend in ("auto", "lerobot"):
        try:
            import lerobot.datasets.lerobot_dataset  # noqa: F401
        except ModuleNotFoundError as exc:
            if backend == "lerobot":
                raise ModuleNotFoundError(
                    "LeRobot import with backend='lerobot' requires the 'lerobot' extra. "
                    "Install it with: pip install 'avala[lerobot]'"
                ) from exc
        else:
            return _LibrarySource(rid, root)
    try:
        import pyarrow  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without either extra
        raise ModuleNotFoundError(
            "LeRobot import requires either the torch-free 'lerobot-core-video' extra "
            "(pip install 'avala[lerobot-core-video]') or the 'lerobot' extra (pip install 'avala[lerobot]')"
        ) from exc
    return _CoreSource(repo_id, root)


def import_lerobot(
    client: "Client",
    *,
    name: str,
    slug: str,
    repo_id: Optional[str] = None,
    root: Optional[str] = None,
    episodes: Optional[Sequence[int]] = None,
    camera_keys: Optional[Sequence[str]] = None,
    state_keys: Optional[Sequence[str]] = None,
    fps: Optional[float] = None,
    visibility: str = "private",
    owner_name: Optional[str] = None,
    industry: Optional[int] = None,
    license: Optional[int] = None,
    workers: int = 8,
    on_progress: "Optional[Callable[[str, int], None]]" = None,
    wait: bool = False,
    wait_timeout: float = 3600.0,
    backend: str = "auto",
) -> "Dataset":
    """Import a LeRobot dataset into Avala as an MCAP dataset.

    Provide ``repo_id`` (downloaded from the Hugging Face Hub) and/or ``root`` (a local
    dataset directory). Each episode is converted to one ``.mcap`` file and uploaded; the
    resulting Avala dataset has ``data_type="mcap"``.

    ``camera_keys`` / ``state_keys`` default to the dataset's camera features and
    ``observation.state`` + ``action``. ``episodes`` limits the export to specific episode
    indices (default: all). ``backend`` picks the reader (see the module docstring).
    """
    import os
    import tempfile

    from avala.converters.lerobot_v3.mcap import build_frame, episode_metadata, plan_topics

    if not repo_id and not root:
        raise ValueError("provide repo_id (Hugging Face Hub) and/or root (local dataset path)")

    rid = repo_id or slug

    # Read metadata first (cheap, no video download) to resolve features and validate the
    # episode selection before pulling any media.
    meta = _open_source(repo_id, root, rid, backend)
    available_cameras = list(meta.camera_keys)
    if camera_keys is not None:
        unknown = [k for k in camera_keys if k not in available_cameras]
        if unknown:
            raise ValueError(f"unknown camera keys {unknown}; available cameras: {sorted(available_cameras)}")
    resolved_cameras: Sequence[str] = list(camera_keys) if camera_keys is not None else available_cameras
    if not resolved_cameras:
        raise ValueError("no camera features found in the LeRobot dataset; pass camera_keys explicitly")
    if state_keys is not None:
        unknown_states = [k for k in state_keys if k not in meta.features]
        if unknown_states:
            raise ValueError(f"unknown state keys {unknown_states}; available features: {sorted(meta.features)}")
    resolved_states: Sequence[str] = (
        list(state_keys)
        if state_keys is not None
        else [k for k in ("observation.state", "action") if k in meta.features]
    )
    resolved_fps: float = float(fps if fps is not None else meta.fps)

    total_episodes = int(meta.total_episodes)
    selected = list(episodes) if episodes is not None else list(range(total_episodes))
    invalid = [ep for ep in selected if not 0 <= ep < total_episodes]
    if invalid:
        raise ValueError(f"episode indices {invalid} out of range [0, {total_episodes})")
    if not selected:
        raise ValueError("no episodes selected to import")

    # Every other non-image per-frame column (control_source, episode_uuid, failure_type,
    # rewards, ...) is carried on its own topic too — see avala.converters.lerobot_v3.mcap.
    plan = plan_topics(meta.features, resolved_cameras, resolved_states)

    with tempfile.TemporaryDirectory(prefix="avala-lerobot-") as tmp:
        written = 0
        for ep, samples in meta.episode_samples(selected, plan.keys):
            out_path = os.path.join(tmp, f"episode_{ep:06d}.mcap")
            frames = (build_frame(sample, plan, resolved_fps) for sample in samples)
            metadata = episode_metadata(
                plan,
                fps=resolved_fps,
                episode_index=ep,
                robot_type=meta.robot_type,
                codebase_version=meta.codebase_version,
            )
            if write_episode_mcap(out_path, frames, metadata=metadata) > 0:
                written += 1
            else:
                os.remove(out_path)

        if written == 0:
            raise ValueError("no non-empty episodes to import")

        return client.datasets.create_from_local(
            source=tmp,
            name=name,
            slug=slug,
            data_type="mcap",
            visibility=visibility,
            owner_name=owner_name,
            industry=industry,
            license=license,
            workers=workers,
            on_progress=on_progress,
            wait=wait,
            wait_timeout=wait_timeout,
        )


register_importer("lerobot", import_lerobot)
