"""Convert Avala sequence datasets to the LeRobot v3 dataset format.

No export step: reads sequences/frames directly through the SDK and writes a
`LeRobot <https://github.com/huggingface/lerobot>`_ v3 dataset using the official
``lerobot`` library (``LeRobotDataset.create`` → ``add_frame`` → ``save_episode``
→ ``finalize``).

One Avala **sequence** becomes one LeRobot **episode**. Each camera maps to an
``observation.images.<cam>`` feature; ``timestamp``/``frame_index``/
``episode_index`` are derived by lerobot from ``--fps``; the ``task`` string comes
from the caller.

Avala sequence frames expose a flat ``image_urls`` list (one signed URL per camera,
no inline dimensions), plus the LiDAR ``asset_url`` and calibration. Cameras are
therefore named positionally (``cam0``, ``cam1`` …) and each camera's resolution is
probed once from frame 0.

**Honest scope.** Avala sequence datasets are annotation-centric: they reliably
provide camera frames + calibration, but not robot proprioception. So by default
this produces a *perception / vision-language* dataset. Robot ``observation.state``
and ``action`` are emitted **only** when an explicit ``state_key``/``action_key``
resolves to a numeric vector inside the raw frame dict (or, opt-in, the camera-rig
ego pose) — they are never fabricated as zeros.

Requires the optional ``lerobot`` extra (Python 3.12+)::

    pip install "avala[lerobot]"

Example::

    from avala import Client
    from avala.lerobot import export_dataset

    export_dataset(
        Client(), "my-org", "my-dataset",
        repo_id="my-hf-user/my-dataset", output_dir="./lerobot-out", fps=30,
        task="pick up the cube",
    )
"""

from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple

import httpx

if TYPE_CHECKING:
    from avala._client import Client
    from avala.types.dataset import DatasetSequence

__all__ = [
    "export_dataset",
    "build_features",
    "iter_frames",
    "convert_sequence",
    "discover_camera_specs",
]

_MEDIA_TIMEOUT = 30.0
_IMG_PREFIX = "observation.images."
_INSTALL_HINT = (
    'avala.lerobot requires the lerobot library (Python 3.12+). Install it with: pip install "avala[lerobot]"'
)

# A camera spec resolved from frame 0: (name, height, width).
CameraSpec = Tuple[str, int, int]


# ── lazy optional imports (kept out of module import so the helpers stay usable,
#    and the heavy/py3.12-only lerobot stack is only required at write) ──
def _numpy() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via install extra
        raise ModuleNotFoundError(_INSTALL_HINT) from exc
    return np


def _lerobot_dataset_cls() -> Any:
    try:
        from lerobot.datasets import LeRobotDataset
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via install extra
        raise ModuleNotFoundError(_INSTALL_HINT) from exc
    return LeRobotDataset


def _decode_image(content: bytes) -> Any:
    """Decode image bytes to an HWC uint8 numpy array (RGB)."""
    np = _numpy()
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via install extra
        raise ModuleNotFoundError(_INSTALL_HINT) from exc
    image = Image.open(io.BytesIO(content)).convert("RGB")
    return np.asarray(image)


# ── frame helpers (work on the raw sequence.frames dicts) ──
def _frames(sequence: "DatasetSequence") -> List[Dict[str, Any]]:
    return sequence.frames or []


def _camera_urls(frame: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Return ``(camera_name, image_url)`` per camera for a frame.

    The canonical Avala sequence-frame shape exposes a flat ``image_urls`` list
    (one signed URL per camera, no inline names/dimensions); cameras are named
    positionally (``cam0``, ``cam1`` …). A legacy ``images: [{image_url, ...}]``
    shape is tolerated as a fallback.
    """
    urls = frame.get("image_urls")
    if isinstance(urls, list) and urls:
        return [(f"cam{i}", url) for i, url in enumerate(urls) if url]
    out: List[Tuple[str, str]] = []
    for i, image in enumerate(frame.get("images") or []):
        url = image.get("image_url") if isinstance(image, dict) else None
        if url:
            out.append((f"cam{i}", url))
    return out


def _resolve_vector(frame: Dict[str, Any], dotted_key: str) -> List[float]:
    """Resolve a dotted key into the raw frame dict to a numeric vector.

    Raises ``KeyError`` if the path is absent and ``ValueError`` if it does not
    resolve to a flat list of numbers. State/action are all-or-nothing: a missing
    or malformed key is an error, never a silent zero-fill.
    """
    cur: Any = frame
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"key {dotted_key!r} not found in frame (missing {part!r})")
        cur = cur[part]
    if (
        isinstance(cur, (list, tuple))
        and cur
        and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in cur)
    ):
        return [float(x) for x in cur]
    raise ValueError(f"key {dotted_key!r} did not resolve to a non-empty numeric vector")


def _has_ego_pose(frame: Dict[str, Any]) -> bool:
    return isinstance(frame.get("device_position"), dict) and isinstance(frame.get("device_heading"), dict)


def _ego_pose(frame: Dict[str, Any], np: Any) -> Any:
    """7-dim camera-rig pose [x,y,z,qw,qx,qy,qz] from device_position+heading."""
    pos = frame.get("device_position")
    head = frame.get("device_heading")
    if not isinstance(pos, dict) or not isinstance(head, dict):
        raise ValueError("ego pose requested but frame has no device_position/device_heading")
    try:
        values = [pos["x"], pos["y"], pos["z"], head["w"], head["x"], head["y"], head["z"]]
    except KeyError as exc:
        raise ValueError(f"ego pose is missing component {exc}") from exc
    return np.asarray(values, dtype="float32")


def _expected_from_features(
    features: Dict[str, Dict[str, Any]],
) -> "Tuple[Dict[str, Tuple[int, int]], Optional[int], Optional[int]]":
    """Derive the per-frame schema to enforce: camera -> (H, W), state dim, action dim."""
    cam_hw: Dict[str, Tuple[int, int]] = {}
    for key, spec in features.items():
        if key.startswith(_IMG_PREFIX):
            _, height, width = spec["shape"]
            cam_hw[key[len(_IMG_PREFIX) :]] = (int(height), int(width))
    state_dim = features["observation.state"]["shape"][0] if "observation.state" in features else None
    action_dim = features["action"]["shape"][0] if "action" in features else None
    return cam_hw, state_dim, action_dim


def discover_camera_specs(
    frame: Dict[str, Any],
    media_client: httpx.Client,
    *,
    camera_keys: Optional[List[str]] = None,
) -> List[CameraSpec]:
    """Probe a frame's images to resolve ``(camera_name, height, width)`` per camera.

    Avala frames carry no inline image dimensions, so each camera's image is fetched
    and decoded once to determine its resolution. ``camera_keys`` (e.g. ``["cam0"]``)
    restricts to a subset.
    """
    cams = _camera_urls(frame)
    if not cams:
        raise ValueError("frame has no image_urls; cannot build camera features")
    if camera_keys is not None:
        wanted = set(camera_keys)
        cams = [(name, url) for name, url in cams if name in wanted]
        if not cams:
            available = [name for name, _ in _camera_urls(frame)]
            raise ValueError(f"none of the requested cameras {sorted(wanted)} found in {available}")
    specs: List[CameraSpec] = []
    for name, url in cams:
        resp = media_client.get(url)
        resp.raise_for_status()
        try:
            decoded = _decode_image(resp.content)
        except Exception as exc:
            raise ValueError(f"failed to decode image for camera {name!r} ({url}): {exc}") from exc
        specs.append((name, int(decoded.shape[0]), int(decoded.shape[1])))
    return specs


def build_features(
    sequence: "DatasetSequence",
    *,
    camera_specs: List[CameraSpec],
    state_key: Optional[str] = None,
    action_key: Optional[str] = None,
    include_ego_pose: bool = False,
    use_videos: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Build the LeRobot v3 ``features`` dict from resolved camera specs + frame 0.

    One ``observation.images.<cam>`` entry per camera (``dtype`` ``video`` or
    ``image``, ``shape`` ``(3, H, W)`` from ``camera_specs``). Adds
    ``observation.state``/``action`` only when the corresponding key resolves to a
    numeric vector on frame 0, or a 7-dim ego-pose ``observation.state`` when
    ``include_ego_pose``. Camera specs come from :func:`discover_camera_specs`.
    """
    if state_key and include_ego_pose:
        raise ValueError("use either state_key or include_ego_pose, not both (both map to observation.state)")
    if not camera_specs:
        raise ValueError("no camera_specs provided; nothing to build features from")

    frames = _frames(sequence)
    if not frames:
        raise ValueError("sequence has no frames; nothing to build features from")
    first = frames[0]

    names = [name for name, _, _ in camera_specs]
    collisions = sorted({name for name in names if names.count(name) > 1})
    if collisions:
        raise ValueError(f"camera names collide: {collisions}; deduplicate camera_specs")

    features: Dict[str, Dict[str, Any]] = {}
    dtype = "video" if use_videos else "image"
    for name, height, width in camera_specs:
        features[f"{_IMG_PREFIX}{name}"] = {
            "dtype": dtype,
            "shape": (3, int(height), int(width)),
            "names": ["channels", "height", "width"],
        }

    if state_key:
        vector = _resolve_vector(first, state_key)
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (len(vector),),
            "names": [f"state_{i}" for i in range(len(vector))],
        }
    if include_ego_pose:
        if not _has_ego_pose(first):
            raise ValueError("include_ego_pose is set but frame 0 has no device_position/device_heading")
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "qw", "qx", "qy", "qz"],
        }
    if action_key:
        vector = _resolve_vector(first, action_key)
        features["action"] = {
            "dtype": "float32",
            "shape": (len(vector),),
            "names": [f"action_{i}" for i in range(len(vector))],
        }
    return features


def iter_frames(
    sequence: "DatasetSequence",
    *,
    media_client: httpx.Client,
    features: Dict[str, Dict[str, Any]],
    state_key: Optional[str] = None,
    action_key: Optional[str] = None,
    include_ego_pose: bool = False,
    task: str,
) -> Iterator[Dict[str, Any]]:
    """Yield ``add_frame``-ready dicts for one sequence (one episode).

    Validates every frame against the schema in ``features`` (camera set + image
    H/W, state/action vector dims) so a later frame that drifts — a different
    resolution, a dropped/added camera, or a different-length state vector — fails
    with a clear error instead of silently corrupting the dataset.
    """
    np = _numpy()
    cam_hw, state_dim, action_dim = _expected_from_features(features)
    expected = set(cam_hw)
    for frame_index, frame in enumerate(_frames(sequence)):
        by_name = dict(_camera_urls(frame))
        if set(by_name) != expected:
            raise ValueError(
                f"frame {frame_index} cameras {sorted(by_name)} do not match the dataset cameras {sorted(expected)}"
            )
        sample: Dict[str, Any] = {"task": task}
        for cam, (height, width) in cam_hw.items():
            resp = media_client.get(by_name[cam])
            resp.raise_for_status()
            try:
                decoded = _decode_image(resp.content)
            except Exception as exc:
                raise ValueError(
                    f"failed to decode image for camera {cam!r} on frame {frame_index} ({by_name[cam]}): {exc}"
                ) from exc
            if decoded.shape[:2] != (height, width):
                raise ValueError(
                    f"camera {cam!r} frame {frame_index} is {decoded.shape[1]}x{decoded.shape[0]}, "
                    f"but the dataset declares {width}x{height} (resolution must be constant per camera)"
                )
            sample[f"{_IMG_PREFIX}{cam}"] = decoded
        if state_key:
            vector = _resolve_vector(frame, state_key)
            if state_dim is not None and len(vector) != state_dim:
                raise ValueError(
                    f"observation.state on frame {frame_index} has length {len(vector)}, expected {state_dim}"
                )
            sample["observation.state"] = np.asarray(vector, dtype="float32")
        if include_ego_pose:
            sample["observation.state"] = _ego_pose(frame, np)
        if action_key:
            vector = _resolve_vector(frame, action_key)
            if action_dim is not None and len(vector) != action_dim:
                raise ValueError(f"action on frame {frame_index} has length {len(vector)}, expected {action_dim}")
            sample["action"] = np.asarray(vector, dtype="float32")
        yield sample


def convert_sequence(
    ds: Any,
    sequence: "DatasetSequence",
    *,
    media_client: httpx.Client,
    features: Dict[str, Dict[str, Any]],
    state_key: Optional[str] = None,
    action_key: Optional[str] = None,
    include_ego_pose: bool = False,
    task: str,
) -> int:
    """Append one Avala sequence to an open LeRobotDataset as one episode.

    Returns the number of frames written. Does not call ``finalize`` — the caller
    owns the dataset lifecycle so multiple sequences share one dataset.
    """
    count = 0
    for sample in iter_frames(
        sequence,
        media_client=media_client,
        features=features,
        state_key=state_key,
        action_key=action_key,
        include_ego_pose=include_ego_pose,
        task=task,
    ):
        ds.add_frame(sample)
        count += 1
    if count:
        ds.save_episode()
    return count


def _list_sequences(client: "Client", owner: str, slug: str, limit: Optional[int]) -> List[Any]:
    out: List[Any] = []
    cursor: Optional[str] = None
    while True:
        page = client.datasets.list_sequences(owner, slug, cursor=cursor)
        for seq in page.items:
            out.append(seq)
            if limit is not None and len(out) >= limit:
                return out
        if not page.has_more or not page.next_cursor:
            return out
        cursor = page.next_cursor


def export_dataset(
    client: "Client",
    owner: str,
    slug: str,
    *,
    repo_id: str,
    output_dir: "str | Path",
    fps: int = 30,
    task: str = "avala sequence",
    camera_keys: Optional[List[str]] = None,
    state_key: Optional[str] = None,
    action_key: Optional[str] = None,
    include_ego_pose: bool = False,
    robot_type: Optional[str] = None,
    use_videos: bool = True,
    limit: Optional[int] = None,
    push: bool = False,
    tags: Optional[List[str]] = None,
    repo_license: Optional[str] = None,
) -> Path:
    """Convert an Avala sequence dataset to a LeRobot v3 dataset on disk.

    Each sequence becomes one episode. Returns the output root path. Set
    ``push=True`` to upload to the Hugging Face Hub (requires HF auth) — off by
    default to avoid accidental publication. ``tags`` are appended to the dataset
    card (lerobot always adds ``LeRobot``/``robotics``; we always add ``avala``);
    ``repo_license`` overrides lerobot's default card license.
    """
    if state_key and include_ego_pose:
        raise ValueError("use either state_key or include_ego_pose, not both (both map to observation.state)")
    if "/" not in repo_id:
        raise ValueError("repo_id must be '<hf_user>/<name>'")
    if not task or not task.strip():
        raise ValueError("task must be a non-empty string")

    lerobot_dataset_cls = _lerobot_dataset_cls()
    out = Path(output_dir)

    sequences = _list_sequences(client, owner, slug, limit)
    if not sequences:
        raise ValueError(f"dataset {owner}/{slug} has no sequences to convert")

    media_client = httpx.Client(timeout=_MEDIA_TIMEOUT)
    ds = None
    try:
        # Derive the schema from the first NON-empty sequence (an empty leading
        # sequence must not block an otherwise-valid dataset). Frames are embedded
        # on the per-sequence GET, so always fetch the full sequence first.
        schema_sequence = None
        for seq_meta in sequences:
            candidate = client.datasets.get_sequence(owner, slug, seq_meta.uid)
            if _frames(candidate):
                schema_sequence = candidate
                break
        if schema_sequence is None:
            raise ValueError(f"dataset {owner}/{slug} has no non-empty sequences to convert")

        camera_specs = discover_camera_specs(_frames(schema_sequence)[0], media_client, camera_keys=camera_keys)
        features = build_features(
            schema_sequence,
            camera_specs=camera_specs,
            state_key=state_key,
            action_key=action_key,
            include_ego_pose=include_ego_pose,
            use_videos=use_videos,
        )
        if "observation.state" not in features and "action" not in features:
            warnings.warn(
                "No observation.state/action features detected — this is a perception-only "
                "(vision-language) dataset, not a policy-training dataset. Pass --state-key/"
                "--action-key (or --ego-pose-state) if the source has proprioception.",
                stacklevel=2,
            )

        ds = lerobot_dataset_cls.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=out,
            robot_type=robot_type,
            use_videos=use_videos,
        )

        for seq_meta in sequences:
            full = client.datasets.get_sequence(owner, slug, seq_meta.uid)
            if not _frames(full):
                warnings.warn(f"sequence {seq_meta.uid} has no frames; skipping", stacklevel=2)
                continue
            convert_sequence(
                ds,
                full,
                media_client=media_client,
                features=features,
                state_key=state_key,
                action_key=action_key,
                include_ego_pose=include_ego_pose,
                task=task,
            )
    except Exception:
        # Footer whatever episodes were already saved so a mid-batch failure does
        # not leave an unreadable (un-finalized) dataset; never mask the real error.
        if ds is not None:
            try:
                ds.finalize()
            except Exception:  # pragma: no cover - best effort
                pass
        raise
    else:
        # Mandatory in v3: without finalize() the parquet footers are never written
        # and the dataset is unreadable.
        ds.finalize()
    finally:
        media_client.close()

    if push:
        push_kwargs: Dict[str, Any] = {"tags": ["avala", *(tags or [])]}
        if repo_license:
            push_kwargs["license"] = repo_license
        ds.push_to_hub(**push_kwargs)
    return out
