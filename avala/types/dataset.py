from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class Dataset(BaseModel):
    uid: str
    name: str
    slug: str
    item_count: int = 0
    status: Optional[str] = None
    data_type: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class DatasetItem(BaseModel):
    id: Optional[int] = None
    uid: str
    key: Optional[str] = None
    dataset: Optional[str] = None
    url: Optional[str] = None
    gpu_texture_url: Optional[str] = None
    thumbnails: Optional[List[str]] = None
    video_thumbnail: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    export_snippet: Optional[Dict[str, Any]] = None
    annotations: Optional[Dict[str, Any]] = None
    crop_data: Optional[Dict[str, Any]] = None
    related_items: Optional[List[Dict[str, Any]]] = None
    related_sequence_uid: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class DatasetSequence(BaseModel):
    uid: str
    key: Optional[str] = None
    custom_uuid: Optional[str] = None
    status: Optional[str] = None
    featured_image: Optional[str] = None
    number_of_frames: Optional[int] = None
    views: Optional[List[Dict[str, Any]]] = None
    crop_data: Optional[Dict[str, Any]] = None
    predefined_labels: Optional[List[Dict[str, Any]]] = None
    frames: Optional[List[Dict[str, Any]]] = None
    metrics: Optional[Dict[str, Any]] = None
    dataset_uid: Optional[str] = None
    allow_lidar_calibration: Optional[bool] = None
    lidar_calibration_enabled: Optional[bool] = None
    camera_calibration_enabled: Optional[bool] = None


class Vec3(BaseModel):
    x: float
    y: float
    z: float


class Quat(BaseModel):
    x: float
    y: float
    z: float
    w: float


class FrameImage(BaseModel):
    """Per-camera view inside a frame. Fields mirror the frame JSON schema."""

    image_url: Optional[str] = None
    position: Optional[Vec3] = None
    heading: Optional[Quat] = None
    width: Optional[int] = None
    height: Optional[int] = None
    # Pinhole intrinsics (rectified cams)
    fx: Optional[float] = None
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None
    # Camera projection model — "pinhole" or "doublesphere"
    model: Optional[str] = None
    camera_model: Optional[str] = None
    xi: Optional[float] = None
    alpha: Optional[float] = None


class DatasetFrame(BaseModel):
    """A single frame's LiDAR JSON metadata — the blob Mission Control loads."""

    frame_index: int
    key: Optional[str] = None
    model: Optional[str] = None
    camera_model: Optional[str] = None
    xi: Optional[float] = None
    alpha: Optional[float] = None
    device_position: Optional[Vec3] = None
    device_heading: Optional[Quat] = None
    images: Optional[List[FrameImage]] = None
    raw: Dict[str, Any] = {}
    """Full untyped frame dict from the server (everything the response contains)."""


class CameraCalibration(BaseModel):
    """Per-camera rig entry distilled from frame[0]."""

    camera_id: Optional[str] = None
    position: Optional[Vec3] = None
    heading: Optional[Quat] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fx: Optional[float] = None
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None
    model: Optional[str] = None
    xi: Optional[float] = None
    alpha: Optional[float] = None


class DatasetCalibration(BaseModel):
    """Canonicalized calibration view for a sequence.

    Derived client-side from the first frame's JSON — the server does not
    ship a dedicated calibration readback endpoint, but every frame carries
    the rig so reading it from frame[0] gives an accurate snapshot.
    """

    sequence_uid: str
    cameras: List[CameraCalibration] = []
    dataset_level_camera_calibration: Optional[Dict[str, Any]] = None


class SequenceHealth(BaseModel):
    uid: str
    key: Optional[str] = None
    status: Optional[str] = None
    frame_count: int = 0
    has_lidar_calibration: bool = False
    has_camera_calibration: bool = False


class DatasetHealth(BaseModel):
    """Read-only health snapshot from `GET /datasets/<owner>/<slug>/health/`."""

    dataset_uid: str
    dataset_slug: str
    dataset_status: Optional[str] = None
    item_count: int = 0
    sequence_count: int = 0
    total_frames: int = 0
    s3_prefix: Optional[str] = None
    last_item_updated_at: Optional[datetime] = None
    sequences: List[SequenceHealth] = []
    ingest_ok: bool = False
    issues: List[str] = []
