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
