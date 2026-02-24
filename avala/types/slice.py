from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class Slice(BaseModel):
    uid: str
    name: str
    slug: Optional[str] = None
    owner_name: Optional[str] = None
    organization: Optional[Dict[str, Any]] = None
    visibility: Optional[str] = None
    status: Optional[str] = None
    item_count: Optional[int] = None
    sub_slices: Optional[List[Dict[str, Any]]] = None
    source_data: Optional[Any] = None
    featured_slice_item_urls: Optional[List[str]] = None


class SliceItem(BaseModel):
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
