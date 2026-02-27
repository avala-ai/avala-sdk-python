"""Pydantic models for fleet management objects."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel


class Device(BaseModel):
    uid: str
    name: str
    type: Optional[str] = None
    status: Optional[str] = None
    tags: List[str] = []
    firmware_version: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    last_seen_at: Optional[datetime] = None
    device_token: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class Recording(BaseModel):
    uid: str
    device: Optional[str] = None
    status: Optional[str] = None
    duration_seconds: Optional[float] = None
    size_bytes: Optional[int] = None
    topic_count: int = 0
    tags: List[str] = []
    topics: Optional[List[Dict[str, Any]]] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class FleetEvent(BaseModel):
    uid: str
    recording: Optional[str] = None
    device: Optional[str] = None
    type: Optional[str] = None
    label: Optional[str] = None
    description: Optional[str] = None
    timestamp: Optional[datetime] = None
    duration_ms: Optional[int] = None
    tags: List[str] = []
    metadata: Optional[Dict[str, Any]] = None
    severity: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class Rule(BaseModel):
    uid: str
    name: str
    description: Optional[str] = None
    enabled: bool = True
    condition: Optional[Dict[str, Any]] = None
    actions: List[Dict[str, Any]] = []
    scope: Optional[Dict[str, Any]] = None
    hit_count: int = 0
    last_hit_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class Alert(BaseModel):
    uid: str
    rule: Optional[str] = None
    device: Optional[str] = None
    recording: Optional[str] = None
    severity: Optional[str] = None
    status: Optional[str] = None
    message: Optional[str] = None
    triggered_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolution_note: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class AlertChannel(BaseModel):
    uid: str
    name: str
    type: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class _BatchEventRequired(TypedDict):
    recording: str
    device: str
    label: str
    type: str
    timestamp: str


class BatchEventParams(_BatchEventRequired, total=False):
    """Parameters for a single event in a batch create request."""

    severity: str
    description: str
    duration_ms: int
    tags: List[str]
    metadata: Dict[str, Any]
