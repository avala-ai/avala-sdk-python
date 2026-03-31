"""Pydantic models for fleet upload operations."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel


class UploadSession(BaseModel):
    uid: str
    total_files: int = 0
    total_bytes: int = 0
    confirmed_files: int = 0
    confirmed_bytes: int = 0
    s3_prefix: Optional[str] = None
    status: Optional[str] = None


class UploadUrlEntry(BaseModel):
    path: str
    put_url: str
    s3_key: str
    headers: Dict[str, str] = {}


class UploadUrlsResponse(BaseModel):
    urls: List[UploadUrlEntry] = []


class UploadConfirmResponse(BaseModel):
    confirmed: int = 0
    total_confirmed: int = 0
    total_files: int = 0


class UploadStatusResponse(BaseModel):
    session_uid: str
    status: str
    total_files: int = 0
    confirmed_files: int = 0
    total_bytes: int = 0
    confirmed_bytes: int = 0
    pending_paths: List[str] = []


class UploadProgress(BaseModel):
    """Progress snapshot emitted during upload."""

    total_files: int
    uploaded_files: int
    total_bytes: int
    uploaded_bytes: int
    failed_files: int = 0
