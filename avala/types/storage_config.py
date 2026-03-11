from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class StorageConfig(BaseModel):
    uid: str
    name: str
    provider: str
    s3_bucket_name: Optional[str] = None
    s3_bucket_region: Optional[str] = None
    s3_bucket_prefix: Optional[str] = None
    s3_auth_method: Optional[str] = None
    s3_is_accelerated: bool = False
    gc_storage_bucket_name: Optional[str] = None
    gc_storage_prefix: Optional[str] = None
    is_verified: bool = False
    last_verified_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class StorageConfigSetupInfo(BaseModel):
    avala_aws_account_id: str
    external_id: str
