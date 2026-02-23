from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel


class InferenceProvider(BaseModel):
    uid: str
    name: str
    description: Optional[str] = None
    provider_type: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    is_active: bool = True
    project: Optional[str] = None
    last_test_at: Optional[datetime] = None
    last_test_success: Optional[bool] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
