from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class Webhook(BaseModel):
    uid: str
    target_url: str
    events: List[str] = []
    is_active: bool = True
    secret: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class WebhookDelivery(BaseModel):
    uid: str
    subscription: str
    event_type: str
    payload: Optional[Dict[str, Any]] = None
    response_status: Optional[int] = None
    response_body: Optional[str] = None
    attempts: int = 0
    next_retry_at: Optional[datetime] = None
    status: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
