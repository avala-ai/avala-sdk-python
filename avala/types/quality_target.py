from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class QualityTarget(BaseModel):
    uid: str
    name: str
    metric: str
    operator: str
    threshold: float
    severity: Optional[str] = None
    is_active: bool = True
    notify_webhook: bool = True
    notify_emails: List[str] = []
    last_evaluated_at: Optional[datetime] = None
    last_value: Optional[float] = None
    is_breached: bool = False
    breach_count: int = 0
    last_breached_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class QualityTargetEvaluation(BaseModel):
    uid: str
    name: str
    metric: str
    threshold: float
    operator: str
    current_value: float
    is_breached: bool
    severity: Optional[str] = None
