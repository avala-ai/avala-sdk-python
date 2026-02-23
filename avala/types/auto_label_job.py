from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class AutoLabelJob(BaseModel):
    uid: str
    status: Optional[str] = None
    model_type: Optional[str] = None
    confidence_threshold: Optional[float] = None
    labels: List[str] = []
    dry_run: bool = False
    total_items: int = 0
    processed_items: int = 0
    successful_items: int = 0
    failed_items: int = 0
    skipped_items: int = 0
    progress_pct: Optional[float] = None
    error_message: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
