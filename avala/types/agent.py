from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class Agent(BaseModel):
    uid: str
    name: str
    description: Optional[str] = None
    events: List[str] = []
    callback_url: Optional[str] = None
    is_active: bool = True
    project: Optional[str] = None
    task_types: List[str] = []
    secret: Optional[str] = None
    execution_stats: Optional[Dict[str, int]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class AgentExecution(BaseModel):
    uid: str
    registration: str
    event_type: str
    task: Optional[str] = None
    result: Optional[str] = None
    status: Optional[str] = None
    action: Optional[str] = None
    event_payload: Optional[Dict[str, Any]] = None
    response_payload: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
