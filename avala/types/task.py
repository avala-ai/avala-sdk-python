from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class Task(BaseModel):
    uid: str
    type: Optional[str] = None
    name: Optional[str] = None
    status: Optional[str] = None
    project: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
