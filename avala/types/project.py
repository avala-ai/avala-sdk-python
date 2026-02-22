from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class Project(BaseModel):
    uid: str
    name: str
    status: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
