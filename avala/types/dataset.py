from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class Dataset(BaseModel):
    uid: str
    name: str
    slug: str
    item_count: int = 0
    data_type: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
