from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ConsensusConfig(BaseModel):
    uid: str
    iou_threshold: float = 0.5
    min_agreement_ratio: float = 0.5
    min_annotations: int = 2
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ConsensusScore(BaseModel):
    uid: str
    dataset_item_uid: str
    task_name: str
    score_type: Optional[str] = None
    score: float
    annotator_count: int
    details: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None


class ConsensusSummary(BaseModel):
    mean_score: float
    median_score: float
    min_score: float
    max_score: float
    total_items: int
    items_with_consensus: int
    score_distribution: Optional[Dict[str, Any]] = None
    by_task_name: Optional[List[Any]] = None


class ConsensusComputeResult(BaseModel):
    computed: int
    skipped: int
    error_count: int
