"""Behavioral outcome label of a dataset sequence.

Technical validity (sensor integrity, sync, calibration, ...) is a separate axis
reported by quality control; this type is only what the actor did.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel

OUTCOMES = (
    "expert_success",
    "slow_success",
    "partial_success",
    "mistake_and_recovery",
    "intervention",
    "unsafe",
    "aborted",
    "failure",
    "novel_strategy",
    "inefficient_strategy",
)
AUTONOMY_LEVELS = ("human_demonstration", "teleoperation", "autonomous", "shared_autonomy", "intervention")
EVALUATION_MEMBERSHIPS = ("train", "held_out_eval", "none")
OUTCOME_SOURCES = ("human", "model", "imported")
LEAKAGE_GROUP_KEYS = ("location", "object", "mechanism", "operator", "environment_family")


class SequenceOutcomeSubtask(BaseModel):
    """One subtask; timestamps are seconds from the start of the sequence."""

    label: str
    start_ts: float
    end_ts: float
    outcome: Optional[str] = None


class SequenceOutcome(BaseModel):
    """One version of a sequence's outcome label (``is_current`` marks the live one)."""

    uid: str
    sequence_uid: str
    version: int
    is_current: bool
    outcome: str
    progress: Optional[float] = None
    quality: Optional[int] = None
    speed: Optional[int] = None
    subtasks: List[SequenceOutcomeSubtask] = []
    mistake_type: str = ""
    recovery_type: str = ""
    failure_stage: str = ""
    autonomy_level: str = ""
    model_version: str = ""
    evaluation_membership: str = ""
    leakage_groups: Dict[str, str] = {}
    source: str = "human"
    labeled_by: Optional[str] = None
    confidence: Optional[float] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
