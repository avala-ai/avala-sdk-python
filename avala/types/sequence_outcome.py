"""Behavioral outcome label of a dataset sequence.

Technical validity (sensor integrity, sync, calibration, ...) is a separate axis
reported by quality control; this type is only what the actor did.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Union

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
HANDS = ("left", "right")
# Server-side cap on items per hand (one hour of ~1 Hz labels).
MAX_HAND_ACTIONS_PER_HAND = 3600


class SequenceOutcomeSubtask(BaseModel):
    """One subtask; timestamps are seconds from the start of the sequence."""

    label: str
    start_ts: float
    end_ts: float
    outcome: Optional[str] = None


# ``source_metadata`` values: flat scalars only (the server rejects nested objects, lists and null).
SourceMetadataValue = Union[bool, int, float, str]


class SequenceHandAction(BaseModel):
    """One action of one hand over ``[start_ts, end_ts]`` seconds from the start of the sequence.

    ``action`` is natural language ("holding the aluminum piston"); ``object`` and
    ``verb`` are optional structured slots; ``contact`` is whether the hand touches
    the object (``None`` when unknown).
    """

    start_ts: float
    end_ts: float
    action: str
    object: Optional[str] = None
    verb: Optional[str] = None
    contact: Optional[bool] = None


class SequenceHandActions(BaseModel):
    """Two independent per-hand timelines. Within a hand, items are ordered and do not overlap."""

    left: List[SequenceHandAction] = []
    right: List[SequenceHandAction] = []


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
    hand_actions: SequenceHandActions = SequenceHandActions()
    mistake_type: str = ""
    recovery_type: str = ""
    failure_stage: str = ""
    autonomy_level: str = ""
    model_version: str = ""
    evaluation_membership: str = ""
    leakage_groups: Dict[str, str] = {}
    source: str = "human"
    source_metadata: Dict[str, SourceMetadataValue] = {}
    labeled_by: Optional[str] = None
    confidence: Optional[float] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
