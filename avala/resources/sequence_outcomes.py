"""Sequence outcome labels: the behavioral outcome of each dataset sequence."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from avala._pagination import CursorPage
from avala.resources._base import BaseAsyncResource, BaseSyncResource
from avala.types.sequence_outcome import SequenceOutcome

FilterValue = Union[str, Sequence[str], None]


def _outcome_url(owner: str, slug: str, sequence_uid: str) -> str:
    return f"/datasets/{owner}/{slug}/sequences/{sequence_uid}/outcome/"


def _join(value: FilterValue) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return ",".join(value)


def _list_params(
    *,
    outcome: FilterValue,
    evaluation_membership: FilterValue,
    source: FilterValue,
    autonomy_level: FilterValue,
    model_version: Optional[str],
    limit: Optional[int],
    cursor: Optional[str],
) -> Optional[Dict[str, Any]]:
    params: Dict[str, Any] = {}
    for key, value in (
        ("outcome", _join(outcome)),
        ("evaluation_membership", _join(evaluation_membership)),
        ("source", _join(source)),
        ("autonomy_level", _join(autonomy_level)),
        ("model_version", model_version),
        ("limit", limit),
        ("cursor", cursor),
    ):
        if value is not None:
            params[key] = value
    return params or None


def _set_payload(
    outcome: str,
    *,
    progress: Optional[float],
    quality: Optional[int],
    speed: Optional[int],
    subtasks: Optional[List[Mapping[str, Any]]],
    mistake_type: Optional[str],
    recovery_type: Optional[str],
    failure_stage: Optional[str],
    autonomy_level: Optional[str],
    model_version: Optional[str],
    evaluation_membership: Optional[str],
    leakage_groups: Optional[Mapping[str, str]],
    source: Optional[str],
    confidence: Optional[float],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"outcome": outcome}
    optional: Dict[str, Any] = {
        "progress": progress,
        "quality": quality,
        "speed": speed,
        "subtasks": [dict(subtask) for subtask in subtasks] if subtasks is not None else None,
        "mistake_type": mistake_type,
        "recovery_type": recovery_type,
        "failure_stage": failure_stage,
        "autonomy_level": autonomy_level,
        "model_version": model_version,
        "evaluation_membership": evaluation_membership,
        "leakage_groups": dict(leakage_groups) if leakage_groups is not None else None,
        "source": source,
        "confidence": confidence,
    }
    payload.update({key: value for key, value in optional.items() if value is not None})
    return payload


class SequenceOutcomes(BaseSyncResource):
    """Get, set and list sequence outcome labels.

    ``set`` replaces the current label wholesale and records a new version: any
    optional field you omit is stored as unset, not carried over from the
    previous version.
    """

    def get(self, owner: str, slug: str, sequence_uid: str) -> SequenceOutcome:
        """Current label of one sequence. Raises ``NotFoundError`` when it is unlabeled."""
        data = self._transport.request("GET", _outcome_url(owner, slug, sequence_uid))
        return SequenceOutcome.model_validate(data)

    def set(
        self,
        owner: str,
        slug: str,
        sequence_uid: str,
        *,
        outcome: str,
        progress: Optional[float] = None,
        quality: Optional[int] = None,
        speed: Optional[int] = None,
        subtasks: Optional[List[Mapping[str, Any]]] = None,
        mistake_type: Optional[str] = None,
        recovery_type: Optional[str] = None,
        failure_stage: Optional[str] = None,
        autonomy_level: Optional[str] = None,
        model_version: Optional[str] = None,
        evaluation_membership: Optional[str] = None,
        leakage_groups: Optional[Mapping[str, str]] = None,
        source: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> SequenceOutcome:
        """Record a new current label (requires edit access to the dataset)."""
        payload = _set_payload(
            outcome,
            progress=progress,
            quality=quality,
            speed=speed,
            subtasks=subtasks,
            mistake_type=mistake_type,
            recovery_type=recovery_type,
            failure_stage=failure_stage,
            autonomy_level=autonomy_level,
            model_version=model_version,
            evaluation_membership=evaluation_membership,
            leakage_groups=leakage_groups,
            source=source,
            confidence=confidence,
        )
        data = self._transport.request("PUT", _outcome_url(owner, slug, sequence_uid), json=payload)
        return SequenceOutcome.model_validate(data)

    def history(self, owner: str, slug: str, sequence_uid: str) -> List[SequenceOutcome]:
        """Every version of one sequence's label, newest first."""
        return self._transport.request_list(f"{_outcome_url(owner, slug, sequence_uid)}history/", SequenceOutcome)

    def list(
        self,
        owner: str,
        slug: str,
        *,
        outcome: FilterValue = None,
        evaluation_membership: FilterValue = None,
        source: FilterValue = None,
        autonomy_level: FilterValue = None,
        model_version: Optional[str] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> CursorPage[SequenceOutcome]:
        """Current labels in a dataset. Enum filters accept one value or a list (OR)."""
        params = _list_params(
            outcome=outcome,
            evaluation_membership=evaluation_membership,
            source=source,
            autonomy_level=autonomy_level,
            model_version=model_version,
            limit=limit,
            cursor=cursor,
        )
        return self._transport.request_page(
            f"/datasets/{owner}/{slug}/sequence-outcomes/", SequenceOutcome, params=params
        )


class AsyncSequenceOutcomes(BaseAsyncResource):
    """Async variant of :class:`SequenceOutcomes`."""

    async def get(self, owner: str, slug: str, sequence_uid: str) -> SequenceOutcome:
        data = await self._transport.request("GET", _outcome_url(owner, slug, sequence_uid))
        return SequenceOutcome.model_validate(data)

    async def set(
        self,
        owner: str,
        slug: str,
        sequence_uid: str,
        *,
        outcome: str,
        progress: Optional[float] = None,
        quality: Optional[int] = None,
        speed: Optional[int] = None,
        subtasks: Optional[List[Mapping[str, Any]]] = None,
        mistake_type: Optional[str] = None,
        recovery_type: Optional[str] = None,
        failure_stage: Optional[str] = None,
        autonomy_level: Optional[str] = None,
        model_version: Optional[str] = None,
        evaluation_membership: Optional[str] = None,
        leakage_groups: Optional[Mapping[str, str]] = None,
        source: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> SequenceOutcome:
        payload = _set_payload(
            outcome,
            progress=progress,
            quality=quality,
            speed=speed,
            subtasks=subtasks,
            mistake_type=mistake_type,
            recovery_type=recovery_type,
            failure_stage=failure_stage,
            autonomy_level=autonomy_level,
            model_version=model_version,
            evaluation_membership=evaluation_membership,
            leakage_groups=leakage_groups,
            source=source,
            confidence=confidence,
        )
        data = await self._transport.request("PUT", _outcome_url(owner, slug, sequence_uid), json=payload)
        return SequenceOutcome.model_validate(data)

    async def history(self, owner: str, slug: str, sequence_uid: str) -> List[SequenceOutcome]:
        return await self._transport.request_list(f"{_outcome_url(owner, slug, sequence_uid)}history/", SequenceOutcome)

    async def list(
        self,
        owner: str,
        slug: str,
        *,
        outcome: FilterValue = None,
        evaluation_membership: FilterValue = None,
        source: FilterValue = None,
        autonomy_level: FilterValue = None,
        model_version: Optional[str] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
    ) -> CursorPage[SequenceOutcome]:
        params = _list_params(
            outcome=outcome,
            evaluation_membership=evaluation_membership,
            source=source,
            autonomy_level=autonomy_level,
            model_version=model_version,
            limit=limit,
            cursor=cursor,
        )
        return await self._transport.request_page(
            f"/datasets/{owner}/{slug}/sequence-outcomes/", SequenceOutcome, params=params
        )
