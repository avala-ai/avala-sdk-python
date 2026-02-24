"""Annotation Issues resource."""

from __future__ import annotations

from typing import Any, Dict, List

from avala.resources._base import BaseAsyncResource, BaseSyncResource
from avala.types.annotation_issue import AnnotationIssue, AnnotationIssueMetrics, AnnotationIssueToolDetail


class AnnotationIssues(BaseSyncResource):
    # ── Sequence-scoped ──────────────────────────────────────

    def list_by_sequence(
        self,
        sequence_uid: str,
        *,
        dataset_item_uid: str | None = None,
        project_uid: str | None = None,
    ) -> List[AnnotationIssue]:
        params: dict[str, Any] = {}
        if dataset_item_uid is not None:
            params["dataset_item_uid"] = dataset_item_uid
        if project_uid is not None:
            params["project_uid"] = project_uid
        return self._transport.request_list(
            f"/sequences/{sequence_uid}/annotation-issues/",
            AnnotationIssue,
            params=params or None,
        )

    def create(
        self,
        sequence_uid: str,
        *,
        tool_uid: str,
        problem_uid: str,
        dataset_item_uid: str | None = None,
        project_uid: str | None = None,
        priority: str | None = None,
        severity: str | None = None,
        description: str | None = None,
        wrong_class: str | None = None,
        correct_class: str | None = None,
        should_re_annotate: bool | None = None,
        should_delete: bool | None = None,
        frames_affected: str | None = None,
        coordinates: Any | None = None,
        query_params: Dict[str, Any] | None = None,
        object_uid: str | None = None,
    ) -> AnnotationIssue:
        payload: dict[str, Any] = {
            "tool_uid": tool_uid,
            "problem_uid": problem_uid,
            "sequence_uid": sequence_uid,
        }
        if dataset_item_uid is not None:
            payload["dataset_item_uid"] = dataset_item_uid
        if project_uid is not None:
            payload["project_uid"] = project_uid
        if priority is not None:
            payload["priority"] = priority
        if severity is not None:
            payload["severity"] = severity
        if description is not None:
            payload["description"] = description
        if wrong_class is not None:
            payload["wrong_class"] = wrong_class
        if correct_class is not None:
            payload["correct_class"] = correct_class
        if should_re_annotate is not None:
            payload["should_re_annotate"] = should_re_annotate
        if should_delete is not None:
            payload["should_delete"] = should_delete
        if frames_affected is not None:
            payload["frames_affected"] = frames_affected
        if coordinates is not None:
            payload["coordinates"] = coordinates
        if query_params is not None:
            payload["query_params"] = query_params
        if object_uid is not None:
            payload["object_uid"] = object_uid
        data = self._transport.request("POST", f"/sequences/{sequence_uid}/annotation-issues/", json=payload)
        return AnnotationIssue.model_validate(data)

    def update(
        self,
        sequence_uid: str,
        issue_uid: str,
        *,
        status: str | None = None,
        priority: str | None = None,
        severity: str | None = None,
        description: str | None = None,
        tool_uid: str | None = None,
        problem_uid: str | None = None,
        wrong_class: str | None = None,
        frames_affected: str | None = None,
    ) -> AnnotationIssue:
        payload: dict[str, Any] = {}
        if status is not None:
            payload["status"] = status
        if priority is not None:
            payload["priority"] = priority
        if severity is not None:
            payload["severity"] = severity
        if description is not None:
            payload["description"] = description
        if tool_uid is not None:
            payload["tool_uid"] = tool_uid
        if problem_uid is not None:
            payload["problem_uid"] = problem_uid
        if wrong_class is not None:
            payload["wrong_class"] = wrong_class
        if frames_affected is not None:
            payload["frames_affected"] = frames_affected
        data = self._transport.request(
            "PATCH", f"/sequences/{sequence_uid}/annotation-issues/{issue_uid}/", json=payload
        )
        return AnnotationIssue.model_validate(data)

    def delete(self, sequence_uid: str, issue_uid: str) -> None:
        self._transport.request("DELETE", f"/sequences/{sequence_uid}/annotation-issues/{issue_uid}/")

    # ── Dataset-scoped ───────────────────────────────────────

    def list_by_dataset(
        self,
        owner: str,
        dataset_slug: str,
        *,
        sequence_uid: str | None = None,
    ) -> List[AnnotationIssue]:
        params: dict[str, Any] = {}
        if sequence_uid is not None:
            params["sequence_uid"] = sequence_uid
        return self._transport.request_list(
            f"/datasets/{owner}/{dataset_slug}/annotation-issues/",
            AnnotationIssue,
            params=params or None,
        )

    def get_metrics(
        self,
        owner: str,
        dataset_slug: str,
        *,
        sequence_uid: str | None = None,
    ) -> AnnotationIssueMetrics:
        params: dict[str, Any] = {}
        if sequence_uid is not None:
            params["sequence_uid"] = sequence_uid
        data = self._transport.request(
            "GET",
            f"/datasets/{owner}/{dataset_slug}/annotation-issues/metrics/",
            params=params or None,
        )
        return AnnotationIssueMetrics.model_validate(data)

    # ── Tools ────────────────────────────────────────────────

    def list_tools(self, *, dataset_type: str) -> List[AnnotationIssueToolDetail]:
        params: dict[str, Any] = {"dataset_type": dataset_type}
        return self._transport.request_list("/qc-available-tools/", AnnotationIssueToolDetail, params=params)


class AsyncAnnotationIssues(BaseAsyncResource):
    # ── Sequence-scoped ──────────────────────────────────────

    async def list_by_sequence(
        self,
        sequence_uid: str,
        *,
        dataset_item_uid: str | None = None,
        project_uid: str | None = None,
    ) -> List[AnnotationIssue]:
        params: dict[str, Any] = {}
        if dataset_item_uid is not None:
            params["dataset_item_uid"] = dataset_item_uid
        if project_uid is not None:
            params["project_uid"] = project_uid
        return await self._transport.request_list(
            f"/sequences/{sequence_uid}/annotation-issues/",
            AnnotationIssue,
            params=params or None,
        )

    async def create(
        self,
        sequence_uid: str,
        *,
        tool_uid: str,
        problem_uid: str,
        dataset_item_uid: str | None = None,
        project_uid: str | None = None,
        priority: str | None = None,
        severity: str | None = None,
        description: str | None = None,
        wrong_class: str | None = None,
        correct_class: str | None = None,
        should_re_annotate: bool | None = None,
        should_delete: bool | None = None,
        frames_affected: str | None = None,
        coordinates: Any | None = None,
        query_params: Dict[str, Any] | None = None,
        object_uid: str | None = None,
    ) -> AnnotationIssue:
        payload: dict[str, Any] = {
            "tool_uid": tool_uid,
            "problem_uid": problem_uid,
            "sequence_uid": sequence_uid,
        }
        if dataset_item_uid is not None:
            payload["dataset_item_uid"] = dataset_item_uid
        if project_uid is not None:
            payload["project_uid"] = project_uid
        if priority is not None:
            payload["priority"] = priority
        if severity is not None:
            payload["severity"] = severity
        if description is not None:
            payload["description"] = description
        if wrong_class is not None:
            payload["wrong_class"] = wrong_class
        if correct_class is not None:
            payload["correct_class"] = correct_class
        if should_re_annotate is not None:
            payload["should_re_annotate"] = should_re_annotate
        if should_delete is not None:
            payload["should_delete"] = should_delete
        if frames_affected is not None:
            payload["frames_affected"] = frames_affected
        if coordinates is not None:
            payload["coordinates"] = coordinates
        if query_params is not None:
            payload["query_params"] = query_params
        if object_uid is not None:
            payload["object_uid"] = object_uid
        data = await self._transport.request("POST", f"/sequences/{sequence_uid}/annotation-issues/", json=payload)
        return AnnotationIssue.model_validate(data)

    async def update(
        self,
        sequence_uid: str,
        issue_uid: str,
        *,
        status: str | None = None,
        priority: str | None = None,
        severity: str | None = None,
        description: str | None = None,
        tool_uid: str | None = None,
        problem_uid: str | None = None,
        wrong_class: str | None = None,
        frames_affected: str | None = None,
    ) -> AnnotationIssue:
        payload: dict[str, Any] = {}
        if status is not None:
            payload["status"] = status
        if priority is not None:
            payload["priority"] = priority
        if severity is not None:
            payload["severity"] = severity
        if description is not None:
            payload["description"] = description
        if tool_uid is not None:
            payload["tool_uid"] = tool_uid
        if problem_uid is not None:
            payload["problem_uid"] = problem_uid
        if wrong_class is not None:
            payload["wrong_class"] = wrong_class
        if frames_affected is not None:
            payload["frames_affected"] = frames_affected
        data = await self._transport.request(
            "PATCH", f"/sequences/{sequence_uid}/annotation-issues/{issue_uid}/", json=payload
        )
        return AnnotationIssue.model_validate(data)

    async def delete(self, sequence_uid: str, issue_uid: str) -> None:
        await self._transport.request("DELETE", f"/sequences/{sequence_uid}/annotation-issues/{issue_uid}/")

    # ── Dataset-scoped ───────────────────────────────────────

    async def list_by_dataset(
        self,
        owner: str,
        dataset_slug: str,
        *,
        sequence_uid: str | None = None,
    ) -> List[AnnotationIssue]:
        params: dict[str, Any] = {}
        if sequence_uid is not None:
            params["sequence_uid"] = sequence_uid
        return await self._transport.request_list(
            f"/datasets/{owner}/{dataset_slug}/annotation-issues/",
            AnnotationIssue,
            params=params or None,
        )

    async def get_metrics(
        self,
        owner: str,
        dataset_slug: str,
        *,
        sequence_uid: str | None = None,
    ) -> AnnotationIssueMetrics:
        params: dict[str, Any] = {}
        if sequence_uid is not None:
            params["sequence_uid"] = sequence_uid
        data = await self._transport.request(
            "GET",
            f"/datasets/{owner}/{dataset_slug}/annotation-issues/metrics/",
            params=params or None,
        )
        return AnnotationIssueMetrics.model_validate(data)

    # ── Tools ────────────────────────────────────────────────

    async def list_tools(self, *, dataset_type: str) -> List[AnnotationIssueToolDetail]:
        params: dict[str, Any] = {"dataset_type": dataset_type}
        return await self._transport.request_list("/qc-available-tools/", AnnotationIssueToolDetail, params=params)
