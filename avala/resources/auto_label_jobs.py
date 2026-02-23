"""Auto-label jobs resource."""

from __future__ import annotations

from typing import Any, List

from avala._pagination import CursorPage
from avala.resources._base import BaseAsyncResource, BaseSyncResource
from avala.types.auto_label_job import AutoLabelJob


class AutoLabelJobs(BaseSyncResource):
    def list(
        self,
        *,
        project: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> CursorPage[AutoLabelJob]:
        params: dict[str, Any] = {}
        if project is not None:
            params["project"] = project
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page("/auto-label-jobs/", AutoLabelJob, params=params or None)

    def get(self, uid: str) -> AutoLabelJob:
        data = self._transport.request("GET", f"/auto-label-jobs/{uid}/")
        return AutoLabelJob.model_validate(data)

    def create(
        self,
        project_uid: str,
        *,
        model_type: str | None = None,
        confidence_threshold: float | None = None,
        labels: List[str] | None = None,
        dry_run: bool | None = None,
    ) -> AutoLabelJob:
        payload: dict[str, Any] = {}
        if model_type is not None:
            payload["model_type"] = model_type
        if confidence_threshold is not None:
            payload["confidence_threshold"] = confidence_threshold
        if labels is not None:
            payload["labels"] = labels
        if dry_run is not None:
            payload["dry_run"] = dry_run
        data = self._transport.request("POST", f"/projects/{project_uid}/auto-label/", json=payload)
        return AutoLabelJob.model_validate(data)

    def cancel(self, uid: str) -> None:
        self._transport.request("DELETE", f"/auto-label-jobs/{uid}/")


class AsyncAutoLabelJobs(BaseAsyncResource):
    async def list(
        self,
        *,
        project: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> CursorPage[AutoLabelJob]:
        params: dict[str, Any] = {}
        if project is not None:
            params["project"] = project
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page("/auto-label-jobs/", AutoLabelJob, params=params or None)

    async def get(self, uid: str) -> AutoLabelJob:
        data = await self._transport.request("GET", f"/auto-label-jobs/{uid}/")
        return AutoLabelJob.model_validate(data)

    async def create(
        self,
        project_uid: str,
        *,
        model_type: str | None = None,
        confidence_threshold: float | None = None,
        labels: List[str] | None = None,
        dry_run: bool | None = None,
    ) -> AutoLabelJob:
        payload: dict[str, Any] = {}
        if model_type is not None:
            payload["model_type"] = model_type
        if confidence_threshold is not None:
            payload["confidence_threshold"] = confidence_threshold
        if labels is not None:
            payload["labels"] = labels
        if dry_run is not None:
            payload["dry_run"] = dry_run
        data = await self._transport.request("POST", f"/projects/{project_uid}/auto-label/", json=payload)
        return AutoLabelJob.model_validate(data)

    async def cancel(self, uid: str) -> None:
        await self._transport.request("DELETE", f"/auto-label-jobs/{uid}/")
