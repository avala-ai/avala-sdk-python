"""Quality targets resource (project-scoped)."""

from __future__ import annotations

from typing import Any, List

from avala._pagination import CursorPage
from avala.resources._base import BaseAsyncResource, BaseSyncResource
from avala.types.quality_target import QualityTarget, QualityTargetEvaluation


class QualityTargets(BaseSyncResource):
    def list(
        self, project_uid: str, *, limit: int | None = None, cursor: str | None = None
    ) -> CursorPage[QualityTarget]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page(
            f"/projects/{project_uid}/quality-targets/", QualityTarget, params=params or None
        )

    def get(self, project_uid: str, uid: str) -> QualityTarget:
        data = self._transport.request("GET", f"/projects/{project_uid}/quality-targets/{uid}/")
        return QualityTarget.model_validate(data)

    def create(
        self,
        project_uid: str,
        *,
        name: str,
        metric: str,
        threshold: float,
        operator: str | None = None,
        severity: str | None = None,
        is_active: bool | None = None,
        notify_webhook: bool | None = None,
        notify_emails: List[str] | None = None,
    ) -> QualityTarget:
        payload: dict[str, Any] = {"name": name, "metric": metric, "threshold": threshold}
        if operator is not None:
            payload["operator"] = operator
        if severity is not None:
            payload["severity"] = severity
        if is_active is not None:
            payload["is_active"] = is_active
        if notify_webhook is not None:
            payload["notify_webhook"] = notify_webhook
        if notify_emails is not None:
            payload["notify_emails"] = notify_emails
        data = self._transport.request("POST", f"/projects/{project_uid}/quality-targets/", json=payload)
        return QualityTarget.model_validate(data)

    def update(
        self,
        project_uid: str,
        uid: str,
        *,
        name: str | None = None,
        metric: str | None = None,
        threshold: float | None = None,
        operator: str | None = None,
        severity: str | None = None,
        is_active: bool | None = None,
        notify_webhook: bool | None = None,
        notify_emails: List[str] | None = None,
    ) -> QualityTarget:
        payload: dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if metric is not None:
            payload["metric"] = metric
        if threshold is not None:
            payload["threshold"] = threshold
        if operator is not None:
            payload["operator"] = operator
        if severity is not None:
            payload["severity"] = severity
        if is_active is not None:
            payload["is_active"] = is_active
        if notify_webhook is not None:
            payload["notify_webhook"] = notify_webhook
        if notify_emails is not None:
            payload["notify_emails"] = notify_emails
        data = self._transport.request("PATCH", f"/projects/{project_uid}/quality-targets/{uid}/", json=payload)
        return QualityTarget.model_validate(data)

    def delete(self, project_uid: str, uid: str) -> None:
        self._transport.request("DELETE", f"/projects/{project_uid}/quality-targets/{uid}/")

    def evaluate(self, project_uid: str) -> List[QualityTargetEvaluation]:
        data = self._transport.request("POST", f"/projects/{project_uid}/quality-targets/evaluate/")
        return [QualityTargetEvaluation.model_validate(item) for item in data]


class AsyncQualityTargets(BaseAsyncResource):
    async def list(
        self, project_uid: str, *, limit: int | None = None, cursor: str | None = None
    ) -> CursorPage[QualityTarget]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page(
            f"/projects/{project_uid}/quality-targets/", QualityTarget, params=params or None
        )

    async def get(self, project_uid: str, uid: str) -> QualityTarget:
        data = await self._transport.request("GET", f"/projects/{project_uid}/quality-targets/{uid}/")
        return QualityTarget.model_validate(data)

    async def create(
        self,
        project_uid: str,
        *,
        name: str,
        metric: str,
        threshold: float,
        operator: str | None = None,
        severity: str | None = None,
        is_active: bool | None = None,
        notify_webhook: bool | None = None,
        notify_emails: List[str] | None = None,
    ) -> QualityTarget:
        payload: dict[str, Any] = {"name": name, "metric": metric, "threshold": threshold}
        if operator is not None:
            payload["operator"] = operator
        if severity is not None:
            payload["severity"] = severity
        if is_active is not None:
            payload["is_active"] = is_active
        if notify_webhook is not None:
            payload["notify_webhook"] = notify_webhook
        if notify_emails is not None:
            payload["notify_emails"] = notify_emails
        data = await self._transport.request("POST", f"/projects/{project_uid}/quality-targets/", json=payload)
        return QualityTarget.model_validate(data)

    async def update(
        self,
        project_uid: str,
        uid: str,
        *,
        name: str | None = None,
        metric: str | None = None,
        threshold: float | None = None,
        operator: str | None = None,
        severity: str | None = None,
        is_active: bool | None = None,
        notify_webhook: bool | None = None,
        notify_emails: List[str] | None = None,
    ) -> QualityTarget:
        payload: dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if metric is not None:
            payload["metric"] = metric
        if threshold is not None:
            payload["threshold"] = threshold
        if operator is not None:
            payload["operator"] = operator
        if severity is not None:
            payload["severity"] = severity
        if is_active is not None:
            payload["is_active"] = is_active
        if notify_webhook is not None:
            payload["notify_webhook"] = notify_webhook
        if notify_emails is not None:
            payload["notify_emails"] = notify_emails
        data = await self._transport.request("PATCH", f"/projects/{project_uid}/quality-targets/{uid}/", json=payload)
        return QualityTarget.model_validate(data)

    async def delete(self, project_uid: str, uid: str) -> None:
        await self._transport.request("DELETE", f"/projects/{project_uid}/quality-targets/{uid}/")

    async def evaluate(self, project_uid: str) -> List[QualityTargetEvaluation]:
        data = await self._transport.request("POST", f"/projects/{project_uid}/quality-targets/evaluate/")
        return [QualityTargetEvaluation.model_validate(item) for item in data]
