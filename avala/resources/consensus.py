"""Consensus resource (project-scoped)."""

from __future__ import annotations

from typing import Any

from avala._pagination import CursorPage
from avala.resources._base import BaseAsyncResource, BaseSyncResource
from avala.types.consensus import ConsensusComputeResult, ConsensusConfig, ConsensusScore, ConsensusSummary


class Consensus(BaseSyncResource):
    def get_summary(self, project_uid: str) -> ConsensusSummary:
        data = self._transport.request("GET", f"/projects/{project_uid}/consensus/")
        return ConsensusSummary.model_validate(data)

    def list_scores(
        self, project_uid: str, *, limit: int | None = None, cursor: str | None = None
    ) -> CursorPage[ConsensusScore]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page(
            f"/projects/{project_uid}/consensus/scores/", ConsensusScore, params=params or None
        )

    def compute(self, project_uid: str) -> ConsensusComputeResult:
        data = self._transport.request("POST", f"/projects/{project_uid}/consensus/compute/")
        return ConsensusComputeResult.model_validate(data)

    def get_config(self, project_uid: str) -> ConsensusConfig:
        data = self._transport.request("GET", f"/projects/{project_uid}/consensus/config/")
        return ConsensusConfig.model_validate(data)

    def update_config(
        self,
        project_uid: str,
        *,
        iou_threshold: float | None = None,
        min_agreement_ratio: float | None = None,
        min_annotations: int | None = None,
    ) -> ConsensusConfig:
        payload: dict[str, Any] = {}
        if iou_threshold is not None:
            payload["iou_threshold"] = iou_threshold
        if min_agreement_ratio is not None:
            payload["min_agreement_ratio"] = min_agreement_ratio
        if min_annotations is not None:
            payload["min_annotations"] = min_annotations
        data = self._transport.request("PUT", f"/projects/{project_uid}/consensus/config/", json=payload)
        return ConsensusConfig.model_validate(data)


class AsyncConsensus(BaseAsyncResource):
    async def get_summary(self, project_uid: str) -> ConsensusSummary:
        data = await self._transport.request("GET", f"/projects/{project_uid}/consensus/")
        return ConsensusSummary.model_validate(data)

    async def list_scores(
        self, project_uid: str, *, limit: int | None = None, cursor: str | None = None
    ) -> CursorPage[ConsensusScore]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page(
            f"/projects/{project_uid}/consensus/scores/", ConsensusScore, params=params or None
        )

    async def compute(self, project_uid: str) -> ConsensusComputeResult:
        data = await self._transport.request("POST", f"/projects/{project_uid}/consensus/compute/")
        return ConsensusComputeResult.model_validate(data)

    async def get_config(self, project_uid: str) -> ConsensusConfig:
        data = await self._transport.request("GET", f"/projects/{project_uid}/consensus/config/")
        return ConsensusConfig.model_validate(data)

    async def update_config(
        self,
        project_uid: str,
        *,
        iou_threshold: float | None = None,
        min_agreement_ratio: float | None = None,
        min_annotations: int | None = None,
    ) -> ConsensusConfig:
        payload: dict[str, Any] = {}
        if iou_threshold is not None:
            payload["iou_threshold"] = iou_threshold
        if min_agreement_ratio is not None:
            payload["min_agreement_ratio"] = min_agreement_ratio
        if min_annotations is not None:
            payload["min_annotations"] = min_annotations
        data = await self._transport.request("PUT", f"/projects/{project_uid}/consensus/config/", json=payload)
        return ConsensusConfig.model_validate(data)
