from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class AnnotationIssueProject(BaseModel):
    uid: str
    name: str


class AnnotationIssueReporter(BaseModel):
    username: Optional[str] = None
    picture: Optional[str] = None
    full_name: Optional[str] = None
    type: Optional[str] = None
    is_staff: Optional[bool] = None


class AnnotationIssueTool(BaseModel):
    uid: str
    name: str
    default: Optional[bool] = None


class AnnotationIssueProblem(BaseModel):
    uid: str
    title: str


class AnnotationIssue(BaseModel):
    uid: str
    dataset_item_uid: Optional[str] = None
    sequence_uid: Optional[str] = None
    project: Optional[AnnotationIssueProject] = None
    reporter: Optional[AnnotationIssueReporter] = None
    priority: Optional[str] = None
    severity: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    tool: Optional[AnnotationIssueTool] = None
    problem: Optional[AnnotationIssueProblem] = None
    wrong_class: Optional[str] = None
    correct_class: Optional[str] = None
    should_re_annotate: Optional[bool] = None
    should_delete: Optional[bool] = None
    frames_affected: Optional[str] = None
    coordinates: Optional[Any] = None
    query_params: Optional[Any] = None
    created_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    object_uid: Optional[str] = None


class AnnotationIssueMetrics(BaseModel):
    status_count: Optional[Dict[str, Any]] = None
    priority_count: Optional[Dict[str, Any]] = None
    severity_count: Optional[Dict[str, Any]] = None
    mean_seconds_close_time_all: Optional[int] = None
    mean_seconds_close_time_customer: Optional[int] = None
    mean_unresolved_issue_age_all: Optional[int] = None
    mean_unresolved_issue_age_customer: Optional[int] = None
    object_count_by_annotation_issue_problem_uid: Optional[List[Dict[str, Any]]] = None


class AnnotationIssueToolProblem(BaseModel):
    uid: str
    title: str


class AnnotationIssueToolDetail(BaseModel):
    uid: str
    name: str
    dataset_type: Optional[str] = None
    default: Optional[bool] = None
    problems: Optional[List[AnnotationIssueToolProblem]] = None
