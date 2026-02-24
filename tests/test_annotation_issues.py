"""Tests for AnnotationIssues resource."""

from __future__ import annotations

import json

import httpx
import respx

from avala import Client

BASE_URL = "https://api.avala.ai/api/v1"

ISSUE_FIXTURE = {
    "uid": "issue-001",
    "dataset_item_uid": "item-001",
    "sequence_uid": "seq-001",
    "project": {"uid": "proj-001", "name": "My Project"},
    "reporter": {
        "username": "alice",
        "picture": None,
        "full_name": "Alice Smith",
        "type": "customer",
        "is_staff": False,
    },
    "priority": "high",
    "severity": "major",
    "description": "Wrong label applied",
    "status": "open",
    "tool": {"uid": "tool-001", "name": "Bounding Box", "default": True},
    "problem": {"uid": "prob-001", "title": "Wrong class"},
    "wrong_class": "cat",
    "correct_class": "dog",
    "should_re_annotate": True,
    "should_delete": False,
    "frames_affected": "1-5",
    "coordinates": None,
    "query_params": None,
    "created_at": "2026-01-01T10:00:00Z",
    "closed_at": None,
    "object_uid": "obj-001",
}

METRICS_FIXTURE = {
    "status_count": {"open": 10, "closed": 5},
    "priority_count": {"high": 8, "low": 7},
    "severity_count": {"major": 6, "minor": 9},
    "mean_seconds_close_time_all": 3600,
    "mean_seconds_close_time_customer": 7200,
    "mean_unresolved_issue_age_all": 86400,
    "mean_unresolved_issue_age_customer": 172800,
    "object_count_by_annotation_issue_problem_uid": [{"problem_uid": "prob-001", "count": 5}],
}

TOOL_FIXTURE = {
    "uid": "tool-001",
    "name": "Bounding Box",
    "dataset_type": "image",
    "default": True,
    "problems": [
        {"uid": "prob-001", "title": "Wrong class"},
        {"uid": "prob-002", "title": "Missing annotation"},
    ],
}


@respx.mock
def test_list_by_sequence():
    """AnnotationIssues.list_by_sequence() returns a list of AnnotationIssue objects."""
    seq_uid = "seq-001"
    respx.get(f"{BASE_URL}/sequences/{seq_uid}/annotation-issues/").mock(
        return_value=httpx.Response(200, json=[ISSUE_FIXTURE])
    )
    client = Client(api_key="test-key")
    issues = client.annotation_issues.list_by_sequence(seq_uid)
    assert len(issues) == 1
    issue = issues[0]
    assert issue.uid == "issue-001"
    assert issue.sequence_uid == "seq-001"
    assert issue.priority == "high"
    assert issue.severity == "major"
    assert issue.status == "open"
    assert issue.project is not None
    assert issue.project.name == "My Project"
    assert issue.reporter is not None
    assert issue.reporter.username == "alice"
    assert issue.tool is not None
    assert issue.tool.name == "Bounding Box"
    assert issue.problem is not None
    assert issue.problem.title == "Wrong class"
    client.close()


@respx.mock
def test_list_by_sequence_with_filters():
    """AnnotationIssues.list_by_sequence() forwards optional filter params."""
    seq_uid = "seq-001"
    route = respx.get(f"{BASE_URL}/sequences/{seq_uid}/annotation-issues/").mock(
        return_value=httpx.Response(200, json=[])
    )
    client = Client(api_key="test-key")
    client.annotation_issues.list_by_sequence(seq_uid, dataset_item_uid="item-001", project_uid="proj-001")
    assert route.called
    request = route.calls.last.request
    assert b"dataset_item_uid=item-001" in request.url.query
    assert b"project_uid=proj-001" in request.url.query
    client.close()


@respx.mock
def test_create():
    """AnnotationIssues.create() sends correct payload and returns AnnotationIssue."""
    seq_uid = "seq-001"
    route = respx.post(f"{BASE_URL}/sequences/{seq_uid}/annotation-issues/").mock(
        return_value=httpx.Response(201, json=ISSUE_FIXTURE)
    )
    client = Client(api_key="test-key")
    issue = client.annotation_issues.create(
        seq_uid,
        tool_uid="tool-001",
        problem_uid="prob-001",
        priority="high",
        severity="major",
        description="Wrong label applied",
        wrong_class="cat",
        correct_class="dog",
        should_re_annotate=True,
        should_delete=False,
        object_uid="obj-001",
    )
    assert issue.uid == "issue-001"
    assert issue.priority == "high"
    request_body = json.loads(route.calls.last.request.content)
    assert request_body["tool_uid"] == "tool-001"
    assert request_body["problem_uid"] == "prob-001"
    assert request_body["sequence_uid"] == "seq-001"
    assert request_body["priority"] == "high"
    assert request_body["wrong_class"] == "cat"
    assert request_body["correct_class"] == "dog"
    assert request_body["should_re_annotate"] is True
    assert request_body["should_delete"] is False
    assert request_body["object_uid"] == "obj-001"
    client.close()


@respx.mock
def test_update():
    """AnnotationIssues.update() sends PATCH and returns updated AnnotationIssue."""
    seq_uid = "seq-001"
    issue_uid = "issue-001"
    updated = {**ISSUE_FIXTURE, "status": "closed", "priority": "low"}
    route = respx.patch(f"{BASE_URL}/sequences/{seq_uid}/annotation-issues/{issue_uid}/").mock(
        return_value=httpx.Response(200, json=updated)
    )
    client = Client(api_key="test-key")
    issue = client.annotation_issues.update(seq_uid, issue_uid, status="closed", priority="low")
    assert issue.status == "closed"
    assert issue.priority == "low"
    request_body = json.loads(route.calls.last.request.content)
    assert request_body["status"] == "closed"
    assert request_body["priority"] == "low"
    client.close()


@respx.mock
def test_delete():
    """AnnotationIssues.delete() sends DELETE request and returns None."""
    seq_uid = "seq-001"
    issue_uid = "issue-001"
    route = respx.delete(f"{BASE_URL}/sequences/{seq_uid}/annotation-issues/{issue_uid}/").mock(
        return_value=httpx.Response(204)
    )
    client = Client(api_key="test-key")
    result = client.annotation_issues.delete(seq_uid, issue_uid)
    assert result is None
    assert route.called
    client.close()


@respx.mock
def test_list_by_dataset():
    """AnnotationIssues.list_by_dataset() returns a list of AnnotationIssue objects."""
    owner = "my-org"
    dataset_slug = "my-dataset"
    respx.get(f"{BASE_URL}/datasets/{owner}/{dataset_slug}/annotation-issues/").mock(
        return_value=httpx.Response(200, json=[ISSUE_FIXTURE])
    )
    client = Client(api_key="test-key")
    issues = client.annotation_issues.list_by_dataset(owner, dataset_slug)
    assert len(issues) == 1
    assert issues[0].uid == "issue-001"
    client.close()


@respx.mock
def test_list_by_dataset_with_sequence_filter():
    """AnnotationIssues.list_by_dataset() forwards sequence_uid filter param."""
    owner = "my-org"
    dataset_slug = "my-dataset"
    route = respx.get(f"{BASE_URL}/datasets/{owner}/{dataset_slug}/annotation-issues/").mock(
        return_value=httpx.Response(200, json=[])
    )
    client = Client(api_key="test-key")
    client.annotation_issues.list_by_dataset(owner, dataset_slug, sequence_uid="seq-001")
    assert route.called
    assert b"sequence_uid=seq-001" in route.calls.last.request.url.query
    client.close()


@respx.mock
def test_get_metrics():
    """AnnotationIssues.get_metrics() returns an AnnotationIssueMetrics object."""
    owner = "my-org"
    dataset_slug = "my-dataset"
    respx.get(f"{BASE_URL}/datasets/{owner}/{dataset_slug}/annotation-issues/metrics/").mock(
        return_value=httpx.Response(200, json=METRICS_FIXTURE)
    )
    client = Client(api_key="test-key")
    metrics = client.annotation_issues.get_metrics(owner, dataset_slug)
    assert metrics.status_count == {"open": 10, "closed": 5}
    assert metrics.priority_count == {"high": 8, "low": 7}
    assert metrics.mean_seconds_close_time_all == 3600
    assert metrics.mean_unresolved_issue_age_customer == 172800
    assert metrics.object_count_by_annotation_issue_problem_uid is not None
    assert len(metrics.object_count_by_annotation_issue_problem_uid) == 1
    client.close()


@respx.mock
def test_list_tools():
    """AnnotationIssues.list_tools() returns a list of AnnotationIssueToolDetail objects."""
    route = respx.get(f"{BASE_URL}/qc-available-tools/").mock(return_value=httpx.Response(200, json=[TOOL_FIXTURE]))
    client = Client(api_key="test-key")
    tools = client.annotation_issues.list_tools(dataset_type="image")
    assert len(tools) == 1
    tool = tools[0]
    assert tool.uid == "tool-001"
    assert tool.name == "Bounding Box"
    assert tool.dataset_type == "image"
    assert tool.default is True
    assert tool.problems is not None
    assert len(tool.problems) == 2
    assert tool.problems[0].title == "Wrong class"
    assert b"dataset_type=image" in route.calls.last.request.url.query
    client.close()
