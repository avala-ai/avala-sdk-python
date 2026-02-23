"""Tests for Pydantic model validation."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError as PydanticValidationError

from avala.types.dataset import Dataset
from avala.types.export import Export
from avala.types.project import Project
from avala.types.task import Task


class TestDatasetModel:
    def test_required_fields(self):
        """Dataset requires uid, name, slug."""
        ds = Dataset(uid="uid-1", name="My Dataset", slug="my-dataset")
        assert ds.uid == "uid-1"
        assert ds.name == "My Dataset"
        assert ds.slug == "my-dataset"

    def test_optional_defaults(self):
        """item_count defaults to 0; data_type, created_at, updated_at default to None."""
        ds = Dataset(uid="uid-1", name="My Dataset", slug="my-dataset")
        assert ds.item_count == 0
        assert ds.data_type is None
        assert ds.created_at is None
        assert ds.updated_at is None

    def test_datetime_parsing(self):
        """ISO datetime strings are parsed to datetime objects."""
        ds = Dataset(
            uid="uid-1",
            name="My Dataset",
            slug="my-dataset",
            created_at="2024-06-01T12:00:00Z",
            updated_at="2024-06-02T08:30:00Z",
        )
        assert isinstance(ds.created_at, datetime)
        assert isinstance(ds.updated_at, datetime)

    def test_missing_required_raises(self):
        """Missing required field (uid) raises a Pydantic ValidationError."""
        with pytest.raises(PydanticValidationError):
            Dataset(name="My Dataset", slug="my-dataset")  # type: ignore[call-arg]

    def test_extra_fields_ignored(self):
        """Extra fields in the payload are ignored (Pydantic v2 default behavior)."""
        ds = Dataset.model_validate(
            {"uid": "uid-1", "name": "My Dataset", "slug": "my-dataset", "unknown_field": "value"}
        )
        assert ds.uid == "uid-1"
        assert not hasattr(ds, "unknown_field")
        assert "unknown_field" not in ds.model_dump()


class TestProjectModel:
    def test_required_fields(self):
        """Project requires uid and name."""
        project = Project(uid="proj-uid", name="My Project")
        assert project.uid == "proj-uid"
        assert project.name == "My Project"

    def test_optional_defaults(self):
        """status, created_at, updated_at default to None."""
        project = Project(uid="proj-uid", name="My Project")
        assert project.status is None
        assert project.created_at is None
        assert project.updated_at is None

    def test_missing_required_raises(self):
        """Missing required field (name) raises a Pydantic ValidationError."""
        with pytest.raises(PydanticValidationError):
            Project(uid="proj-uid")  # type: ignore[call-arg]


class TestExportModel:
    def test_required_fields(self):
        """Export requires uid."""
        export = Export(uid="export-uid")
        assert export.uid == "export-uid"

    def test_optional_defaults(self):
        """status, download_url, created_at, updated_at default to None."""
        export = Export(uid="export-uid")
        assert export.status is None
        assert export.download_url is None
        assert export.created_at is None
        assert export.updated_at is None

    def test_missing_required_raises(self):
        """Missing required field (uid) raises a Pydantic ValidationError."""
        with pytest.raises(PydanticValidationError):
            Export()  # type: ignore[call-arg]


class TestTaskModel:
    def test_required_fields(self):
        """Task requires uid."""
        task = Task(uid="task-uid")
        assert task.uid == "task-uid"

    def test_optional_defaults(self):
        """type, name, status, project, created_at, updated_at default to None."""
        task = Task(uid="task-uid")
        assert task.type is None
        assert task.name is None
        assert task.status is None
        assert task.project is None
        assert task.created_at is None
        assert task.updated_at is None

    def test_model_validate_full(self):
        """All fields can be set and round-trip through model_validate."""
        data = {
            "uid": "task-uid",
            "type": "annotation",
            "name": "My Task",
            "status": "pending",
            "project": "proj-uid",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-02T00:00:00Z",
        }
        task = Task.model_validate(data)
        assert task.uid == "task-uid"
        assert task.type == "annotation"
        assert isinstance(task.created_at, datetime)
