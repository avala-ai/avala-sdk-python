"""Tests for Pydantic model validation."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError as PydanticValidationError

from avala.types.dataset import Dataset, DatasetItem, DatasetSequence
from avala.types.export import Export
from avala.types.project import Project
from avala.types.slice import SliceItem
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


class TestDatasetSequenceModel:
    def test_workflow_fields_are_parsed_when_exposed(self) -> None:
        sequence = DatasetSequence.model_validate(
            {
                "uid": "sequence-uid",
                "is_workflow_terminal": True,
                "sequence_status_workflow": {
                    "workflow_revision_uid": "revision-uid",
                    "initial_status": "labeling",
                    "terminal_status": "complete",
                    "statuses": [],
                },
                "sequence_deliverable_workflow": {
                    "schema_version": 2,
                    "workflow_revision_uid": "revision-uid",
                    "is_complete": True,
                    "deliverables": [],
                },
            }
        )

        assert sequence.is_workflow_terminal is True
        assert sequence.sequence_status_workflow is not None
        assert sequence.sequence_status_workflow["initial_status"] == "labeling"
        assert sequence.sequence_deliverable_workflow is not None
        assert sequence.sequence_deliverable_workflow["schema_version"] == 2

    def test_workflow_fields_default_to_none_when_hidden(self) -> None:
        sequence = DatasetSequence(uid="sequence-uid")

        assert sequence.is_workflow_terminal is None
        assert sequence.sequence_status_workflow is None
        assert sequence.sequence_deliverable_workflow is None


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


@pytest.mark.parametrize("model", [DatasetItem, DatasetSequence, SliceItem])
@pytest.mark.parametrize("hidden", [True, False])
def test_hidden_state_survives_response_parsing(model: type, hidden: bool) -> None:
    item = model.model_validate({"uid": "item-uid", "is_hidden": hidden})
    assert item.is_hidden is hidden
    assert item.model_dump()["is_hidden"] is hidden


@pytest.mark.parametrize("model", [DatasetItem, DatasetSequence, SliceItem])
def test_hidden_state_defaults_for_older_server_responses(model: type) -> None:
    item = model.model_validate({"uid": "item-uid"})
    assert item.is_hidden is False
    assert "is_hidden" not in item.model_fields_set


@pytest.mark.parametrize("model", [DatasetItem, DatasetSequence, SliceItem])
def test_hidden_state_rejects_null_like_server_boolean_field(model: type) -> None:
    with pytest.raises(PydanticValidationError):
        model.model_validate({"uid": "item-uid", "is_hidden": None})
