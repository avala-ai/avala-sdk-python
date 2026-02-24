"""Tests for the Avala CLI."""

import pytest

pytest.importorskip("click", reason="CLI dependencies not installed (pip install avala[cli])")

import httpx  # noqa: E402
import respx  # noqa: E402
from click.testing import CliRunner  # noqa: E402

from avala.cli import main  # noqa: E402


@respx.mock
def test_datasets_list():
    respx.get("https://api.avala.ai/api/v1/datasets/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "uid": "abc123",
                        "name": "My Dataset",
                        "slug": "my-dataset",
                        "item_count": 42,
                        "data_type": "image",
                    }
                ],
                "next": None,
                "previous": None,
            },
        )
    )
    runner = CliRunner()
    result = runner.invoke(main, ["--api-key", "test-key", "datasets", "list"])
    assert result.exit_code == 0
    assert "My Dataset" in result.output
    assert "abc123" in result.output


@respx.mock
def test_datasets_get():
    uid = "abc123"
    respx.get(f"https://api.avala.ai/api/v1/datasets/{uid}/").mock(
        return_value=httpx.Response(
            200,
            json={
                "uid": uid,
                "name": "My Dataset",
                "slug": "my-dataset",
                "item_count": 42,
                "data_type": "image",
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-02T00:00:00Z",
            },
        )
    )
    runner = CliRunner()
    result = runner.invoke(main, ["--api-key", "test-key", "datasets", "get", uid])
    assert result.exit_code == 0
    assert "My Dataset" in result.output


@respx.mock
def test_projects_list():
    respx.get("https://api.avala.ai/api/v1/projects/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "uid": "proj123",
                        "name": "My Project",
                        "status": "active",
                        "created_at": "2026-01-01T00:00:00Z",
                        "updated_at": "2026-01-02T00:00:00Z",
                    }
                ],
                "next": None,
                "previous": None,
            },
        )
    )
    runner = CliRunner()
    result = runner.invoke(main, ["--api-key", "test-key", "projects", "list"])
    assert result.exit_code == 0
    assert "My Project" in result.output


@respx.mock
def test_storage_configs_list():
    respx.get("https://api.avala.ai/api/v1/storage-configs/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "uid": "sc123",
                        "name": "My S3 Bucket",
                        "provider": "aws_s3",
                        "is_verified": True,
                        "created_at": "2026-01-01T00:00:00Z",
                        "updated_at": "2026-01-02T00:00:00Z",
                    }
                ],
                "next": None,
                "previous": None,
            },
        )
    )
    runner = CliRunner()
    result = runner.invoke(main, ["--api-key", "test-key", "storage-configs", "list"])
    assert result.exit_code == 0
    assert "My S3 Bucket" in result.output


@respx.mock
def test_exports_list():
    respx.get("https://api.avala.ai/api/v1/exports/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "uid": "exp123",
                        "status": "exported",
                        "download_url": "https://example.com/export.zip",
                        "created_at": "2026-01-01T00:00:00Z",
                        "updated_at": "2026-01-02T00:00:00Z",
                    }
                ],
                "next": None,
                "previous": None,
            },
        )
    )
    runner = CliRunner()
    result = runner.invoke(main, ["--api-key", "test-key", "exports", "list"])
    assert result.exit_code == 0
    assert "exp123" in result.output


@respx.mock
def test_tasks_list():
    respx.get("https://api.avala.ai/api/v1/tasks/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "uid": "task123",
                        "name": "Box Task",
                        "type": "box",
                        "status": "completed",
                        "project": "proj123",
                        "created_at": "2026-01-01T00:00:00Z",
                        "updated_at": "2026-01-02T00:00:00Z",
                    }
                ],
                "next": None,
                "previous": None,
            },
        )
    )
    runner = CliRunner()
    result = runner.invoke(main, ["--api-key", "test-key", "tasks", "list"])
    assert result.exit_code == 0
    assert "Box Task" in result.output


def test_missing_api_key():
    runner = CliRunner(env={"AVALA_API_KEY": ""})
    result = runner.invoke(main, ["datasets", "list"])
    assert result.exit_code != 0


@respx.mock
def test_api_key_via_env():
    respx.get("https://api.avala.ai/api/v1/datasets/").mock(
        return_value=httpx.Response(
            200,
            json={"results": [], "next": None, "previous": None},
        )
    )
    runner = CliRunner(env={"AVALA_API_KEY": "env-test-key"})
    result = runner.invoke(main, ["datasets", "list"])
    assert result.exit_code == 0


@respx.mock
def test_auth_failure():
    respx.get("https://api.avala.ai/api/v1/datasets/").mock(
        return_value=httpx.Response(401, json={"detail": "Invalid API key"})
    )
    runner = CliRunner()
    result = runner.invoke(main, ["--api-key", "bad-key", "datasets", "list"])
    assert result.exit_code != 0


def test_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Avala CLI" in result.output


def test_datasets_list_help():
    runner = CliRunner()
    result = runner.invoke(main, ["datasets", "list", "--help"])
    assert result.exit_code == 0
    assert "--limit" in result.output
