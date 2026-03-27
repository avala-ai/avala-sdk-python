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
def test_datasets_list_with_filters():
    route = respx.get("https://api.avala.ai/api/v1/datasets/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "uid": "abc123",
                        "name": "Highway MCAP",
                        "slug": "highway-mcap",
                        "item_count": 50,
                        "data_type": "mcap",
                    }
                ],
                "next": None,
                "previous": None,
            },
        )
    )
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--api-key",
            "test-key",
            "datasets",
            "list",
            "--data-type",
            "mcap",
            "--name",
            "highway",
            "--status",
            "created",
            "--visibility",
            "private",
        ],
    )
    assert result.exit_code == 0
    assert "Highway MCAP" in result.output
    assert route.called
    request = route.calls[0].request
    assert request.url.params["data_type"] == "mcap"
    assert request.url.params["name"] == "highway"
    assert request.url.params["status"] == "created"
    assert request.url.params["visibility"] == "private"


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


@respx.mock
def test_datasets_create():
    respx.post("https://api.avala.ai/api/v1/datasets/").mock(
        return_value=httpx.Response(
            201,
            json={
                "uid": "new-ds-uid",
                "name": "New Dataset",
                "slug": "new-dataset",
                "item_count": 0,
                "data_type": "lidar",
            },
        )
    )
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--api-key",
            "test-key",
            "datasets",
            "create",
            "--name",
            "New Dataset",
            "--slug",
            "new-dataset",
            "--data-type",
            "lidar",
        ],
    )
    assert result.exit_code == 0
    assert "new-ds-uid" in result.output
    assert "New Dataset" in result.output


@respx.mock
def test_datasets_create_with_provider_config():
    respx.post("https://api.avala.ai/api/v1/datasets/").mock(
        return_value=httpx.Response(
            201,
            json={
                "uid": "s3-ds-uid",
                "name": "S3 Dataset",
                "slug": "s3-dataset",
                "item_count": 0,
                "data_type": "image",
            },
        )
    )
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--api-key",
            "test-key",
            "datasets",
            "create",
            "--name",
            "S3 Dataset",
            "--slug",
            "s3-dataset",
            "--data-type",
            "image",
            "--is-sequence",
            "--provider-config",
            '{"provider": "aws_s3", "s3_bucket_name": "my-bucket"}',
            "--owner",
            "my-org",
        ],
    )
    assert result.exit_code == 0
    assert "s3-ds-uid" in result.output


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


def test_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Avala CLI" in result.output


def test_datasets_list_help():
    runner = CliRunner()
    result = runner.invoke(main, ["datasets", "list", "--help"])
    assert result.exit_code == 0
    assert "--data-type" in result.output
    assert "--name" in result.output
    assert "--status" in result.output
    assert "--visibility" in result.output
    assert "--limit" in result.output


def test_version_flag():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "avala, version" in result.output


def test_shell_completion_bash():
    runner = CliRunner()
    result = runner.invoke(main, ["shell-completion", "bash"])
    assert result.exit_code == 0
    assert "_AVALA_COMPLETE=bash_source" in result.output


def test_shell_completion_zsh():
    runner = CliRunner()
    result = runner.invoke(main, ["shell-completion", "zsh"])
    assert result.exit_code == 0
    assert "_AVALA_COMPLETE=zsh_source" in result.output


def test_shell_completion_fish():
    runner = CliRunner()
    result = runner.invoke(main, ["shell-completion", "fish"])
    assert result.exit_code == 0
    assert "_AVALA_COMPLETE=fish_source" in result.output


@respx.mock
def test_auth_error_shows_friendly_message():
    respx.get("https://api.avala.ai/api/v1/datasets/").mock(
        return_value=httpx.Response(401, json={"detail": "Invalid API key"})
    )
    runner = CliRunner()
    result = runner.invoke(main, ["--api-key", "bad-key", "datasets", "list"])
    assert result.exit_code != 0
    assert "AVALA_API_KEY" in result.output
    assert "avala configure" in result.output


def test_missing_key_shows_friendly_message():
    runner = CliRunner(env={"AVALA_API_KEY": ""})
    result = runner.invoke(main, ["datasets", "list"])
    assert result.exit_code != 0
    assert "No API key provided" in result.output
    assert "avala configure" in result.output


def test_shell_completion_autodetect():
    runner = CliRunner(env={"SHELL": "/bin/zsh"})
    result = runner.invoke(main, ["shell-completion"])
    assert result.exit_code == 0
    assert "_AVALA_COMPLETE=zsh_source" in result.output


def test_bad_base_url_shows_friendly_message():
    runner = CliRunner()
    result = runner.invoke(main, ["--api-key", "test", "--base-url", "ftp://bad", "datasets", "list"])
    assert result.exit_code != 0
    assert "HTTPS" in result.output or "Invalid" in result.output


# ── JSON output tests ───────────────────────────────────────────────────────


_DATASETS_LIST_RESPONSE = {
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
}

_DATASET_DETAIL_RESPONSE = {
    "uid": "abc123",
    "name": "My Dataset",
    "slug": "my-dataset",
    "item_count": 42,
    "data_type": "image",
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-02T00:00:00Z",
}


@respx.mock
def test_datasets_list_json():
    import json as _json

    respx.get("https://api.avala.ai/api/v1/datasets/").mock(
        return_value=httpx.Response(200, json=_DATASETS_LIST_RESPONSE)
    )
    runner = CliRunner()
    result = runner.invoke(main, ["--api-key", "test-key", "-o", "json", "datasets", "list"])
    assert result.exit_code == 0
    data = _json.loads(result.output)
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["uid"] == "abc123"
    assert data[0]["name"] == "My Dataset"
    assert data[0]["item_count"] == "42"
    assert data[0]["data_type"] == "image"


@respx.mock
def test_datasets_get_json():
    import json as _json

    uid = "abc123"
    respx.get(f"https://api.avala.ai/api/v1/datasets/{uid}/").mock(
        return_value=httpx.Response(200, json=_DATASET_DETAIL_RESPONSE)
    )
    runner = CliRunner()
    result = runner.invoke(main, ["--api-key", "test-key", "-o", "json", "datasets", "get", uid])
    assert result.exit_code == 0
    data = _json.loads(result.output)
    assert isinstance(data, dict)
    assert data["name"] == "My Dataset"
    assert data["uid"] == "abc123"
    assert data["item_count"] == "42"
    assert data["data_type"] == "image"


@respx.mock
def test_default_output_is_table():
    respx.get("https://api.avala.ai/api/v1/datasets/").mock(
        return_value=httpx.Response(200, json=_DATASETS_LIST_RESPONSE)
    )
    runner = CliRunner()
    result = runner.invoke(main, ["--api-key", "test-key", "datasets", "list"])
    assert result.exit_code == 0
    # Table output uses rich formatting; should NOT be valid JSON
    import json as _json

    try:
        _json.loads(result.output)
        is_json = True
    except _json.JSONDecodeError:
        is_json = False
    assert not is_json, "Default output should be table format, not JSON"


# ── status command tests ────────────────────────────────────────────────────


_ORGS_LIST_RESPONSE = {
    "results": [{"uid": "org-1", "name": "Acme Robotics", "slug": "acme-robotics"}],
    "next": None,
    "previous": None,
}

_DATASETS_LIST_RESPONSE_STATUS = {
    "results": [
        {"uid": "ds-1", "name": "Highway LiDAR", "slug": "highway-lidar", "item_count": 100, "data_type": "lidar"},
        {"uid": "ds-2", "name": "Urban Camera", "slug": "urban-camera", "item_count": 50, "data_type": "image"},
    ],
    "next": None,
    "previous": None,
}

_PROJECTS_LIST_RESPONSE_STATUS = {
    "results": [
        {"uid": "proj-1", "name": "Lane Detection", "status": "active"},
        {"uid": "proj-2", "name": "Object Tracking", "status": "paused"},
    ],
    "next": None,
    "previous": None,
}

_EXPORTS_LIST_RESPONSE_STATUS = {
    "results": [
        {"uid": "exp-1", "status": "pending"},
        {"uid": "exp-2", "status": "exported"},
    ],
    "next": None,
    "previous": None,
}

_FLEET_DEVICES_RESPONSE = {
    "results": [
        {"uid": "dev-1", "name": "Robot A", "status": "online", "tags": []},
        {"uid": "dev-2", "name": "Robot B", "status": "offline", "tags": []},
        {"uid": "dev-3", "name": "Robot C", "status": "online", "tags": []},
    ],
    "next": None,
    "previous": None,
}


def _mock_status_endpoints(*, fleet_error: bool = False) -> None:
    """Set up respx mocks for all status dashboard endpoints."""
    respx.get("https://api.avala.ai/api/v1/organizations/").mock(
        return_value=httpx.Response(200, json=_ORGS_LIST_RESPONSE)
    )
    respx.get("https://api.avala.ai/api/v1/datasets/").mock(
        return_value=httpx.Response(200, json=_DATASETS_LIST_RESPONSE_STATUS)
    )
    respx.get("https://api.avala.ai/api/v1/projects/").mock(
        return_value=httpx.Response(200, json=_PROJECTS_LIST_RESPONSE_STATUS)
    )
    respx.get("https://api.avala.ai/api/v1/exports/").mock(
        return_value=httpx.Response(200, json=_EXPORTS_LIST_RESPONSE_STATUS)
    )
    if fleet_error:
        respx.get("https://api.avala.ai/api/v1/fleet/devices/").mock(
            return_value=httpx.Response(403, json={"detail": "Fleet not enabled"})
        )
    else:
        respx.get("https://api.avala.ai/api/v1/fleet/devices/").mock(
            return_value=httpx.Response(200, json=_FLEET_DEVICES_RESPONSE)
        )


@respx.mock
def test_status_command():
    _mock_status_endpoints()
    runner = CliRunner()
    result = runner.invoke(main, ["--api-key", "test-key", "status"])
    assert result.exit_code == 0
    assert "Acme Robotics" in result.output
    assert "Highway LiDAR" in result.output
    assert "Urban Camera" in result.output
    assert "Lane Detection" in result.output
    assert "Object Tracking" in result.output


@respx.mock
def test_status_json():
    import json as _json

    _mock_status_endpoints()
    runner = CliRunner()
    result = runner.invoke(main, ["--api-key", "test-key", "-o", "json", "status"])
    assert result.exit_code == 0
    data = _json.loads(result.output)
    assert data["organizations"][0]["name"] == "Acme Robotics"
    assert len(data["datasets"]["latest"]) == 2
    assert len(data["projects"]["latest"]) == 2
    assert data["exports"]["pending"][0]["uid"] == "exp-1"
    assert data["fleet"]["online"] == 2
    assert data["fleet"]["offline"] == 1


@respx.mock
def test_status_without_fleet():
    _mock_status_endpoints(fleet_error=True)
    runner = CliRunner()
    result = runner.invoke(main, ["--api-key", "test-key", "status"])
    assert result.exit_code == 0
    assert "Acme Robotics" in result.output
    # Fleet section should be skipped gracefully
    assert "Highway LiDAR" in result.output


# ── exports wait CLI tests ──────────────────────────────────────────────────


@respx.mock
def test_exports_wait_success():
    """CLI 'exports wait' polls until exported and shows details."""
    uid = "exp-wait-001"
    respx.get(f"https://api.avala.ai/api/v1/exports/{uid}/").mock(
        side_effect=[
            httpx.Response(200, json={"uid": uid, "status": "pending"}),
            httpx.Response(
                200,
                json={
                    "uid": uid,
                    "status": "exported",
                    "download_url": "https://example.com/out.zip",
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-02T00:00:00Z",
                },
            ),
        ]
    )
    from unittest.mock import patch

    with patch("avala.resources.exports.time.sleep", return_value=None):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--api-key", "test-key", "exports", "wait", uid, "--interval", "0.1"],
        )
    assert result.exit_code == 0
    assert uid in result.output
    assert "exported" in result.output
    assert "https://example.com/out.zip" in result.output


@respx.mock
def test_exports_wait_timeout():
    """CLI 'exports wait' shows error on timeout."""
    uid = "exp-timeout-001"
    respx.get(f"https://api.avala.ai/api/v1/exports/{uid}/").mock(
        return_value=httpx.Response(200, json={"uid": uid, "status": "pending"})
    )
    from unittest.mock import patch

    with (
        patch("avala.resources.exports.time.sleep", return_value=None),
        patch("avala.resources.exports.time.monotonic", side_effect=[0.0, 1.0, 999.0]),
    ):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--api-key", "test-key", "exports", "wait", uid, "--timeout", "5"],
        )
    assert result.exit_code != 0
    assert "did not complete" in result.output
