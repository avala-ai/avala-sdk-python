"""Tests for the `avala annotations bulk-edit` CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("click", reason="CLI dependencies not installed (pip install avala[cli])")

import httpx  # noqa: E402
import respx  # noqa: E402
from click.testing import CliRunner  # noqa: E402

from avala.cli import main  # noqa: E402

BASE_URL = "https://api.avala.ai/api/v1"
OWNER = "serve.robotics@avala.ai"
SLUG = "poc2"


def _frame_edit(item_uid: str, object_uuid: str, name: str = "vehicle") -> dict:
    return {
        "dataset_item_uid": item_uid,
        "sequence_uid": None,
        "action": "upsert",
        "duration": 0,
        "object_uuid": object_uuid,
        "object_type": "cuboid",
        "object_name": name,
        "object_name_plural": f"{name}s",
        "object_data": {
            "coordinate_system": "frame",
            "label": name,
            "object_id": object_uuid,
            "position": {"px": 0.0, "py": 0.0, "pz": 0.0},
            "dimensions": {"dx": 1.0, "dy": 1.0, "dz": 1.0},
            "rotation": {"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0},
        },
    }


def _write_jsonl(path: Path, edits: list[dict]) -> None:
    with path.open("w") as fh:
        for edit in edits:
            fh.write(json.dumps(edit) + "\n")


def _write_json_array(path: Path, edits: list[dict]) -> None:
    with path.open("w") as fh:
        json.dump(edits, fh)


def _mock_dataset_detail(data_type: str = "lidar", is_sequence: bool = True) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "uid": "ds-uid",
            "name": "POC2",
            "slug": SLUG,
            "data_type": data_type,
            "is_sequence": is_sequence,
        },
    )


def _mock_items_page(uids: list[str]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "results": [{"uid": u} for u in uids],
            "next": None,
            "previous": None,
        },
    )


@respx.mock
def test_bulk_edit_dry_run_validates_only(tmp_path: Path):
    edits_path = tmp_path / "edits.jsonl"
    _write_jsonl(edits_path, [_frame_edit("item-a", "obj-1"), _frame_edit("item-b", "obj-2")])

    bulk_route = respx.post(f"{BASE_URL}/datasets/{OWNER}/{SLUG}/bulk-edition/").mock(
        return_value=httpx.Response(200, json={})
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--api-key",
            "test-key",
            "annotations",
            "bulk-edit",
            "--owner",
            OWNER,
            "--slug",
            SLUG,
            "--input",
            str(edits_path),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output + (result.exception and str(result.exception) or "")
    assert "DRY RUN" in result.output
    assert not bulk_route.called


@respx.mock
def test_bulk_edit_check_only_runs_preflight_no_post(tmp_path: Path):
    edits_path = tmp_path / "edits.json"
    _write_json_array(edits_path, [_frame_edit("item-a", "obj-1"), _frame_edit("item-b", "obj-2")])

    detail_route = respx.get(f"{BASE_URL}/datasets/{OWNER}/{SLUG}/").mock(return_value=_mock_dataset_detail())
    items_route = respx.get(f"{BASE_URL}/datasets/{OWNER}/{SLUG}/items/").mock(
        return_value=_mock_items_page(["item-a", "item-b", "item-c"])
    )
    bulk_route = respx.post(f"{BASE_URL}/datasets/{OWNER}/{SLUG}/bulk-edition/").mock(
        return_value=httpx.Response(200, json={})
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--api-key",
            "test-key",
            "annotations",
            "bulk-edit",
            "--owner",
            OWNER,
            "--slug",
            SLUG,
            "--input",
            str(edits_path),
            "--check-only",
        ],
    )
    assert result.exit_code == 0, result.output
    assert detail_route.called
    assert items_route.called
    assert not bulk_route.called
    assert "CHECK ONLY" in result.output


@respx.mock
def test_bulk_edit_check_only_fails_on_non_lidar(tmp_path: Path):
    edits_path = tmp_path / "edits.jsonl"
    _write_jsonl(edits_path, [_frame_edit("item-a", "obj-1")])

    respx.get(f"{BASE_URL}/datasets/{OWNER}/{SLUG}/").mock(return_value=_mock_dataset_detail(data_type="image"))
    bulk_route = respx.post(f"{BASE_URL}/datasets/{OWNER}/{SLUG}/bulk-edition/").mock(
        return_value=httpx.Response(200, json={})
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--api-key",
            "test-key",
            "annotations",
            "bulk-edit",
            "--owner",
            OWNER,
            "--slug",
            SLUG,
            "--input",
            str(edits_path),
            "--check-only",
        ],
    )
    assert result.exit_code != 0
    assert "lidar" in result.output.lower()
    assert not bulk_route.called


@respx.mock
def test_bulk_edit_posts_chunked(tmp_path: Path):
    edits_path = tmp_path / "edits.jsonl"
    edits = [_frame_edit(f"item-{i}", f"obj-{i}") for i in range(5)]
    _write_jsonl(edits_path, edits)

    respx.get(f"{BASE_URL}/datasets/{OWNER}/{SLUG}/").mock(return_value=_mock_dataset_detail())
    respx.get(f"{BASE_URL}/datasets/{OWNER}/{SLUG}/items/").mock(
        return_value=_mock_items_page([f"item-{i}" for i in range(5)])
    )
    bulk_route = respx.post(f"{BASE_URL}/datasets/{OWNER}/{SLUG}/bulk-edition/").mock(
        return_value=httpx.Response(200, json={})
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--api-key",
            "test-key",
            "annotations",
            "bulk-edit",
            "--owner",
            OWNER,
            "--slug",
            SLUG,
            "--input",
            str(edits_path),
            "--chunk-size",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert bulk_route.call_count == 3
    assert [len(json.loads(c.request.content)) for c in bulk_route.calls] == [2, 2, 1]


@respx.mock
def test_bulk_edit_idempotent_re_run(tmp_path: Path):
    edits_path = tmp_path / "edits.jsonl"
    _write_jsonl(edits_path, [_frame_edit("item-a", "obj-1")])

    respx.get(f"{BASE_URL}/datasets/{OWNER}/{SLUG}/").mock(return_value=_mock_dataset_detail())
    respx.get(f"{BASE_URL}/datasets/{OWNER}/{SLUG}/items/").mock(return_value=_mock_items_page(["item-a"]))
    bulk_route = respx.post(f"{BASE_URL}/datasets/{OWNER}/{SLUG}/bulk-edition/").mock(
        return_value=httpx.Response(200, json={})
    )

    runner = CliRunner()
    args = [
        "--api-key",
        "test-key",
        "annotations",
        "bulk-edit",
        "--owner",
        OWNER,
        "--slug",
        SLUG,
        "--input",
        str(edits_path),
    ]
    first = runner.invoke(main, args)
    second = runner.invoke(main, args)
    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert bulk_route.call_count == 2


@respx.mock
def test_bulk_edit_rejects_invalid_action(tmp_path: Path):
    bad = _frame_edit("item-a", "obj-1")
    bad["action"] = "delete-everything"
    edits_path = tmp_path / "edits.jsonl"
    _write_jsonl(edits_path, [bad])

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--api-key",
            "test-key",
            "annotations",
            "bulk-edit",
            "--owner",
            OWNER,
            "--slug",
            SLUG,
            "--input",
            str(edits_path),
            "--dry-run",
        ],
    )
    assert result.exit_code != 0
    assert "action" in result.output.lower()
