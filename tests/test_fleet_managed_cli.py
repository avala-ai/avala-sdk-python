"""Explicit managed CLI selection preserves exact publication and local evidence."""

import json
import inspect
from types import SimpleNamespace

import httpx
import pytest
from click.testing import CliRunner

from avala import Client
from avala.cli import main
from avala.cli.fleet import fleet
from avala.cli import _fleet_managed as managed_cli
from avala._fleet_uploads import SourceFile, SourceInventory
from avala.errors import UploadStateError
from avala.resources.fleet import uploads
from avala.resources.fleet._managed_consumer import validate_managed_inventory
from avala.resources.fleet import _managed_consumer as consumer
from tests.test_fleet_managed_upload import transfer as transfer


def _runner():
    kwargs = {"mix_stderr": False} if "mix_stderr" in inspect.signature(CliRunner).parameters else {}
    return CliRunner(**kwargs)


def command(transfer, *extra, client=None, output="table"):
    return _runner().invoke(
        fleet,
        [
            "recordings",
            "upload",
            "--managed",
            "--source",
            str(transfer.source),
            "--recording",
            transfer.recording,
            *extra,
        ],
        obj={"client": client, "output_format": output},
    )


def test_managed_cli_publishes_and_resumes_without_legacy_recording_calls(transfer):
    with Client(api_key="synthetic-api-key") as client:
        first = command(transfer, "--wait", client=client, output="json")
        assert first.exit_code == 0, first.stdout + first.stderr
        assert "Published" in first.stderr
        data = json.loads(first.stdout)
        assert data["publication_uid"] == transfer.finalization["publication_uid"]
        assert data["dataset_uid"] == transfer.finalization["dataset_uid"]
        puts = list(transfer.put_bytes)
        second = command(transfer, client=client)
        assert second.exit_code == 0, second.stdout + second.stderr
        assert transfer.put_bytes == puts
    assert transfer.receipt().exists()
    assert "ready" not in (first.stdout + first.stderr).lower()


def test_managed_cli_dry_run_without_credentials_has_no_network_or_receipt(transfer, monkeypatch):
    monkeypatch.delenv("AVALA_API_KEY", raising=False)
    result = _runner().invoke(
        main,
        [
            "fleet",
            "recordings",
            "upload",
            "--managed",
            "--source",
            str(transfer.source),
            "--recording",
            transfer.recording,
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "clip.mcap" in (result.stdout + result.stderr)
    assert not transfer.route.called and not transfer.put_route.called
    assert not uploads._STATE_DIR.exists()


@pytest.mark.parametrize("wait", [False, True])
def test_managed_cli_pending_is_not_published(transfer, wait):
    transfer.complete_on_poll = False
    with Client(api_key="synthetic-api-key") as client:
        result = command(transfer, *(["--wait", "--wait-timeout", "0"] if wait else []), client=client)
    assert result.exit_code == (1 if wait else 0), result.stdout + result.stderr
    assert "Published" not in (result.stdout + result.stderr) and "ready" not in (result.stdout + result.stderr).lower()
    assert ("Timed out" if wait else "submitted") in (result.stdout + result.stderr)
    assert transfer.receipt().exists()


@pytest.mark.parametrize(
    "extra, text",
    [
        (["--storage-config", "customer-bucket"], "cannot be combined"),
        (["--workers", "0"], "between one and four"),
        (["--workers", "5"], "between one and four"),
        (["--wait", "--wait-timeout", "-1"], "between zero and 3600"),
        (["--wait", "--wait-timeout", "3601"], "between zero and 3600"),
        (["--wait", "--wait-timeout", "nan"], "between zero and 3600"),
        (["--wait", "--wait-timeout", "inf"], "between zero and 3600"),
        (["--recording", "not-a-uuid"], "canonical UUID"),
    ],
)
@pytest.mark.parametrize("preview", [False, True])
def test_managed_invalid_options_fail_before_source_or_remote_work(transfer, monkeypatch, extra, text, preview):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid options must not inventory files")

    monkeypatch.setattr(managed_cli, "collect_source", forbidden)
    result = command(transfer, *extra, *(["--dry-run"] if preview else []))
    assert result.exit_code != 0 and text in (result.stdout + result.stderr)
    assert not transfer.route.called and not transfer.put_route.called
    assert not uploads._STATE_DIR.exists()


@pytest.mark.parametrize(
    "name, payload",
    [
        ("clip.mcap.gz", b"compressed"),
        ("clip.bag", b"rosbag"),
        ("clip.mcap", b""),
        ("__derived__/clip.mcap", b"derived"),
        ("e\u0301.mcap", b"decomposed"),
        ("escape\x1b[31m.mcap", b"control"),
    ],
)
def test_managed_preview_rejects_unsupported_originals_locally(transfer, name, payload):
    (transfer.source / "clip.mcap").unlink()
    original = transfer.source / name
    original.parent.mkdir(parents=True, exist_ok=True)
    original.write_bytes(payload)
    result = command(transfer, "--dry-run")
    assert result.exit_code != 0
    assert "one to 64 uncompressed MCAP originals" in (result.stdout + result.stderr)
    assert "\x1b" not in (result.stdout + result.stderr)
    assert not transfer.route.called and not uploads._STATE_DIR.exists()


@pytest.mark.parametrize(
    "count,size,accepted",
    [
        (1, 8 * 1024**3, True),
        (1, 8 * 1024**3 + 1, False),
        (8, 8 * 1024**3, True),
        (9, 8 * 1024**3, False),
        (64, 1, True),
        (65, 1, False),
    ],
)
def test_managed_preview_uses_exact_shared_size_and_count_profile(transfer, monkeypatch, count, size, accepted):
    inventory = SourceInventory(
        transfer.source,
        tuple(SourceFile(f"{number}.mcap", size, "a" * 64) for number in range(count)),
    )
    monkeypatch.setattr(managed_cli, "collect_source", lambda *args, **kwargs: inventory)
    if accepted:
        validate_managed_inventory(inventory)
    else:
        with pytest.raises(UploadStateError):
            validate_managed_inventory(inventory)
    result = command(transfer, "--dry-run")
    assert (result.exit_code == 0) == accepted, result.stdout + result.stderr
    assert not transfer.route.called and not uploads._STATE_DIR.exists()


def test_managed_preview_excludes_state_and_escapes_source_control_characters(transfer, monkeypatch):
    renamed = transfer.source.with_name("source\x1b[31m")
    transfer.source.rename(renamed)
    transfer.source = renamed
    state = renamed / ".avala" / "uploads"
    state.mkdir(parents=True)
    receipt = state / "synthetic-other-recording.json"
    receipt.write_text("synthetic-private-receipt")
    monkeypatch.setattr(uploads, "_STATE_DIR", state)
    before = sorted(state.rglob("*"))
    result = command(transfer, "--dry-run")
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "\\u001b" in (result.stdout + result.stderr) and "\x1b" not in (result.stdout + result.stderr)
    assert "synthetic-other-recording" not in (result.stdout + result.stderr) and "synthetic-private-receipt" not in (
        result.stdout + result.stderr
    )
    assert sorted(state.rglob("*")) == before
    assert receipt.read_text() == "synthetic-private-receipt"
    assert not transfer.route.called and not transfer.put_route.called


@pytest.mark.parametrize("status_code", [401, 403, 404])
def test_managed_refusal_never_falls_back_or_exposes_server_detail(transfer, status_code):
    transfer.route.mock(return_value=httpx.Response(status_code, json={"detail": "private-token-signature"}))
    with Client(api_key="synthetic-api-key", max_retries=0) as client:
        result = command(transfer, client=client)
    assert result.exit_code != 0
    assert "private-token-signature" not in (result.stdout + result.stderr)
    assert "Retain the receipt" in (result.stdout + result.stderr)
    assert transfer.receipt().exists() and not transfer.put_route.called
    assert all("/managed-upload/admit/" in str(call.request.url) for call in transfer.route.calls)


@pytest.mark.parametrize("error", [RuntimeError, UploadStateError])
def test_managed_cli_sanitizes_transport_and_sdk_exception_details(transfer, monkeypatch, error):
    def failing(*args, **kwargs):
        raise error("https://signed.example/?X-Amz-Signature=synthetic-secret")

    with Client(api_key="synthetic-api-key") as client:
        monkeypatch.setattr(client.fleet.uploads, "upload_managed_recording", failing)
        result = command(transfer, client=client)
    assert result.exit_code != 0
    assert "synthetic-secret" not in (result.stdout + result.stderr) and "signed.example" not in (
        result.stdout + result.stderr
    )


@pytest.mark.parametrize("code", ["content_mismatch", "invalid_mcap", "attempts_exhausted"])
def test_managed_failed_publication_has_nonzero_exit_and_retains_originals(transfer, code):
    transfer.complete_on_poll = False
    transfer.upload()
    transfer.finalization.update(state="failed", code=code)
    with Client(api_key="synthetic-api-key") as client:
        result = command(transfer, client=client, output="json")
    assert result.exit_code == 1, result.stdout + result.stderr
    assert json.loads(result.stdout)["code"] == code
    assert "publication failed" in (result.stdout + result.stderr) and "Published" not in (
        result.stdout + result.stderr
    )
    assert transfer.receipt().exists() and (transfer.source / "clip.mcap").exists()


def test_managed_json_is_a_bounded_whitelist(transfer):
    with Client(api_key="synthetic-api-key") as client:
        result = command(transfer, client=client, output="json")
    assert result.exit_code == 0, result.stdout + result.stderr
    data = json.loads(result.stdout)
    assert set(data) == {
        "protocol",
        "recording_uid",
        "session_uid",
        "upload_status",
        "confirmed_files",
        "total_files",
        "confirmed_bytes",
        "total_bytes",
        "finalization_state",
        "code",
        "publication_uid",
        "dataset_uid",
    }
    assert "clip.mcap" not in result.stdout and "sha256" not in result.stdout
    assert "r2.cloudflarestorage.com" not in (result.stdout + result.stderr) and "X-Amz" not in (
        result.stdout + result.stderr
    )


def test_managed_partial_transfer_resumes_same_session(transfer):
    transfer.put_route.mock(side_effect=httpx.ConnectError("private-upload-url"))
    with Client(api_key="synthetic-api-key") as client:
        first = command(transfer, client=client)
        assert first.exit_code == 1 and "private-upload-url" not in (first.stdout + first.stderr)
        session = json.loads(transfer.receipt().read_text())["session_uid"]
        transfer.put_route.mock(side_effect=transfer.put)
        second = command(transfer, client=client)
        assert second.exit_code == 0, second.stdout + second.stderr
    assert json.loads(transfer.receipt().read_text())["session_uid"] == session
    assert sum(action == "admit/" for action, _ in transfer.requests) == 1
    assert sum(action == "multipart/init/" for action, _ in transfer.requests) == 1


def test_managed_upload_requires_credentials_after_successful_preview(transfer):
    result = command(transfer)
    assert result.exit_code == 1 and "No API key provided" in (result.stdout + result.stderr)
    assert not transfer.route.called and not uploads._STATE_DIR.exists()


def test_managed_wait_observes_publication_without_recording_readiness(transfer, monkeypatch):
    transfer.complete_on_poll = False
    original_status = transfer.status
    polls = []

    def delayed_publication():
        if transfer.finalization:
            polls.append(transfer.finalization["state"])
            transfer.complete_on_poll = len(polls) >= 3
        return original_status()

    monkeypatch.setattr(transfer, "status", delayed_publication)
    monkeypatch.setattr(consumer.time, "sleep", lambda seconds: None)
    with Client(api_key="synthetic-api-key") as client:
        result = command(transfer, "--wait", "--wait-timeout", "10", client=client)
    assert result.exit_code == 0, result.stdout + result.stderr
    assert len(polls) == 3 and "Published dataset" in (result.stdout + result.stderr)
    assert "ready" not in (result.stdout + result.stderr).lower()


@pytest.mark.parametrize("storage_config", [None, "11111111-1111-4111-8111-111111111111"])
def test_default_and_byob_keep_legacy_selection(transfer, monkeypatch, storage_config):
    calls = []

    def forbidden(*args, **kwargs):
        pytest.fail("Legacy CLI must not select managed upload")

    def legacy(recording_uid, source, **kwargs):
        calls.append((recording_uid, source, kwargs))
        return SimpleNamespace(confirmed_files=1, total_files=1, status="completed")

    with Client(api_key="synthetic-api-key") as client:
        monkeypatch.setattr(client.fleet.uploads, "upload_managed_recording", forbidden)
        monkeypatch.setattr(client.fleet.uploads, "upload_recording", legacy)
        monkeypatch.setattr(
            client.fleet.recordings,
            "get",
            lambda uid: SimpleNamespace(uid=uid, status="processing", size_bytes=13),
        )
        result = _runner().invoke(
            fleet,
            [
                "recordings",
                "upload",
                "--source",
                str(transfer.source),
                "--recording",
                transfer.recording,
                *(["--storage-config", storage_config] if storage_config else []),
            ],
            obj={"client": client},
        )
    assert result.exit_code == 0, result.stdout + result.stderr
    assert len(calls) == 1 and calls[0][2]["storage_config_uid"] == storage_config
    assert calls[0][2]["max_workers"] == 4
    assert not transfer.route.called and not uploads._STATE_DIR.exists()
