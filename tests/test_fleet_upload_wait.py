"""Read-only waits bound hashing without weakening terminal source checks."""

import json
from types import SimpleNamespace

import pytest

from avala import Client
from avala.errors import UploadStateError
from tests.test_fleet_upload_safety import _cli
from tests.test_fleet_upload_safety import transfer as transfer


@pytest.mark.parametrize("poll_count", [1, 8])
def test_cli_wait_hashes_source_twice_independent_of_poll_count(transfer, monkeypatch, poll_count):
    result = transfer.upload()
    observations = []
    scans = []
    with Client(api_key="synthetic credential") as client:
        manager = client.fleet.uploads
        collect = manager.collect_files

        def collect_files(source):
            scans.append(source)
            return collect(source)

        def recording(uid):
            observations.append(uid)
            return SimpleNamespace(
                uid=uid, status="ready" if len(observations) > poll_count else "processing", size_bytes=7
            )

        def upload(*args, **kwargs):
            # Exclude the CLI's existing pre-upload inventory preview.
            scans.clear()
            return result

        monkeypatch.setattr(manager, "collect_files", collect_files)
        monkeypatch.setattr(manager, "upload_recording", upload)
        monkeypatch.setattr(client.fleet.recordings, "get", recording)
        command = _cli(transfer, client, "--wait")
    assert command.exit_code == 0, command.output
    assert len(observations) == poll_count + 1
    assert len(scans) == 2
    assert (
        transfer.init_route.call_count == transfer.confirm_route.call_count == transfer.finalize_route.call_count == 1
    )


def test_ready_recording_does_not_repeat_hashes_while_upload_still_completes(transfer, monkeypatch):
    transfer.retained(phase="finalizing")
    transfer.confirmed = set(transfer.files)
    transfer.phase = "completing"
    with Client(api_key="synthetic credential") as client:
        manager = client.fleet.uploads
        collect = manager.collect_files
        scans = []

        def collect_files(source):
            scans.append(source)
            return collect(source)

        monkeypatch.setattr(manager, "collect_files", collect_files)
        observer = manager._upload_observer("recording-1", "session-1", transfer.source)
        for _ in range(8):
            assert observer.poll().status == "completing"
        assert len(scans) == 1
        transfer.phase = "completed"
        assert observer.poll().status == "completed"
        assert len(scans) == 2
    assert json.loads(transfer.receipt().read_text())["phase"] == "completed"


@pytest.mark.parametrize("mutation", ["changed", "added", "missing"])
def test_wait_refuses_source_mutation_before_ready(transfer, monkeypatch, mutation):
    result = transfer.upload()
    observations = []

    def recording(uid):
        observations.append(uid)
        if len(observations) == 3:
            if mutation == "changed":
                (transfer.source / "capture.mcap").write_bytes(b"CAPTURE")
            elif mutation == "added":
                (transfer.source / "extra.mcap").write_bytes(b"extra")
            else:
                (transfer.source / "capture.mcap").unlink()
        return SimpleNamespace(uid=uid, status="ready" if len(observations) == 3 else "processing", size_bytes=7)

    with Client(api_key="synthetic credential") as client:
        monkeypatch.setattr(client.fleet.uploads, "upload_recording", lambda *args, **kwargs: result)
        monkeypatch.setattr(client.fleet.recordings, "get", recording)
        command = _cli(transfer, client, "--wait")
    assert command.exit_code != 0
    assert "is ready" not in command.output
    assert transfer.receipt().exists()
    assert (
        transfer.init_route.call_count == transfer.confirm_route.call_count == transfer.finalize_route.call_count == 1
    )


def test_observer_never_settles_receipt_without_terminal_source_check(transfer):
    transfer.retained(phase="finalizing")
    transfer.confirmed = set(transfer.files)
    transfer.phase = "completed"
    with Client(api_key="synthetic credential") as client:
        observer = client.fleet.uploads._upload_observer("recording-1", "session-1", transfer.source)
        assert observer.poll(verify_source=False).status == "completed"
        assert json.loads(transfer.receipt().read_text())["phase"] == "finalizing"
        (transfer.source / "capture.mcap").write_bytes(b"CAPTURE")
        with pytest.raises(UploadStateError):
            observer.poll()
    assert json.loads(transfer.receipt().read_text())["phase"] == "finalizing"


@pytest.mark.parametrize("mutation", ["missing", "corrupt"])
def test_observer_rereads_retained_receipt_on_every_poll(transfer, mutation):
    transfer.retained(phase="finalizing")
    transfer.confirmed = set(transfer.files)
    transfer.phase = "completing"
    with Client(api_key="synthetic credential") as client:
        observer = client.fleet.uploads._upload_observer("recording-1", "session-1", transfer.source)
        assert observer.poll(verify_source=False).status == "completing"
        receipt = transfer.receipt()
        if mutation == "missing":
            receipt.unlink()
        else:
            receipt.write_text("corrupt retained evidence")
        with pytest.raises(UploadStateError):
            observer.poll(verify_source=False)
    assert transfer.status_route.call_count == 1
    assert not transfer.init_route.called and not transfer.confirm_route.called and not transfer.finalize_route.called


def test_terminal_hash_cannot_publish_after_authority_replacement(transfer, monkeypatch):
    transfer.retained(phase="finalizing")
    transfer.confirmed = set(transfer.files)
    transfer.phase = "completed"
    with Client(api_key="synthetic credential") as client, Client(api_key="replacement credential") as replacement:
        manager = client.fleet.uploads
        observer = manager._upload_observer("recording-1", "session-1", transfer.source)
        collect = manager.collect_files

        def collect_files(source):
            inventory = collect(source)
            manager._transport = replacement.fleet.uploads._transport
            return inventory

        monkeypatch.setattr(manager, "collect_files", collect_files)
        with pytest.raises(UploadStateError):
            observer.poll()
    assert json.loads(transfer.receipt().read_text())["phase"] == "finalizing"


@pytest.mark.parametrize("when", ["before", "during"])
@pytest.mark.parametrize("field", ["credentials", "api"])
def test_observer_refuses_changed_authority_before_or_during_status(transfer, when, field):
    transfer.retained(phase="finalizing")
    transfer.confirmed = set(transfer.files)
    transfer.phase = "completed"
    replacement = Client(
        api_key="replacement credential" if field == "credentials" else "synthetic credential",
        base_url="https://example.invalid/api/v1" if field == "api" else "https://api.avala.ai/api/v1",
    )
    with Client(api_key="synthetic credential") as client, replacement:
        manager = client.fleet.uploads
        observer = manager._upload_observer("recording-1", "session-1", transfer.source)
        if when == "before":
            manager._transport = replacement.fleet.uploads._transport
        else:

            def status(request):
                manager._transport = replacement.fleet.uploads._transport
                return transfer.status(request)

            transfer.status_route.mock(side_effect=status)
        with pytest.raises(UploadStateError):
            observer.poll()
    assert transfer.status_route.call_count == (0 if when == "before" else 1)
    assert json.loads(transfer.receipt().read_text())["phase"] == "finalizing"
