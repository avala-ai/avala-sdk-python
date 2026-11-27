"""#24265. Confirmed originals are reused within a named vendor intake."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx
from click.testing import CliRunner

from avala import Client
from avala.cli import main

BASE = "https://api.avala.ai/api/v1"
STORAGE = "https://s3.us-east-1.amazonaws.com/upload"
# Synthetic transport payload, deliberately not a claim of decodable media.
VIDEO_BYTES = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"original-video-payload"


@respx.mock
def test_contribution_recovers_receipt_and_closes_original_run(tmp_path: Path) -> None:
    """#24265: a fresh invocation reuses the original run for the same destination and source."""
    import hashlib
    from uuid import uuid4

    destination, upload, original_run = (str(uuid4()) for _ in range(3))
    name = "Kitchen"
    dataset_path = "/@vendor/datasets/kitchen"
    respx.post(f"{BASE}/vendor-intakes/named/", json={"name": name}).mock(
        return_value=httpx.Response(
            200, json={"destination_uid": destination, "dataset_path": dataset_path, "intake_name": name}
        )
    )
    prefix = f"{BASE}/vendor-intakes/{destination}"
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "sample.mp4").write_bytes(VIDEO_BYTES)
    (tmp_path / ".DS_Store").write_bytes(b"sidecar")
    identity = {"run_uid": original_run, "upload_uid": upload}
    source = respx.post(f"{prefix}/sources/").mock(
        side_effect=[
            httpx.Response(200, json={**identity, "complete": False, "method": "PUT", "url": STORAGE}),
            httpx.Response(200, json={**identity, "complete": True}),
        ]
    )
    put = respx.put(STORAGE).mock(return_value=httpx.Response(200))
    complete = respx.post(f"{prefix}/uploads/{upload}/complete/").mock(return_value=httpx.Response(200, json={}))
    close = respx.post(f"{prefix}/runs/{original_run}/close/").mock(return_value=httpx.Response(200, json={}))
    for root in (nested, tmp_path):
        with Client(api_key="test-key") as client:
            assert client.datasets.contribute_videos(source=str(root), name=name) == dataset_path
    receipts = [json.loads(call.request.content) for call in source.calls]
    assert receipts[0]["run_uid"] != receipts[1]["run_uid"]
    assert [r["source_path"] for r in receipts] == ["sample.mp4", "nested/sample.mp4"]
    for receipt in receipts:
        assert receipt["size"] == len(VIDEO_BYTES)
        assert receipt["sha256"] == hashlib.sha256(VIDEO_BYTES).hexdigest()
    assert put.call_count == 1
    assert complete.call_count == close.call_count == 2


@respx.mock
def test_named_contribution_requires_server_confirmation(tmp_path: Path) -> None:
    """A server that ignores --name must never receive files destined for the named intake."""
    from uuid import uuid4

    (tmp_path / "sample.mp4").write_bytes(VIDEO_BYTES)
    respx.post(f"{BASE}/vendor-intakes/named/", json={"name": "Kitchen"}).mock(
        return_value=httpx.Response(200, json={"destination_uid": str(uuid4()), "dataset_path": "/default"})
    )
    with Client(api_key="test-key") as client, pytest.raises(ValueError, match="did not confirm"):
        client.datasets.contribute_videos(source=str(tmp_path), name="Kitchen")


def test_contribute_cli_requires_dataset_name(tmp_path: Path) -> None:
    """A vendor cannot upload to an implicit dataset when the name is missing."""
    result = CliRunner().invoke(
        main, ["import", "contribute", "--source", str(tmp_path)], env={"AVALA_API_KEY": "test-key"}
    )
    assert result.exit_code == 2
    assert "Missing option '--name'" in result.output


@respx.mock
@pytest.mark.parametrize("already_received", [False, True])
@pytest.mark.parametrize("verified", [False, True])
def test_contribution_reports_verification_from_completion(
    tmp_path: Path, already_received: bool, verified: bool
) -> None:
    """#24265: POST complete reports digest verification independently of source reuse."""
    from uuid import uuid4

    destination, upload, run = (str(uuid4()) for _ in range(3))
    folder = tmp_path / "nested"
    folder.mkdir()
    (folder / "3.mov").write_bytes(VIDEO_BYTES)
    respx.post(f"{BASE}/vendor-intakes/named/").mock(
        return_value=httpx.Response(
            200, json={"destination_uid": destination, "dataset_path": "/digits-2", "intake_name": "Digits 2"}
        )
    )
    endpoint = f"{BASE}/vendor-intakes/{destination}"
    respx.post(f"{endpoint}/sources/").mock(
        return_value=httpx.Response(
            200,
            json={
                "run_uid": run,
                "upload_uid": upload,
                "complete": already_received,
                "sha256_verified": False,
                "method": "PUT",
                "url": STORAGE,
            },
        )
    )
    put = respx.put(STORAGE).mock(return_value=httpx.Response(200))
    respx.post(f"{endpoint}/uploads/{upload}/complete/").mock(
        return_value=httpx.Response(200, json={"sha256_verified": verified})
    )
    respx.post(f"{endpoint}/runs/{run}/close/").mock(return_value=httpx.Response(200, json={}))
    result = CliRunner().invoke(
        main, ["--api-key", "test-key", "import", "contribute", "--source", str(tmp_path), "--name", "Digits 2"]
    )
    assert result.exit_code == 0, result.output
    label = "Already received" if already_received else "Received"
    if not verified:
        label = "Received, verification pending"
    assert f"{label}: nested/3.mov" in result.output.splitlines()
    assert put.call_count == int(not already_received)


@pytest.mark.parametrize("resource_name", ["datasets", "slices"])
@respx.mock
def test_item_detail_preserves_optional_vendor_playback_bounds(resource_name: str) -> None:
    """DatasetItemSerializer omits ordinary bounds and returns null or a verified saved interval."""
    bounds = {
        "origin_ns": "0",
        "duration_ns": 4_125_000_000,
        "start_offset_ns": 100_000_000,
        "end_offset_ns": 2_000_000_000,
    }
    respx.get(f"{BASE}/{resource_name}/org/data/items/item/").mock(
        side_effect=[
            httpx.Response(200, json={"uid": "item", **fields})
            for fields in ({}, {"playback_bounds": None}, {"playback_bounds": bounds})
        ]
    )
    with Client(api_key="test-key") as client:
        get_item = getattr(client, resource_name).get_item
        ordinary, untrimmed, trimmed = [get_item("org", "data", "item") for _ in range(3)]
    assert ordinary.playback_bounds is None and "playback_bounds" not in ordinary.model_fields_set
    assert untrimmed.playback_bounds is None and "playback_bounds" in untrimmed.model_fields_set
    assert trimmed.playback_bounds.model_dump() == bounds


@pytest.mark.parametrize("replace", [True, False])
def test_changing_recording_never_registers_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replace: bool
) -> None:
    """source_for_file binds path/digest permanently. An unstable hash cannot create that binding."""
    import hashlib
    from unittest.mock import Mock
    from uuid import uuid4

    from avala._vendor_contribution import contribute_file

    source = tmp_path / "source.mp4"
    source.write_bytes(VIDEO_BYTES)
    digest = hashlib.sha256()
    changing = Mock(wraps=digest)

    def change(block: bytes) -> None:
        digest.update(block)
        if replace:
            replacement = tmp_path / "replacement"
            replacement.write_bytes(VIDEO_BYTES[::-1])
            replacement.replace(source)
        else:
            with source.open("ab") as recording:
                recording.write(b"more")

    changing.update.side_effect = change
    monkeypatch.setattr(hashlib, "sha256", lambda: changing)
    request = Mock()
    with pytest.raises(ValueError, match="while hashing"):
        contribute_file(str(source), source.name, uuid4(), str(uuid4()), request, httpx.Timeout(10))
    request.assert_not_called()


@respx.mock
def test_retry_refreshes_vendor_grants_without_rebinding_changed_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SDK upload contract retries transient control/data-plane failures using fresh grants and one source identity."""
    import hashlib
    from unittest.mock import Mock
    from uuid import uuid4

    import avala._vendor_contribution as contribution

    backoff = Mock()
    monkeypatch.setattr(contribution, "sleep_backoff", backoff)
    destination, upload, run = (str(uuid4()) for _ in range(3))
    endpoint = f"{BASE}/vendor-intakes/{destination}"
    file = tmp_path / "clip.mp4"
    file.write_bytes(VIDEO_BYTES)
    intake = {"intake_name": "Kitchen", "destination_uid": destination, "dataset_path": "/kitchen"}
    respx.post(f"{BASE}/vendor-intakes/named/").mock(
        side_effect=[httpx.Response(503), httpx.Response(200, json=intake)]
    )
    payloads = []
    grants = [STORAGE + suffix for suffix in ("1", "2", "3")]

    def register(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        file.write_bytes(VIDEO_BYTES + b"new bytes")
        if len(payloads) == 1:
            return httpx.Response(429)
        idx = len(payloads) - 2
        return httpx.Response(
            200,
            json={
                "run_uid": run,
                "upload_uid": upload,
                "complete": False,
                "method": "PUT",
                "url": grants[idx],
            },
        )

    respx.post(f"{endpoint}/sources/").mock(side_effect=register)
    puts = [respx.put(url).mock(return_value=httpx.Response(code)) for url, code in zip(grants, (503, 200, 412))]
    complete = respx.post(f"{endpoint}/uploads/{upload}/complete/").mock(
        side_effect=[httpx.ReadTimeout("transient"), httpx.Response(200, json={"sha256_verified": True})]
    )
    close = respx.post(f"{endpoint}/runs/{run}/close/").mock(
        side_effect=[httpx.Response(502), httpx.Response(200, json={})]
    )
    received = Mock()
    with Client(api_key="test-key") as client:
        assert client.datasets.contribute_videos(str(file), name="Kitchen", on_received=received) == "/kitchen"
    assert len(payloads) == 4 and all(payload == payloads[0] for payload in payloads)
    assert payloads[0]["sha256"] == hashlib.sha256(VIDEO_BYTES).hexdigest() and payloads[0]["size"] == len(VIDEO_BYTES)
    assert all(route.call_count == 1 and route.calls.last.request.content == VIDEO_BYTES for route in puts)
    assert complete.call_count == close.call_count == 2
    assert backoff.call_count == 5
    received.assert_called_once_with(file.name, False)


@respx.mock
def test_vendor_multipart_retry_resumes_signed_parts_from_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh server descriptor owns part state. A retry sends only its unconfirmed parts."""
    from unittest.mock import Mock
    from uuid import uuid4

    import avala._vendor_contribution as contribution

    monkeypatch.setattr(contribution, "sleep_backoff", Mock())
    destination, upload, run = uuid4(), uuid4(), uuid4()
    part_size = 5 * 1024 * 1024
    source = tmp_path / "sample"
    source.write_bytes(b"a" * part_size + b"tail")
    endpoint = f"{BASE}/vendor-intakes/{destination}"
    descriptor = {
        "run_uid": str(run),
        "upload_uid": str(upload),
        "complete": False,
        "method": "MULTIPART",
        "part_size": part_size,
        "part_count": 2,
        "uploaded_parts": [{"part_number": 1, "size": part_size}],
    }
    registrations = respx.post(f"{endpoint}/sources/").mock(return_value=httpx.Response(200, json=descriptor))
    signatures = respx.post(f"{endpoint}/uploads/{upload}/parts/", json={"part_numbers": [2]}).mock(
        side_effect=[httpx.Response(503), httpx.Response(200, json={"parts": [{"part_number": 2, "url": STORAGE}]})]
    )
    put = respx.put(STORAGE).mock(return_value=httpx.Response(200))
    respx.post(f"{endpoint}/uploads/{upload}/complete/").mock(
        return_value=httpx.Response(200, json={"sha256_verified": True})
    )
    with Client(api_key="test-key") as client:
        assert contribution.contribute_file(
            str(source), source.name, destination, str(run), client.datasets._transport.request, httpx.Timeout(10)
        ) == (str(run), False, True)
    assert registrations.call_count == signatures.call_count == 2
    assert put.call_count == 1 and put.calls.last.request.content == b"tail"


@pytest.mark.parametrize("status,attempts", [(403, 1), (409, 1), (503, 8)])
def test_vendor_retry_respects_permanent_errors_and_budget(
    status: int, attempts: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SDK upload retry policy preserves authorization/source conflicts and bounds transient retries."""
    from unittest.mock import Mock

    import avala._vendor_contribution as contribution

    operation = Mock(
        side_effect=httpx.HTTPStatusError(
            "storage", request=httpx.Request("PUT", STORAGE), response=httpx.Response(status)
        )
    )
    backoff = Mock()
    monkeypatch.setattr(contribution, "sleep_backoff", backoff)
    with pytest.raises(httpx.HTTPStatusError):
        contribution.retry_upload_operation(operation)
    assert operation.call_count == attempts and backoff.call_count == attempts - 1
