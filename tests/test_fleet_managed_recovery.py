"""Adversarial HTTP observations must preserve managed upload recovery identity."""

from __future__ import annotations

import json
import logging
import traceback
from dataclasses import replace
from uuid import uuid4

import httpx
import pytest

from avala import Avala
from avala.errors import UploadStateError
from avala.resources.fleet import _managed_consumer as consumer
from tests import test_fleet_managed_upload as upload_fixtures

BASE = upload_fixtures.BASE
transfer = upload_fixtures.transfer


@pytest.fixture(autouse=True)
def no_managed_retry_sleep(monkeypatch):
    monkeypatch.setattr(consumer.time, "sleep", lambda seconds: None)


def _action(request):
    return request.url.path.split("/managed-upload/", 1)[1]


def _count(transfer, action):
    return sum(name == action for name, _ in transfer.requests)


def _lose_initialization(transfer, *, state="active", number=1):
    def response(request):
        result = transfer.api(request)
        if _action(request) == "multipart/init/":
            generation = transfer.originals[0]["generation"]
            transfer.originals[0]["generation"] = (
                dict(generation, state=state, number=number) if state is not None else None
            )
            raise httpx.ReadTimeout("synthetic initialization acknowledgment lost")
        return result

    transfer.route.mock(side_effect=response)
    with pytest.raises(UploadStateError):
        transfer.upload()
    assert _count(transfer, "multipart/init/") == 1
    assert not transfer.put_route.called
    assert _count(transfer, "finalize/") == 0
    retained = json.loads(transfer.receipt().read_text())
    assert retained["originals"]["clip.mcap"]["initializing"] is True
    assert retained["originals"]["clip.mcap"]["generation_uid"] is None
    transfer.route.mock(side_effect=transfer.api)


def test_lost_successful_initialization_recovers_exact_active_generation_without_reinitializing(transfer):
    _lose_initialization(transfer)
    session = transfer.session
    generation = transfer.originals[0]["generation"]["generation_uid"]
    result = transfer.upload(max_workers=2, wait_timeout=0)
    assert result.finalization.state == "succeeded"
    assert result.session_uid == session
    assert _count(transfer, "multipart/init/") == 1
    assert _count(transfer, "admit/") == 1
    assert result.files[0].generation.generation_uid == generation
    assert len(transfer.put_bytes) == 1


@pytest.mark.parametrize("state", [None, "initializing", "abandoned"])
def test_uncertain_initialization_cannot_allocate_a_successor_on_any_resume(transfer, state):
    _lose_initialization(transfer, state=state)
    session = transfer.session
    for _ in range(2):
        with pytest.raises(UploadStateError):
            transfer.upload()
    assert transfer.session == session
    assert _count(transfer, "admit/") == 1
    assert _count(transfer, "multipart/init/") == 1
    assert _count(transfer, "multipart/parts/") == 0
    assert _count(transfer, "finalize/") == 0
    assert not transfer.put_route.called


def test_lost_initialization_cannot_adopt_a_later_generation(transfer):
    _lose_initialization(transfer, number=2)
    with pytest.raises(UploadStateError):
        transfer.upload()
    assert _count(transfer, "multipart/init/") == 1
    assert not transfer.put_route.called


@pytest.mark.parametrize("field", ["protocol", "session_uid", "file_uid", "generation_uid", "number"])
def test_malformed_initialization_response_retains_intent_and_cannot_send_bytes(transfer, field):
    def response(request):
        result = transfer.api(request)
        if _action(request) == "multipart/init/":
            data = result.json()
            data[field] = 2 if field == "number" else str(uuid4())
            return httpx.Response(200, json=data)
        return result

    transfer.route.mock(side_effect=response)
    with pytest.raises(UploadStateError):
        transfer.upload()
    retained = json.loads(transfer.receipt().read_text())
    assert retained["originals"]["clip.mcap"]["initializing"] is True
    assert not transfer.put_route.called
    assert _count(transfer, "finalize/") == 0
    assert _count(transfer, "multipart/init/") == 1


@pytest.mark.parametrize("stage", ["admit/", "multipart/init/", "multipart/parts/"])
@pytest.mark.parametrize("change", ["api_key", "base_url"])
def test_changed_authority_after_response_blocks_subsequent_work(transfer, stage, change):
    with Avala(api_key="synthetic-api-key", base_url=BASE) as client:
        manager = client.fleet.uploads

        def response(request):
            result = transfer.api(request)
            if _action(request) == stage:
                config = manager._transport._config
                value = "different-synthetic-key" if change == "api_key" else "https://elsewhere.invalid/api/v1"
                manager._transport._config = replace(config, **{change: value})
            return result

        transfer.route.mock(side_effect=response)
        with pytest.raises(UploadStateError):
            manager.upload_managed_recording(transfer.recording, transfer.source)
    assert not transfer.put_route.called
    assert _count(transfer, "finalize/") == 0
    assert _count(transfer, stage) == 1
    assert transfer.receipt().exists()


def test_changed_credentials_on_resume_fail_before_any_api_request(transfer):
    _lose_initialization(transfer)
    before = list(transfer.requests)
    receipt = transfer.receipt().read_bytes()
    with Avala(api_key="different-synthetic-key", base_url=BASE) as client:
        with pytest.raises(UploadStateError):
            client.fleet.uploads.upload_managed_recording(transfer.recording, transfer.source)
    assert transfer.requests == before
    assert transfer.receipt().read_bytes() == receipt
    assert not transfer.put_route.called


def test_changed_original_on_resume_fails_before_any_api_request(transfer):
    _lose_initialization(transfer)
    before = list(transfer.requests)
    receipt = transfer.receipt().read_bytes()
    source = transfer.source / "clip.mcap"
    original = source.read_bytes()
    source.write_bytes(b"x" * len(original))
    with pytest.raises(UploadStateError):
        transfer.upload()
    assert transfer.requests == before
    assert transfer.receipt().read_bytes() == receipt
    assert not transfer.put_route.called
    source.write_bytes(original)
    assert transfer.upload().finalization.state == "succeeded"


def test_original_mutation_during_put_cannot_finalize(transfer):
    def put(request):
        result = transfer.put(request)
        (transfer.source / "clip.mcap").write_bytes(b"different-capture")
        return result

    transfer.put_route.mock(side_effect=put)
    with pytest.raises(UploadStateError):
        transfer.upload()
    assert len(transfer.put_bytes) == 1
    assert _count(transfer, "finalize/") == 0
    assert transfer.receipt().exists()


@pytest.mark.parametrize("field", ["protocol", "session_uid", "recording_uid", "file_uid", "path", "content_sha256"])
def test_substituted_status_cannot_authorize_bytes(transfer, field):
    _lose_initialization(transfer)

    def response(request):
        result = transfer.api(request)
        if _action(request) == "status/":
            data = result.json()
            if field in {"protocol", "session_uid", "recording_uid"}:
                data[field] = str(uuid4())
            else:
                data["files"][0][field] = (
                    "b" * 64 if field == "content_sha256" else "other.mcap" if field == "path" else str(uuid4())
                )
            return httpx.Response(200, json=data)
        return result

    transfer.route.mock(side_effect=response)
    with pytest.raises(UploadStateError):
        transfer.upload()
    assert not transfer.put_route.called
    assert _count(transfer, "multipart/init/") == 1
    assert _count(transfer, "finalize/") == 0


def test_retained_active_generation_cannot_be_substituted_on_resume(transfer):
    def unavailable(request):
        result = transfer.api(request)
        if _action(request) == "multipart/parts/":
            return httpx.Response(503, json={"detail": "synthetic unavailable"})
        return result

    transfer.route.mock(side_effect=unavailable)
    with pytest.raises(UploadStateError):
        transfer.upload()
    retained = json.loads(transfer.receipt().read_text())
    generation = retained["originals"]["clip.mcap"]["generation_uid"]
    assert generation == transfer.originals[0]["generation"]["generation_uid"]
    transfer.originals[0]["generation"]["generation_uid"] = str(uuid4())
    transfer.route.mock(side_effect=transfer.api)
    with pytest.raises(UploadStateError):
        transfer.upload()
    assert not transfer.put_route.called
    assert _count(transfer, "multipart/init/") == 1
    assert _count(transfer, "finalize/") == 0


@pytest.mark.parametrize("stage", ["multipart/progress/", "multipart/parts/"])
@pytest.mark.parametrize("field", ["protocol", "session_uid", "file_uid", "generation_uid", "missing"])
def test_malformed_or_substituted_part_envelope_cannot_send_bytes(transfer, stage, field):
    def response(request):
        result = transfer.api(request)
        if _action(request) == stage:
            data = result.json()
            if field == "missing":
                del data["file_uid"]
            else:
                data[field] = str(uuid4())
            return httpx.Response(200, json=data)
        return result

    transfer.route.mock(side_effect=response)
    with pytest.raises(UploadStateError):
        transfer.upload()
    assert not transfer.put_route.called
    assert _count(transfer, "finalize/") == 0


def test_unsafe_part_headers_are_rejected_by_public_consumer_before_put(transfer):
    def response(request):
        result = transfer.api(request)
        if _action(request) == "multipart/parts/":
            data = result.json()
            data["parts"][0]["headers"]["Authorization"] = "synthetic-secret-header"
            return httpx.Response(200, json=data)
        return result

    transfer.route.mock(side_effect=response)
    with pytest.raises(UploadStateError):
        transfer.upload()
    assert not transfer.put_route.called
    assert _count(transfer, "finalize/") == 0


@pytest.mark.parametrize("outcome", ["success", "lost_put", "api_failure", "malformed_capability"])
def test_signed_capabilities_are_absent_from_errors_logs_and_checkpoints(transfer, caplog, outcome):
    caplog.set_level(logging.DEBUG)
    marker = "synthetic-private-response-material"

    def response(request):
        result = transfer.api(request)
        if _action(request) == "multipart/parts/":
            data = result.json()
            logging.getLogger("httpcore.http11").debug("private response %s", data)
            if outcome == "api_failure":
                return httpx.Response(503, json={"detail": marker, "url": data["parts"][0]["url"]})
            if outcome == "malformed_capability":
                data["parts"][0]["headers"]["Authorization"] = marker
                return httpx.Response(200, json=data)
        return result

    def put(request):
        logging.getLogger("httpcore.http11").debug("private URL %s", request.url)
        result = transfer.put(request)
        if outcome == "lost_put":
            raise httpx.ReadTimeout(f"{marker} {request.url}")
        return result

    transfer.route.mock(side_effect=response)
    transfer.put_route.mock(side_effect=put)
    rendered = ""
    if outcome == "success":
        assert transfer.upload().finalization.state == "succeeded"
    else:
        with pytest.raises(UploadStateError) as caught:
            transfer.upload()
        rendered = "".join(traceback.format_exception(caught.type, caught.value, caught.tb))
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None
        frame = caught.value.__traceback__
        while frame is not None:
            # CI also names its checkout directory "avala". Match the package
            # identity so the test's own synthetic secrets are not diagnostics.
            if frame.tb_frame.f_globals.get("__name__", "").startswith("avala."):
                rendered += repr(frame.tb_frame.f_locals)
            frame = frame.tb_next
    logging.getLogger("httpx").warning("unrelated request remains observable")
    assert "unrelated request remains observable" in caplog.text
    retained = transfer.receipt().read_text()
    for private in (marker, "X-Amz-", "private-provider-", "synthetic-api-key"):
        assert private not in rendered
        assert private not in caplog.text
        assert private not in retained
