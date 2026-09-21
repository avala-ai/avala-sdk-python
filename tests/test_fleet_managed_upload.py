"""Synthetic HTTP checks for negotiated uploads and immutable recovery evidence."""

from datetime import datetime, timezone
from contextlib import contextmanager
from urllib.parse import quote, urlencode
from uuid import uuid4
import json

import httpx
import pytest
import respx

from avala import Avala
from avala.errors import UploadStateError
from avala.resources.fleet import uploads

BASE = "https://api.avala.ai/api/v1"
PROTOCOL = "fleet-managed-mcap-v1"
PART_SIZE = 64 * 1024**2
SYNTHETIC_API_KEY = "synthetic-api-key"


@pytest.fixture
def transfer(tmp_path, monkeypatch):
    class Transfer:
        def __init__(self):
            self.source = tmp_path / "source"
            self.source.mkdir()
            (self.source / "clip.mcap").write_bytes(b"synthetic-mcap")
            self.recording = str(uuid4())
            self.organization = str(uuid4())
            self.session = None
            self.originals = []
            self.parts = {}
            self.finalization = None
            self.complete_on_poll = True
            self.put_bytes = []
            self.requests = []
            self.endpoint = f"{BASE}/fleet/recordings/{self.recording}/managed-upload"

        def status(self):
            complete = self.finalization is not None and self.complete_on_poll
            if complete and self.finalization["state"] != "succeeded":
                self.finalization.update(
                    state="succeeded",
                    stage="publish",
                    code="published",
                    failures=0,
                    publication_uid=str(uuid4()),
                    dataset_uid=str(uuid4()),
                )
            files = []
            for file in self.originals:
                row = dict(file)
                row["verified"] = complete
                if complete:
                    row["generation"] = dict(row["generation"], state="completed")
                files.append(row)
            return dict(
                protocol=PROTOCOL,
                session_uid=self.session,
                recording_uid=self.recording,
                status="completed" if complete else "initiated",
                files=files,
                total_files=len(files),
                total_bytes=sum(f["size_bytes"] for f in files),
                confirmed_files=len(files) if complete else 0,
                confirmed_bytes=sum(f["size_bytes"] for f in files) if complete else 0,
                finalization=self.finalization,
            )

        def api(self, request):
            action = str(request.url).split("/managed-upload/", 1)[1].split("?", 1)[0]
            data = dict(request.url.params) if request.method == "GET" else json.loads(request.content)
            self.requests.append((action, data))
            assert data["protocol"] == PROTOCOL
            if action == "admit/":
                if self.session is None:
                    self.session = data["session_uid"]
                    for file in data["files"]:
                        self.originals.append(
                            dict(
                                file,
                                file_uid=str(uuid4()),
                                content_type="application/octet-stream",
                                content_encoding="",
                                part_size=PART_SIZE,
                                part_count=(file["size_bytes"] + PART_SIZE - 1) // PART_SIZE,
                                verified=False,
                                generation=None,
                            )
                        )
                assert self.session == data["session_uid"]
                return httpx.Response(201, json=self.status())
            assert data["session_uid"] == self.session
            if action == "status/":
                return httpx.Response(200, json=self.status())
            if action == "finalize/":
                expected = [
                    {"file_uid": row["file_uid"], "generation_uid": row["generation"]["generation_uid"]}
                    for row in self.originals
                ]
                assert sorted(data["files"], key=lambda f: f["file_uid"]) == sorted(
                    expected, key=lambda f: f["file_uid"]
                )
                assert len(self.parts) == sum(row["part_count"] for row in self.originals)
                self.finalization = self.finalization or dict(
                    uid=str(uuid4()),
                    state="pending",
                    stage="complete",
                    code="pending",
                    attempts=0,
                    failures=0,
                    available_at=datetime.now(timezone.utc).isoformat(),
                    publication_uid=None,
                    dataset_uid=None,
                )
                return httpx.Response(
                    202, json=dict(protocol=PROTOCOL, session_uid=self.session, finalization=self.finalization)
                )
            row = next(row for row in self.originals if row["file_uid"] == data["file_uid"])
            if action == "multipart/init/":
                row["generation"] = row["generation"] or dict(generation_uid=str(uuid4()), number=1, state="active")
                return httpx.Response(
                    200,
                    json=dict(
                        protocol=PROTOCOL,
                        session_uid=self.session,
                        file_uid=row["file_uid"],
                        **row["generation"],
                        part_size=PART_SIZE,
                        part_count=row["part_count"],
                        initialization_deadline=datetime.now(timezone.utc).isoformat(),
                    ),
                )
            assert data["generation_uid"] == row["generation"]["generation_uid"]
            envelope = dict(
                protocol=PROTOCOL,
                session_uid=self.session,
                file_uid=row["file_uid"],
                generation_uid=data["generation_uid"],
            )
            if action == "multipart/progress/":
                present = [
                    dict(part_number=n, size=len(payload))
                    for (uid, n), payload in sorted(self.parts.items())
                    if uid == row["file_uid"] and n > int(data["after"])
                ]
                return httpx.Response(
                    200,
                    json=dict(
                        envelope,
                        parts=present[:100],
                        next_cursor=present[99]["part_number"] if len(present) > 100 else None,
                    ),
                )
            assert action == "multipart/parts/"
            signed = []
            for number in data["numbers"]:
                size = min(PART_SIZE, row["size_bytes"] - (number - 1) * PART_SIZE)
                date = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                key = f"__o__=/{self.organization}/__cloud__/{self.session}/{row['path']}"
                query = urlencode(
                    dict(
                        uploadId="private-provider-" + row["file_uid"],
                        partNumber=str(number),
                        **{
                            "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
                            "X-Amz-Credential": f"{'b' * 32}/{date[:8]}/auto/s3/aws4_request",
                            "X-Amz-Date": date,
                            "X-Amz-Expires": "300",
                            "X-Amz-SignedHeaders": "content-length;host",
                            "X-Amz-Signature": "a" * 64,
                        },
                    )
                )
                url = f"https://{'0' * 32}.r2.cloudflarestorage.com/avala-user-managed/{quote(key, safe='/')}?{query}"
                signed.append(
                    dict(
                        grant_uid=str(uuid4()),
                        part_number=number,
                        offset=(number - 1) * PART_SIZE,
                        size_bytes=size,
                        url=url,
                        headers={"Content-Length": str(size)},
                        expires_at=datetime.fromtimestamp(
                            datetime.strptime(date, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp() + 300,
                            timezone.utc,
                        ).isoformat(),
                    )
                )
            return httpx.Response(200, json=dict(envelope, parts=signed))

        def put(self, request):
            uid = request.url.params["uploadId"].removeprefix("private-provider-")
            number = int(request.url.params["partNumber"])
            assert "X-Avala-Api-Key" not in request.headers
            assert len(request.content) == int(request.headers["Content-Length"])
            self.put_bytes.append(len(request.content))
            self.parts[(uid, number)] = request.content
            return httpx.Response(200, headers={"ETag": '"synthetic-part"'})

        def upload(self, **kwargs):
            with Avala(api_key=SYNTHETIC_API_KEY, base_url=BASE) as client:
                return client.fleet.uploads.upload_managed_recording(self.recording, self.source, **kwargs)

        def receipt(self):
            return next((uploads._STATE_DIR / "fleet-v1").glob("*.json"))

    item = Transfer()
    with respx.mock(assert_all_called=False) as router:
        item.route = router.route(url__startswith=item.endpoint).mock(side_effect=item.api)
        item.put_route = router.put(url__startswith=f"https://{'0' * 32}.r2.cloudflarestorage.com/").mock(
            side_effect=item.put
        )
        yield item


def test_new_upload_publishes_and_retains_exact_identity(transfer):
    result = transfer.upload()
    assert result.finalization.state == "succeeded"
    assert result.finalization.dataset_uid and result.finalization.publication_uid
    assert result.confirmed_files == 1
    receipt = transfer.receipt().read_text()
    assert result.session_uid in receipt and result.finalization.uid in receipt
    assert "synthetic-api-key" not in receipt and "private-provider-" not in receipt and "X-Amz" not in receipt
    before = list(transfer.put_bytes)
    assert transfer.upload() == result
    assert transfer.put_bytes == before


@pytest.mark.parametrize("action", ["admit/", "finalize/"])
def test_lost_idempotent_ack_reuses_precommitted_identity(transfer, monkeypatch, action):
    from avala.resources.fleet import _managed_consumer as consumer

    monkeypatch.setattr(consumer.time, "sleep", lambda *_: None)
    lost = False

    def response(request):
        nonlocal lost
        payload = json.loads(request.content) if request.method == "POST" else {}
        if str(request.url).endswith(action):
            persisted = json.loads(transfer.receipt().read_text())
            assert persisted["session_uid"] == payload["session_uid"]
            if action == "finalize/":
                assert persisted["finalizing"]
        result = transfer.api(request)
        if str(request.url).endswith(action) and not lost:
            lost = True
            raise httpx.ReadTimeout("synthetic lost acknowledgement")
        return result

    transfer.route.mock(side_effect=response)
    result = transfer.upload()
    attempts = [body for name, body in transfer.requests if name == action]
    assert len(attempts) == 2 and attempts[0] == attempts[1]
    assert result.finalization.state == "succeeded" and len(transfer.put_bytes) == 1


def test_pending_publication_is_returned_without_readiness_claim(transfer):
    transfer.complete_on_poll = False
    result = transfer.upload(wait_timeout=0)
    assert result.finalization.state == "pending"
    assert result.finalization.dataset_uid is result.finalization.publication_uid is None
    assert result.status == "initiated" and result.confirmed_files == 0
    before = list(transfer.put_bytes)
    assert transfer.upload(wait_timeout=0) == result
    assert transfer.put_bytes == before
    assert sum(action == "finalize/" for action, _ in transfer.requests) == 1


@pytest.mark.parametrize(
    "key",
    [
        "version",
        "protocol",
        "binding",
        "evidence",
        "session_uid",
        "originals",
        "finalizing",
        "finalization_uid",
        "publication_uid",
        "dataset_uid",
    ],
)
def test_missing_receipt_fields_never_infer_upload_intent(transfer, key):
    transfer.upload()
    path = transfer.receipt()
    data = json.loads(path.read_text())
    del data[key]
    path.write_text(json.dumps(data))
    calls = len(transfer.requests)
    with pytest.raises(UploadStateError):
        transfer.upload()
    assert len(transfer.requests) == calls and json.loads(path.read_text()) == data


@pytest.mark.parametrize("key", ["file_uid", "initializing", "generation_uid", "number", "transport_pin", "verified"])
def test_missing_original_receipt_fields_cannot_change_recovery(transfer, key):
    transfer.upload()
    path = transfer.receipt()
    data = json.loads(path.read_text())
    del data["originals"]["clip.mcap"][key]
    path.write_text(json.dumps(data))
    calls = len(transfer.requests)
    with pytest.raises(UploadStateError):
        transfer.upload()
    assert len(transfer.requests) == calls


@pytest.mark.parametrize("version", [2, 3.0, True, None, "3"])
def test_legacy_or_inexact_receipt_version_is_not_adopted(transfer, version):
    transfer.upload()
    path = transfer.receipt()
    data = json.loads(path.read_text())
    data["version"] = version
    path.write_text(json.dumps(data))
    calls = len(transfer.requests)
    with pytest.raises(UploadStateError):
        transfer.upload()
    assert len(transfer.requests) == calls and json.loads(path.read_text()) == data


@pytest.mark.parametrize(
    "field,value",
    [("part_size", float(PART_SIZE)), ("part_size", True), ("part_count", 1.0), ("size_bytes", True), ("verified", 0)],
)
def test_raw_geometry_must_have_exact_scalar_types(transfer, field, value):
    from pydantic import ValidationError
    from avala.types.fleet_managed_upload import ManagedFleetUploadStatus

    transfer.upload()
    raw = transfer.status()
    raw["files"][0][field] = value
    with pytest.raises(ValidationError):
        ManagedFleetUploadStatus.model_validate(raw)


@pytest.mark.parametrize(
    "options",
    [
        {"max_workers": 0},
        {"max_workers": 5},
        {"max_workers": True},
        {"wait_timeout": -1},
        {"wait_timeout": float("nan")},
        {"wait_timeout": float("inf")},
        {"wait_timeout": True},
    ],
)
def test_invalid_local_options_make_no_remote_requests(transfer, options):
    with pytest.raises(UploadStateError):
        transfer.upload(**options)
    assert transfer.requests == [] and transfer.put_bytes == []


def test_missing_parts_resume_fixed_geometry_with_bounded_reads(transfer, monkeypatch):
    from avala.resources.fleet import _managed_consumer as consumer

    original = transfer.source / "clip.mcap"
    with original.open("wb") as handle:
        handle.write(b"x" * PART_SIZE)
        handle.write(b"tail")
    original_open = consumer._open_regular_file
    reads = []

    @contextmanager
    def bounded(path):
        with original_open(path) as handle:

            class Checked:
                def read(self, size):
                    assert 0 < size <= PART_SIZE
                    reads.append(size)
                    return handle.read(size)

            yield Checked()

    monkeypatch.setattr(consumer, "_open_regular_file", bounded)
    failed = False

    def put(request):
        nonlocal failed
        if request.url.params["partNumber"] == "2" and not failed:
            failed = True
            return httpx.Response(503)
        return transfer.put(request)

    transfer.put_route.mock(side_effect=put)
    with pytest.raises(UploadStateError):
        transfer.upload(max_workers=1)
    assert transfer.put_bytes == [PART_SIZE]
    result = transfer.upload(max_workers=1)
    assert result.finalization.state == "succeeded"
    assert transfer.put_bytes == [PART_SIZE, 4]
    assert max(reads) == PART_SIZE and reads.count(PART_SIZE) == 2
    assert sum(action == "multipart/init/" for action, _ in transfer.requests) == 1


def test_lost_part_ack_is_observed_before_any_repeated_put(transfer):
    def accepted(request):
        transfer.put(request)
        raise httpx.ReadTimeout("synthetic accepted part response lost")

    transfer.put_route.mock(side_effect=accepted)
    with pytest.raises(UploadStateError):
        transfer.upload()
    transfer.put_route.mock(side_effect=transfer.put)
    assert transfer.upload().finalization.state == "succeeded"
    assert len(transfer.put_bytes) == 1


@pytest.mark.parametrize("state", ["retry_wait", "failed"])
def test_finalizing_resume_repairs_exact_missing_part_and_retries_recoverable_failure(transfer, state):
    transfer.complete_on_poll = False
    initial = transfer.upload()
    original = transfer.originals[0]
    original["generation"]["state"] = "completing"
    transfer.parts.clear()
    transfer.finalization.update(
        state=state,
        failures=5 if state == "failed" else 1,
        code="attempts_exhausted" if state == "failed" else "source_unavailable",
    )
    receipt_before = json.loads(transfer.receipt().read_text())

    def put(request):
        response = transfer.put(request)
        if state == "retry_wait":
            transfer.complete_on_poll = True
        return response

    def api(request):
        if str(request.url).endswith("/finalize/"):
            assert state == "failed"
            assert len(transfer.parts) == 1
            transfer.finalization.update(state="pending", code="pending", failures=0)
            transfer.complete_on_poll = True
        return transfer.api(request)

    transfer.put_route.mock(side_effect=put)
    transfer.route.mock(side_effect=api)
    result = transfer.upload()
    assert result.finalization.state == "succeeded"
    assert result.finalization.uid == initial.finalization.uid
    assert len(transfer.put_bytes) == 2
    assert sum(action == "multipart/init/" for action, _ in transfer.requests) == 1
    assert sum(action == "finalize/" for action, _ in transfer.requests) == (2 if state == "failed" else 1)
    receipt_after = json.loads(transfer.receipt().read_text())
    for key in ("file_uid", "generation_uid", "number", "transport_pin"):
        assert receipt_before["originals"]["clip.mcap"][key] == receipt_after["originals"]["clip.mcap"][key]


@pytest.mark.parametrize("code", ["content_mismatch", "invalid_mcap"])
def test_terminal_source_failure_never_repairs_or_restarts(transfer, code):
    transfer.complete_on_poll = False
    transfer.upload()
    transfer.parts.clear()
    transfer.finalization.update(state="failed", failures=1, code=code)
    before = len(transfer.requests)
    result = transfer.upload()
    assert result.finalization.code == code and result.finalization.state == "failed"
    assert all(action == "status/" for action, _ in transfer.requests[before:])
    assert len(transfer.put_bytes) == 1


def test_multiple_originals_keep_independent_file_generation_maps(transfer):
    (transfer.source / "another.mcap").write_bytes(b"another-original")
    result = transfer.upload()
    assert result.total_files == result.confirmed_files == 2
    assert result.finalization.state == "succeeded"
    assert len({file.generation.generation_uid for file in result.files}) == 2
    assert len(transfer.put_bytes) == 2


def test_managed_and_legacy_receipts_share_the_discovery_lock_without_adoption(transfer):
    from avala._fleet_uploads import fleet_checkpoint, upload_binding

    with Avala(api_key="synthetic-api-key", base_url=BASE) as client:
        inventory = client.fleet.uploads.collect_files(transfer.source)
        binding = upload_binding(
            inventory,
            base_url=BASE,
            api_key="synthetic-api-key",
            recording_uid=transfer.recording,
            storage_config_uid=None,
        )
        with fleet_checkpoint(uploads._STATE_DIR, binding) as checkpoint:
            checkpoint.write("initializing")
        retained = transfer.receipt().read_bytes()
        with pytest.raises(UploadStateError):
            transfer.upload()
        assert transfer.receipt().read_bytes() == retained
        assert transfer.requests == []


def test_legacy_uploader_cannot_ignore_managed_receipt(transfer):
    transfer.upload()
    calls, retained = len(transfer.requests), transfer.receipt().read_bytes()
    with Avala(api_key="synthetic-api-key", base_url=BASE) as client:
        with pytest.raises(UploadStateError):
            client.fleet.uploads.upload_recording(transfer.recording, transfer.source)
    assert len(transfer.requests) == calls and transfer.receipt().read_bytes() == retained


def test_resumption_survives_new_enrollment_closing(transfer):
    def lost(request):
        transfer.put(request)
        raise httpx.ReadTimeout("lost successful PUT")

    transfer.put_route.mock(side_effect=lost)
    with pytest.raises(UploadStateError):
        transfer.upload()

    def no_admission(request):
        assert not str(request.url).endswith("/admit/")
        return transfer.api(request)

    transfer.route.mock(side_effect=no_admission)
    assert transfer.upload().finalization.state == "succeeded"
    assert len(transfer.put_bytes) == 1


def test_positive_wait_stops_at_its_poll_budget(transfer, monkeypatch):
    from avala.resources.fleet import _managed_consumer as consumer
    from types import SimpleNamespace

    transfer.complete_on_poll = False
    clock = [10.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(consumer, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep))
    result = transfer.upload(wait_timeout=3)
    assert result.finalization.state == "pending"
    assert sleeps == [2, 1] and clock[0] == 13
