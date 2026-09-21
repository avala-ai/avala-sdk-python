"""Fleet upload recovery must not replace ambiguous remote work."""

import json
import inspect
import logging
import re
import traceback
from types import SimpleNamespace
from urllib.parse import quote

import httpx
import pytest
import respx
from click.testing import CliRunner

from avala import Client
from avala.errors import UploadStateError
from avala.resources.fleet import uploads
from avala.resources.fleet import _upload_guard as guard
from avala._fleet_uploads import collect_source, fleet_checkpoint, upload_binding
from avala.cli.fleet import fleet

BASE = "https://api.avala.ai/api/v1/fleet/recordings/recording-1/upload"


@respx.mock
def test_ambiguous_initialization_cannot_initialize_again(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "capture.mcap").write_bytes(b"capture")
    state = tmp_path / "state"
    monkeypatch.setattr(uploads, "_STATE_DIR", state)
    respx.get(f"{BASE}/status/").respond(404, json={"detail": "No upload session found for this recording."})
    initialize = respx.post(f"{BASE}/init/").mock(side_effect=httpx.ReadTimeout("synthetic response lost"))
    with Client(api_key="synthetic credential") as client:
        for _ in range(2):
            with pytest.raises(UploadStateError):
                client.fleet.uploads.upload_recording("recording-1", source)
    assert initialize.call_count == 1
    receipt = next((state / "fleet-v1").glob("*.json"))
    assert json.loads(receipt.read_text())["phase"] == "initializing"


class Transfer:
    """Synthetic server state, with real request bodies and non-idempotent confirms."""

    def __init__(self, source, state, router):
        self.source, self.state, self.router = source, state, router
        self.files = {"capture.mcap": b"capture"}
        self.source.mkdir()
        (source / "capture.mcap").write_bytes(b"capture")
        self.uid = None
        self.prefix = "fleet/org/device/recording-1/"
        self.phase = "initiated"
        self.confirmed = set()
        self.sent = []
        self.headers = {}
        self.init_route = router.post(f"{BASE}/init/").mock(side_effect=self.initialize)
        self.status_route = router.get(f"{BASE}/status/").mock(side_effect=self.status)
        self.grants_route = router.post(f"{BASE}/urls/").mock(side_effect=self.grants)
        self.put_route = router.put(re.compile(r"https://.*amazonaws.com/.*")).mock(side_effect=self.put)
        self.confirm_route = router.post(f"{BASE}/confirm/").mock(side_effect=self.confirm)
        self.finalize_route = router.post(f"{BASE}/finalize/").mock(side_effect=self.finalize)

    def counts(self):
        return dict(
            total_files=len(self.files),
            total_bytes=sum(map(len, self.files.values())),
            confirmed_files=len(self.confirmed),
            confirmed_bytes=sum(len(self.files[path]) for path in self.confirmed),
        )

    def body(self):
        return dict(
            session_uid=self.uid,
            status=self.phase,
            pending_paths=sorted(set(self.files) - self.confirmed),
            **self.counts(),
        )

    def receipt(self):
        return next((self.state / "fleet-v1").glob("*.json"))

    def initialize(self, request):
        assert json.loads(self.receipt().read_text())["phase"] == "initializing"
        files = json.loads(request.content)["files"]
        assert files == [{"path": path, "size_bytes": len(data)} for path, data in sorted(self.files.items())]
        self.uid = "session-1"
        return httpx.Response(
            201,
            json=dict(
                uid=self.uid,
                status="initiated",
                s3_prefix=self.prefix,
                total_files=len(self.files),
                total_bytes=sum(map(len, self.files.values())),
            ),
        )

    def status(self, request):
        if self.uid is None:
            return httpx.Response(404, json={"detail": "No upload session found for this recording."})
        return httpx.Response(200, json=self.body())

    def grants(self, request):
        payload = json.loads(request.content)
        assert payload["session_uid"] == self.uid
        return httpx.Response(
            200,
            json={
                "urls": [
                    dict(
                        path=path,
                        s3_key=self.prefix + path,
                        headers=self.headers,
                        put_url="https://bucket.s3.us-west-2.amazonaws.com/"
                        + quote(self.prefix + path, safe="/")
                        + "?signature=synthetic-private",
                    )
                    for path in payload["file_paths"]
                ]
            },
        )

    def put(self, request):
        self.sent.append((request.url, request.read(), dict(request.headers)))
        return httpx.Response(200, headers={"ETag": '"synthetic-etag"'})

    def confirm(self, request):
        payload = json.loads(request.content)
        assert payload["session_uid"] == self.uid
        for file in payload["files"]:
            assert file["path"] not in self.confirmed, "confirm must not be blindly retried"
            assert file["size_bytes"] == len(self.files[file["path"]])
            self.confirmed.add(file["path"])
        self.phase = "uploading"
        return httpx.Response(200, json=dict(session_uid=self.uid, **self.counts()))

    def finalize(self, request):
        assert self.confirmed == set(self.files)
        assert json.loads(self.receipt().read_text())["phase"] == "finalizing"
        self.phase = "completed"
        return httpx.Response(202, json={"detail": "Finalization started.", "session_uid": self.uid})

    def retained(self, *, phase="active"):
        self.uid = "session-1"
        self.phase = "uploading"
        binding = upload_binding(
            collect_source(self.source, state_dir=self.state),
            base_url="https://api.avala.ai/api/v1",
            api_key="synthetic credential",
            recording_uid="recording-1",
            storage_config_uid=None,
        )
        with fleet_checkpoint(self.state, binding) as checkpoint:
            checkpoint.write("initializing")
            checkpoint.write("active", self.uid, s3_prefix=self.prefix)
            if phase in {"finalizing", "completed"}:
                checkpoint.write("finalizing", self.uid)
            if phase == "completed":
                checkpoint.write("completed", self.uid)

    def upload(self, **kwargs):
        with Client(api_key="synthetic credential") as client:
            return client.fleet.uploads.upload_recording("recording-1", self.source, **kwargs)


@pytest.fixture
def transfer(tmp_path, monkeypatch):
    monkeypatch.setattr(uploads, "_STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(guard.time, "sleep", lambda _: None)
    with respx.mock(assert_all_called=False) as router:
        yield Transfer(tmp_path / "source", tmp_path / "state", router)


def test_happy_upload_and_completed_retry_keep_pinned_receipt(transfer):
    events = []
    result = transfer.upload(on_progress=events.append)
    assert result.status == "completed"
    assert transfer.sent[0][1] == b"capture"
    assert transfer.sent[0][2]["content-type"] == "application/octet-stream"
    assert "x-avala-api-key" not in transfer.sent[0][2]
    assert events[-1].uploaded_bytes == 7
    assert json.loads(transfer.receipt().read_text())["s3_prefix"] == transfer.prefix
    assert transfer.upload().status == "completed"
    assert (
        transfer.init_route.call_count == transfer.confirm_route.call_count == transfer.finalize_route.call_count == 1
    )


@pytest.mark.parametrize(
    "status,body",
    [
        (404, {"detail": "Recording not found."}),
        (404, {"detail": "No upload session found for this recording.", "extra": True}),
        (401, {}),
        (403, {}),
        (429, {}),
        (500, {}),
        (200, {}),
    ],
)
def test_only_explicit_session_absence_can_initialize(transfer, status, body):
    transfer.status_route.respond(status, json=body)
    with pytest.raises(UploadStateError):
        transfer.upload()
    assert not transfer.init_route.called


@pytest.mark.parametrize("phase", ["initiated", "uploading", "completing", "completed", "abandoned"])
def test_remote_session_without_local_evidence_never_restarts(transfer, phase):
    transfer.uid = "session-remote"
    transfer.phase = phase
    with pytest.raises(UploadStateError):
        transfer.upload()
    assert not transfer.init_route.called


@pytest.mark.parametrize("field", ["uid", "total_files", "total_bytes", "s3_prefix", "status"])
def test_missing_init_field_retains_ambiguous_intent(transfer, field):
    def response(request):
        payload = json.loads(transfer.initialize(request).content)
        del payload[field]
        return httpx.Response(201, json=payload)

    transfer.init_route.mock(side_effect=response)
    for _ in range(2):
        with pytest.raises(UploadStateError):
            transfer.upload()
    assert transfer.init_route.call_count == 1
    assert json.loads(transfer.receipt().read_text())["phase"] == "initializing"


@pytest.mark.parametrize(
    "field",
    ["session_uid", "status", "total_files", "total_bytes", "confirmed_files", "confirmed_bytes", "pending_paths"],
)
def test_missing_status_field_never_becomes_completion(transfer, field):
    transfer.retained()
    body = transfer.body()
    del body[field]
    transfer.status_route.respond(200, json=body)
    with pytest.raises(UploadStateError):
        transfer.upload()
    assert not transfer.init_route.called and not transfer.grants_route.called


@pytest.mark.parametrize("value", [True, 1.0, 1.5, "1", -1, 2, None])
@pytest.mark.parametrize("field", ["total_files", "confirmed_files", "total_bytes", "confirmed_bytes"])
def test_status_counters_are_exact_integers(transfer, field, value):
    transfer.retained()
    body = transfer.body()
    body[field] = value
    transfer.status_route.respond(200, json=body)
    with pytest.raises(UploadStateError):
        transfer.upload()
    assert not transfer.grants_route.called


@pytest.mark.parametrize("pending", [None, "capture.mcap", [None], ["unknown"], ["capture.mcap", "capture.mcap"]])
def test_invalid_pending_inventory_refuses(transfer, pending):
    transfer.retained()
    body = transfer.body()
    body["pending_paths"] = pending
    transfer.status_route.respond(200, json=body)
    with pytest.raises(UploadStateError):
        transfer.upload()
    assert not transfer.grants_route.called


def test_capped_inventory_refuses_even_with_plausible_duplicate_confirm_counters(transfer):
    (transfer.source / "capture.mcap").unlink()
    transfer.files = {f"{i}.mcap": b"x" for i in range(1500)}
    for path, data in transfer.files.items():
        (transfer.source / path).write_bytes(data)
    transfer.retained()
    body = dict(
        session_uid=transfer.uid,
        status="uploading",
        total_files=1500,
        total_bytes=1500,
        confirmed_files=500,
        confirmed_bytes=500,
        pending_paths=list(transfer.files)[:1000],
    )
    transfer.status_route.respond(200, json=body)
    with pytest.raises(UploadStateError, match="1000-path cap"):
        transfer.upload()
    assert not transfer.grants_route.called and not transfer.finalize_route.called


def test_lost_confirmation_response_resumes_without_double_confirm(transfer):
    def response(request):
        transfer.confirm(request)
        raise httpx.ReadTimeout("synthetic lost confirmation")

    transfer.confirm_route.mock(side_effect=response)
    with pytest.raises(UploadStateError):
        transfer.upload()
    assert json.loads(transfer.receipt().read_text())["phase"] == "confirming"
    assert transfer.upload().status == "completed"
    assert transfer.confirm_route.call_count == 1
    assert len(transfer.sent) == 1


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "duplicate", "wrong_prefix", "wrong_path", "missing_headers", "unsafe_header"]
)
def test_entire_presign_batch_is_validated_before_any_put(transfer, mutation):
    def response(request):
        body = json.loads(transfer.grants(request).content)
        grant = body["urls"][0]
        if mutation == "missing":
            body["urls"] = []
        elif mutation == "extra":
            body["urls"].append(dict(grant, path="other"))
        elif mutation == "duplicate":
            body["urls"].append(grant)
        elif mutation == "wrong_prefix":
            grant["s3_key"] = grant["s3_key"].replace("/org/", "/other/")
            grant["put_url"] = grant["put_url"].replace("/org/", "/other/")
        elif mutation == "wrong_path":
            grant["put_url"] = grant["put_url"].replace("capture.mcap", "different.mcap")
        elif mutation == "missing_headers":
            del grant["headers"]
        else:
            grant["headers"] = {"Authorization": "synthetic private header"}
        return httpx.Response(200, json=body)

    transfer.grants_route.mock(side_effect=response)
    with pytest.raises(UploadStateError):
        transfer.upload()
    assert not transfer.put_route.called and not transfer.confirm_route.called


@pytest.mark.parametrize(
    "headers",
    [
        {"Cache-Control": "private, immutable, max-age=31536000", "Content-Encoding": "gzip"},
        {"Cache-Control": "private, immutable, max-age=31536000"},
        {"Cache-Control": "no-cache"},
        {"Content-Type": "application/octet-stream"},
    ],
)
def test_actual_server_headers_are_preserved(transfer, headers):
    transfer.headers = headers
    assert transfer.upload().status == "completed"
    for name, value in headers.items():
        assert transfer.sent[0][2][name.lower()] == value


@pytest.mark.parametrize("phase", ["finalizing", "completed"])
@pytest.mark.parametrize("server", ["uploading", "abandoned", "different"])
def test_terminal_receipt_never_resumes_mutations_on_conflicting_status(transfer, phase, server):
    transfer.retained(phase=phase)
    transfer.confirmed = set(transfer.files)
    transfer.phase = "completed" if server == "different" else server
    if server == "different":
        transfer.uid = "other-session"
    with pytest.raises(UploadStateError):
        transfer.upload()
    assert not transfer.init_route.called and not transfer.grants_route.called and not transfer.finalize_route.called


@pytest.mark.parametrize(
    "outcome", ["completing", "uploading", "different", "unavailable", "malformed_ack", "lost_ack"]
)
def test_finalize_ambiguity_never_discards_receipt(transfer, outcome):
    def finalize(request):
        transfer.phase = "completing"
        if outcome == "lost_ack":
            raise httpx.ReadTimeout("synthetic lost finalize")
        if outcome == "malformed_ack":
            return httpx.Response(202, json={})
        if outcome == "different":
            transfer.uid = "other-session"
        if outcome == "uploading":
            transfer.phase = "uploading"
        if outcome == "unavailable":
            transfer.status_route.respond(503, json={})
        return httpx.Response(200, json={"detail": "Upload session is already finalizing or complete."})

    transfer.finalize_route.mock(side_effect=finalize)
    if outcome == "completing":
        assert transfer.upload().status == "completing"
    else:
        with pytest.raises(UploadStateError):
            transfer.upload()
    assert json.loads(transfer.receipt().read_text())["phase"] == "finalizing"


def test_logs_and_error_output_exclude_api_and_storage_secrets(transfer, caplog):
    caplog.set_level(logging.DEBUG)

    def status(request):
        logging.getLogger("httpcore.http11").debug("response headers: synthetic-private-api-cookie")
        return transfer.status(request)

    transfer.status_route.mock(side_effect=status)
    transfer.put_route.mock(side_effect=httpx.ConnectError("synthetic-private-storage-token"))
    with pytest.raises(UploadStateError) as caught:
        transfer.upload()
    rendered = "".join(traceback.format_exception(type(caught.value), caught.value, caught.value.__traceback__))
    for secret in ["synthetic-private-api-cookie", "synthetic-private-storage-token", "signature=synthetic-private"]:
        assert secret not in caplog.text and secret not in rendered
    logging.getLogger("httpx").info("unrelated request logging restored")
    assert "unrelated request logging restored" in caplog.text


def test_delayed_confirmation_commit_cannot_be_sent_twice(transfer):
    delayed = []

    def timeout_before_commit(request):
        delayed.append(request)
        assert json.loads(transfer.receipt().read_text())["phase"] == "confirming"
        raise httpx.ReadTimeout("synthetic request may still commit")

    transfer.confirm_route.mock(side_effect=timeout_before_commit)
    with pytest.raises(UploadStateError):
        transfer.upload()
    with pytest.raises(UploadStateError, match="may still execute"):
        transfer.upload()
    assert transfer.confirm_route.call_count == 1 and len(transfer.sent) == 1
    transfer.confirm(delayed[0])
    assert transfer.upload().status == "completed"
    assert transfer.confirm_route.call_count == 1


@pytest.mark.parametrize("phase", ["active", "finalizing", "completed"])
def test_changed_confirmed_source_during_status_blocks_finalize_or_success(transfer, phase):
    transfer.retained(phase=phase)
    transfer.confirmed = set(transfer.files)
    transfer.phase = "uploading" if phase == "active" else "completed"

    def mutate_then_respond(request):
        response = transfer.status(request)
        (transfer.source / "capture.mcap").write_bytes(b"changed")
        return response

    transfer.status_route.mock(side_effect=mutate_then_respond)
    with pytest.raises(UploadStateError):
        transfer.upload()
    assert not transfer.finalize_route.called and not transfer.grants_route.called
    assert json.loads(transfer.receipt().read_text())["phase"] == phase


def test_changed_source_while_awaiting_final_result_keeps_finalizing(transfer):
    def finalize(request):
        response = transfer.finalize(request)
        (transfer.source / "capture.mcap").write_bytes(b"changed")
        return response

    transfer.finalize_route.mock(side_effect=finalize)
    with pytest.raises(UploadStateError):
        transfer.upload()
    assert json.loads(transfer.receipt().read_text())["phase"] == "finalizing"


def test_put_success_without_consuming_exact_stream_cannot_confirm(transfer, monkeypatch):
    def no_read(url, **kwargs):
        return httpx.Response(200, headers={"ETag": "synthetic-etag"}, request=httpx.Request("PUT", url))

    monkeypatch.setattr(guard.httpx, "put", no_read)
    with pytest.raises(UploadStateError):
        transfer.upload()
    assert not transfer.confirm_route.called


def test_changed_bytes_during_put_cannot_confirm(transfer, monkeypatch):
    monkeypatch.setattr(guard, "_CHUNK_SIZE", 3)

    def consume_changed_bytes(url, *, content, **kwargs):
        chunks = iter(content)
        next(chunks)
        with (transfer.source / "capture.mcap").open("r+b") as handle:
            handle.seek(3)
            handle.write(b"xxxx")
        list(chunks)
        return httpx.Response(200, headers={"ETag": "synthetic-etag"}, request=httpx.Request("PUT", url))

    monkeypatch.setattr(guard.httpx, "put", consume_changed_bytes)
    with pytest.raises(UploadStateError):
        transfer.upload()
    assert not transfer.confirm_route.called and not transfer.finalize_route.called


@pytest.mark.parametrize("path_style", [False, True])
def test_exact_unicode_space_plus_and_percent_key_encoding_is_supported(transfer, path_style):
    original = transfer.source / "capture.mcap"
    name = "camera 😀 + %FF.mcap"
    original.rename(transfer.source / name)
    transfer.files = {name: b"capture"}

    def grants(request):
        body = json.loads(transfer.grants(request).content)
        if path_style:
            body["urls"][0]["put_url"] = body["urls"][0]["put_url"].replace(
                "bucket.s3.us-west-2.amazonaws.com/", "s3.us-west-2.amazonaws.com/dotted.bucket/"
            )
        return httpx.Response(200, json=body)

    transfer.grants_route.mock(side_effect=grants)
    assert transfer.upload().status == "completed"
    assert transfer.sent[0][1] == b"capture"


@pytest.mark.parametrize("mutation", ["dot", "extra_segment", "invalid_utf8", "invalid_escape", "lowercase_escape"])
def test_url_normalization_never_selects_another_object(transfer, mutation):
    def grants(request):
        body = json.loads(transfer.grants(request).content)
        grant = body["urls"][0]
        url = grant["put_url"]
        if mutation == "dot":
            url = url.replace("/fleet/", "/different/../fleet/")
        elif mutation == "extra_segment":
            url = url.replace("/fleet/", "/extra/fleet/")
        elif mutation == "invalid_utf8":
            url = url.replace("capture.mcap", "%FF.mcap")
        elif mutation == "invalid_escape":
            url = url.replace("capture.mcap", "%ZZ.mcap")
        else:
            url = url.replace("capture.mcap", "%63apture.mcap")
        grant["put_url"] = url
        return httpx.Response(200, json=body)

    transfer.grants_route.mock(side_effect=grants)
    with pytest.raises(UploadStateError):
        transfer.upload()
    assert not transfer.put_route.called


def test_wrong_prefix_after_restart_is_rejected(transfer):
    transfer.retained()
    transfer.prefix = "fleet/other/device/recording-1/"
    with pytest.raises(UploadStateError):
        transfer.upload()
    assert not transfer.put_route.called


def test_fresh_many_file_upload_uses_linear_full_inventory_scans(transfer, monkeypatch):
    (transfer.source / "capture.mcap").unlink()
    transfer.files = {f"{i:04}.mcap": b"x" for i in range(1001)}
    for path, data in transfer.files.items():
        (transfer.source / path).write_bytes(data)
    collect = uploads.FleetUploadManager.collect_files
    scans = []

    def counted(manager, source):
        scans.append(source)
        return collect(manager, source)

    monkeypatch.setattr(uploads.FleetUploadManager, "collect_files", counted)
    assert transfer.upload().status == "completed"
    assert len(scans) == 4  # Initial, pre-init, pre-finalize, final result; independent of batch count.
    assert transfer.grants_route.call_count == 11


def test_prefixless_version_one_receipt_is_retained_and_refused(transfer):
    transfer.retained()
    receipt = transfer.receipt()
    payload = json.loads(receipt.read_text())
    payload["version"] = 1
    del payload["s3_prefix"]
    del payload["confirmation_paths"]
    receipt.write_text(json.dumps(payload))
    retained = receipt.read_bytes()
    with pytest.raises(UploadStateError):
        transfer.upload()
    assert receipt.read_bytes() == retained and not transfer.status_route.called


def _cli(transfer, client, *extra):
    kwargs = {"mix_stderr": False} if "mix_stderr" in inspect.signature(CliRunner).parameters else {}
    runner = CliRunner(**kwargs)
    return runner.invoke(
        fleet,
        ["recordings", "upload", "--source", str(transfer.source), "--recording", "recording-1", *extra],
        obj={"client": client},
    )


def test_cli_preview_and_upload_share_state_exclusion_without_preview_network(transfer, monkeypatch):
    transfer.state = transfer.source / ".avala" / "uploads"
    transfer.state.mkdir(parents=True)
    (transfer.state / "other-recording.json").write_text("synthetic private checkpoint")
    monkeypatch.setattr(uploads, "_STATE_DIR", transfer.state)
    with Client(api_key="synthetic credential") as client:
        monkeypatch.setattr(
            client.fleet.recordings, "get", lambda uid: SimpleNamespace(uid=uid, status="uploading", size_bytes=0)
        )
        preview = _cli(transfer, client, "--dry-run")
        assert preview.exit_code == 0, preview.output
        output = preview.stdout + preview.stderr
        assert "1 files" in output and "capture.mcap" in output
        assert "other-recording" not in output and "private checkpoint" not in output
        assert not transfer.status_route.called and not transfer.init_route.called
        result = _cli(transfer, client)
        assert result.exit_code == 0, result.output
    assert transfer.init_route.call_count == 1


def test_cli_partial_transfer_is_an_error_and_does_not_expose_transport_details(transfer, monkeypatch):
    transfer.put_route.mock(side_effect=httpx.ConnectError("synthetic-private-cli-token"))
    with Client(api_key="synthetic credential") as client:
        monkeypatch.setattr(
            client.fleet.recordings, "get", lambda uid: SimpleNamespace(uid=uid, status="uploading", size_bytes=0)
        )
        result = _cli(transfer, client)
    assert result.exit_code != 0
    output = result.stdout + result.stderr
    assert "synthetic-private-cli-token" not in output and "signature=" not in output and "Done" not in output
    assert transfer.receipt().exists() and not transfer.finalize_route.called


def test_cli_wait_does_not_equate_upload_completion_with_recording_ready(transfer, monkeypatch):
    observations = []

    def recording(uid):
        observations.append(uid)
        return SimpleNamespace(uid=uid, status="uploading" if len(observations) == 1 else "ready", size_bytes=7)

    with Client(api_key="synthetic credential") as client:
        monkeypatch.setattr(client.fleet.recordings, "get", recording)
        result = _cli(transfer, client, "--wait")
    assert result.exit_code == 0, result.output
    assert len(observations) == 2
    assert "is ready" in result.stdout + result.stderr


@pytest.mark.parametrize("host", ["s3-example.s3.us-east-1.amazonaws.com", "s3.s3.us-east-1.amazonaws.com"])
@pytest.mark.parametrize("extra", ["", "other-bucket/"])
def test_s3_named_virtual_bucket_is_not_a_path_style_endpoint(transfer, host, extra):
    def grants(request):
        body = json.loads(transfer.grants(request).content)
        body["urls"][0]["put_url"] = "https://" + host + "/" + extra + quote(transfer.prefix + "capture.mcap", safe="/")
        return httpx.Response(200, json=body)

    transfer.grants_route.mock(side_effect=grants)
    if extra:
        with pytest.raises(UploadStateError):
            transfer.upload()
        assert not transfer.put_route.called
    else:
        assert transfer.upload().status == "completed"


@pytest.mark.parametrize(
    "host",
    [
        "s3.amazonaws.com",
        "s3-us-east-1.amazonaws.com",
        "s3.us-east-1.amazonaws.com",
        "s3.dualstack.us-east-1.amazonaws.com",
    ],
)
def test_genuine_s3_path_style_endpoint_keeps_bucket_segment(transfer, host):
    def grants(request):
        body = json.loads(transfer.grants(request).content)
        body["urls"][0]["put_url"] = "https://" + host + "/bucket/" + quote(transfer.prefix + "capture.mcap", safe="/")
        return httpx.Response(200, json=body)

    transfer.grants_route.mock(side_effect=grants)
    assert transfer.upload().status == "completed"


@pytest.mark.parametrize(
    "host",
    [
        "s3.amazonaws.com",
        "s3-us-east-1.amazonaws.com",
        "s3.us-east-1.amazonaws.com",
        "s3.dualstack.us-east-1.amazonaws.com",
    ],
)
def test_path_style_endpoint_requires_bucket_before_exact_key(transfer, host):
    def grants(request):
        body = json.loads(transfer.grants(request).content)
        body["urls"][0]["put_url"] = "https://" + host + "/" + quote(transfer.prefix + "capture.mcap", safe="/")
        return httpx.Response(200, json=body)

    transfer.grants_route.mock(side_effect=grants)
    with pytest.raises(UploadStateError):
        transfer.upload()
    assert not transfer.put_route.called


@pytest.mark.parametrize(
    "field,value",
    [
        ("session_uid", "other"),
        ("confirmed_files", None),
        ("confirmed_files", True),
        ("confirmed_bytes", 7.0),
        ("total_files", 2),
        ("total_bytes", 8),
    ],
)
def test_malformed_confirm_ack_keeps_exact_confirmation_intent(transfer, field, value):
    def confirm(request):
        payload = json.loads(transfer.confirm(request).content)
        payload[field] = value
        return httpx.Response(200, json=payload)

    transfer.confirm_route.mock(side_effect=confirm)
    with pytest.raises(UploadStateError):
        transfer.upload()
    receipt = json.loads(transfer.receipt().read_text())
    assert receipt["phase"] == "confirming" and receipt["confirmation_paths"] == ["capture.mcap"]
    assert not transfer.finalize_route.called


def test_changed_original_source_cannot_finalize_old_confirmed_session(transfer):
    transfer.retained()
    transfer.confirmed = set(transfer.files)
    (transfer.source / "capture.mcap").write_bytes(b"changed")
    with pytest.raises(UploadStateError):
        transfer.upload()
    assert not transfer.status_route.called and not transfer.init_route.called and not transfer.finalize_route.called


@pytest.mark.parametrize("when", ["initial", "poll", "final_session"])
def test_cli_never_switches_requested_recording_or_pinned_session(transfer, monkeypatch, when):
    calls = []

    def recording(uid):
        calls.append(uid)
        if when == "final_session" and len(calls) > 1:
            transfer.uid = "other-session"
        changed = when == "initial" or (when == "poll" and len(calls) > 1)
        return SimpleNamespace(uid="other-recording" if changed else uid, status="ready", size_bytes=7)

    with Client(api_key="synthetic credential") as client:
        monkeypatch.setattr(client.fleet.recordings, "get", recording)
        result = _cli(transfer, client, "--wait")
    assert result.exit_code != 0
    assert "is ready" not in result.stdout + result.stderr
    if when == "initial":
        assert not transfer.init_route.called
    else:
        assert json.loads(transfer.receipt().read_text())["session_uid"] == "session-1"


def test_cli_wait_continues_until_exact_session_and_recording_both_complete(transfer, monkeypatch):
    def finalize(request):
        transfer.phase = "completing"
        return httpx.Response(202, json={"detail": "Finalization started.", "session_uid": transfer.uid})

    transfer.finalize_route.mock(side_effect=finalize)
    polls = []

    def recording(uid):
        polls.append(uid)
        if len(polls) == 3:
            transfer.phase = "completed"
        return SimpleNamespace(uid=uid, status="ready", size_bytes=7)

    with Client(api_key="synthetic credential") as client:
        monkeypatch.setattr(client.fleet.recordings, "get", recording)
        result = _cli(transfer, client, "--wait")
    assert result.exit_code == 0, result.output
    assert len(polls) == 3
    assert "upload=completing" in result.stdout + result.stderr
    assert (
        transfer.init_route.call_count == transfer.confirm_route.call_count == transfer.finalize_route.call_count == 1
    )


def test_cli_same_session_completing_respects_timeout(transfer, monkeypatch):
    def finalize(request):
        transfer.phase = "completing"
        return httpx.Response(202, json={"detail": "Finalization started.", "session_uid": transfer.uid})

    transfer.finalize_route.mock(side_effect=finalize)
    with Client(api_key="synthetic credential") as client:
        monkeypatch.setattr(
            client.fleet.recordings, "get", lambda uid: SimpleNamespace(uid=uid, status="ready", size_bytes=7)
        )
        result = _cli(transfer, client, "--wait", "--wait-timeout", "0")
    assert result.exit_code != 0 and "Timed out" in result.stdout + result.stderr
    assert "is ready" not in result.stdout + result.stderr
    assert json.loads(transfer.receipt().read_text())["phase"] == "finalizing"


def test_read_only_observation_cannot_initialize_after_receipt_disappears(transfer):
    result = transfer.upload()
    transfer.receipt().unlink()
    transfer.uid = None
    status_calls = transfer.status_route.call_count
    with Client(api_key="synthetic credential") as client:
        with pytest.raises(UploadStateError):
            client.fleet.uploads._observe_upload("recording-1", result.uid, transfer.source)
    assert transfer.status_route.call_count == status_calls
    assert (
        transfer.init_route.call_count == transfer.confirm_route.call_count == transfer.finalize_route.call_count == 1
    )


@pytest.mark.parametrize("value", ["nan", "inf", "-1"])
def test_cli_invalid_timeout_fails_before_upload(transfer, value):
    with Client(api_key="synthetic credential") as client:
        result = _cli(transfer, client, "--wait", "--wait-timeout", value)
    assert result.exit_code != 0 and not transfer.init_route.called


@pytest.mark.parametrize("etag", ["x" * 129, " leading", "trailing "])
def test_storage_etag_must_fit_the_exact_server_confirmation_field(transfer, etag):
    transfer.put_route.respond(200, headers={"ETag": etag})
    with pytest.raises(UploadStateError):
        transfer.upload()
    assert not transfer.confirm_route.called
    assert json.loads(transfer.receipt().read_text())["phase"] == "active"
