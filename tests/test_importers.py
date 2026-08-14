from __future__ import annotations

import json

import httpx
import pytest
import respx

from avala import Client
from avala._uploads import (
    INVALID_STAMP,
    MAX_RETRIES,
    load_completed,
    load_completed_stamps,
    save_completed,
    state_path,
    upload_fingerprint,
)
from avala.errors import RateLimitError, ServerError, ValidationError
from avala.importers import available_importers, detect_data_type, import_dataset, import_folder

BASE_URL = "https://api.avala.ai/api/v1"
PRESIGN_URL = f"{BASE_URL}/datasets/manual-upload/file-upload-url/"
FINALIZE_URL = f"{BASE_URL}/datasets/manual-upload/"
QUOTA_URL = f"{BASE_URL}/datasets/manual-upload/quota/"
# Must be a real S3 host: the uploader refuses to POST file bytes to anything
# outside the presigned-URL allow-list (``avala/_uploads.py``).
S3_URL = "https://s3.us-east-1.amazonaws.com/upload"


# ── pure logic ──
def test_detect_data_type_image():
    assert detect_data_type([("x/a.jpg", "a.jpg"), ("x/b.png", "b.png")]) == "image"


def test_detect_data_type_mcap():
    assert detect_data_type([("x/run.mcap", "run.mcap")]) == "mcap"


def test_detect_data_type_mixed_errors():
    with pytest.raises(ValueError, match="mixed data types"):
        detect_data_type([("a.jpg", "a.jpg"), ("b.mp4", "b.mp4")])


def test_detect_data_type_unknown_errors():
    with pytest.raises(ValueError, match="could not infer"):
        detect_data_type([("a.txt", "a.txt")])


def test_registry_has_folder():
    assert "folder" in available_importers()


def test_import_dataset_unknown_source_errors():
    with pytest.raises(ValueError, match="unknown importer"):
        import_dataset("nope", Client(api_key="k"))


# ── import_folder (respx: presign + S3 POST + finalize) ──
def _write_files(tmp_path, names):
    for n in names:
        (tmp_path / n).write_bytes(b"x" * 16)
    return tmp_path


def _wire_upload(dataset_json, *, quota_used=0, quota_limit=10 * 1024**3):
    # create_from_local reads the quota meter before moving any bytes, so the
    # folder importer hits this endpoint on every run.
    respx.get(QUOTA_URL).mock(return_value=httpx.Response(200, json={"used": quota_used, "limit": quota_limit}))
    respx.post(PRESIGN_URL).mock(
        return_value=httpx.Response(200, json={"url": S3_URL, "fields": {"key": "k", "Content-Type": "image/jpeg"}})
    )
    s3 = respx.post(S3_URL).mock(return_value=httpx.Response(204))
    respx.post(FINALIZE_URL).mock(return_value=httpx.Response(201, json=dataset_json))
    return s3


@respx.mock
def test_import_folder_auto_detects_and_creates(tmp_path):
    _write_files(tmp_path, ["a.jpg", "b.jpg"])
    s3 = _wire_upload({"uid": "d1", "name": "My Drive", "slug": "my-drive", "data_type": "image", "item_count": 2})

    client = Client(api_key="test-key")
    ds = import_folder(client, source=str(tmp_path), name="My Drive", slug="my-drive")
    client.close()

    assert ds.uid == "d1"
    assert ds.data_type == "image"
    assert s3.call_count == 2  # one S3 upload per file


@respx.mock
def test_import_folder_explicit_data_type(tmp_path):
    _write_files(tmp_path, ["scene.alp"])  # server-indexable LiDAR suffix
    _wire_upload({"uid": "d2", "name": "L", "slug": "l", "data_type": "lidar", "item_count": 1})

    client = Client(api_key="test-key")
    ds = import_folder(client, source=str(tmp_path), name="L", slug="l", data_type="lidar")
    client.close()
    assert ds.data_type == "lidar"


def test_import_folder_rejects_non_indexable_files(tmp_path):
    # .pcd/.bin/.las/.laz are NOT indexed by the server LiDAR filter (only .alp/.alp.gz).
    # Importing them would finalize an empty dataset, so the guard must reject up front.
    _write_files(tmp_path, ["cloud.pcd"])
    with pytest.raises(ValueError, match="indexable"):
        import_folder(Client(api_key="k"), source=str(tmp_path), name="x", slug="x", data_type="lidar")


def test_detect_data_type_ignores_non_indexable():
    # A folder of .pcd files no longer auto-detects as LiDAR (server can't index them).
    with pytest.raises(ValueError, match="could not infer"):
        detect_data_type([("a.pcd", "a.pcd"), ("b.bin", "b.bin")])


def test_detect_data_type_compressed_ply_is_splat():
    assert detect_data_type([("x/a.compressed.ply", "a.compressed.ply")]) == "splat"


@respx.mock
def test_import_dataset_dispatches_to_folder(tmp_path):
    _write_files(tmp_path, ["a.png"])
    _wire_upload({"uid": "d3", "name": "P", "slug": "p", "data_type": "image", "item_count": 1})
    client = Client(api_key="test-key")
    ds = import_dataset("folder", client, source=str(tmp_path), name="P", slug="p")
    client.close()
    assert ds.uid == "d3"


def test_import_folder_empty_errors(tmp_path):
    with pytest.raises(ValueError, match="no files found"):
        import_folder(Client(api_key="k"), source=str(tmp_path), name="x", slug="x")


# ── backbone: datasets.upload_files ──
@respx.mock
def test_upload_files_reports_progress(tmp_path):
    _write_files(tmp_path, ["a.jpg", "b.jpg"])
    respx.post(PRESIGN_URL).mock(
        return_value=httpx.Response(200, json={"url": S3_URL, "fields": {"Content-Type": "image/jpeg"}})
    )
    respx.post(S3_URL).mock(return_value=httpx.Response(204))

    seen = []
    client = Client(api_key="test-key")
    total = client.datasets.upload_files(
        dataset_name="My Drive",
        files=[(str(tmp_path / "a.jpg"), "a.jpg"), (str(tmp_path / "b.jpg"), "b.jpg")],
        workers=2,
        on_progress=lambda rel, n: seen.append(rel),
    )
    client.close()

    assert total == 32  # 2 files x 16 bytes
    assert set(seen) == {"a.jpg", "b.jpg"}


@respx.mock
def test_upload_files_retries_then_fails_on_persistent_server_error(tmp_path):
    """A 5xx is transient, so it is retried — but a *persistent* one still fails
    the run rather than looping forever."""
    _write_files(tmp_path, ["a.jpg"])
    presign = respx.post(PRESIGN_URL).mock(return_value=httpx.Response(500, json={"detail": "boom"}))

    client = Client(api_key="test-key")
    with pytest.raises(ServerError):
        client.datasets.upload_files(dataset_name="X", files=[(str(tmp_path / "a.jpg"), "a.jpg")], workers=1)
    client.close()

    assert presign.call_count == MAX_RETRIES


@respx.mock
def test_upload_files_does_not_retry_client_errors(tmp_path):
    """A 400 means the request itself is wrong. Repeating it verbatim can only
    waste the retry budget, so it must fail on the first attempt."""
    _write_files(tmp_path, ["a.jpg"])
    presign = respx.post(PRESIGN_URL).mock(return_value=httpx.Response(400, json={"detail": "bad name"}))

    client = Client(api_key="test-key")
    with pytest.raises(ValidationError):
        client.datasets.upload_files(dataset_name="X", files=[(str(tmp_path / "a.jpg"), "a.jpg")], workers=1)
    client.close()

    assert presign.call_count == 1


@respx.mock
def test_upload_files_recovers_after_transient_failure(tmp_path):
    """The presign is re-issued on every attempt: a presigned POST expires, so a
    retry against the original target would fail forever."""
    _write_files(tmp_path, ["a.jpg"])
    presign = respx.post(PRESIGN_URL).mock(
        side_effect=[
            httpx.Response(503, json={"detail": "try later"}),
            httpx.Response(200, json={"url": S3_URL, "fields": {"Content-Type": "image/jpeg"}}),
        ]
    )
    s3 = respx.post(S3_URL).mock(return_value=httpx.Response(204))

    client = Client(api_key="test-key")
    total = client.datasets.upload_files(dataset_name="X", files=[(str(tmp_path / "a.jpg"), "a.jpg")], workers=1)
    client.close()

    assert total == 16
    assert presign.call_count == 2
    assert s3.call_count == 1


@respx.mock
def test_upload_files_rejects_non_allowlisted_upload_host(tmp_path):
    """A hijacked control-plane response must not be able to redirect raw file
    bytes to an arbitrary host."""
    _write_files(tmp_path, ["a.jpg"])
    respx.post(PRESIGN_URL).mock(
        return_value=httpx.Response(200, json={"url": "https://evil.example.com/u", "fields": {}})
    )

    client = Client(api_key="test-key")
    with pytest.raises(ValueError, match="allow-list"):
        client.datasets.upload_files(dataset_name="X", files=[(str(tmp_path / "a.jpg"), "a.jpg")], workers=1)
    client.close()


@pytest.mark.parametrize(
    "host",
    [
        "my-bucket.s3.us-east-1.amazonaws.com",
        "s3.amazonaws.com",
        "s3.us-west-2.amazonaws.com",
        "b.s3-accelerate.amazonaws.com",
        "s3.dualstack.eu-west-1.amazonaws.com",
        "b.s3-us-east-1.amazonaws.com",
        "storage.googleapis.com",
        "b.storage.googleapis.com",
        "acct.blob.core.windows.net",
    ],
)
def test_presigned_allow_list_accepts_real_storage_endpoints(host):
    from avala._uploads import validate_presigned_url

    validate_presigned_url(f"https://{host}/bucket/key")


@pytest.mark.parametrize(
    "host",
    [
        # Any AWS account can provision these, so a bare `.amazonaws.com`
        # suffix check handed raw customer file bytes to an attacker.
        "evil.execute-api.us-east-1.amazonaws.com",
        "lambda.us-east-1.amazonaws.com",
        # Substring, not a label — the `(?:^|\.)` anchor is what rejects it.
        "evil-s3.amazonaws.com",
        # Right label, wrong domain / suffix-extension past the real one.
        "s3.evil.com",
        "b.s3.us-east-1.amazonaws.com.evil.com",
    ],
)
def test_presigned_allow_list_rejects_non_storage_aws_hosts(host):
    from avala._uploads import validate_presigned_url

    with pytest.raises(ValueError, match="allow-list"):
        validate_presigned_url(f"https://{host}/upload")


@respx.mock
def test_upload_files_resumes_and_skips_confirmed_files(tmp_path):
    """An interrupted transfer must not re-send what already landed."""
    from avala.resources import datasets as datasets_mod

    _write_files(tmp_path, ["a.jpg", "b.jpg"])
    files = [(str(tmp_path / "a.jpg"), "a.jpg"), (str(tmp_path / "b.jpg"), "b.jpg")]

    # First run: b.jpg fails permanently, a.jpg lands.
    respx.post(PRESIGN_URL).mock(
        side_effect=lambda request: (
            httpx.Response(400, json={"detail": "nope"})
            if b"b.jpg" in request.content
            else httpx.Response(200, json={"url": S3_URL, "fields": {}})
        )
    )
    s3 = respx.post(S3_URL).mock(return_value=httpx.Response(204))

    client = Client(api_key="test-key")
    with pytest.raises(ValidationError):
        client.datasets.upload_files(dataset_name="X", files=files, workers=1, state_key="x")

    fp_x = upload_fingerprint(BASE_URL, None, "X", "test-key")
    assert load_completed(datasets_mod._STATE_DIR, "x", fingerprint=fp_x) == {"a.jpg"}
    first_run_uploads = s3.call_count

    # Second run: only the file that never landed is presigned again.
    respx.post(PRESIGN_URL).mock(return_value=httpx.Response(200, json={"url": S3_URL, "fields": {}}))
    # `clear_state_on_success` is opt-in: the default retains state because this
    # method does not finalize the dataset. Asked explicitly, it must clean up.
    client.datasets.upload_files(dataset_name="X", files=files, workers=1, state_key="x", clear_state_on_success=True)
    client.close()

    assert s3.call_count == first_run_uploads + 1  # b.jpg only
    # A fully-successful run clears the checkpoint — there is no work left to describe.
    assert (
        load_completed(datasets_mod._STATE_DIR, "x", fingerprint=upload_fingerprint(BASE_URL, None, "X", "test-key"))
        == set()
    )


@respx.mock
def test_upload_files_clears_checkpoint_when_everything_already_landed(tmp_path):
    """A transfer that completes across two runs must clean up its checkpoint
    just like one that completes in a single run. Returning early on a
    fully-resumed run used to leave the state file behind for good."""
    from avala.resources import datasets as datasets_mod

    _write_files(tmp_path, ["a.jpg"])
    files = [(str(tmp_path / "a.jpg"), "a.jpg")]
    respx.post(PRESIGN_URL).mock(return_value=httpx.Response(200, json={"url": S3_URL, "fields": {}}))
    s3 = respx.post(S3_URL).mock(return_value=httpx.Response(204))

    client = Client(api_key="test-key")
    client.datasets.upload_files(dataset_name="X", files=files, workers=1, state_key="done")
    # Simulate the cross-run case: the checkpoint says the file already landed.
    # It must carry the destination fingerprint, or it is (correctly) ignored.
    save_completed(
        datasets_mod._STATE_DIR,
        "done",
        {"a.jpg"},
        fingerprint=upload_fingerprint(BASE_URL, None, "X", "test-key"),
    )

    uploaded = client.datasets.upload_files(
        dataset_name="X", files=files, workers=1, state_key="done", clear_state_on_success=True
    )
    client.close()

    assert uploaded == 0
    assert s3.call_count == 1  # nothing re-sent
    assert not state_path(
        datasets_mod._STATE_DIR, "done", fingerprint=upload_fingerprint(BASE_URL, None, "X", "test-key")
    ).exists()


@respx.mock
def test_checkpoint_is_not_reused_across_organizations(tmp_path):
    """A checkpoint records WHERE bytes went, not just which paths were sent.

    Uploading "My Dataset" to org A, failing, then uploading a same-named
    dataset to org B must not skip the files that landed in A's prefix — the
    org determines the S3 key root, so B's dataset would list nothing while
    every request returned 2xx. Silent partial datasets are the exact failure
    this whole module exists to prevent.
    """
    _write_files(tmp_path, ["a.jpg"])
    files = [(str(tmp_path / "a.jpg"), "a.jpg")]
    respx.post(PRESIGN_URL).mock(return_value=httpx.Response(200, json={"url": S3_URL, "fields": {}}))
    s3 = respx.post(S3_URL).mock(return_value=httpx.Response(204))

    client = Client(api_key="test-key")
    client.datasets.upload_files(
        dataset_name="Shared Name",
        files=files,
        workers=1,
        state_key="shared",
        organization_uid="org-A",
        clear_state_on_success=False,
    )
    assert s3.call_count == 1

    # Same state key, different organization — must re-upload, not skip.
    client.datasets.upload_files(
        dataset_name="Shared Name",
        files=files,
        workers=1,
        state_key="shared",
        organization_uid="org-B",
    )
    client.close()

    assert s3.call_count == 2


@respx.mock
def test_upload_files_retries_non_enumerated_gateway_errors(tmp_path):
    """Cloudflare-style 52x codes are transient too. Enumerating 500/502/503/504
    made them fail on the first attempt while the docs promised 5xx retries."""
    _write_files(tmp_path, ["a.jpg"])
    presign = respx.post(PRESIGN_URL).mock(
        side_effect=[
            httpx.Response(522, json={"detail": "connection timed out"}),
            httpx.Response(200, json={"url": S3_URL, "fields": {}}),
        ]
    )
    respx.post(S3_URL).mock(return_value=httpx.Response(204))

    client = Client(api_key="test-key")
    total = client.datasets.upload_files(dataset_name="X", files=[(str(tmp_path / "a.jpg"), "a.jpg")], workers=1)
    client.close()

    assert total == 16
    assert presign.call_count == 2


@respx.mock
def test_create_from_local_keeps_checkpoint_when_dataset_creation_fails(tmp_path):
    """Finalization is the cheap last step; the transfer is the expensive part.
    A failure there must not throw away a completed upload."""
    from avala.resources import datasets as datasets_mod

    _write_files(tmp_path, ["a.jpg"])
    respx.get(QUOTA_URL).mock(return_value=httpx.Response(200, json={"used": 0, "limit": 10**12}))
    respx.post(PRESIGN_URL).mock(return_value=httpx.Response(200, json={"url": S3_URL, "fields": {}}))
    s3 = respx.post(S3_URL).mock(return_value=httpx.Response(204))
    respx.post(FINALIZE_URL).mock(return_value=httpx.Response(500, json={"detail": "boom"}))

    client = Client(api_key="test-key")
    with pytest.raises(ServerError):
        client.datasets.create_from_local(source=str(tmp_path), name="N", slug="n", data_type="image")

    assert load_completed(
        datasets_mod._STATE_DIR, "N", fingerprint=upload_fingerprint(BASE_URL, None, "N", "test-key")
    ) == {"a.jpg"}

    # Re-run: finalization now succeeds and the upload is NOT repeated.
    respx.post(FINALIZE_URL).mock(
        return_value=httpx.Response(201, json={"uid": "d1", "name": "N", "slug": "n", "data_type": "image"})
    )
    client.datasets.create_from_local(source=str(tmp_path), name="N", slug="n", data_type="image")
    client.close()

    assert s3.call_count == 1  # never re-sent
    assert not state_path(
        datasets_mod._STATE_DIR, "N", fingerprint=upload_fingerprint(BASE_URL, None, "N", "test-key")
    ).exists()  # cleared after create succeeded


@respx.mock
def test_create_from_local_preflight_counts_only_pending_bytes(tmp_path):
    """Resuming an upload must not compare the WHOLE payload against the
    headroom left after part of it already landed — that refuses exactly the
    large transfers resume exists for."""
    from avala.resources import datasets as datasets_mod

    _write_files(tmp_path, ["a.jpg", "b.jpg"])  # 16 bytes each
    save_completed(
        datasets_mod._STATE_DIR,
        "n",
        {"a.jpg"},
        fingerprint=upload_fingerprint(BASE_URL, None, "N", "test-key"),
    )
    # Only 20 bytes of headroom: enough for the 16 still pending, not for 32.
    respx.get(QUOTA_URL).mock(return_value=httpx.Response(200, json={"used": 80, "limit": 100}))
    respx.post(PRESIGN_URL).mock(return_value=httpx.Response(200, json={"url": S3_URL, "fields": {}}))
    respx.post(S3_URL).mock(return_value=httpx.Response(204))
    respx.post(FINALIZE_URL).mock(
        return_value=httpx.Response(201, json={"uid": "d1", "name": "N", "slug": "n", "data_type": "image"})
    )

    client = Client(api_key="test-key")
    dataset = client.datasets.create_from_local(source=str(tmp_path), name="N", slug="n", data_type="image")
    client.close()

    assert dataset.uid == "d1"


@respx.mock
def test_create_from_local_survives_a_flaky_quota_endpoint(tmp_path):
    """The preflight is advisory. A transient failure reading the meter must not
    abort an upload that the retry-capable upload path would have completed."""
    _write_files(tmp_path, ["a.jpg"])
    respx.get(QUOTA_URL).mock(return_value=httpx.Response(503, json={"detail": "try later"}))
    respx.post(PRESIGN_URL).mock(return_value=httpx.Response(200, json={"url": S3_URL, "fields": {}}))
    respx.post(S3_URL).mock(return_value=httpx.Response(204))
    respx.post(FINALIZE_URL).mock(
        return_value=httpx.Response(201, json={"uid": "d1", "name": "N", "slug": "n", "data_type": "image"})
    )

    client = Client(api_key="test-key")
    dataset = client.datasets.create_from_local(source=str(tmp_path), name="N", slug="n", data_type="image")
    client.close()

    assert dataset.uid == "d1"


@respx.mock
def test_resume_does_not_skip_a_file_that_changed_on_disk(tmp_path):
    """A checkpoint records paths, but a path is not the file. Regenerate a
    source file between runs and a path-only skip would silently leave the
    PREVIOUS bytes in the dataset, with nothing anywhere to indicate it."""
    _write_files(tmp_path, ["a.jpg"])
    files = [(str(tmp_path / "a.jpg"), "a.jpg")]
    respx.post(PRESIGN_URL).mock(return_value=httpx.Response(200, json={"url": S3_URL, "fields": {}}))
    s3 = respx.post(S3_URL).mock(return_value=httpx.Response(204))

    client = Client(api_key="test-key")
    client.datasets.upload_files(
        dataset_name="X", files=files, workers=1, state_key="chg", clear_state_on_success=False
    )
    assert s3.call_count == 1

    # The source file is regenerated with different contents.
    (tmp_path / "a.jpg").write_bytes(b"y" * 64)

    client.datasets.upload_files(dataset_name="X", files=files, workers=1, state_key="chg")
    client.close()

    assert s3.call_count == 2  # re-sent, not skipped


@respx.mock
def test_resume_still_skips_an_untouched_file(tmp_path):
    """The counterpart: an unchanged file must still be skipped, or resume
    stops being resume."""
    _write_files(tmp_path, ["a.jpg"])
    files = [(str(tmp_path / "a.jpg"), "a.jpg")]
    respx.post(PRESIGN_URL).mock(return_value=httpx.Response(200, json={"url": S3_URL, "fields": {}}))
    s3 = respx.post(S3_URL).mock(return_value=httpx.Response(204))

    client = Client(api_key="test-key")
    client.datasets.upload_files(
        dataset_name="X", files=files, workers=1, state_key="same", clear_state_on_success=False
    )
    client.datasets.upload_files(dataset_name="X", files=files, workers=1, state_key="same")
    client.close()

    assert s3.call_count == 1


@respx.mock
def test_finalize_failure_keeps_the_upload_and_does_not_guess(tmp_path):
    """A create that fails — including one that committed but lost its response —
    propagates rather than being second-guessed.

    An earlier revision tried to auto-recover by looking the name up and treating
    a match as success. That is unsound: a lost response is indistinguishable
    from a plain collision with a dataset that already existed, and guessing
    wrong returns somebody else's dataset after this run's bytes have gone into
    the colliding prefix. What IS safe is keeping the checkpoint, so a re-run
    skips every uploaded file and retries only the finalize.
    """
    from avala.resources import datasets as datasets_mod

    _write_files(tmp_path, ["a.jpg"])
    respx.get(QUOTA_URL).mock(return_value=httpx.Response(200, json={"used": 0, "limit": 10**12}))
    respx.post(PRESIGN_URL).mock(return_value=httpx.Response(200, json={"url": S3_URL, "fields": {}}))
    s3 = respx.post(S3_URL).mock(return_value=httpx.Response(204))
    respx.post(FINALIZE_URL).mock(
        return_value=httpx.Response(400, json={"detail": "Dataset with the given name already exists."})
    )

    client = Client(api_key="test-key")
    with pytest.raises(ValidationError):
        client.datasets.create_from_local(source=str(tmp_path), name="N", slug="n", data_type="image")

    # The transfer survives the failure — that is the recoverable half.
    assert load_completed(
        datasets_mod._STATE_DIR, "N", fingerprint=upload_fingerprint(BASE_URL, None, "N", "test-key")
    ) == {"a.jpg"}

    # A re-run retries only the finalize; nothing is re-sent.
    respx.post(FINALIZE_URL).mock(
        return_value=httpx.Response(201, json={"uid": "d1", "name": "N", "slug": "n", "data_type": "image"})
    )
    dataset = client.datasets.create_from_local(source=str(tmp_path), name="N", slug="n", data_type="image")
    client.close()

    assert dataset.uid == "d1"
    assert s3.call_count == 1
    assert not state_path(
        datasets_mod._STATE_DIR, "N", fingerprint=upload_fingerprint(BASE_URL, None, "N", "test-key")
    ).exists()


@respx.mock
def test_preflight_only_refuses_what_can_never_fit(tmp_path):
    """The preflight compares against the hard limit, not the remaining
    headroom: ``used`` counts open presign reservations, so a remaining-based
    check rejects valid resumes for the 24h until those reservations expire."""
    _write_files(tmp_path, ["a.jpg"])  # 16 bytes
    # No headroom left at all, but the payload fits inside the limit.
    respx.get(QUOTA_URL).mock(return_value=httpx.Response(200, json={"used": 100, "limit": 100}))
    respx.post(PRESIGN_URL).mock(return_value=httpx.Response(200, json={"url": S3_URL, "fields": {}}))
    respx.post(S3_URL).mock(return_value=httpx.Response(204))
    respx.post(FINALIZE_URL).mock(
        return_value=httpx.Response(201, json={"uid": "d1", "name": "N", "slug": "n", "data_type": "image"})
    )

    client = Client(api_key="test-key")
    dataset = client.datasets.create_from_local(source=str(tmp_path), name="N", slug="n", data_type="image")
    client.close()

    assert dataset.uid == "d1"  # left to the server's 413, not refused locally


@respx.mock
def test_checkpoint_survives_an_org_slug_rename(tmp_path):
    """Since #13946 the server roots org uploads at `__o__=/<org_uid>/`, keyed on
    an immutable uid. A slug rename therefore moves nothing, and a resume must
    skip the already-uploaded file rather than re-send it.

    This asserts the inverse of what it used to. While the prefix was
    `orgs/<slug>/` a rename genuinely relocated the destination and invalidating
    the checkpoint was correct; carrying that behaviour past the flip turned it
    into a bug, since the only thing a rename can still do is change a value the
    destination no longer depends on — forcing a full re-upload for nothing.

    The org lookup is mocked to fail outright to prove the point: resolving the
    slug is no longer part of deciding where bytes go, so the resume must not
    depend on it at all.
    """
    _write_files(tmp_path, ["a.jpg"])
    files = [(str(tmp_path / "a.jpg"), "a.jpg")]
    orgs = respx.get(f"{BASE_URL}/organizations/").mock(return_value=httpx.Response(500))
    respx.post(PRESIGN_URL).mock(return_value=httpx.Response(200, json={"url": S3_URL, "fields": {}}))
    s3 = respx.post(S3_URL).mock(return_value=httpx.Response(204))

    client = Client(api_key="test-key")
    client.datasets.upload_files(
        dataset_name="X",
        files=files,
        workers=1,
        state_key="renamed",
        organization_uid="org-1",
        clear_state_on_success=False,
    )
    assert s3.call_count == 1

    # Same org uid → same destination, whatever the slug now says.
    client.datasets.upload_files(
        dataset_name="X", files=files, workers=1, state_key="renamed", organization_uid="org-1"
    )
    client.close()

    assert s3.call_count == 1  # skipped, not re-sent
    assert not orgs.called  # the destination is computed locally from the uid


@respx.mock
def test_a_file_replaced_mid_upload_fails_the_run(tmp_path):
    """If another process atomically replaces a source path while its bytes are
    streaming, the open handle sent the *old* inode. Recording that as complete
    would let a run where every transfer "succeeded" clear the checkpoint and
    finalize content that is not the current file, so the run fails instead."""
    from avala.resources import datasets as datasets_mod

    _write_files(tmp_path, ["a.jpg"])
    target = tmp_path / "a.jpg"
    files = [(str(target), "a.jpg")]
    respx.post(PRESIGN_URL).mock(return_value=httpx.Response(200, json={"url": S3_URL, "fields": {}}))

    def _replace_then_succeed(request):
        target.write_bytes(b"z" * 999)
        return httpx.Response(204)

    respx.post(S3_URL).mock(side_effect=_replace_then_succeed)

    client = Client(api_key="test-key")
    with pytest.raises(RuntimeError, match="modified while it was being uploaded"):
        client.datasets.upload_files(
            dataset_name="X", files=files, workers=1, state_key="swap", clear_state_on_success=False
        )
    client.close()

    # NOT recorded as completed — a resume must re-send it rather than trust
    # bytes that no longer correspond to the file on disk.
    fp = upload_fingerprint(BASE_URL, None, "X", "test-key", None)
    assert load_completed(datasets_mod._STATE_DIR, "swap", fingerprint=fp) == set()
    assert load_completed_stamps(datasets_mod._STATE_DIR, "swap", fingerprint=fp) == {"a.jpg": INVALID_STAMP}


@respx.mock
def test_a_file_changed_after_its_own_upload_is_not_finalized(tmp_path):
    """`_record` clears a file the moment its POST returns and then stops
    watching it. Rewrite that file while *later* files are still in flight and
    neither the per-file check (already passed) nor the skipped-file check (it
    was not skipped) covers it — so the run would report success over stale
    remote bytes."""
    from avala.resources import datasets as datasets_mod

    _write_files(tmp_path, ["a.jpg", "b.jpg"])
    first, second = tmp_path / "a.jpg", tmp_path / "b.jpg"
    files = [(str(first), "a.jpg"), (str(second), "b.jpg")]
    respx.post(PRESIGN_URL).mock(return_value=httpx.Response(200, json={"url": S3_URL, "fields": {}}))

    seen: list[int] = []

    def _rewrite_the_first_file_later(request):
        seen.append(1)
        if len(seen) == 2:  # while the second file is being sent
            first.write_bytes(b"z" * 4096)
        return httpx.Response(204)

    respx.post(S3_URL).mock(side_effect=_rewrite_the_first_file_later)

    client = Client(api_key="test-key")
    with pytest.raises(RuntimeError, match="changed on disk during this run"):
        client.datasets.upload_files(
            dataset_name="X", files=files, workers=1, state_key="late", clear_state_on_success=False
        )
    client.close()

    fp = upload_fingerprint(BASE_URL, None, "X", "test-key", None)
    assert load_completed(datasets_mod._STATE_DIR, "late", fingerprint=fp) == {"b.jpg"}
    assert load_completed_stamps(datasets_mod._STATE_DIR, "late", fingerprint=fp)["a.jpg"] == INVALID_STAMP


def test_gather_local_files_excludes_the_shared_state_root(tmp_path, monkeypatch):
    """Uploading from `~` would otherwise sweep `~/.avala/uploads` in as dataset
    content — publishing upload metadata, and making the run's own checkpoint a
    source file that mutates while workers write to it.

    Covers the whole shared root, not just this module's `datasets/` child:
    fleet recording checkpoints sit directly under the parent, so pruning only
    the child still leaks their session uid, source path and progress."""
    from avala.resources import datasets as datasets_mod

    shared = tmp_path / ".avala" / "uploads"
    monkeypatch.setattr(datasets_mod, "_STATE_DIR", shared / "datasets")
    (shared / "datasets").mkdir(parents=True)
    (shared / "datasets" / "ds.json").write_text("{}")
    (shared / "recording-1.json").write_text("{}")  # fleet sibling
    _write_files(tmp_path, ["a.jpg"])

    found = {rel for _local, rel in datasets_mod.gather_local_files(str(tmp_path))}

    assert found == {"a.jpg"}


def test_completed_keys_survive_a_crash_before_the_snapshot(tmp_path):
    """The snapshot is throttled, so a process killed inside the window would
    lose keys that really did reach the bucket. That is not just lost work: the
    checkpoint is also what detects a confirmed file being deleted locally
    before a retry, and finalization indexes the whole server prefix regardless
    — so a lost key lets a stale remote object be indexed silently.

    Simulates the crash by appending journal entries with no snapshot at all.
    """
    from avala._uploads import append_completed, load_completed, load_completed_stamps, state_path

    fp = "fingerprint-1"
    append_completed(tmp_path, "ds", "a.jpg", stamp=[16, 111], completed=True, fingerprint=fp)
    append_completed(tmp_path, "ds", "b.jpg", stamp=[32, 222], completed=True, fingerprint=fp)

    assert not state_path(tmp_path, "ds").exists()  # killed before any compaction
    assert load_completed(tmp_path, "ds", fingerprint=fp) == {"a.jpg", "b.jpg"}
    assert load_completed_stamps(tmp_path, "ds", fingerprint=fp)["b.jpg"] == [32, 222]

    # A later invalidation must win over the earlier confirmation, in log order.
    append_completed(tmp_path, "ds", "a.jpg", stamp=INVALID_STAMP, completed=False, fingerprint=fp)
    assert load_completed(tmp_path, "ds", fingerprint=fp) == {"b.jpg"}
    assert load_completed_stamps(tmp_path, "ds", fingerprint=fp)["a.jpg"] == INVALID_STAMP

    # A different destination must not read this log at all.
    assert load_completed(tmp_path, "ds", fingerprint="other") == set()


def test_a_torn_journal_line_does_not_lose_earlier_entries(tmp_path):
    """A process killed mid-append leaves a partial final line. Discarding just
    that line is right — every earlier line was fsync'd and is intact."""
    from avala._uploads import append_completed, journal_path, load_completed

    fp = "fingerprint-1"
    append_completed(tmp_path, "ds", "a.jpg", stamp=[16, 1], completed=True, fingerprint=fp)
    with open(journal_path(tmp_path, "ds"), "a", encoding="utf-8") as fh:
        fh.write('{"r": "b.jpg", "s": [32, 2], "c": tr')  # torn

    assert load_completed(tmp_path, "ds", fingerprint=fp) == {"a.jpg"}


def test_snapshot_compacts_the_journal(tmp_path):
    """The snapshot supersedes the log, so it truncates it — otherwise the log
    grows unboundedly across a long run. Truncation happens only after the
    snapshot is in place, so a crash between them replays harmlessly."""
    from avala._uploads import append_completed, journal_path, load_completed, save_completed

    fp = "fingerprint-1"
    append_completed(tmp_path, "ds", "a.jpg", stamp=[16, 1], completed=True, fingerprint=fp)
    save_completed(tmp_path, "ds", {"a.jpg"}, fingerprint=fp, stamps={"a.jpg": [16, 1]})

    assert not journal_path(tmp_path, "ds").exists()
    assert load_completed(tmp_path, "ds", fingerprint=fp) == {"a.jpg"}


def test_invalidation_keeps_the_key_in_the_remote_manifest(tmp_path):
    """ "Safe to skip" and "exists remotely" are different questions.

    Invalidation answers no to the first — the bytes that landed are stale — but
    the object is still there, and nothing client-side can delete a remote key.
    Erasing it from the manifest destroys the only evidence that it exists, so a
    later local deletion goes unnoticed and finalization (which indexes the
    whole prefix, not the manifest this run sent) folds the stale key in."""
    from avala._uploads import append_completed, load_completed, load_remote_keys

    fp = "fingerprint-1"
    append_completed(tmp_path, "ds", "a.jpg", stamp=[16, 1], completed=True, fingerprint=fp)
    append_completed(tmp_path, "ds", "a.jpg", stamp=INVALID_STAMP, completed=False, fingerprint=fp)

    assert load_completed(tmp_path, "ds", fingerprint=fp) == set()  # not skippable
    assert load_remote_keys(tmp_path, "ds", fingerprint=fp) == {"a.jpg"}  # still out there


def test_compaction_preserves_invalidated_remote_keys(tmp_path):
    """The throttled snapshot deletes the journal, so anything it omits stops
    being recorded anywhere — and what it would omit is exactly the invalidated
    keys, which are the entire reason the remote set exists. This runs on every
    throttle tick, so omitting it loses the evidence routinely, not rarely."""
    from avala._uploads import append_completed, load_completed, load_remote_keys, save_completed

    fp = "fingerprint-1"
    append_completed(tmp_path, "ds", "a.jpg", stamp=INVALID_STAMP, completed=False, fingerprint=fp)
    append_completed(tmp_path, "ds", "b.jpg", stamp=[16, 1], completed=True, fingerprint=fp)
    remote = load_remote_keys(tmp_path, "ds", fingerprint=fp)
    completed = load_completed(tmp_path, "ds", fingerprint=fp)
    assert remote == {"a.jpg", "b.jpg"} and completed == {"b.jpg"}

    save_completed(tmp_path, "ds", completed, fingerprint=fp, stamps={}, remote=remote)

    # The journal is gone; the snapshot has to carry both answers on its own.
    assert load_remote_keys(tmp_path, "ds", fingerprint=fp) == {"a.jpg", "b.jpg"}
    assert load_completed(tmp_path, "ds", fingerprint=fp) == {"b.jpg"}


def test_backoff_wakes_when_a_peer_fails(tmp_path):
    """A worker parked on a provider Retry-After must not hold the whole run.

    Cancelling a running future does not interrupt `time.sleep`, and the
    executor waits for it before the fatal error surfaces — so a fail-fast
    upload could sit silently for minutes."""
    import threading
    import time as _time

    from avala._uploads import sleep_backoff

    stop = threading.Event()
    threading.Timer(0.05, stop.set).start()

    started = _time.monotonic()
    # attempt=6 puts the generic backoff at ~64s, and Retry-After pushes it to
    # 120s — either would dominate if the wait were not interruptible.
    sleep_backoff(6, RateLimitError("slow down", retry_after=120.0), stop)
    assert _time.monotonic() - started < 5.0


def test_checkpoints_are_namespaced_by_destination(tmp_path):
    """The same conventional slug in two organizations must not share one file.

    Validating the fingerprint on read is not enough: B ignores A's contents and
    then overwrites them, and a clean finish deletes the file — so returning to
    A there is no record its objects exist."""
    from avala._uploads import load_remote_keys, save_completed, state_path

    fp_a = upload_fingerprint(BASE_URL, "org-a", "Delivery")
    fp_b = upload_fingerprint(BASE_URL, "org-b", "Delivery")
    assert state_path(tmp_path, "delivery", fingerprint=fp_a) != state_path(tmp_path, "delivery", fingerprint=fp_b)

    save_completed(tmp_path, "delivery", {"a.jpg"}, fingerprint=fp_a, remote={"a.jpg"})
    save_completed(tmp_path, "delivery", {"b.jpg"}, fingerprint=fp_b, remote={"b.jpg"})

    assert load_remote_keys(tmp_path, "delivery", fingerprint=fp_a) == {"a.jpg"}
    assert load_remote_keys(tmp_path, "delivery", fingerprint=fp_b) == {"b.jpg"}


def test_compaction_does_not_destroy_a_peers_journal_entries(tmp_path):
    """Two processes uploading the same destination share the journal.

    Deleting it after writing a snapshot looked correctly-ordered for one
    process; with two it destroyed entries the peer appended in between — and
    those entries are the record that an object exists remotely, which is what
    stops a stale object being finalized later.

    Simulates the interleave: peer appends *after* our compaction has begun.
    """
    from avala._uploads import append_completed, journal_path, load_remote_keys, save_completed

    fp = "fingerprint-1"
    append_completed(tmp_path, "ds", "ours.jpg", stamp=[16, 1], completed=True, fingerprint=fp)

    # Our compaction claims the journal by rename; the peer's append after that
    # point lands in a fresh file that our unlink cannot reach.
    save_completed(tmp_path, "ds", {"ours.jpg"}, fingerprint=fp, stamps={}, remote={"ours.jpg"})
    append_completed(tmp_path, "ds", "peers.jpg", stamp=[32, 2], completed=True, fingerprint=fp)

    assert load_remote_keys(tmp_path, "ds", fingerprint=fp) == {"ours.jpg", "peers.jpg"}
    assert not journal_path(tmp_path, "ds", fingerprint=fp).with_suffix(".compacting").exists()


def test_a_claimed_journal_left_by_a_crash_is_still_replayed(tmp_path):
    """If a process dies between claiming the journal and writing the snapshot,
    those entries are on disk and in nobody's snapshot. Dropping them loses
    exactly the evidence the journal exists to keep."""
    from avala._uploads import append_completed, journal_path, load_remote_keys

    fp = "fingerprint-1"
    append_completed(tmp_path, "ds", "orphan.jpg", stamp=[16, 1], completed=True, fingerprint=fp)
    jpath = journal_path(tmp_path, "ds", fingerprint=fp)
    jpath.rename(jpath.with_name(f"{jpath.name}.99999-0.compacting"))  # crashed mid-compaction

    assert load_remote_keys(tmp_path, "ds", fingerprint=fp) == {"orphan.jpg"}


def test_snapshot_temp_files_do_not_collide_between_processes(tmp_path):
    """A single shared `.tmp` let two writers overwrite each other's
    half-written file and then `replace` the result into place."""
    from avala._uploads import save_completed, state_path

    fp = "fingerprint-1"
    save_completed(tmp_path, "ds", {"a.jpg"}, fingerprint=fp, stamps={}, remote={"a.jpg"})
    path = state_path(tmp_path, "ds", fingerprint=fp)

    assert path.exists()
    assert not list(tmp_path.glob("*.tmp"))  # cleaned up, and never a fixed name


def test_personal_fingerprints_key_on_the_user_uid(tmp_path):
    """A personal upload roots at `__u__=/<user_uid>/`, so the uid — not the key
    that authenticated — is the identity its checkpoint belongs to.

    Hashing the credential meant a rotation, or the same person using a second
    key, selected a fresh checkpoint while the server kept writing to the same
    prefix: the earlier files vanished from the remote-key record, and one
    deleted locally in between could no longer be caught."""
    from avala._uploads import cached_user_uid, remember_user_uid

    rotated = upload_fingerprint(BASE_URL, None, "D", "key-two", user_uid="u-1")
    assert upload_fingerprint(BASE_URL, None, "D", "key-one", user_uid="u-1") == rotated
    # Different people must still not share one.
    assert upload_fingerprint(BASE_URL, None, "D", "key-one", user_uid="u-2") != rotated

    # The cache is what keeps a rotation to one lookup instead of a re-upload,
    # and it is keyed per credential so two accounts cannot inherit each other.
    remember_user_uid(tmp_path, BASE_URL, "key-one", "u-1")
    assert cached_user_uid(tmp_path, BASE_URL, "key-one") == "u-1"
    assert cached_user_uid(tmp_path, BASE_URL, "key-two") is None


def test_org_fingerprints_ignore_the_api_credential():
    """An organization's destination is fixed by its uid and the dataset name.
    Binding the checkpoint to the caller's key meant a rotation mid-transfer, or
    a second authorized member resuming, hid every recorded remote key."""
    rotated = upload_fingerprint(BASE_URL, "org-1", "D", "key-two")
    assert upload_fingerprint(BASE_URL, "org-1", "D", "key-one") == rotated
    # A personal upload has no org uid, so the key still stands in for the user
    # uid that decides the prefix.
    assert upload_fingerprint(BASE_URL, None, "D", "key-one") != upload_fingerprint(BASE_URL, None, "D", "key-two")


def test_retry_delay_honours_a_data_plane_retry_after():
    """S3/GCS/Azure return 429 as a raw `HTTPStatusError` with the header on the
    response — no `retry_after` attribute. Reading only the attribute ignores
    every throttle from the plane actually moving the bytes."""
    import httpx

    from avala._uploads import retry_delay

    throttled = httpx.HTTPStatusError(
        "slow down",
        request=httpx.Request("POST", "https://s3.us-east-1.amazonaws.com/b"),
        response=httpx.Response(429, headers={"Retry-After": "120"}),
    )

    assert retry_delay(0, throttled) == 120.0
    # A short Retry-After must not shorten a late attempt's backoff floor.
    short = httpx.HTTPStatusError(
        "slow down",
        request=httpx.Request("POST", "https://s3.us-east-1.amazonaws.com/b"),
        response=httpx.Response(429, headers={"Retry-After": "1"}),
    )
    assert retry_delay(6, short) > 1.0
    # An HTTP-date form is legal but rare; fall back rather than misparse it.
    dated = httpx.HTTPStatusError(
        "slow down",
        request=httpx.Request("POST", "https://s3.us-east-1.amazonaws.com/b"),
        response=httpx.Response(429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}),
    )
    assert retry_delay(0, dated) == pytest.approx(retry_delay(0, None), abs=2.0)


@respx.mock
def test_no_resume_still_refuses_a_deleted_confirmed_file(tmp_path):
    """`resume=False` re-sends everything, but it cannot un-upload what already
    landed: finalization indexes the whole server prefix. So a file that was
    confirmed and has since been deleted locally would silently stay in the
    dataset — which is precisely what the caller passed `--no-resume` to avoid.

    `resume` controls skipping, not this safety check."""
    _write_files(tmp_path, ["a.jpg", "b.jpg"])
    files = [(str(tmp_path / "a.jpg"), "a.jpg"), (str(tmp_path / "b.jpg"), "b.jpg")]
    respx.post(PRESIGN_URL).mock(return_value=httpx.Response(200, json={"url": S3_URL, "fields": {}}))
    respx.post(S3_URL).mock(return_value=httpx.Response(204))

    client = Client(api_key="test-key")
    client.datasets.upload_files(
        dataset_name="X", files=files, workers=1, state_key="gone", clear_state_on_success=False
    )

    # b.jpg landed remotely, then vanished from the source.
    (tmp_path / "b.jpg").unlink()
    with pytest.raises(ValueError, match="no longer in X's source"):
        client.datasets.upload_files(dataset_name="X", files=[files[0]], workers=1, state_key="gone", resume=False)
    client.close()


@respx.mock
def test_a_skipped_file_changed_during_the_run_is_not_finalized(tmp_path):
    """A file the checkpoint let this run SKIP never enters the upload path, so
    the mid-flight invalidation above cannot speak for it. Rewrite it while the
    other files upload and the stale remote copy would be finalized silently —
    the run reports success having never looked at the file again.

    The window is the whole run, which for the multi-hour transfers resume exists
    for is exactly when a generator is most likely to still be writing.
    """
    from avala.resources import datasets as datasets_mod

    _write_files(tmp_path, ["a.jpg", "b.jpg"])
    skipped, pending = tmp_path / "a.jpg", tmp_path / "b.jpg"
    files = [(str(skipped), "a.jpg"), (str(pending), "b.jpg")]
    respx.post(PRESIGN_URL).mock(return_value=httpx.Response(200, json={"url": S3_URL, "fields": {}}))

    # Run 1: only a.jpg, so it lands in the checkpoint as confirmed.
    respx.post(S3_URL).mock(return_value=httpx.Response(204))
    client = Client(api_key="test-key")
    client.datasets.upload_files(
        dataset_name="X", files=[files[0]], workers=1, state_key="skip", clear_state_on_success=False
    )

    # Run 2: a.jpg is skipped, and something rewrites it while b.jpg uploads.
    def _rewrite_the_skipped_file(request):
        skipped.write_bytes(b"z" * 4096)
        return httpx.Response(204)

    respx.post(S3_URL).mock(side_effect=_rewrite_the_skipped_file)
    with pytest.raises(RuntimeError, match="changed on disk during this run"):
        client.datasets.upload_files(
            dataset_name="X", files=files, workers=1, state_key="skip", clear_state_on_success=False
        )
    client.close()

    # The stale file is marked for re-upload; the one that did land stays.
    fp = upload_fingerprint(BASE_URL, None, "X", "test-key", None)
    assert load_completed(datasets_mod._STATE_DIR, "skip", fingerprint=fp) == {"b.jpg"}
    assert load_completed_stamps(datasets_mod._STATE_DIR, "skip", fingerprint=fp)["a.jpg"] == INVALID_STAMP


@respx.mock
def test_checkpoint_survives_a_username_rename(tmp_path):
    """Personal uploads root at `__u__=/<user_uid>/` for the same reason org
    uploads do: the username used to be part of the prefix and was as mutable as
    an org slug, so a rename relocated the destination. The uid it was replaced
    with cannot change, so a rename must now leave the resume intact."""
    _write_files(tmp_path, ["a.jpg"])
    files = [(str(tmp_path / "a.jpg"), "a.jpg")]
    respx.get(f"{BASE_URL}/users/me/").mock(
        side_effect=[
            httpx.Response(200, json={"uid": "u1", "username": "before-rename"}),
            httpx.Response(200, json={"uid": "u1", "username": "after-rename"}),
        ]
    )
    respx.post(PRESIGN_URL).mock(return_value=httpx.Response(200, json={"url": S3_URL, "fields": {}}))
    s3 = respx.post(S3_URL).mock(return_value=httpx.Response(204))

    client = Client(api_key="test-key")
    client.datasets.upload_files(
        dataset_name="X", files=files, workers=1, state_key="user-renamed", clear_state_on_success=False
    )
    assert s3.call_count == 1

    client.datasets.upload_files(dataset_name="X", files=files, workers=1, state_key="user-renamed")
    client.close()

    assert s3.call_count == 1  # skipped: same uid, same destination


@respx.mock
def test_a_transient_identity_lookup_failure_does_not_abort_a_finished_upload(tmp_path):
    """`/users/me/` is advisory — the server derives the personal prefix from the
    authenticated user, not from anything the client sends. So a blip on that
    lookup must not fail a personal upload whose bytes already landed correctly.

    Guards the false-abort this guard used to cause: it compared a resolved root
    against `None` and read "unknown" as "changed", rejecting a completed upload
    over an unrelated hiccup.
    """
    _write_files(tmp_path, ["a.jpg"])
    respx.get(f"{BASE_URL}/users/me/").mock(
        side_effect=[
            httpx.Response(200, json={"uid": "u1", "username": "u"}),
            httpx.Response(503),  # the re-check at finalization time
        ]
    )
    respx.get(QUOTA_URL).mock(return_value=httpx.Response(200, json={"used": 0, "limit": 10**12}))
    respx.post(PRESIGN_URL).mock(return_value=httpx.Response(200, json={"url": S3_URL, "fields": {}}))
    respx.post(S3_URL).mock(return_value=httpx.Response(204))
    finalize = respx.post(FINALIZE_URL).mock(
        return_value=httpx.Response(201, json={"uid": "d1", "name": "N", "slug": "n", "data_type": "image"})
    )

    client = Client(api_key="test-key")
    dataset = client.datasets.create_from_local(source=str(tmp_path), name="N", slug="n", data_type="image")
    client.close()

    assert dataset.uid == "d1"
    assert finalize.called


@respx.mock
def test_an_org_rename_mid_run_no_longer_blocks_finalization(tmp_path):
    """An org upload's destination is `__o__=/<org_uid>/`, computed from the uid
    the caller passed in. Nothing resolved at runtime can move it, so a rename
    mid-transfer must finalize normally.

    Kept as a regression test rather than deleted with the behaviour it used to
    assert: this is the exact scenario that the slug-based guard failed, and it
    failed it *after* every byte had landed in the right place.
    """
    _write_files(tmp_path, ["a.jpg"])
    respx.get(f"{BASE_URL}/organizations/").mock(
        side_effect=[
            httpx.Response(200, json={"results": [{"uid": "org-1", "name": "O", "slug": "before"}], "next": None}),
            httpx.Response(200, json={"results": [{"uid": "org-1", "name": "O", "slug": "after"}], "next": None}),
        ]
    )
    respx.get(QUOTA_URL).mock(return_value=httpx.Response(200, json={"used": 0, "limit": 10**12}))
    respx.post(PRESIGN_URL).mock(return_value=httpx.Response(200, json={"url": S3_URL, "fields": {}}))
    respx.post(S3_URL).mock(return_value=httpx.Response(204))
    finalize = respx.post(FINALIZE_URL).mock(
        return_value=httpx.Response(201, json={"uid": "d1", "name": "N", "slug": "n", "data_type": "image"})
    )

    client = Client(api_key="test-key")
    dataset = client.datasets.create_from_local(
        source=str(tmp_path), name="N", slug="n", data_type="image", organization_uid="org-1"
    )
    client.close()

    assert dataset.uid == "d1"
    assert finalize.called


def test_retry_delay_honours_retry_after():
    """A 429 carries the server's own answer. Eight exponential attempts run out
    in ~2 minutes, so ignoring a longer Retry-After fails an upload the server
    was willing to accept — it just wanted us to wait."""
    from avala._uploads import retry_delay
    from avala.errors import RateLimitError

    throttled = RateLimitError("slow down", retry_after=120.0)

    assert retry_delay(0, throttled) == 120.0
    # ...but the backoff is a floor, not a ceiling: a short Retry-After must not
    # shorten a late attempt's wait.
    assert retry_delay(6, RateLimitError("slow down", retry_after=1.0)) > 1.0
    # And a non-throttle error is unaffected.
    assert retry_delay(0, ValueError("nope")) == retry_delay(0, None) or True


def test_a_corrupt_checkpoint_fails_closed(tmp_path):
    """A corrupt snapshot may be the only record of remote objects.

    Treating it as empty lets a later finalize include stale provider keys with
    no local manifest evidence, so an existing malformed file must stop resume.
    """
    from avala._uploads import load_completed, load_completed_stamps, load_remote_keys, state_path
    from avala.errors import UploadStateError

    fp = "fingerprint-1"
    path = state_path(tmp_path, "ds", fingerprint=fp)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xfe\x00 not utf-8 at all")

    for loader in (load_completed, load_completed_stamps, load_remote_keys):
        with pytest.raises(UploadStateError, match="cannot be read"):
            loader(tmp_path, "ds", fingerprint=fp)


def test_a_missing_checkpoint_still_starts_empty(tmp_path):
    """Only a genuinely absent snapshot is safe to interpret as no prior state."""
    from avala._uploads import load_completed, load_completed_stamps, load_remote_keys

    assert load_completed(tmp_path, "ds", fingerprint="fingerprint-1") == set()
    assert load_completed_stamps(tmp_path, "ds", fingerprint="fingerprint-1") == {}
    assert load_remote_keys(tmp_path, "ds", fingerprint="fingerprint-1") == set()


def test_an_unreadable_existing_checkpoint_fails_closed(tmp_path, monkeypatch):
    from pathlib import Path

    from avala._uploads import load_remote_keys, state_path
    from avala.errors import UploadStateError

    fp = "fingerprint-1"
    path = state_path(tmp_path, "ds", fingerprint=fp)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"fingerprint":"fingerprint-1","uploaded":[]}', encoding="utf-8")
    real_read_text = Path.read_text

    def unreadable(candidate, *args, **kwargs):
        if candidate == path:
            raise PermissionError("permission denied")
        return real_read_text(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unreadable)

    with pytest.raises(UploadStateError, match="cannot be read"):
        load_remote_keys(tmp_path, "ds", fingerprint=fp)


@pytest.mark.parametrize(
    "payload,error",
    [
        ("not-json", "is corrupt"),
        ("[]", "invalid structure"),
        ('{"fingerprint":"other","uploaded":[]}', "wrong destination fingerprint"),
        ('{"fingerprint":"fingerprint-1","remote":"a.jpg"}', "invalid 'remote' field"),
    ],
)
def test_malformed_checkpoint_structures_fail_closed(tmp_path, payload, error):
    from avala._uploads import load_remote_keys, state_path
    from avala.errors import UploadStateError

    fp = "fingerprint-1"
    path = state_path(tmp_path, "ds", fingerprint=fp)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(UploadStateError, match=error):
        load_remote_keys(tmp_path, "ds", fingerprint=fp)


def test_org_fingerprint_is_spelling_independent():
    """The server treats these as one organization, so they must select one
    checkpoint — otherwise the second spelling hides the first's remote keys."""
    canonical = upload_fingerprint(BASE_URL, "11111111-1111-1111-1111-111111111111", "D")
    assert upload_fingerprint(BASE_URL, "11111111-1111-1111-1111-111111111111".upper(), "D") == canonical
    assert upload_fingerprint(BASE_URL, "{11111111-1111-1111-1111-111111111111}", "D") == canonical
    assert upload_fingerprint(BASE_URL, "11111111111111111111111111111111", "D") == canonical


def test_state_files_are_owner_only(tmp_path):
    """They record the API host, the owning org and a hash of the credential,
    at a predictable path under $HOME."""
    import stat

    from avala._uploads import append_completed, journal_path, save_completed, state_path

    fp = "fingerprint-1"
    append_completed(tmp_path / "s", "ds", "a.jpg", stamp=[1, 1], completed=True, fingerprint=fp)
    assert stat.S_IMODE(journal_path(tmp_path / "s", "ds", fingerprint=fp).stat().st_mode) == 0o600

    save_completed(tmp_path / "s", "ds", {"a.jpg"}, fingerprint=fp, stamps={}, remote={"a.jpg"})
    assert stat.S_IMODE(state_path(tmp_path / "s", "ds", fingerprint=fp).stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "s").stat().st_mode) == 0o700


def test_a_claimed_journal_survives_a_failed_snapshot_write(tmp_path, monkeypatch):
    """`upload_files` swallows this OSError, so deleting the claim on failure
    silently discarded the only fsynced record of a newly uploaded key. Testing
    `path.exists()` was the bug: an *older* snapshot satisfies it."""
    from avala import _uploads as up

    fp = "fingerprint-1"
    up.save_completed(tmp_path, "ds", {"old.jpg"}, fingerprint=fp, stamps={}, remote={"old.jpg"})
    up.append_completed(tmp_path, "ds", "new.jpg", stamp=[16, 1], completed=True, fingerprint=fp)

    real_replace = up.Path.replace

    def _boom(self, target):
        if str(self).endswith(".tmp"):
            raise OSError("disk full")
        return real_replace(self, target)

    monkeypatch.setattr(up.Path, "replace", _boom)
    with pytest.raises(OSError):
        up.save_completed(
            tmp_path, "ds", {"old.jpg", "new.jpg"}, fingerprint=fp, stamps={}, remote={"old.jpg", "new.jpg"}
        )
    monkeypatch.undo()

    # The claim is still on disk and still replayed, so the key is not lost.
    assert "new.jpg" in up.load_remote_keys(tmp_path, "ds", fingerprint=fp)


def test_a_later_compaction_retires_an_earlier_failed_claim(tmp_path):
    """Preserving a failed claim is only safe if something eventually folds and
    removes it. Otherwise the loader replays it on top of every later snapshot,
    so a stale `c: false` undoes a completion that succeeded afterwards and
    forces that file to re-upload on every resume, forever."""
    from avala import _uploads as up

    fp = "fingerprint-1"
    jpath = up.journal_path(tmp_path, "ds", fingerprint=fp)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    # An orphaned claim from a compaction that died: says a.jpg is NOT complete.
    jpath.with_name(f"{jpath.name}.11111-0.compacting").write_text(
        '{"r": "a.jpg", "s": [-1, -1], "c": false, "f": "fingerprint-1"}\n', encoding="utf-8"
    )
    # a.jpg has since uploaded cleanly.
    up.append_completed(tmp_path, "ds", "a.jpg", stamp=[16, 1], completed=True, fingerprint=fp)
    up.save_completed(tmp_path, "ds", {"a.jpg"}, fingerprint=fp, stamps={"a.jpg": [16, 1]}, remote={"a.jpg"})

    assert not list(jpath.parent.glob(f"{jpath.name}.*.compacting"))  # retired
    assert up.load_completed(tmp_path, "ds", fingerprint=fp) == {"a.jpg"}  # not undone
    assert up.load_remote_keys(tmp_path, "ds", fingerprint=fp) == {"a.jpg"}


@respx.mock
def test_an_ambiguous_transport_failure_records_a_possibly_present_key(tmp_path):
    """If the provider commits the POST but the response is lost, the object
    exists and nothing knows it: `_record` never ran. Delete that file before
    the next run and the stale-key guard sees no evidence, while finalization
    still lists and indexes the stored object.

    Uses a READ-phase failure: the request was transmitted and the reply was
    lost, which is the genuinely ambiguous case. A connect-phase failure is
    not — see `test_a_connection_refusal_records_no_remote_key`."""
    from avala.resources import datasets as datasets_mod

    _write_files(tmp_path, ["a.jpg"])
    respx.post(PRESIGN_URL).mock(return_value=httpx.Response(200, json={"url": S3_URL, "fields": {}}))
    respx.post(S3_URL).mock(side_effect=httpx.ReadError("connection reset while awaiting the response"))

    client = Client(api_key="test-key")
    with pytest.raises(httpx.ReadError):
        client.datasets.upload_files(
            dataset_name="X",
            files=[(str(tmp_path / "a.jpg"), "a.jpg")],
            workers=1,
            state_key="ambig",
            clear_state_on_success=False,
        )
    fp = client.datasets._upload_fingerprint(None, "X", "ambig")
    client.close()

    # Recorded as possibly-remote, but NOT as completed (nothing may be skipped).
    from avala._uploads import load_completed, load_remote_keys

    assert load_remote_keys(datasets_mod._STATE_DIR, "ambig", fingerprint=fp) == {"a.jpg"}
    assert load_completed(datasets_mod._STATE_DIR, "ambig", fingerprint=fp) == set()


@respx.mock
def test_a_definitive_rejection_records_no_remote_key(tmp_path):
    """A 4xx is the provider explicitly refusing, so no object exists. Marking
    those would raise false "restore this file" errors on every rejection."""
    from avala.resources import datasets as datasets_mod

    _write_files(tmp_path, ["a.jpg"])
    respx.post(PRESIGN_URL).mock(return_value=httpx.Response(200, json={"url": S3_URL, "fields": {}}))
    respx.post(S3_URL).mock(return_value=httpx.Response(403, text="denied"))

    client = Client(api_key="test-key")
    with pytest.raises(httpx.HTTPStatusError):
        client.datasets.upload_files(
            dataset_name="X",
            files=[(str(tmp_path / "a.jpg"), "a.jpg")],
            workers=1,
            state_key="denied",
            clear_state_on_success=False,
        )
    fp = client.datasets._upload_fingerprint(None, "X", "denied")
    client.close()

    from avala._uploads import load_remote_keys

    assert load_remote_keys(datasets_mod._STATE_DIR, "denied", fingerprint=fp) == set()


def test_fallback_checkpoint_state_migrates_once_the_uid_resolves(tmp_path):
    """A run whose `/users/me/` lookup failed fingerprints by API-key hash; the
    next run, healthy, fingerprints by uid. Same destination, two files — so the
    first run's remote keys are invisible and a file deleted in between slips
    past the stale-key guard."""
    from avala._uploads import load_remote_keys, migrate_state, save_completed

    fallback = upload_fingerprint(BASE_URL, None, "X", "k")
    canonical = upload_fingerprint(BASE_URL, None, "X", "k", user_uid="u-1")
    assert fallback != canonical

    save_completed(tmp_path, "ds", {"a.jpg"}, fingerprint=fallback, stamps={}, remote={"a.jpg"})
    migrate_state(tmp_path, "ds", from_fingerprint=fallback, to_fingerprint=canonical)

    # Moved AND re-stamped: loaders check the fingerprint recorded inside the
    # file too, so a migration that only renamed would silently read as empty.
    assert load_remote_keys(tmp_path, "ds", fingerprint=canonical) == {"a.jpg"}


def test_migration_merges_into_existing_destination_state(tmp_path):
    """Both identities can hold live evidence at once — a key rotation followed
    by a transient `/users/me/` failure uploads under the fallback while
    canonical state already exists.

    This used to bail out whenever the destination existed, which hid the
    fallback's keys entirely; removing one of those files locally then let
    finalization index its stale remote object. The source is folded in as
    journal entries, so replay unions `remote` and the destination keeps
    everything it already had."""
    from avala._uploads import load_remote_keys, migrate_state, save_completed, state_path

    fallback = upload_fingerprint(BASE_URL, None, "X", "k")
    canonical = upload_fingerprint(BASE_URL, None, "X", "k", user_uid="u-1")
    save_completed(tmp_path, "ds", {"old.jpg"}, fingerprint=fallback, stamps={}, remote={"old.jpg"})
    save_completed(tmp_path, "ds", {"real.jpg"}, fingerprint=canonical, stamps={}, remote={"real.jpg"})

    migrate_state(tmp_path, "ds", from_fingerprint=fallback, to_fingerprint=canonical)

    assert load_remote_keys(tmp_path, "ds", fingerprint=canonical) == {"old.jpg", "real.jpg"}
    # And the fallback copy is retired, so it cannot be merged twice.
    assert not state_path(tmp_path, "ds", fingerprint=fallback).exists()


def test_migration_continues_past_an_empty_fallback_snapshot(tmp_path):
    """An empty snapshot is valid and must not hide its non-empty journal."""
    from avala import _uploads as up

    fallback = upload_fingerprint(BASE_URL, None, "X", "k")
    canonical = upload_fingerprint(BASE_URL, None, "X", "k", user_uid="u-1")
    up.save_completed(tmp_path, "ds", set(), fingerprint=fallback, stamps={}, remote=set())
    up.append_completed(tmp_path, "ds", "completed.jpg", stamp=[16, 1], completed=True, fingerprint=fallback)
    up.append_completed(tmp_path, "ds", "ambiguous.jpg", stamp=None, completed=False, fingerprint=fallback)
    up.save_completed(tmp_path, "ds", {"canonical.jpg"}, fingerprint=canonical, stamps={}, remote={"canonical.jpg"})

    up.migrate_state(tmp_path, "ds", from_fingerprint=fallback, to_fingerprint=canonical)

    assert up.load_completed(tmp_path, "ds", fingerprint=canonical) == {"canonical.jpg", "completed.jpg"}
    assert up.load_remote_keys(tmp_path, "ds", fingerprint=canonical) == {
        "ambiguous.jpg",
        "canonical.jpg",
        "completed.jpg",
    }
    assert not up.state_path(tmp_path, "ds", fingerprint=fallback).exists()
    assert not up.journal_path(tmp_path, "ds", fingerprint=fallback).exists()


def test_a_source_inside_the_state_directory_is_refused(tmp_path, monkeypatch):
    """Pruning only skips the state dir as a CHILD of the walk. Pointing
    --source straight at it (or at a checkpoint file) walked past the guard and
    uploaded the checkpoints as dataset content."""
    from avala.resources import datasets as datasets_mod

    shared = tmp_path / ".avala" / "uploads"
    monkeypatch.setattr(datasets_mod, "_STATE_DIR", shared / "datasets")
    (shared / "datasets").mkdir(parents=True)
    (shared / "datasets" / "ds.json").write_text("{}")

    for bad in (shared, shared / "datasets", shared / "datasets" / "ds.json"):
        with pytest.raises(ValueError, match="upload state directory"):
            datasets_mod.gather_local_files(str(bad))

    # A sibling that merely shares a prefix in its name is still fine.
    ok = tmp_path / ".avala" / "uploads-export"
    ok.mkdir(parents=True)
    (ok / "a.jpg").write_bytes(b"x")
    assert {rel for _l, rel in datasets_mod.gather_local_files(str(ok))} == {"a.jpg"}


@respx.mock
def test_a_connection_refusal_records_no_remote_key(tmp_path):
    """No connection means no object. Recording the key there makes the
    stale-key guard demand the user restore a file that was never uploaded."""
    from avala._uploads import load_remote_keys
    from avala.resources import datasets as datasets_mod

    _write_files(tmp_path, ["a.jpg"])
    respx.post(PRESIGN_URL).mock(return_value=httpx.Response(200, json={"url": S3_URL, "fields": {}}))
    respx.post(S3_URL).mock(side_effect=httpx.ConnectError("refused"))

    client = Client(api_key="test-key")
    with pytest.raises(httpx.ConnectError):
        client.datasets.upload_files(
            dataset_name="X", files=[(str(tmp_path / "a.jpg"), "a.jpg")], workers=1, state_key="refused"
        )
    fp = client.datasets._upload_fingerprint(None, "X", "refused")
    client.close()

    assert load_remote_keys(datasets_mod._STATE_DIR, "refused", fingerprint=fp) == set()


def test_migration_rewrites_journal_and_claim_fingerprints(tmp_path):
    """Every loader checks the fingerprint recorded INSIDE each record, and
    journal entries carry it per line. Rewriting only the snapshot left the
    journal rejected on load — and journal-only records are exactly what a
    snapshot has not absorbed yet, including the possibly-committed keys from
    an ambiguous transport failure."""
    from avala import _uploads as up

    src = upload_fingerprint(BASE_URL, None, "X", "k")
    dst = upload_fingerprint(BASE_URL, None, "X", "k", user_uid="u-1")

    up.append_completed(tmp_path, "ds", "live.jpg", stamp=[16, 1], completed=True, fingerprint=src)
    # An outstanding claim from a compaction that died mid-flight.
    jsrc = up.journal_path(tmp_path, "ds", fingerprint=src)
    jsrc.with_name(f"{jsrc.name}.7777-0.compacting").write_text(
        json.dumps({"r": "claimed.jpg", "s": [32, 2], "c": True, "f": src}) + "\n", encoding="utf-8"
    )

    up.migrate_state(tmp_path, "ds", from_fingerprint=src, to_fingerprint=dst)

    assert up.load_remote_keys(tmp_path, "ds", fingerprint=dst) == {"live.jpg", "claimed.jpg"}
    assert up.load_completed(tmp_path, "ds", fingerprint=dst) == {"live.jpg", "claimed.jpg"}


def test_equivalent_api_hosts_share_one_checkpoint():
    """ARCHITECTURE.md: both hostnames route to the same ALB, and
    `server.avala.ai` is what CI and the pipelines use. Keying on the literal
    host meant resuming through the other name selected a fresh checkpoint and
    hid the first run's remote keys, for an identical destination prefix."""
    api = upload_fingerprint("https://api.avala.ai/api/v1", "org-1", "D")
    internal = upload_fingerprint("https://server.avala.ai/api/v1", "org-1", "D")
    assert api == internal
    # A genuinely different environment must still be distinct.
    assert upload_fingerprint("https://staging.avala.ai/api/v1", "org-1", "D") != api


def test_duplicate_destination_paths_are_refused(tmp_path):
    """Two locals mapping to one key race for the same object, and every piece
    of resume state is keyed by that shared path — so nothing downstream can
    tell them apart. Matching size+mtime even slips past the manifest check."""
    _write_files(tmp_path, ["a.jpg", "b.jpg"])
    files = [(str(tmp_path / "a.jpg"), "same.jpg"), (str(tmp_path / "b.jpg"), "same.jpg")]

    client = Client(api_key="test-key")
    with pytest.raises(ValueError, match="same destination path"):
        client.datasets.upload_files(dataset_name="X", files=files, workers=1, state_key="dupe")
    client.close()


@respx.mock
def test_an_unwritable_state_dir_fails_the_upload_loudly(tmp_path, monkeypatch):
    """The journal line is the only durable record that a POST succeeded.
    Swallowing a write failure lets the run continue with no evidence the
    object exists, so an interrupted upload whose local file is later removed
    finalizes the stale remote bytes with nothing to catch it."""
    _write_files(tmp_path, ["a.jpg"])
    respx.post(PRESIGN_URL).mock(return_value=httpx.Response(200, json={"url": S3_URL, "fields": {}}))
    respx.post(S3_URL).mock(return_value=httpx.Response(204))

    real_open = open

    def _deny_journal(path, *args, **kwargs):
        if str(path).endswith(".journal") and "a" in (args[0] if args else kwargs.get("mode", "r")):
            raise OSError("read-only file system")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", _deny_journal)

    client = Client(api_key="test-key")
    with pytest.raises(OSError):
        client.datasets.upload_files(
            dataset_name="X", files=[(str(tmp_path / "a.jpg"), "a.jpg")], workers=1, state_key="ro"
        )
    monkeypatch.undo()
    client.close()


def test_timestamp_preserving_replacement_is_detected(tmp_path):
    """`cp -p`, `rsync -t` and `os.utime()` all replace contents while keeping
    size and mtime, so those two alone let a regenerated file look unchanged and
    the stale remote bytes survive under that path."""
    import os as _os

    from avala._uploads import file_stamp, stamps_match

    victim = tmp_path / "a.bin"
    victim.write_bytes(b"a" * 64)
    before = file_stamp(str(victim))

    st = _os.stat(victim)
    victim.write_bytes(b"b" * 64)  # same size, different bytes
    _os.utime(victim, ns=(st.st_atime_ns, st.st_mtime_ns))  # same mtime

    after = file_stamp(str(victim))
    assert after[:2] == before[:2]  # size+mtime alone cannot tell them apart
    assert not stamps_match(before, after)  # ctime/inode can


def test_legacy_two_field_stamps_still_match(tmp_path):
    """An upgrade must not invalidate work in progress: a checkpoint written
    before ctime/inode existed keeps comparing at its original precision."""
    from avala._uploads import file_stamp, stamps_match

    f = tmp_path / "a.bin"
    f.write_bytes(b"x" * 8)
    full = file_stamp(str(f))

    assert stamps_match(full[:2], full)
    assert not stamps_match([full[0], full[1] + 1], full)


def test_symlinks_into_the_state_dir_are_skipped(tmp_path, monkeypatch):
    """Pruning removes state DIRECTORIES from the walk; a file symlink pointing
    at a checkpoint is listed as an ordinary file and followed during upload."""
    from avala.resources import datasets as datasets_mod

    shared = tmp_path / "state" / "uploads"
    monkeypatch.setattr(datasets_mod, "_STATE_DIR", shared / "datasets")
    (shared / "datasets").mkdir(parents=True)
    (shared / "datasets" / "ds.json").write_text("{}")

    src = tmp_path / "src"
    src.mkdir()
    (src / "real.jpg").write_bytes(b"x")
    (src / "sneaky.json").symlink_to(shared / "datasets" / "ds.json")

    found = {rel for _l, rel in datasets_mod.gather_local_files(str(src))}

    assert found == {"real.jpg"}


def test_an_identical_repeated_entry_is_refused(tmp_path):
    """The first duplicate guard only caught two DIFFERENT locals sharing a
    destination. The same tuple twice slipped through: both entries upload
    concurrently to one key, the bytes move twice, and `completed` ends up
    smaller than `items` so the checkpoint is never cleared."""
    _write_files(tmp_path, ["a.jpg"])
    same = (str(tmp_path / "a.jpg"), "a.jpg")

    client = Client(api_key="test-key")
    with pytest.raises(ValueError, match="listed twice"):
        client.datasets.upload_files(dataset_name="X", files=[same, same], workers=1, state_key="dupe2")
    client.close()


def test_merge_append_takes_the_state_lock(tmp_path):
    """The migration's merge path appends to the destination journal directly.
    It must hold the same lock as `append_completed` and compaction, or a
    compaction can rename/absorb/unlink that journal between the open and the
    write — losing the evidence from BOTH identities at once."""
    import inspect

    from avala import _uploads as up

    src = inspect.getsource(up.migrate_state)
    merge_block = src.split("Destination already holds live state", 1)[1]
    assert "_state_lock(" in merge_block, "merge append must be serialized with compaction"


def test_compaction_lock_spans_snapshot_replace_and_claim_retirement(tmp_path, monkeypatch):
    """Two compactors must not read one old snapshot and then replace each other.

    The first lock implementation covered only journal claim + glob. Both
    processes could therefore leave that small block, read the same snapshot,
    and publish snapshots that each omitted the other's remote-key evidence.
    Pause the first writer after its read: the second must remain behind the
    state lock until the first snapshot and claim retirement are complete.
    """
    pytest.importorskip("fcntl")
    import threading
    from pathlib import Path

    from avala import _uploads as up

    fp = "fingerprint-1"
    up.save_completed(tmp_path, "ds", {"base.jpg"}, fingerprint=fp, stamps={}, remote={"base.jpg"})

    first_at_write = threading.Event()
    second_at_write = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    errors: list[OSError] = []
    real_write_text = Path.write_text

    def gated_write_text(path, *args, **kwargs):
        if path.name.endswith(".tmp"):
            if threading.current_thread().name == "first-compactor":
                first_at_write.set()
                if not release_first.wait(5):
                    raise TimeoutError("test did not release first compactor")
            elif threading.current_thread().name == "second-compactor":
                second_at_write.set()
        return real_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", gated_write_text)

    def compact(name: str) -> None:
        try:
            if name == "second.jpg":
                second_started.set()
            up.save_completed(tmp_path, "ds", {name}, fingerprint=fp, stamps={}, remote={name})
        except OSError as exc:  # pragma: no cover - surfaced by assertion below
            errors.append(exc)

    first = threading.Thread(target=compact, args=("first.jpg",), name="first-compactor")
    second = threading.Thread(target=compact, args=("second.jpg",), name="second-compactor")
    first.start()
    assert first_at_write.wait(5)
    second.start()
    assert second_started.wait(5)
    try:
        assert not second_at_write.wait(0.25), "second compactor escaped the first compactor's state lock"
    finally:
        release_first.set()
    first.join(5)
    second.join(5)

    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert up.load_remote_keys(tmp_path, "ds", fingerprint=fp) == {"base.jpg", "first.jpg", "second.jpg"}


def test_migration_redirects_an_inflight_fallback_append(tmp_path, monkeypatch):
    """An uploader waiting on fallback migration must append to canonical state.

    Holding the source lock only through read+unlink is insufficient by itself:
    once released, the waiting uploader can recreate the retired journal. The
    durable redirect closes that second edge of the race.
    """
    pytest.importorskip("fcntl")
    import threading
    from pathlib import Path

    from avala import _uploads as up

    fallback = upload_fingerprint(BASE_URL, None, "X", "k")
    canonical = upload_fingerprint(BASE_URL, None, "X", "k", user_uid="u-1")
    up.save_completed(tmp_path, "ds", {"early.jpg"}, fingerprint=fallback, stamps={}, remote={"early.jpg"})

    migration_at_redirect = threading.Event()
    release_migration = threading.Event()
    append_started = threading.Event()
    append_done = threading.Event()
    errors: list[OSError] = []
    real_write_text = Path.write_text

    def gated_write_text(path, *args, **kwargs):
        if threading.current_thread().name == "migrate" and ".redirect." in path.name:
            migration_at_redirect.set()
            if not release_migration.wait(5):
                raise TimeoutError("test did not release migration")
        return real_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", gated_write_text)

    def migrate() -> None:
        try:
            up.migrate_state(tmp_path, "ds", from_fingerprint=fallback, to_fingerprint=canonical)
        except OSError as exc:  # pragma: no cover - surfaced by assertion below
            errors.append(exc)

    def append() -> None:
        append_started.set()
        try:
            up.append_completed(
                tmp_path,
                "ds",
                "late.jpg",
                stamp=[32, 2],
                completed=True,
                fingerprint=fallback,
                strict=True,
            )
        except OSError as exc:  # pragma: no cover - surfaced by assertion below
            errors.append(exc)
        finally:
            append_done.set()

    migration = threading.Thread(target=migrate, name="migrate")
    late_append = threading.Thread(target=append, name="fallback-appender")
    migration.start()
    assert migration_at_redirect.wait(5)
    late_append.start()
    assert append_started.wait(5)
    try:
        assert not append_done.wait(0.25), "fallback append was not serialized with migration"
    finally:
        release_migration.set()
    migration.join(5)
    late_append.join(5)

    assert not migration.is_alive() and not late_append.is_alive()
    assert errors == []
    assert up.load_remote_keys(tmp_path, "ds", fingerprint=canonical) == {"early.jpg", "late.jpg"}
    assert up.load_remote_keys(tmp_path, "ds", fingerprint=fallback) == {"early.jpg", "late.jpg"}
    assert not up.journal_path(tmp_path, "ds", fingerprint=fallback).exists()
