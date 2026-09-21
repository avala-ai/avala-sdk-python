from uuid import uuid4

import httpx
import pytest
import respx

from avala._managed_upload import transfer_managed_upload
from avala._uploads import managed_upload_batch, migrate_state

URL = "https://0123456789abcdef0123456789abcdef.eu.r2.cloudflarestorage.com/b/key"


@respx.mock
def test_put_replay_verifies_completion_and_sends_length(tmp_path):
    source = tmp_path / "file"
    source.write_bytes(b"content")
    put = respx.put(URL).mock(return_value=httpx.Response(412))
    calls = []
    transfer_managed_upload(
        str(source),
        {"method": "PUT", "upload_uid": str(uuid4()), "url": URL},
        request=lambda *args, **kwargs: calls.append((args, kwargs)),
        timeout=httpx.Timeout(10),
        cancelled=lambda: False,
    )
    assert put.calls.last.request.headers["content-length"] == "7"
    assert put.calls.last.request.content == b"content"
    assert calls[0][0][1].endswith("/complete/")


@respx.mock
def test_multipart_resume_skips_verified_parts_and_sends_short_tail(tmp_path):
    part_size = 5 * 1024 * 1024
    source = tmp_path / "file"
    source.write_bytes(b"a" * part_size + b"tail")
    put = respx.put(URL).mock(return_value=httpx.Response(200))
    calls = []

    def request(method, path, **kwargs):
        calls.append((path, kwargs["json"]))
        return {"parts": [{"part_number": 2, "url": URL}]} if path.endswith("/parts/") else {}

    transferred = transfer_managed_upload(
        str(source),
        {
            "method": "MULTIPART",
            "upload_uid": str(uuid4()),
            "part_size": part_size,
            "part_count": 2,
            "uploaded_parts": [{"part_number": 1, "size": part_size, "etag": "verified"}],
        },
        request=request,
        timeout=httpx.Timeout(10),
        cancelled=lambda: False,
    )
    assert transferred == 4
    assert put.call_count == 1
    assert put.calls.last.request.content == b"tail"
    assert calls[0][1] == {"part_numbers": [2]}
    assert calls[1][0].endswith("/complete/")


@pytest.mark.parametrize("part_size,part_count", [(1, 2), (64 * 1024 * 1024, 10001), (5 * 1024 * 1024, 2)])
def test_invalid_multipart_geometry_never_signs(tmp_path, part_size, part_count):
    source = tmp_path / "file"
    source.write_bytes(b"a")

    def request(*args, **kwargs):
        pytest.fail("Invalid geometry must not issue requests")

    with pytest.raises(ValueError, match="geometry"):
        transfer_managed_upload(
            str(source),
            {"method": "MULTIPART", "upload_uid": str(uuid4()), "part_size": part_size, "part_count": part_count},
            request=request,
            timeout=httpx.Timeout(10),
            cancelled=lambda: False,
        )


def test_batch_identity_survives_retries_and_credential_identity_migration(tmp_path):
    batch = managed_upload_batch(tmp_path, "dataset", fingerprint="provisional")
    assert managed_upload_batch(tmp_path, "dataset", fingerprint="provisional") == batch
    migrate_state(tmp_path, "dataset", from_fingerprint="provisional", to_fingerprint="canonical")
    assert managed_upload_batch(tmp_path, "dataset", fingerprint="canonical") == batch
    assert managed_upload_batch(tmp_path, "dataset", fingerprint="provisional") == batch


def test_source_stamp_change_cannot_reuse_managed_parts(tmp_path):
    from avala._uploads import bind_managed_source
    from avala.errors import UploadStateError

    bind_managed_source(tmp_path, "batch", "file.mcap", [100, 123, 456])
    bind_managed_source(tmp_path, "batch", "file.mcap", [100, 123, 456])
    with pytest.raises(UploadStateError, match="changed"):
        bind_managed_source(tmp_path, "batch", "file.mcap", [100, 124, 456])


def test_completed_multipart_only_rechecks_completion(tmp_path):
    calls = []
    transfer_managed_upload(
        str(tmp_path / "not-read"),
        {"method": "MULTIPART", "upload_uid": str(uuid4()), "complete": True},
        request=lambda *args, **kwargs: calls.append((args, kwargs)),
        timeout=httpx.Timeout(10),
        cancelled=lambda: False,
    )
    assert len(calls) == 1
    assert calls[0][0][1].endswith("/complete/")


def test_cancelled_put_never_opens_source_or_issues_request(tmp_path):
    with pytest.raises(InterruptedError, match="before sending"):
        transfer_managed_upload(
            str(tmp_path / "not-read"),
            {},
            request=lambda *args, **kwargs: pytest.fail("unexpected request"),
            timeout=httpx.Timeout(10),
            cancelled=lambda: True,
        )


def test_different_batch_cannot_inherit_completed_file_checkpoint(tmp_path):
    from avala._uploads import bind_managed_batch
    from avala.errors import UploadStateError

    original = str(uuid4())
    bind_managed_batch(tmp_path, "dataset", fingerprint="owner", batch=original)
    bind_managed_batch(tmp_path, "dataset", fingerprint="owner", batch=original)
    for replacement in [str(uuid4()), None]:
        with pytest.raises(UploadStateError, match="differs"):
            bind_managed_batch(tmp_path, "dataset", fingerprint="owner", batch=replacement)


def test_no_resume_rotates_batch_and_invalidates_skips_but_preserves_remote_evidence(tmp_path):
    from avala._uploads import load_completed, load_remote_keys, save_completed

    original = managed_upload_batch(tmp_path, "dataset", fingerprint="owner")
    save_completed(
        tmp_path,
        "dataset",
        {"a"},
        fingerprint="owner",
        expected_batch=original,
        stamps={"a": [1, 2]},
        remote={"a", "b"},
    )
    fresh = managed_upload_batch(tmp_path, "dataset", fingerprint="owner", resume=False)
    assert fresh != original
    assert next(tmp_path.glob("*.retired-batches")).read_text().strip() == original
    assert managed_upload_batch(tmp_path, "dataset", fingerprint="owner") == fresh
    assert load_completed(tmp_path, "dataset", fingerprint="owner") == set()
    assert load_remote_keys(tmp_path, "dataset", fingerprint="owner") == {"a", "b"}


def test_finalization_retires_only_its_batch_source_markers(tmp_path):
    from avala._uploads import _managed_source_directory, bind_managed_source, clear_completed

    batch = managed_upload_batch(tmp_path, "dataset", fingerprint="owner")
    other = managed_upload_batch(tmp_path, "other", fingerprint="owner")
    for relative in ["a", "b", "c"]:
        bind_managed_source(tmp_path, batch, relative, [1, 2, 3])
    bind_managed_source(tmp_path, other, "a", [1, 2, 3])
    assert len(list(_managed_source_directory(tmp_path, batch).iterdir())) == 3
    clear_completed(tmp_path, "dataset", fingerprint="owner", expected_batch=batch)
    assert not _managed_source_directory(tmp_path, batch).exists()
    assert _managed_source_directory(tmp_path, other).exists()
    # Stable locks scale with batches, never with the number of source files.
    assert len(list((tmp_path / "managed-sources").glob("*.lock"))) == 2


def test_old_finalization_cannot_clear_restarted_batch_or_source_guards(tmp_path):
    from avala._uploads import (
        _managed_source_directory,
        bind_managed_source,
        clear_completed,
        load_completed,
        save_completed,
    )
    from avala.errors import UploadStateError

    finalized = managed_upload_batch(tmp_path, "dataset", fingerprint="owner")
    bind_managed_source(tmp_path, finalized, "a", [1, 2, 3])
    newer = managed_upload_batch(tmp_path, "dataset", fingerprint="owner", resume=False)
    bind_managed_source(tmp_path, newer, "a", [1, 4, 5])
    save_completed(tmp_path, "dataset", {"a"}, fingerprint="owner", expected_batch=newer, stamps={"a": [1, 4, 5]})
    # Delayed A finalization races with B having already restarted this target.
    clear_completed(tmp_path, "dataset", fingerprint="owner", expected_batch=finalized)
    clear_completed(tmp_path, "dataset", fingerprint="owner")  # An older legacy client is fenced too.
    assert managed_upload_batch(tmp_path, "dataset", fingerprint="owner") == newer
    assert load_completed(tmp_path, "dataset", fingerprint="owner") == {"a"}
    assert _managed_source_directory(tmp_path, newer).exists()
    with pytest.raises(UploadStateError, match="changed"):
        bind_managed_source(tmp_path, newer, "a", [1, 8, 9])
    clear_completed(tmp_path, "dataset", fingerprint="owner", expected_batch=newer)
    assert not _managed_source_directory(tmp_path, newer).exists()


def test_cleanup_resolves_identity_redirect_before_checking_batch(tmp_path):
    from avala._uploads import _managed_source_directory, bind_managed_source, clear_completed, state_path

    batch = managed_upload_batch(tmp_path, "dataset", fingerprint="provisional")
    bind_managed_source(tmp_path, batch, "a", [1, 2, 3])
    migrate_state(tmp_path, "dataset", from_fingerprint="provisional", to_fingerprint="canonical")
    clear_completed(tmp_path, "dataset", fingerprint="provisional", expected_batch=batch)
    assert not state_path(tmp_path, "dataset", fingerprint="canonical").with_suffix(".batch").exists()
    assert not _managed_source_directory(tmp_path, batch).exists()


@pytest.mark.parametrize("writer", ["journal", "ambiguous", "snapshot"])
def test_rotated_batch_rejects_stale_checkpoint_writes(tmp_path, writer):
    from avala._uploads import append_completed, load_completed, load_remote_keys, save_completed
    from avala.errors import UploadStateError

    old = managed_upload_batch(tmp_path, "dataset", fingerprint="owner")
    new = managed_upload_batch(tmp_path, "dataset", fingerprint="owner", resume=False)
    for expected in [old, None]:
        with pytest.raises(UploadStateError, match="batch changed"):
            if writer in {"journal", "ambiguous"}:
                append_completed(
                    tmp_path,
                    "dataset",
                    "old",
                    stamp=[1, 2],
                    completed=writer == "journal",
                    fingerprint="owner",
                    expected_batch=expected,
                )
            else:
                save_completed(
                    tmp_path,
                    "dataset",
                    {"old"},
                    stamps={"old": [1, 2]},
                    remote={"old"},
                    fingerprint="owner",
                    expected_batch=expected,
                )
    assert managed_upload_batch(tmp_path, "dataset", fingerprint="owner") == new
    assert load_completed(tmp_path, "dataset", fingerprint="owner") == set()
    assert load_remote_keys(tmp_path, "dataset", fingerprint="owner") == set()


@respx.mock
def test_successful_part_bytes_are_reported_even_when_peer_part_fails(tmp_path):
    part_size = 5 * 1024 * 1024
    source = tmp_path / "file"
    source.write_bytes(b"a" * part_size + b"tail")
    respx.put(URL + "1").mock(return_value=httpx.Response(200))
    tail = respx.put(URL + "2").mock(return_value=httpx.Response(503))
    sent = []

    def request(method, path, **kwargs):
        if path.endswith("/parts/"):
            number = kwargs["json"]["part_numbers"][0]
            return {"parts": [{"part_number": number, "url": URL + str(number)}]}
        return {}

    descriptor = {"method": "MULTIPART", "upload_uid": str(uuid4()), "part_size": part_size, "part_count": 2}
    with pytest.raises(httpx.HTTPStatusError):
        transfer_managed_upload(
            str(source),
            descriptor,
            request=request,
            timeout=httpx.Timeout(10),
            cancelled=lambda: False,
            on_sent=sent.append,
        )
    assert sent == [part_size]
    tail.mock(return_value=httpx.Response(200))
    descriptor["uploaded_parts"] = [{"part_number": 1, "size": part_size}]
    transfer_managed_upload(
        str(source),
        descriptor,
        request=request,
        timeout=httpx.Timeout(10),
        cancelled=lambda: False,
        on_sent=sent.append,
    )
    assert sent == [part_size, 4]
