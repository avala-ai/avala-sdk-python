"""Local Fleet evidence must survive ambiguity without selecting a new upload."""

import json
import os
import stat
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from avala import _fleet_uploads, _uploads
from avala._fleet_uploads import CheckpointState, collect_source, fleet_checkpoint, upload_binding
from avala.errors import UploadStateError


def test_inventory_excludes_nested_checkpoint_state(tmp_path: Path) -> None:
    source = tmp_path / "source"
    state = source / ".avala" / "uploads"
    state.mkdir(parents=True)
    (state / "recording.json").write_text("private checkpoint")
    (source / "frame.mcap").write_bytes(b"synthetic capture")

    inventory = collect_source(source, state_dir=state)

    assert [file.path for file in inventory.files] == ["frame.mcap"]


def test_changed_source_cannot_miss_initialization_intent(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    file = source / "frame.mcap"
    file.write_bytes(b"take A")
    state = tmp_path / "state"

    def binding():
        return upload_binding(
            collect_source(source, state_dir=state),
            base_url="https://api.avala.ai/api/v1",
            api_key="synthetic credential",
            recording_uid="recording-1",
            storage_config_uid=None,
        )

    original = binding()
    with fleet_checkpoint(state, original) as checkpoint:
        checkpoint.write("initializing")
    retained = list((state / "fleet-v1").glob("*.json"))[0].read_bytes()
    file.write_bytes(b"take B")
    changed = binding()
    assert changed.scope == original.scope
    assert changed.digest != original.digest
    with pytest.raises(UploadStateError):
        with fleet_checkpoint(state, changed) as checkpoint:
            checkpoint.read()
    assert list((state / "fleet-v1").glob("*.json"))[0].read_bytes() == retained


@pytest.fixture
def evidence(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "capture.mcap").write_bytes(b"synthetic capture")
    state = tmp_path / "state"
    inventory = collect_source(source, state_dir=state)
    arguments = dict(
        base_url="https://api.avala.ai/api/v1",
        api_key="synthetic credential",
        recording_uid="recording-1",
        storage_config_uid=None,
    )
    return source, state, inventory, arguments


def test_manifest_is_sorted_and_inventory_is_immutable(evidence):
    source, state, _, _ = evidence
    (source / "empty").write_bytes(b"")
    (source / "nested").mkdir()
    (source / "nested" / "a").write_bytes(b"a")
    inventory = collect_source(source, state_dir=state)
    assert inventory.manifest() == [
        {"path": "capture.mcap", "size_bytes": 17},
        {"path": "empty", "size_bytes": 0},
        {"path": "nested/a", "size_bytes": 1},
    ]
    assert inventory.total_bytes == 18
    assert inventory.files[1].sha256 == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    with pytest.raises(FrozenInstanceError):
        inventory.files[0].path = "other"
    inventory.manifest().clear()
    assert len(inventory.files) == 3


@pytest.mark.parametrize("kind", ["file", "directory", "root", "internal", "dangling", "loop"])
def test_symlinks_are_rejected_before_inventory(evidence, tmp_path, kind):
    source, state, _, _ = evidence
    if kind == "root":
        link = tmp_path / "source-link"
        link.symlink_to(source, target_is_directory=True)
        source = link
    elif kind == "directory":
        (source / "link").symlink_to(tmp_path, target_is_directory=True)
    elif kind == "loop":
        (source / "link").symlink_to(source / "link")
    else:
        target = source / "capture.mcap" if kind == "internal" else tmp_path / "external"
        if kind == "file":
            target.write_text("private external bytes")
        (source / "link").symlink_to(target)
    with pytest.raises(UploadStateError, match="symlink"):
        collect_source(source, state_dir=state)


def test_alias_to_checkpoint_directory_is_omitted(evidence):
    source, state, _, _ = evidence
    state.mkdir()
    (state / "private.json").write_text("private state")
    (source / "state-alias").symlink_to(state, target_is_directory=True)
    assert len(collect_source(source, state_dir=state).files) == 1


@pytest.mark.parametrize("nested", [False, True])
def test_source_inside_checkpoint_directory_is_rejected(evidence, nested):
    source, _, _, _ = evidence
    with pytest.raises(UploadStateError, match="inside"):
        collect_source(source, state_dir=source.parent if nested else source)


@pytest.mark.skipif(os.name != "posix", reason="POSIX synthetic FIFO")
def test_special_file_never_blocks_or_reads(evidence):
    source, state, _, _ = evidence
    os.mkfifo(source / "pipe")
    with pytest.raises(UploadStateError, match="special"):
        collect_source(source, state_dir=state)


def test_limits_empty_and_unsupported_paths(evidence, monkeypatch):
    source, state, _, _ = evidence
    monkeypatch.setattr(_fleet_uploads, "_MAX_FILES", 0)
    with pytest.raises(UploadStateError, match="limit"):
        collect_source(source, state_dir=state)
    (source / "capture.mcap").unlink()
    with pytest.raises(UploadStateError, match="no uploadable"):
        collect_source(source, state_dir=state)
    if os.name == "posix":
        (source / "bad\\path").write_bytes(b"a")
        with pytest.raises(UploadStateError, match="unsupported"):
            collect_source(source, state_dir=state)


def test_source_replacement_is_rejected_before_descriptor_read(evidence, monkeypatch):
    source, state, _, _ = evidence
    target = source / "capture.mcap"
    real_open = os.open

    def replace_then_open(path, flags, *args, **kwargs):
        target.unlink()
        target.write_bytes(b"replacement")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", replace_then_open)
    with pytest.raises(UploadStateError, match="changed while opening"):
        collect_source(source, state_dir=state)


@pytest.mark.parametrize("change", ["file", "directory", "delete"])
def test_changes_during_scan_do_not_produce_evidence(evidence, monkeypatch, change):
    source, state, _, _ = evidence
    original = _fleet_uploads._open_regular_file

    @contextmanager
    def changing_file(path):
        with original(path) as handle:
            yield handle
            if change == "file":
                path.write_bytes(b"different bytes")
            elif change == "delete":
                path.unlink()
            else:
                (source / "added").write_bytes(b"new file")

    monkeypatch.setattr(_fleet_uploads, "_open_regular_file", changing_file)
    with pytest.raises(UploadStateError):
        collect_source(source, state_dir=state)


@pytest.mark.parametrize(
    "field,value",
    [
        ("api_key", "different synthetic credential"),
        ("storage_config_uid", "storage-2"),
    ],
)
def test_changed_identity_remains_discoverable_and_refuses(evidence, field, value):
    _, state, inventory, arguments = evidence
    original = upload_binding(inventory, **arguments)
    with fleet_checkpoint(state, original) as checkpoint:
        checkpoint.write("initializing")
    arguments[field] = value
    changed = upload_binding(inventory, **arguments)
    assert changed.scope == original.scope
    with fleet_checkpoint(state, changed) as checkpoint:
        with pytest.raises(UploadStateError):
            checkpoint.read()


def test_canonical_api_alias_and_uuid_bind_to_same_scope(evidence):
    _, _, inventory, arguments = evidence
    arguments["recording_uid"] = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    first = upload_binding(inventory, **arguments)
    arguments["recording_uid"] = "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA"
    arguments["base_url"] = "https://server.avala.ai/api/v1/"
    alias = upload_binding(inventory, **arguments)
    assert (first.scope, first.digest) == (alias.scope, alias.digest)
    arguments["base_url"] = "https://example.invalid/api/v1"
    assert upload_binding(inventory, **arguments).scope != first.scope


def test_same_bytes_in_different_source_directory_are_not_same_binding(evidence):
    source, state, inventory, arguments = evidence
    second = source.parent / "second"
    second.mkdir()
    (second / "capture.mcap").write_bytes((source / "capture.mcap").read_bytes())
    before = upload_binding(inventory, **arguments)
    after = upload_binding(collect_source(second, state_dir=state), **arguments)
    assert before.scope == after.scope
    assert before.digest != after.digest


def test_same_size_preserved_mtime_cannot_hide_changed_bytes(evidence):
    source, state, inventory, arguments = evidence
    path = source / "capture.mcap"
    before = path.stat()
    path.write_bytes(b"different capture")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size
    assert upload_binding(inventory, **arguments) != upload_binding(
        collect_source(source, state_dir=state), **arguments
    )


def test_intent_session_and_completion_receipts_are_retained(evidence):
    _, state, inventory, arguments = evidence
    binding = upload_binding(inventory, **arguments)
    with fleet_checkpoint(state, binding) as checkpoint:
        assert checkpoint.read() is None
        checkpoint.write("initializing")
        assert checkpoint.read() == CheckpointState("initializing")
        with pytest.raises(UploadStateError):
            checkpoint.write("initializing")
        checkpoint.write("active", "session-1", s3_prefix="fleet/org/device/recording-1/")
        with pytest.raises(UploadStateError):
            checkpoint.write("active", "other-session")
        checkpoint.write("finalizing", "session-1")
        with pytest.raises(UploadStateError):
            checkpoint.write("active", "session-1", s3_prefix="fleet/org/device/recording-1/")
        checkpoint.write("completed", "session-1")
        assert checkpoint.path.exists()
        assert checkpoint.read() == CheckpointState("completed", "session-1", "fleet/org/device/recording-1/")
        if os.name == "posix":
            assert stat.S_IMODE(checkpoint.path.stat().st_mode) == 0o600
            assert stat.S_IMODE(checkpoint.path.parent.stat().st_mode) == 0o700
        assert arguments["api_key"] not in checkpoint.path.read_text()


@pytest.mark.parametrize("payload", ["", "{", "[]", "null", "{}", "x" * 2049])
def test_corrupt_checkpoint_is_never_absence(evidence, payload):
    _, state, inventory, arguments = evidence
    with fleet_checkpoint(state, upload_binding(inventory, **arguments)) as checkpoint:
        checkpoint.path.write_text(payload)
        with pytest.raises(UploadStateError):
            checkpoint.read()
        with pytest.raises(UploadStateError):
            checkpoint.write("initializing")
        assert checkpoint.path.read_text() == payload


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("version", 1.0),
        ("version", 3),
        ("binding", "different"),
        ("phase", "unknown"),
        ("phase", []),
        ("session_uid", "invented-session"),
        ("extra", "field"),
    ],
)
def test_malformed_receipt_fields_refuse_without_rewrite(evidence, field, value):
    _, state, inventory, arguments = evidence
    with fleet_checkpoint(state, upload_binding(inventory, **arguments)) as checkpoint:
        checkpoint.write("initializing")
        data = json.loads(checkpoint.path.read_text())
        data[field] = value
        payload = json.dumps(data)
        checkpoint.path.write_text(payload)
        with pytest.raises(UploadStateError):
            checkpoint.read()
        assert checkpoint.path.read_text() == payload


@pytest.mark.parametrize("legacy_kind", ["valid", "corrupt", "dangling"])
@pytest.mark.parametrize("suffix", ["json", "tmp"])
def test_legacy_evidence_is_not_hidden_by_new_namespace(evidence, legacy_kind, suffix):
    _, state, inventory, arguments = evidence
    state.mkdir()
    legacy = state / f"recording-1.{suffix}"
    if legacy_kind == "dangling":
        legacy.symlink_to(state / "missing")
    else:
        legacy.write_text('{"session_uid":"old"}' if legacy_kind == "valid" else "{")
    with pytest.raises(UploadStateError, match="legacy"):
        with fleet_checkpoint(state, upload_binding(inventory, **arguments)):
            pytest.fail("must not yield")
    assert os.path.lexists(legacy)


def test_failed_atomic_replace_preserves_initialization_intent(evidence, monkeypatch):
    _, state, inventory, arguments = evidence
    with fleet_checkpoint(state, upload_binding(inventory, **arguments)) as checkpoint:
        checkpoint.write("initializing")
        retained = checkpoint.path.read_bytes()
        attempted = []

        def fail_replace(source, target):
            attempted.append(Path(source))
            assert stat.S_IMODE(Path(source).stat().st_mode) == 0o600
            raise OSError("synthetic private failure")

        monkeypatch.setattr(Path, "replace", fail_replace)
        for _ in range(2):
            with pytest.raises(UploadStateError, match="do not initialize again"):
                checkpoint.write("active", "session-1", s3_prefix="fleet/org/device/recording-1/")
        assert checkpoint.path.read_bytes() == retained
        assert len(set(attempted)) == 2
        assert not any(path.exists() for path in attempted)


@pytest.mark.skipif(os.name != "posix", reason="Actual POSIX interprocess lock")
def test_strict_lock_excludes_other_process_and_releases_after_exception(evidence):
    _, state, inventory, arguments = evidence
    binding = upload_binding(inventory, **arguments)
    script = """
import sys
from pathlib import Path
from avala._uploads import _state_lock
from avala.errors import UploadStateError
try:
    with _state_lock(Path(sys.argv[1]), sys.argv[2], None, strict=True):
        print("acquired")
except UploadStateError:
    print("busy")
"""

    def probe():
        return subprocess.run(
            [sys.executable, "-c", script, str(state / "fleet-v1"), binding.scope],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()

    with pytest.raises(RuntimeError):
        with fleet_checkpoint(state, binding):
            assert probe() == "busy"
            raise RuntimeError("release on caller failure")
    assert probe() == "acquired"
    assert list((state / "fleet-v1").glob("*.lock"))


def test_windows_backend_locks_and_unlocks_same_byte_without_writing(tmp_path, monkeypatch):
    calls = []
    with (tmp_path / "lock").open("w+") as handle:
        backend = SimpleNamespace(
            LK_NBLCK=2,
            LK_UNLCK=0,
            locking=lambda fd, mode, size: calls.append((fd, mode, size, handle.tell())),
        )
        monkeypatch.setitem(sys.modules, "msvcrt", backend)
        handle.seek(20)
        release = _uploads._acquire_strict_lock(handle, platform="nt")
        handle.seek(40)
        release()
        assert calls == [(handle.fileno(), 2, 1, 0), (handle.fileno(), 0, 1, 0)]
        assert os.fstat(handle.fileno()).st_size == 0


def test_strict_lock_failure_never_yields_and_sanitizes_errors(tmp_path, monkeypatch):
    def unavailable(handle):
        raise OSError("synthetic private lock details")

    monkeypatch.setattr(_uploads, "_acquire_strict_lock", unavailable)
    with pytest.raises(UploadStateError) as caught:
        with _uploads._state_lock(tmp_path, "recording", None, strict=True):
            pytest.fail("must not yield unlocked")
    assert "private lock details" not in str(caught.value)


def test_checkpoint_handle_cannot_outlive_exclusive_scope(evidence):
    _, state, inventory, arguments = evidence
    with fleet_checkpoint(state, upload_binding(inventory, **arguments)) as checkpoint:
        checkpoint.write("initializing")
    retained = checkpoint.path.read_bytes()
    with pytest.raises(UploadStateError, match="ownership has ended"):
        checkpoint.read()
    with pytest.raises(UploadStateError, match="ownership has ended"):
        checkpoint.write("active", "session-1", s3_prefix="fleet/org/device/recording-1/")
    assert checkpoint.path.read_bytes() == retained


@pytest.mark.parametrize("kind", ["state", "namespace", "lock", "checkpoint"])
def test_state_aliases_never_authorize_a_write(evidence, tmp_path, kind):
    _, state, inventory, arguments = evidence
    binding = upload_binding(inventory, **arguments)
    external = tmp_path / "external"
    external.mkdir()
    if kind == "state":
        state.symlink_to(external, target_is_directory=True)
    elif kind == "namespace":
        state.mkdir()
        (state / "fleet-v1").symlink_to(external, target_is_directory=True)
    else:
        (state / "fleet-v1").mkdir(parents=True)
        target = external / "untouched"
        target.write_text("external evidence")
        suffix = "lock" if kind == "lock" else "json"
        (state / "fleet-v1" / f"{binding.scope}.{suffix}").symlink_to(target)
    with pytest.raises(UploadStateError):
        with fleet_checkpoint(state, binding) as checkpoint:
            checkpoint.write("initializing")
    if kind in {"lock", "checkpoint"}:
        assert target.read_text() == "external evidence"


def test_temporary_cleanup_failure_does_not_mask_persistence_error(evidence, monkeypatch):
    _, state, inventory, arguments = evidence
    with fleet_checkpoint(state, upload_binding(inventory, **arguments)) as checkpoint:
        checkpoint.write("initializing")
        retained = checkpoint.path.read_bytes()

        def failure(*args, **kwargs):
            raise OSError("synthetic private failure")

        monkeypatch.setattr(Path, "replace", failure)
        monkeypatch.setattr(Path, "unlink", failure)
        with pytest.raises(UploadStateError, match="checkpoint could not be persisted") as caught:
            checkpoint.write("active", "session-1", s3_prefix="fleet/org/device/recording-1/")
        assert "synthetic private" not in str(caught.value)
        assert checkpoint.path.read_bytes() == retained


def test_checkpoint_retains_actionable_content_evidence_without_credential(evidence):
    source, state, inventory, arguments = evidence
    with fleet_checkpoint(state, upload_binding(inventory, **arguments)) as checkpoint:
        checkpoint.write("initializing")
        document = json.loads(checkpoint.path.read_text())["evidence"]
        assert document["source"] == str(source)
        assert document["api"] == arguments["base_url"]
        assert document["storage_config_uid"] is None
        assert document["recording_uid"] == "recording-1"
        assert len(document["credential_sha256"]) == 64
        assert document["files"] == [
            {
                "path": "capture.mcap",
                "size_bytes": 17,
                "sha256": inventory.files[0].sha256,
            }
        ]
        assert arguments["api_key"] not in checkpoint.path.read_text()


@pytest.mark.parametrize("mutation", ["fraction", "boolean", "missing", "extra", "hash"])
def test_retained_manifest_must_match_exact_typed_evidence(evidence, mutation):
    _, state, inventory, arguments = evidence
    with fleet_checkpoint(state, upload_binding(inventory, **arguments)) as checkpoint:
        checkpoint.write("initializing")
        document = json.loads(checkpoint.path.read_text())
        file = document["evidence"]["files"][0]
        if mutation == "fraction":
            file["size_bytes"] = 17.0
        elif mutation == "boolean":
            file["size_bytes"] = True
        elif mutation == "missing":
            del file["path"]
        elif mutation == "extra":
            file["unexpected"] = "field"
        else:
            file["sha256"] = "0" * 64
        checkpoint.path.write_text(json.dumps(document))
        with pytest.raises(UploadStateError):
            checkpoint.read()


def test_duplicate_checkpoint_fields_and_oversized_reads_refuse(evidence, monkeypatch):
    _, state, inventory, arguments = evidence
    with fleet_checkpoint(state, upload_binding(inventory, **arguments)) as checkpoint:
        checkpoint.write("initializing")
        payload = checkpoint.path.read_text()
        checkpoint.path.write_text(payload.replace('"version":2', '"version":1,"version":2'))
        with pytest.raises(UploadStateError):
            checkpoint.read()
        checkpoint.path.write_text(payload)
        monkeypatch.setattr(_fleet_uploads, "_MAX_CHECKPOINT_BYTES", 20)
        with pytest.raises(UploadStateError):
            checkpoint.read()


@pytest.mark.parametrize(
    "url",
    [
        "https://user:private@example.invalid/api/v1",
        "https://example.invalid/api/v1?secret=value",
        "https://example.invalid/api/v1#private",
        "https://example.invalid:bad/api/v1",
        "file:///private",
    ],
)
def test_api_identity_cannot_persist_embedded_secrets(evidence, url):
    _, _, inventory, arguments = evidence
    arguments["base_url"] = url
    with pytest.raises(UploadStateError) as caught:
        upload_binding(inventory, **arguments)
    assert "private" not in str(caught.value)
    assert "value" not in str(caught.value)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_strict_permissions_failure_cannot_yield_unprotected_state(evidence, monkeypatch):
    _, state, inventory, arguments = evidence

    def fail_chmod(*args, **kwargs):
        raise OSError("synthetic chmod failure")

    monkeypatch.setattr(Path, "chmod", fail_chmod)
    with pytest.raises(UploadStateError):
        with fleet_checkpoint(state, upload_binding(inventory, **arguments)):
            pytest.fail("must not yield")


def test_checkpoint_write_bound_counts_encoded_bytes_before_replacing(evidence, monkeypatch):
    source, state, _, arguments = evidence
    (source / "multibyte-😀").write_bytes(b"a")
    inventory = collect_source(source, state_dir=state)
    with fleet_checkpoint(state, upload_binding(inventory, **arguments)) as checkpoint:
        checkpoint.write("initializing")
        retained = checkpoint.path.read_bytes()
        assert len(retained) > len(retained.decode("utf-8"))
        monkeypatch.setattr(_fleet_uploads, "_MAX_CHECKPOINT_BYTES", len(retained))
        with pytest.raises(UploadStateError, match="byte limit"):
            checkpoint.write("active", "a" * 128, s3_prefix="fleet/org/device/recording-1/")
        assert checkpoint.path.read_bytes() == retained


@pytest.mark.parametrize("uid", ["", "../other", "private?token=value", "a" * 129])
def test_invalid_storage_identity_cannot_enter_receipt(evidence, uid):
    _, _, inventory, arguments = evidence
    arguments["storage_config_uid"] = uid
    with pytest.raises(UploadStateError, match="storage config UID is invalid"):
        upload_binding(inventory, **arguments)


@pytest.mark.parametrize("name", [" leading", "trailing ", "\tcapture", "capture\n"])
def test_paths_trimmed_by_server_are_refused_before_evidence(evidence, name):
    source, state, _, _ = evidence
    (source / name).write_bytes(b"capture")
    with pytest.raises(UploadStateError, match="unsupported relative path"):
        collect_source(source, state_dir=state)


def test_strict_lock_rechecks_new_path_without_nofollow(tmp_path, monkeypatch):
    external = tmp_path / "external"
    external.write_text("unchanged")
    state = tmp_path / "state"
    original = os.open

    def replace_before_open(path, flags, mode=0o777):
        Path(path).symlink_to(external)
        return original(path, flags & ~getattr(os, "O_NOFOLLOW", 0), mode)

    monkeypatch.setattr(os, "open", replace_before_open)
    with pytest.raises(UploadStateError):
        with _uploads._state_lock(state, "recording", None, strict=True):
            pytest.fail("must not acquire an aliased lock")
    assert external.read_text() == "unchanged"


@pytest.mark.parametrize(
    "old_uid,new_uid",
    [
        ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        ("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
        ("aAaaAAAA-aaaa-aaaa-AAAA-aaaaaaaaaaaa", "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        ("{aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa}", "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
    ],
)
@pytest.mark.parametrize("suffix", ["json", "tmp"])
def test_all_legacy_uuid_spellings_remain_visible(evidence, old_uid, new_uid, suffix):
    _, state, inventory, arguments = evidence
    state.mkdir()
    legacy = state / f"{old_uid}.{suffix}"
    legacy.write_text("private original evidence")
    arguments["recording_uid"] = new_uid
    with pytest.raises(UploadStateError, match="legacy checkpoint"):
        with fleet_checkpoint(state, upload_binding(inventory, **arguments)):
            pytest.fail("must not treat an alternate UUID spelling as no evidence")
    assert legacy.read_text() == "private original evidence"


def test_legacy_inspection_ignores_other_recordings_and_is_bounded(evidence, monkeypatch):
    _, state, inventory, arguments = evidence
    state.mkdir()
    other = state / "other-recording.json"
    other.write_text("unreadable content is immaterial")
    with fleet_checkpoint(state, upload_binding(inventory, **arguments)) as checkpoint:
        assert checkpoint.read() is None
    assert other.read_text() == "unreadable content is immaterial"
    monkeypatch.setattr(_fleet_uploads, "_MAX_LEGACY_ENTRIES", 0)
    with pytest.raises(UploadStateError, match="inspection limit"):
        with fleet_checkpoint(state, upload_binding(inventory, **arguments)):
            pytest.fail("must not infer absence from an incomplete inspection")
