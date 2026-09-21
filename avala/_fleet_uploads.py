"""Private local evidence for Fleet uploads; the uploader must explicitly opt in.

Content hashes bind a local observation, not a remote object or an immutable
filesystem snapshot. Checkpoints retain uncertainty instead of replacing it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal
from urllib.parse import urlsplit, urlunsplit

from avala._uploads import _canonical_base_url, _canonical_uuid, _state_lock, _strict_private_dir
from avala.errors import UploadStateError

_CHUNK_SIZE = 1024 * 1024
_MAX_FILES = 50_000
_MAX_LEGACY_ENTRIES = 100_000
# JSON control-character escaping can use six bytes per path character.
_MAX_CHECKPOINT_BYTES = _MAX_FILES * (6 * 1024 + 160) + 65_536
_RECOVERY = "Keep the checkpoint. Restore the original source and destination, or use a new recording UID."
_Phase = Literal["initializing", "active", "confirming", "finalizing", "completed"]
_PHASES = {"initializing", "active", "confirming", "finalizing", "completed"}


def _failure(message: str) -> UploadStateError:
    return UploadStateError(f"Fleet upload evidence unavailable: {message}. {_RECOVERY}")


def _stamp(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


@contextmanager
def _open_regular_file(path: Path) -> Iterator[BinaryIO]:
    """Check the opened descriptor before reading, including replacement races."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise _failure("source contains a symlink or special file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if not stat.S_ISREG(opened.st_mode) or _stamp(before) != _stamp(opened):
            raise _failure("source changed while opening a file")
        yield handle
        if _stamp(opened) != _stamp(os.fstat(handle.fileno())) or _stamp(opened) != _stamp(path.lstat()):
            raise _failure("source changed while reading a file")


@dataclass(frozen=True)
class SourceFile:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class SourceInventory:
    source: Path
    files: tuple[SourceFile, ...]

    @property
    def total_bytes(self) -> int:
        return sum(file.size_bytes for file in self.files)

    def manifest(self) -> list[dict[str, Any]]:
        return [{"path": file.path, "size_bytes": file.size_bytes} for file in self.files]


def collect_source(source_dir: str | Path, *, state_dir: Path) -> SourceInventory:
    """Hash sorted regular files, omitting SDK state and refusing unsafe entries.

    Each file is checked while it is read. An earlier file can change after its
    observation, so callers must validate again before confirmation/finalization
    and verify bytes consumed by each PUT.
    """
    source = Path(source_dir).absolute()
    state = state_dir.resolve()
    files: list[SourceFile] = []

    def visit(directory: Path) -> None:
        before = directory.lstat()
        if not stat.S_ISDIR(before.st_mode):
            raise _failure("source contains a symlink or special directory")
        with os.scandir(directory) as entries:
            children = sorted(entries, key=lambda entry: entry.name)
        for entry in children:
            path = directory / entry.name
            # Exclude the actual checkpoint directory, including an alias to it.
            if entry.is_symlink():
                try:
                    if path.resolve() == state:
                        continue
                except (OSError, RuntimeError):
                    pass  # A broken/looping link is still an unsupported source.
                raise _failure("source contains a symlink")
            if path.resolve() == state:
                continue
            if entry.is_dir(follow_symlinks=False):
                visit(path)
                continue
            relative = path.relative_to(source).as_posix()
            if (
                len(relative) > 1024
                or relative != relative.strip()
                or "\\" in relative
                or "\x00" in relative
                or ".." in relative.split("/")
            ):
                raise _failure("source contains an unsupported relative path")
            if len(files) >= _MAX_FILES:
                raise _failure("source exceeds the 50000 file limit")
            digest = hashlib.sha256()
            size = 0
            with _open_regular_file(path) as handle:
                while chunk := handle.read(_CHUNK_SIZE):
                    digest.update(chunk)
                    size += len(chunk)
            files.append(SourceFile(relative, size, digest.hexdigest()))
        if _stamp(before) != _stamp(directory.lstat()):
            raise _failure("source directory changed during inventory")

    try:
        if source.is_symlink():
            raise _failure("source directory is a symlink")
        source = source.resolve(strict=True)
        if source == state or state in source.parents:
            raise _failure("source is inside the SDK checkpoint directory")
        visit(source)
    except (OSError, ValueError):
        raise _failure("source could not be safely read") from None
    if not files:
        raise _failure("source has no uploadable files")
    return SourceInventory(source, tuple(sorted(files, key=lambda file: file.path)))


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Repeated checkpoint field")
        result[key] = value
    return result


@dataclass(frozen=True)
class UploadBinding:
    inventory: SourceInventory
    base_url: str
    credential_sha256: str
    recording_uid: str
    storage_config_uid: str | None

    @property
    def scope(self) -> str:
        return _digest([self.base_url, self.recording_uid])

    @property
    def digest(self) -> str:
        return _digest(self.document())

    def document(self) -> dict[str, Any]:
        """Private recovery details; never a credential, grant, or header."""
        return {
            "api": self.base_url,
            "credential_sha256": self.credential_sha256,
            "recording_uid": self.recording_uid,
            "storage_config_uid": self.storage_config_uid,
            "source": str(self.inventory.source),
            "files": [
                {"path": file.path, "size_bytes": file.size_bytes, "sha256": file.sha256}
                for file in self.inventory.files
            ],
        }


def _canonical_api_url(base_url: str) -> str:
    try:
        url = urlsplit(base_url)
        if (
            len(base_url) > 4096
            or url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username is not None
            or url.password is not None
            or url.query
            or url.fragment
        ):
            raise ValueError
        host = url.hostname.lower()
        if ":" in host:
            host = f"[{host}]"
        port = url.port
        authority = host if port is None or (url.scheme, port) in {("http", 80), ("https", 443)} else f"{host}:{port}"
        return _canonical_base_url(urlunsplit((url.scheme, authority, url.path.rstrip("/"), "", "")))
    except ValueError:
        raise _failure("API URL must contain no credentials, query, or fragment") from None


def upload_binding(
    inventory: SourceInventory,
    *,
    base_url: str,
    api_key: str,
    recording_uid: str,
    storage_config_uid: str | None,
) -> UploadBinding:
    """Keep discovery stable when credentials, storage, or source bytes change."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", recording_uid):
        raise _failure("recording UID is invalid")
    if storage_config_uid is not None and not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", storage_config_uid):
        raise _failure("storage config UID is invalid")
    if len(str(inventory.source)) > 4096:
        raise _failure("source directory path is too long")
    return UploadBinding(
        inventory,
        _canonical_api_url(base_url),
        hashlib.sha256(api_key.encode("utf-8")).hexdigest(),
        _canonical_uuid(recording_uid),
        _canonical_uuid(storage_config_uid) if storage_config_uid is not None else None,
    )


@dataclass(frozen=True)
class CheckpointState:
    phase: _Phase
    session_uid: str | None = None
    s3_prefix: str | None = None
    confirmation_paths: tuple[str, ...] = ()


class FleetCheckpoint:
    """One atomic receipt, used only while ``fleet_checkpoint`` owns its lock.

    Write initializing BEFORE POST init. A lost response leaves that intent
    intact; it cannot authorize adopting a latest session or another POST.
    Completed receipts remain as evidence against restarting the same prefix.
    """

    def __init__(self, path: Path, binding: UploadBinding) -> None:
        self.path = path
        self._binding = binding
        self._closed = False

    def read(self) -> CheckpointState | None:
        if self._closed:
            raise _failure("checkpoint ownership has ended")
        if not os.path.lexists(self.path):
            return None
        try:
            with _open_regular_file(self.path) as handle:
                payload = handle.read(_MAX_CHECKPOINT_BYTES + 1)
            if len(payload) > _MAX_CHECKPOINT_BYTES:
                raise ValueError
            data = json.loads(payload, object_pairs_hook=_unique_fields)
            if not isinstance(data, dict) or set(data) != {
                "version",
                "binding",
                "evidence",
                "phase",
                "session_uid",
                "s3_prefix",
                "confirmation_paths",
            }:
                raise ValueError
            if type(data["version"]) is not int or data["version"] != 2 or data["binding"] != self._binding.digest:
                raise ValueError
            if _digest(data["evidence"]) != self._binding.digest:
                raise ValueError
            phase, session = data["phase"], data["session_uid"]
            prefix = data["s3_prefix"]
            paths = data["confirmation_paths"]
            if not isinstance(phase, str) or phase not in _PHASES:
                raise ValueError
            if phase == "initializing":
                if session is not None or prefix is not None:
                    raise ValueError
            elif not isinstance(session, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session):
                raise ValueError
            elif not self._valid_prefix(prefix):
                raise ValueError
            if not isinstance(paths, list) or not self._valid_confirmation(phase, paths):
                raise ValueError
            return CheckpointState(phase, session, prefix, tuple(paths))  # type: ignore[arg-type]
        except (OSError, ValueError, RecursionError, UploadStateError):
            pass
        raise _failure("checkpoint is corrupt or belongs to different upload inputs")

    def _valid_prefix(self, prefix: object) -> bool:
        return (
            isinstance(prefix, str)
            and len(prefix) <= 1024
            and prefix.startswith("fleet/")
            and prefix.endswith(f"/{self._binding.recording_uid}/")
            and "\\" not in prefix
            and not any(ord(char) < 32 for char in prefix)
            and ".." not in prefix.split("/")
        )

    def _valid_confirmation(self, phase: str, paths: list[str]) -> bool:
        if phase != "confirming":
            return paths == []
        return (
            0 < len(paths) <= 100
            and all(isinstance(path, str) for path in paths)
            and len(set(paths)) == len(paths)
            and set(paths) <= {file.path for file in self._binding.inventory.files}
        )

    def write(
        self,
        phase: _Phase,
        session_uid: str | None = None,
        *,
        s3_prefix: str | None = None,
        confirmation_paths: tuple[str, ...] = (),
    ) -> None:
        previous = self.read()
        allowed = {
            None: {"initializing"},
            "initializing": {"active"},
            "active": {"active", "confirming", "finalizing"},
            "confirming": {"active"},
            "finalizing": {"finalizing", "completed"},
            "completed": {"completed"},
        }
        if phase not in allowed[previous.phase if previous else None]:
            raise _failure("checkpoint transition is not safe")
        if phase == "initializing":
            if session_uid is not None:
                raise _failure("initialization intent must not declare a session")
        elif not isinstance(session_uid, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_uid):
            raise _failure("session UID is invalid")
        if previous and previous.session_uid is not None and previous.session_uid != session_uid:
            raise _failure("checkpoint session cannot be replaced")
        if previous is not None and previous.s3_prefix is not None:
            if s3_prefix is not None and s3_prefix != previous.s3_prefix:
                raise _failure("checkpoint storage prefix cannot be replaced")
            s3_prefix = previous.s3_prefix
        if (phase == "initializing" and s3_prefix is not None) or (
            phase != "initializing" and not self._valid_prefix(s3_prefix)
        ):
            raise _failure("checkpoint requires an exact storage prefix for this recording")
        if not self._valid_confirmation(phase, list(confirmation_paths)):
            raise _failure("checkpoint confirmation intent must name exact manifest paths")
        payload = {
            "version": 2,
            "binding": self._binding.digest,
            "evidence": self._binding.document(),
            "phase": phase,
            "session_uid": session_uid,
            "s3_prefix": s3_prefix,
            "confirmation_paths": list(confirmation_paths),
        }
        _write_private_receipt(self.path, payload, maximum=_MAX_CHECKPOINT_BYTES)


def _write_private_receipt(path: Path, payload: dict[str, Any], *, maximum: int) -> None:
    """Publish private evidence atomically while the caller owns its recording lock."""
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(encoded) > maximum:
        raise _failure("serialized checkpoint exceeds the byte limit")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        if os.name == "posix":
            descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except OSError:
        raise _failure("checkpoint could not be persisted; do not initialize again") from None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                # An unused private temporary file cannot authorize another upload.
                pass


@contextmanager
def fleet_checkpoint(state_dir: Path, binding: UploadBinding) -> Iterator[FleetCheckpoint]:
    """Exclusively own one API/recording scope for the entire caller operation.

    Older SDKs and other hosts do not participate in this local advisory lock.
    Legacy recording-only receipts remain discoverable and are never migrated.
    """
    directory = state_dir / "fleet-v1"
    try:
        _strict_private_dir(state_dir)
    except OSError:
        raise _failure("checkpoint directory is unsafe or unavailable") from None
    with _state_lock(directory, binding.scope, None, strict=True):
        try:
            # Old clients used the exact supplied UUID spelling in filenames.
            # Canonicalize names, without reading any other recording's state.
            with os.scandir(state_dir) as entries:
                for index, entry in enumerate(entries):
                    if index >= _MAX_LEGACY_ENTRIES:
                        raise _failure("legacy checkpoint directory exceeds the inspection limit")
                    if (
                        entry.name.endswith((".json", ".tmp"))
                        and _canonical_uuid(entry.name.rsplit(".", 1)[0]) == binding.recording_uid
                    ):
                        raise _failure("legacy checkpoint cannot establish source and destination identity")
        except OSError:
            raise _failure("legacy checkpoint directory could not be inspected") from None
        checkpoint = FleetCheckpoint(directory / f"{binding.scope}.json", binding)
        try:
            yield checkpoint
        finally:
            checkpoint._closed = True
