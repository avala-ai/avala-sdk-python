"""Primitives shared by every SDK upload path (datasets, fleet recordings).

Both paths push raw file bytes to a URL the control plane minted, over links
that flap, for hours at a time. That combination needs the same three things
everywhere — a host allow-list, bounded retries, and a resume checkpoint — so
they live here rather than being reimplemented per resource.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import os
import random
import re
import shutil
import stat
import time
import uuid
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import urlparse

from avala.errors import UploadStateError

# Crash-recovery state for resumable uploads: one JSON file per upload target.
# Callers alias this into their own module namespace so tests can redirect it
# away from the real home directory (see ``tests/test_fleet.py``).
STATE_DIR = Path.home() / ".avala" / "uploads"

# Allow-list of hosts that may receive file contents via presigned URLs.
# Prevents a compromised/misbehaving control-plane response (or MITM) from
# redirecting local file bytes to an attacker-controlled host.
# Extend cautiously — each entry is a trust boundary for raw file data.
#
# These two suffixes are service-specific: `.blob.core.windows.net` is Azure
# Blob and nothing else, and `.storage.googleapis.com` is GCS and nothing else.
# The bare `storage.googleapis.com` form is matched too (see below).
PRESIGNED_URL_HOST_SUFFIXES = (
    ".storage.googleapis.com",  # GCS presigned URLs
    ".blob.core.windows.net",  # Azure Blob SAS URLs
)

# AWS needs a stricter test than a suffix. `.amazonaws.com` spans every AWS
# service, including ones any account can provision on demand: an
# attacker-owned `*.execute-api.<region>.amazonaws.com` API Gateway would have
# passed a suffix check and received the customer's raw file bytes.
#
# So require an actual S3 endpoint label immediately before `amazonaws.com`.
# This admits the real forms — `s3.amazonaws.com`, `s3.<region>.amazonaws.com`,
# `s3-<region>…`, `s3-accelerate…`, `s3.dualstack.<region>…`, each with or
# without a `<bucket>.` prefix — and rejects any other service. The `(?:^|\.)`
# anchor matters: without it `evil-s3.amazonaws.com` would slip through on a
# substring match.
#
# This bounds the blast radius to S3 itself; it cannot distinguish our bucket
# from an attacker's bucket. Only pinning the expected bucket would, and the
# server does not tell the client which bucket to expect.
_S3_ENDPOINT_HOST = re.compile(r"(?:^|\.)s3(?:[-.][a-z0-9-]+)*\.amazonaws\.com$")

# Match account endpoints, including the EU jurisdiction, without trusting
# arbitrary subdomains of Cloudflare or Avala.
_R2_ENDPOINT_HOST = re.compile(r"^[0-9a-f]{32}(?:\.eu)?\.r2\.cloudflarestorage\.com$")
_MANAGED_UPLOAD_HOSTS = frozenset({"data.avala.ai"})

# Distinguishes concurrent writers' temp/claim files. A shared name let two
# processes clobber each other's half-written checkpoint.
_WRITE_SEQ = itertools.count()

# Hostnames that reach the same backend, collapsed to one identity for
# checkpoint purposes. ARCHITECTURE.md: "Both api.avala.ai and server.avala.ai
# route to the same ALB" — `server.avala.ai` is the internal name CI and the
# pipelines use. Keying state on the literal host meant resuming an interrupted
# upload through the other name selected a fresh checkpoint, hiding the first
# run's remote keys even though the destination prefix was identical.
_EQUIVALENT_API_HOSTS = {"server.avala.ai": "api.avala.ai"}


def _canonical_base_url(base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    parsed = urlparse(trimmed)
    canonical = _EQUIVALENT_API_HOSTS.get((parsed.hostname or "").lower())
    if not canonical or not parsed.scheme:
        return trimmed
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{canonical}{port}{parsed.path}"


MAX_RETRIES = 8
_MAX_BACKOFF_SECONDS = 60.0

# Non-5xx codes worth a retry. Everything at 5xx and above is retried too (see
# ``is_retryable``) — enumerating 500/502/503/504 missed the gateway codes CDNs
# in front of the API actually emit, notably Cloudflare's 520/522/524.
# Everything else (400, 403, 404, 413…) means the request itself is wrong and
# repeating it verbatim just burns the clock.
RETRYABLE_STATUS_CODES = frozenset({408, 429})


def _is_retryable_status(status: int | None) -> bool:
    return status is not None and (status in RETRYABLE_STATUS_CODES or status >= 500)


def validate_presigned_url(url: str) -> None:
    """Ensure a server-provided upload URL targets a known cloud-storage host.

    The SDK trusts the server to mint presigned URLs, but a hijacked
    control-plane response would otherwise exfiltrate raw file bytes to any
    host. This enforces ``https://`` and a host-suffix allow-list.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"Upload URL must use HTTPS, got scheme '{parsed.scheme}'.")
    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("Upload URL has no host.")
    if parsed.username is not None or parsed.password is not None or parsed.port is not None or parsed.fragment:
        raise ValueError("Upload URL must not contain credentials, a port, or a fragment.")
    allowed = (
        host in _MANAGED_UPLOAD_HOSTS
        or _R2_ENDPOINT_HOST.fullmatch(host) is not None
        or _S3_ENDPOINT_HOST.search(host) is not None
        or any(host == suffix.lstrip(".") or host.endswith(suffix) for suffix in PRESIGNED_URL_HOST_SUFFIXES)
    )
    if not allowed:
        raise ValueError(
            f"Upload URL host '{host}' is not in the presigned-URL allow-list. "
            "Expected an S3 endpoint (e.g. bucket.s3.us-east-1.amazonaws.com), GCS "
            "(*.storage.googleapis.com), Azure Blob (*.blob.core.windows.net), "
            "an R2 account endpoint, or data.avala.ai."
        )


def is_retryable(exc: BaseException) -> bool:
    """Whether ``exc`` is worth retrying with backoff.

    An upload touches two very different error surfaces and both have to be
    classified here, or half the transient failures look permanent:

    * the **control plane** (presign) goes through the SDK transport, which
      converts status codes into :mod:`avala.errors` types before the caller
      ever sees an httpx exception;
    * the **data plane** (the presigned POST) is a raw ``httpx`` call, so it
      raises ``httpx.HTTPStatusError`` / transport errors directly.

    Import httpx lazily: this module is imported at package-import time, and
    httpx is only needed once bytes actually move.
    """
    import httpx

    from avala.errors import AvalaError

    if isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return _is_retryable_status(exc.response.status_code)
    if isinstance(exc, AvalaError):
        # Covers ServerError (5xx) and RateLimitError (429). A quota 413 or a
        # validation 400 is deliberately excluded — retrying cannot change it.
        return _is_retryable_status(exc.status_code)
    return False


def backoff_seconds(attempt: int) -> float:
    """Exponential backoff with jitter for retry number ``attempt`` (0-based).

    The jitter matters more than the curve: without it, N parallel workers that
    trip the same server-side limit all retry on the same tick and reproduce the
    burst that caused the failure.
    """
    return min(_MAX_BACKOFF_SECONDS, 2.0**attempt) + random.uniform(0, 1.5)


def _retry_after_seconds(exc: BaseException | None) -> float | None:
    """``Retry-After`` from either error surface, in seconds.

    The two planes report it differently and both matter:

    * the **control plane** (presign) goes through the SDK transport, which
      parses the header into ``RateLimitError.retry_after``;
    * the **data plane** (the presigned POST to S3/GCS/Azure) is raw ``httpx``,
      so a 429 arrives as ``HTTPStatusError`` with the header still sitting in
      ``exc.response.headers`` and no ``retry_after`` attribute at all.

    Reading only the attribute silently ignores every throttle the storage
    provider itself sends, which is the plane actually moving the bytes.
    """
    retry_after = getattr(exc, "retry_after", None)
    if isinstance(retry_after, (int, float)):
        return float(retry_after)
    response = getattr(exc, "response", None)
    header = getattr(response, "headers", None)
    raw = header.get("Retry-After") if header is not None else None
    if raw is None:
        return None
    try:
        # Only the delta-seconds form. The HTTP-date form is legal but rare
        # here, and misparsing it into a huge sleep is worse than falling back
        # to the generic backoff.
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return None


def retry_delay(attempt: int, exc: BaseException | None = None) -> float:
    """How long to wait before retry ``attempt``, honouring ``Retry-After``.

    A 429 carries the server's own answer to this question, and it can exceed
    the whole generic budget: eight exponential attempts run out in about two
    minutes, so ignoring a longer ``Retry-After`` fails an upload the server
    was willing to accept — it just wanted us to wait. Take whichever is
    longer; the backoff is a floor, not a substitute.
    """
    delay = backoff_seconds(attempt)
    retry_after = _retry_after_seconds(exc)
    if retry_after is not None and retry_after > delay:
        return retry_after
    return delay


def sleep_backoff(attempt: int, exc: BaseException | None = None, stop: Any = None) -> None:
    """Wait for :func:`retry_delay`. Split out so tests can patch it.

    ``stop`` is an optional ``threading.Event``. When given, the wait is done
    with ``stop.wait(...)`` rather than ``time.sleep``, so a worker parked on a
    long backoff wakes the moment a peer hits a permanent error.

    That matters more than it looks: a provider ``Retry-After`` can be minutes,
    cancelling an already-running future does not interrupt ``time.sleep``, and
    the executor's shutdown waits for the sleeper. So a fail-fast upload — which
    is what the docstring promises and what the CLI reports — could sit silently
    for the whole throttle window before surfacing the error that already
    doomed the run.
    """
    delay = retry_delay(attempt, exc)
    if stop is not None:
        stop.wait(delay)
        return
    time.sleep(delay)


def _ensure_private_dir(state_dir: Path) -> None:
    """Create the state dir owner-only.

    Checkpoints record the API host, the owning organization and a hash of
    the credential, and they live in a predictable path under ``$HOME``. On a
    shared or multi-tenant box the default umask can leave them
    world-readable, which hands any local user a map of who uploads what,
    where.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    try:
        state_dir.chmod(0o700)
    except OSError:  # pragma: no cover - unsupported filesystem
        pass


def _restrict(path: Path) -> None:
    """Owner-only permissions on a state file. Never fatal."""
    try:
        path.chmod(0o600)
    except OSError:  # pragma: no cover - unsupported filesystem
        pass


def _identity_cache_path(state_dir: Path) -> Path:
    return state_dir / "identity.json"


def cached_user_uid(state_dir: Path, base_url: str, api_key: str | None) -> str | None:
    """Remembered user uid for this (host, credential), if one was resolved."""
    if not api_key:
        return None
    try:
        data = json.loads(_identity_cache_path(state_dir).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    value = data.get(_identity_key(base_url, api_key))
    return value if isinstance(value, str) else None


def remember_user_uid(state_dir: Path, base_url: str, api_key: str | None, user_uid: str) -> None:
    """Persist the uid this credential authenticates as. Never raises.

    Keyed on the credential rather than the host alone, so two accounts used
    from the same machine cannot inherit each other's identity. A rotated key is
    a cache miss, which costs exactly one lookup — not a re-upload, because the
    uid it resolves to is the same and so is the resulting fingerprint.
    """
    if not api_key:
        return
    path = _identity_cache_path(state_dir)
    try:
        _ensure_private_dir(state_dir)
        # Re-read immediately before writing, and write through a
        # process-unique temp file. The previous version shared one
        # `identity.tmp` across processes — so two SDK or CLI runs resolving
        # different keys could replace that file under each other, and the last
        # writer persisted a dictionary that had never seen its peer's entry,
        # dropping it. A dropped entry is not cosmetic: the next run for that
        # key falls back to the API-key identity and cannot see the uid-keyed
        # remote set.
        #
        # This narrows the window to the read-modify-write itself rather than
        # closing it — a true fix needs an interprocess lock, which is not worth
        # it for a cache whose miss costs one request. Losing a race now costs a
        # re-lookup, not a wrong answer.
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, dict):
                data = {}
        except (OSError, ValueError):
            data = {}
        data[_identity_key(base_url, api_key)] = user_uid
        tmp = path.with_name(f"{path.name}.{os.getpid()}-{next(_WRITE_SEQ)}.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        _restrict(tmp)
        tmp.replace(path)
    except OSError:
        pass


def _canonical_uuid(value: str) -> str:
    """Canonical lowercase-hyphenated UUID, or ``value`` unchanged.

    Returned as-is when it does not parse, so a non-UUID identifier still
    compares equal to itself rather than being silently mangled.
    """
    import uuid as _uuid

    try:
        return str(_uuid.UUID(str(value)))
    except (ValueError, AttributeError, TypeError):
        return str(value)


def _identity_key(base_url: str, api_key: str) -> str:
    # Same canonicalization as the fingerprint. Normalizing one and not the
    # other meant caching the uid through api.avala.ai and then missing that
    # cache through server.avala.ai — falling back to the API-key identity,
    # which cannot see the uid-keyed remote set.
    return f"{_canonical_base_url(base_url)}|{hashlib.sha256(api_key.encode('utf-8')).hexdigest()[:16]}"


def migrate_state(state_dir: Path, key: str, *, from_fingerprint: str, to_fingerprint: str) -> None:
    """Adopt state written under ``from_fingerprint`` as ``to_fingerprint``.

    Existing canonical state is merged rather than replaced. Never raises:
    failing to migrate costs a re-upload, whereas failing the run costs the
    whole transfer.

    **Destination first, source last.** Every file is written to its new path
    with its embedded fingerprint already corrected, and only then is the source
    removed. Renaming first and rewriting afterwards looked equivalent and was
    not: a rewrite that fails on a full or flaky filesystem leaves the only copy
    at the canonical path still naming the old destination, which every loader
    rejects — while the source it came from is gone. That turns a failed
    optimisation into total loss of the compacted remote-key evidence, which is
    the one thing this migration exists to preserve.
    """
    if from_fingerprint == to_fingerprint:
        return

    def _rewrite_fingerprint(payload: str, *, json_lines: bool) -> str | None:
        """Return ``payload`` with the old fingerprint swapped, or None if absent."""
        if json_lines:
            if not payload.strip():
                return ""
            out: list[str] = []
            changed = False
            for line in payload.splitlines():
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue  # torn tail; dropping it matches the loader
                if isinstance(entry, dict) and entry.get("f") == from_fingerprint:
                    entry["f"] = to_fingerprint
                    changed = True
                out.append(json.dumps(entry, separators=(",", ":")))
            return ("\n".join(out) + "\n") if changed and out else None
        try:
            data = json.loads(payload)
        except ValueError:
            return None
        if not isinstance(data, dict) or data.get("fingerprint") != from_fingerprint:
            return None
        data["fingerprint"] = to_fingerprint
        return json.dumps(data, indent=2)

    def _as_journal_lines(payload: str, *, json_lines: bool) -> str | None:
        """Re-express ``payload`` as journal lines under the new fingerprint.

        A snapshot becomes one line per key it records. That is what lets a
        fallback checkpoint be MERGED into a canonical one that already exists:
        journal replay unions `remote` and applies completed/stamps in order, so
        appending is exactly the merge semantics the loader already implements.
        Refusing to merge — the earlier behaviour — hid the fallback state
        whenever both identities held live evidence, which is precisely the case
        a key rotation plus a transient lookup failure produces.
        """
        if json_lines:
            return _rewrite_fingerprint(payload, json_lines=True)
        try:
            data = json.loads(payload)
        except ValueError:
            return None
        if not isinstance(data, dict) or data.get("fingerprint") != from_fingerprint:
            return None
        completed = {k for k in data.get("uploaded", []) if isinstance(k, str)}
        remote = {k for k in data.get("remote", []) if isinstance(k, str)}
        stamps = data.get("stamps") if isinstance(data.get("stamps"), dict) else {}
        lines = []
        for relative in sorted(remote | completed):
            stamp = stamps.get(relative) if isinstance(stamps, dict) else None
            lines.append(
                json.dumps(
                    {
                        "r": relative,
                        "s": stamp if isinstance(stamp, list) and len(stamp) >= 2 else None,
                        "c": relative in completed,
                        "f": to_fingerprint,
                    },
                    separators=(",", ":"),
                )
            )
        # A valid empty snapshot is a no-op, not a failed migration. Returning
        # ``None`` here short-circuits adoption before a non-empty source
        # journal can contribute its remote-key evidence.
        return ("\n".join(lines) + "\n") if lines else ""

    def _adopt(src: Path, dst: Path, *, json_lines: bool) -> bool:
        if not src.exists():
            return True
        try:
            payload = src.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return False

        if not dst.exists():
            corrected = _rewrite_fingerprint(payload, json_lines=json_lines)
            if corrected is None:
                return False
            try:
                tmp = dst.with_name(f"{dst.name}.{os.getpid()}-{next(_WRITE_SEQ)}.tmp")
                tmp.write_text(corrected, encoding="utf-8")
                _restrict(tmp)
                tmp.replace(dst)  # destination is durable from here on
            except OSError:
                return False
        else:
            # Destination already holds live state. Fold the source in as
            # journal entries rather than dropping it: both sides can contain
            # keys the other has never seen.
            merged = _as_journal_lines(payload, json_lines=json_lines)
            if merged is None:
                return False
            journal = journal_path(state_dir, key, fingerprint=to_fingerprint)
            try:
                _ensure_private_dir(state_dir)
                # Both source and destination locks surround the whole migration
                # below, so this direct append cannot race destination compaction.
                existed = journal.exists()
                with open(journal, "a", encoding="utf-8") as fh:
                    fh.write(merged)
                    fh.flush()
                    os.fsync(fh.fileno())
                if not existed:
                    _restrict(journal)
            except OSError:
                return False
        return True

    # Lock both identities in path order. The deterministic order prevents two
    # concurrent recovery attempts from deadlocking, while holding the SOURCE
    # lock through its reads and retirement closes the append-after-read race.
    fingerprints = sorted(
        {from_fingerprint, to_fingerprint},
        key=lambda value: str(state_path(state_dir, key, fingerprint=value).with_suffix(".lock")),
    )
    try:
        with contextlib.ExitStack() as locks:
            for value in fingerprints:
                locks.enter_context(_state_lock(state_dir, key, value))

            src_snapshot = state_path(state_dir, key, fingerprint=from_fingerprint)
            dst_snapshot = state_path(state_dir, key, fingerprint=to_fingerprint)
            src_journal = journal_path(state_dir, key, fingerprint=from_fingerprint)
            dst_journal = journal_path(state_dir, key, fingerprint=to_fingerprint)
            src_batch = src_snapshot.with_suffix(".batch")
            dst_batch = dst_snapshot.with_suffix(".batch")
            if src_batch.exists():
                if dst_batch.exists() and src_batch.read_text() != dst_batch.read_text():
                    raise UploadStateError(
                        "Two upload batches have different identities; resume each original batch separately."
                    )
                if not dst_batch.exists():
                    dst_batch.write_bytes(src_batch.read_bytes())
                    _restrict(dst_batch)
            try:
                orphans = sorted(src_journal.parent.glob(f"{src_journal.name}.*.compacting"))
            except OSError:
                orphans = []

            work: list[tuple[Path, Path, bool]] = [
                (src_snapshot, dst_snapshot, False),
                (src_journal, dst_journal, True),
            ]
            for orphan in orphans:
                suffix = orphan.name[len(src_journal.name) :]
                work.append((orphan, dst_journal.with_name(f"{dst_journal.name}{suffix}"), True))

            # Copy first, publish the redirect second, retire the source last.
            # If any write fails, every source file remains available for a
            # later retry. Once the marker is durable, a fallback appender that
            # was waiting on this lock follows it to the canonical journal.
            if not all(_adopt(src, dst, json_lines=json_lines) for src, dst, json_lines in work):
                return
            _write_redirect_unlocked(state_dir, key, from_fingerprint, to_fingerprint)
            src_batch.unlink(missing_ok=True)
            for src, _dst, _json_lines in work:
                try:
                    src.unlink(missing_ok=True)
                except OSError:
                    pass  # a duplicate under the old identity is harmless; loss is not
    except OSError:
        return


def state_path(state_dir: Path, key: str, *, fingerprint: str | None = None) -> Path:
    """Path of the checkpoint file for ``key`` under ``state_dir``.

    ``key`` is caller-supplied (a dataset slug, a recording uid). It is
    flattened so a value containing a separator can't write outside
    ``state_dir`` — slugs are validated server-side, but this file is written
    before any server round-trip confirms that.

    ``fingerprint`` gives each **destination** its own file. Validating the
    fingerprint on read is not sufficient on its own: a multi-org operator
    using the same conventional slug in orgs A and B shares one filename, so
    starting B correctly ignores A's contents and then *overwrites* them, and a
    clean finish deletes the file. Coming back to A there is no record that its
    objects exist — and since finalization indexes everything under the prefix,
    a file deleted locally in the meantime is silently included in A's dataset.
    Separate files mean the two runs cannot destroy each other's evidence.
    """
    safe = key.replace("/", "_").replace("\\", "_").strip(". ") or "upload"
    if fingerprint:
        digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:12]
        return state_dir / f"{safe}-{digest}.json"
    return state_dir / f"{safe}.json"


def journal_path(state_dir: Path, key: str, *, fingerprint: str | None = None) -> Path:
    """Append-log companion to the snapshot at :func:`state_path`."""
    return state_path(state_dir, key, fingerprint=fingerprint).with_suffix(".journal")


def redirect_path(state_dir: Path, key: str, *, fingerprint: str) -> Path:
    """Durable forwarding marker for a provisional checkpoint identity.

    A personal upload can begin under an API-key fallback while ``/users/me/``
    is unavailable and later recover the immutable user uid. Merely moving the
    files is racy: a process that was already uploading under the fallback can
    append again as soon as the migration releases its lock, recreating state
    that the canonical identity cannot see. The marker makes every later state
    write follow the migration instead.
    """
    return state_path(state_dir, key, fingerprint=fingerprint).with_suffix(".redirect")


@contextlib.contextmanager
def _open_lock_file(path: Path) -> Iterator[TextIO]:
    """Open a lock file while allowing its lifetime to span a caller's yield."""
    with path.open("a+") as handle:
        yield handle


@contextlib.contextmanager
def _strict_state_lock(lock_path: Path) -> Iterator[None]:
    """Fail closed on contention or unsupported locking; retain the lock inode."""
    lock_stack = contextlib.ExitStack()
    unlock: Callable[[], None] | None = None
    try:
        _strict_private_dir(lock_path.parent)
        before = lock_path.lstat() if os.path.lexists(lock_path) else None
        if before is not None and not stat.S_ISREG(before.st_mode):
            raise OSError("Invalid lock file")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(lock_path, flags, 0o600)
        handle = lock_stack.enter_context(os.fdopen(descriptor, "r+"))
        opened = os.fstat(handle.fileno())
        retained = lock_path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or (before is not None and (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino))
            or not stat.S_ISREG(retained.st_mode)
            or ((retained.st_dev, retained.st_ino) != (opened.st_dev, opened.st_ino))
        ):
            raise OSError("Invalid lock file")
        if os.name == "posix":
            os.fchmod(handle.fileno(), 0o600)
        unlock = _acquire_strict_lock(handle)
    except (ImportError, OSError):
        lock_stack.close()
        raise UploadStateError("Fleet upload is busy or its checkpoint cannot be locked safely. Retry later.") from None
    try:
        yield
    finally:
        try:
            unlock()
        finally:
            lock_stack.close()


def _strict_private_dir(directory: Path) -> None:
    """Refuse directory aliases; require POSIX modes, inherit Windows ACLs."""
    if directory.is_symlink():
        raise OSError("Invalid state directory")
    directory.mkdir(parents=True, exist_ok=True)
    if not stat.S_ISDIR(directory.lstat().st_mode):
        raise OSError("Invalid state directory")
    if os.name == "posix":
        directory.chmod(0o700)


def _acquire_strict_lock(handle: TextIO, *, platform: str = os.name) -> Callable[[], None]:
    """Use a nonblocking native lock, including the Python 3.9 Windows backend."""
    if platform == "nt":
        import msvcrt

        handle.seek(0)
        # The CRT permits locking beyond EOF, so an empty persistent file needs
        # no sentinel write that could race another process's locked byte.
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]  # Windows-only stdlib API

        def unlock() -> None:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]  # Windows-only stdlib API

        return unlock
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def _state_lock(state_dir: Path, key: str, fingerprint: str | None, *, strict: bool = False) -> Iterator[None]:
    """Serialize journal appends against claim/compaction, across processes.

    Two prior rounds narrowed this race (per-process temp files, claim by
    rename) and documented that closing it needed a real lock. It does: an
    appender that has already OPENED the live journal is invisible to the
    rename, so a compaction can claim, absorb and unlink that path while the
    first process still holds the old inode. Its write then succeeds into an
    unlinked file — reported as durable, readable by nobody — and that is the
    one record proving an object exists.

    The default mode retains historical advisory POSIX locking and fail-open
    behavior on unavailable locks or Windows. Opt-in ``strict`` mode uses a
    nonblocking POSIX or Windows lock and fails closed; callers choose the
    lock lifetime.
    """
    if strict:
        lock_path = state_path(state_dir, key, fingerprint=fingerprint).with_suffix(".lock")
        with _strict_state_lock(lock_path):
            yield
        return
    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-POSIX
        yield
        return
    lock_path = state_path(state_dir, key, fingerprint=fingerprint).with_suffix(".lock")
    lock_stack = contextlib.ExitStack()
    try:
        _ensure_private_dir(state_dir)
        handle = lock_stack.enter_context(_open_lock_file(lock_path))
        _restrict(lock_path)
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    except OSError:
        # An unlockable state dir must not fail the upload; the caller's own
        # error handling still applies to the write it was about to make.
        lock_stack.close()
        yield
        return
    try:
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            lock_stack.close()


def _read_redirect_unlocked(state_dir: Path, key: str, fingerprint: str | None) -> str | None:
    """Return a migrated destination while the source state lock is held."""
    if fingerprint is None:
        return None
    try:
        data = json.loads(redirect_path(state_dir, key, fingerprint=fingerprint).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    target = data.get("fingerprint") if isinstance(data, dict) else None
    return target if isinstance(target, str) and target and target != fingerprint else None


def _write_redirect_unlocked(state_dir: Path, key: str, from_fingerprint: str, to_fingerprint: str) -> None:
    """Publish a forwarding marker atomically while both state locks are held."""
    path = redirect_path(state_dir, key, fingerprint=from_fingerprint)
    tmp = path.with_name(f"{path.name}.{os.getpid()}-{next(_WRITE_SEQ)}.tmp")
    try:
        tmp.write_text(json.dumps({"fingerprint": to_fingerprint}, indent=2), encoding="utf-8")
        _restrict(tmp)
        tmp.replace(path)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


@contextlib.contextmanager
def _resolved_state_lock(state_dir: Path, key: str, fingerprint: str | None) -> Iterator[str | None]:
    """Lock the current checkpoint identity, following durable migrations.

    The redirect is checked *after* taking each source lock. Therefore either an
    append wins first and migration copies it, or migration wins first and the
    append follows its marker; there is no gap in which a retired fallback
    journal can be recreated invisibly.
    """
    current = fingerprint
    visited: set[str] = set()
    while True:
        with _state_lock(state_dir, key, current):
            target = _read_redirect_unlocked(state_dir, key, current)
            if target is None or target in visited:
                yield current
                return
            if current is not None:
                visited.add(current)
            current = target


def _check_checkpoint_batch(state_dir: Path, key: str, fingerprint: str | None, expected_batch: str | None) -> None:
    """Fence writes while the caller holds the resolved destination lock."""
    path = state_path(state_dir, key, fingerprint=fingerprint).with_suffix(".batch")
    current = path.read_text().strip() if path.exists() else None
    expected = str(uuid.UUID(expected_batch)) if expected_batch is not None else None
    if current != expected:
        raise UploadStateError("Upload batch changed during this upload; resume the current batch before continuing.")


def append_completed(
    state_dir: Path,
    key: str,
    relative: str,
    *,
    stamp: list[int] | None,
    completed: bool,
    fingerprint: str | None = None,
    strict: bool = False,
    expected_batch: str | None = None,
) -> None:
    """Durably record one file's outcome, in O(1).

    The snapshot in :func:`save_completed` rewrites the entire set, so writing
    it per file is quadratic; throttling it instead leaves a window in which a
    killed process loses keys that really did reach the bucket. That is not
    merely lost work — the checkpoint is also what detects a confirmed file
    being deleted locally before a retry, so losing keys silently weakens that
    check and lets a stale remote object be indexed at finalization.

    One appended line per file gives both properties: constant cost, and a
    record that survives `kill -9` the moment it is written. The snapshot
    becomes a compaction of this log rather than the only copy.

    Batch rotation always raises to prevent stale writers contaminating the
    replacement checkpoint. Filesystem errors raise only when strict is set.
    """
    try:
        _ensure_private_dir(state_dir)
        # Held across redirect resolution + open + write + fsync. A fallback
        # uploader waiting behind identity migration therefore writes to the
        # canonical journal instead of recreating a retired source journal.
        with _resolved_state_lock(state_dir, key, fingerprint) as resolved:
            _check_checkpoint_batch(state_dir, key, resolved, expected_batch)
            line = json.dumps({"r": relative, "s": stamp, "c": completed, "f": resolved}, separators=(",", ":"))
            jpath = journal_path(state_dir, key, fingerprint=resolved)
            existed = jpath.exists()
            with open(jpath, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            if not existed:
                _restrict(jpath)
    except OSError:
        if strict:
            # This line is the durable evidence that a provider POST succeeded.
            # Swallowing the failure — on a full disk, a read-only home, a
            # failed mount — lets the run continue with no record that the
            # object exists, so an interrupted upload whose local file is then
            # removed finalizes the stale remote bytes with nothing to catch it.
            # Callers recording a completed POST pass strict=True and let the
            # upload fail loudly instead.
            raise
        # Best-effort callers (already handling an error of their own) keep the
        # old behaviour: a missing record costs repeated work, not correctness.


def _replay_journal(
    state_dir: Path,
    key: str,
    completed: set[str],
    stamps: dict[str, list[int]],
    fingerprint: str | None,
    remote: set[str] | None = None,
) -> None:
    """Fold journal lines over a loaded snapshot, in write order.

    Tolerates a truncated final line: a process killed mid-append leaves a
    partial record, and discarding just that line is right — every earlier line
    is intact and fsync'd. The caller must hold the resolved state lock across
    both its snapshot read and this replay, so claim discovery and file reads
    describe one atomic checkpoint view.
    """
    jpath = journal_path(state_dir, key, fingerprint=fingerprint)
    # Also replay any journal a compaction claimed and then died before folding
    # in (see ``save_completed``). Those entries are on disk and in nobody's
    # snapshot, so skipping them loses exactly the evidence the journal exists
    # to keep. Sorted so replay order is deterministic; the live journal last,
    # because it is the newest.
    chunks: list[str] = []
    try:
        leftovers = sorted(jpath.parent.glob(f"{jpath.name}.*.compacting"))
    except OSError:
        leftovers = []
    for candidate in [*leftovers, jpath]:
        try:
            chunks.append(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            continue
    for line in "\n".join(chunks).splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn tail, or a line from a crashed write
        if not isinstance(entry, dict):
            continue
        if fingerprint is not None and entry.get("f") != fingerprint:
            continue
        relative = entry.get("r")
        if not isinstance(relative, str):
            continue
        stamp = entry.get("s")
        if isinstance(stamp, list) and len(stamp) >= 2:
            try:
                stamps[relative] = [int(v) for v in stamp]
            except (TypeError, ValueError):
                pass
        if remote is not None:
            # The line exists because a PUT succeeded, so the key exists — even
            # when ``c`` is false because the source changed under the handle.
            remote.add(relative)
        if entry.get("c"):
            completed.add(relative)
        else:
            completed.discard(relative)


def _load_upload_state_unlocked(
    state_dir: Path,
    key: str,
    fingerprint: str | None,
) -> tuple[set[str], dict[str, list[int]], set[str]]:
    """Read snapshot, claims, and live journal while the state lock is held."""
    completed: set[str] = set()
    stamps: dict[str, list[int]] = {}
    remote: set[str] = set()
    path = state_path(state_dir, key, fingerprint=fingerprint)
    try:
        payload = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        data = None
    except (OSError, UnicodeDecodeError) as exc:
        raise UploadStateError(
            f"Upload checkpoint '{path}' exists but cannot be read; refusing to discard resume evidence."
        ) from exc
    else:
        try:
            data = json.loads(payload)
        except ValueError as exc:
            raise UploadStateError(
                f"Upload checkpoint '{path}' is corrupt; refusing to discard resume evidence."
            ) from exc
        if not isinstance(data, dict):
            raise UploadStateError(f"Upload checkpoint '{path}' has an invalid structure; refusing to resume.")
        if fingerprint is not None and data.get("fingerprint") != fingerprint:
            raise UploadStateError(
                f"Upload checkpoint '{path}' has the wrong destination fingerprint; refusing to resume."
            )

        for field in ("uploaded", "remote"):
            value = data.get(field)
            if value is not None and (not isinstance(value, list) or any(not isinstance(item, str) for item in value)):
                raise UploadStateError(
                    f"Upload checkpoint '{path}' has an invalid '{field}' field; refusing to resume."
                )
        stored_stamps = data.get("stamps")
        if stored_stamps is not None:
            if not isinstance(stored_stamps, dict):
                raise UploadStateError(f"Upload checkpoint '{path}' has an invalid 'stamps' field; refusing to resume.")
            for name, value in stored_stamps.items():
                if not isinstance(name, str) or not isinstance(value, list) or len(value) < 2:
                    raise UploadStateError(
                        f"Upload checkpoint '{path}' contains an invalid file stamp; refusing to resume."
                    )
                try:
                    [int(item) for item in value]
                except (TypeError, ValueError) as exc:
                    raise UploadStateError(
                        f"Upload checkpoint '{path}' contains an invalid file stamp; refusing to resume."
                    ) from exc

    if data is not None:
        uploaded = data.get("uploaded")
        if isinstance(uploaded, list):
            completed = {item for item in uploaded if isinstance(item, str)}
            # Pre-``remote`` snapshots still prove that every uploaded key
            # existed at the provider.
            remote |= completed
        stored_remote = data.get("remote")
        if isinstance(stored_remote, list):
            remote |= {item for item in stored_remote if isinstance(item, str)}
        stored_stamps = data.get("stamps")
        if isinstance(stored_stamps, dict):
            for name, value in stored_stamps.items():
                if not isinstance(name, str) or not isinstance(value, list) or len(value) < 2:
                    continue
                try:
                    stamps[name] = [int(v) for v in value]
                except (TypeError, ValueError):
                    continue
    _replay_journal(state_dir, key, completed, stamps, fingerprint, remote=remote)
    return completed, stamps, remote


def load_completed(state_dir: Path, key: str, *, fingerprint: str | None = None) -> set[str]:
    """Return the relative paths already confirmed uploaded for ``key``.

    A missing checkpoint yields an empty set. An existing unreadable or corrupt
    checkpoint fails closed: it may be the only record of remote objects, so
    silently treating it as empty can let finalization index stale provider
    keys after their local files have disappeared.

    ``fingerprint`` identifies **where** those bytes went — the API host and the
    owning organization. Each destination has a fingerprint-specific path, so a
    checkpoint at that path whose embedded fingerprint does not match is corrupt
    and fails closed. Trusting it could skip files that landed in a different
    organization's prefix; ignoring it could discard the only evidence of stale
    remote objects.
    """
    with _resolved_state_lock(state_dir, key, fingerprint) as resolved:
        completed, _stamps, _remote = _load_upload_state_unlocked(state_dir, key, resolved)
        return completed


def load_remote_keys(state_dir: Path, key: str, *, fingerprint: str | None = None) -> set[str]:
    """Every key this destination is known to hold — including invalidated ones.

    Distinct from :func:`load_completed`, and the distinction is load-bearing.
    ``completed`` answers "may this file be skipped?"; this answers "does an
    object already exist under that key?". Invalidation flips the first to no
    while the second stays yes: the bytes were uploaded, they are simply the
    wrong bytes, and nothing client-side can delete a remote key.

    Conflating them loses the only evidence of a stale object. Invalidate a
    file, delete it locally, re-run: with one set the removal is invisible, and
    finalization — which indexes every object under the prefix, not the manifest
    this run sent — silently folds the stale key into the dataset.
    """
    with _resolved_state_lock(state_dir, key, fingerprint) as resolved:
        _completed, _stamps, remote = _load_upload_state_unlocked(state_dir, key, resolved)
        return remote


def load_completed_stamps(state_dir: Path, key: str, *, fingerprint: str | None = None) -> dict[str, list[int]]:
    """Return ``{relative: [size, mtime_ns]}`` for confirmed files.

    Paths alone are not enough to decide a file can be skipped: regenerate or
    edit a source file between runs and a path-only checkpoint silently keeps
    the *previous* bytes in the dataset. Size and mtime are free (one ``stat``
    the caller already needs) and catch every realistic edit, unlike a checksum
    that would cost a full re-read of data we are trying to avoid re-sending.
    """
    with _resolved_state_lock(state_dir, key, fingerprint) as resolved:
        _completed, stamps, _remote = _load_upload_state_unlocked(state_dir, key, resolved)
        return stamps


def file_stamp(local_path: str) -> list[int]:
    """The resume identity of a file: ``[size, mtime_ns, ctime_ns, inode]``.

    Size and mtime alone are forgeable by ordinary tooling, not just by an
    adversary: ``cp -p``, ``rsync -t`` and ``os.utime()`` all replace contents
    while preserving both, so a regenerated file compared on those two looks
    unchanged and the upload is skipped — leaving the previous bytes in the
    dataset under that path.

    ``ctime_ns`` moves on any inode change including a metadata-preserving
    overwrite, and the inode number changes under the write-new-then-rename
    pattern most tools use. Both come from the same ``stat`` the caller already
    needs, so this stays free; a content digest would cost a full re-read of
    exactly the data resume exists to avoid re-sending.

    Two-element stamps from older checkpoints are still honoured on read (see
    ``_stamps_match``) — an upgrade must not invalidate work in progress.
    """
    import os

    info = os.stat(local_path)
    return [info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino]


def stamps_match(recorded: list[int] | None, current: list[int] | None) -> bool:
    """Whether ``current`` is the file ``recorded`` described.

    Compares only as many fields as the older of the two carries, so a
    checkpoint written before ctime/inode were recorded keeps working at its
    original (weaker) precision instead of forcing a full re-upload.
    """
    if recorded is None or current is None:
        return False
    width = min(len(recorded), len(current))
    if width < 2:
        return False
    return list(recorded[:width]) == list(current[:width])


# Recorded instead of dropping a stamp when the uploaded bytes cannot be
# vouched for (the source was replaced mid-transfer). A *missing* stamp has to
# keep meaning "checkpoint written before stamps existed — trust it", or every
# upgrade would re-send everything; this sentinel says the opposite explicitly,
# so invalidation survives into the next run instead of being read as legacy.
INVALID_STAMP = [-1, -1]


def save_completed(
    state_dir: Path,
    key: str,
    completed: Iterable[str],
    *,
    fingerprint: str | None = None,
    stamps: dict[str, list[int]] | None = None,
    remote: Iterable[str] | None = None,
    expected_batch: str | None = None,
    **extra: Any,
) -> None:
    """Persist the confirmed set for ``key``, atomically, and compact the journal.

    Written via a temp file + ``replace`` so a crash mid-write leaves the
    previous checkpoint intact rather than a truncated one. The temp name
    carries this process's pid: a single shared ``.tmp`` let two processes
    uploading the same destination overwrite each other's half-written file and
    then ``replace`` the result into place.

    **Compaction claims the journal by renaming it first.** Deleting it after
    writing the snapshot looked ordered-correctly for one process, but with two
    it destroyed entries the peer appended in between — and those entries are
    the record that an object exists remotely, which is what stops a stale
    object being finalized later. After the rename, a peer's next append lands
    in a brand-new journal that this call will not touch. The claimed file is
    folded into the snapshot here and replayed by the loader if this process
    dies before finishing.

    ``remote`` is merged with what is already on disk rather than replacing it.
    A key that ever existed remotely still does, so union is always the correct
    operation and never resurrects a wrong answer.

    The same state lock spans claim, snapshot/journal reads, snapshot replace,
    and claim retirement. Releasing it after only the rename still let two
    compactors read the same old snapshot and overwrite one another's newer
    remote-key evidence.
    """
    _ensure_private_dir(state_dir)
    unique = f"{os.getpid()}-{next(_WRITE_SEQ)}"

    # This lock is deliberately held until the new snapshot is durable and all
    # claims it absorbed are retired. Snapshot + claims + live journal must be
    # one view, not three individually safe reads separated by writer windows.
    with _resolved_state_lock(state_dir, key, fingerprint) as resolved:
        _check_checkpoint_batch(state_dir, key, resolved, expected_batch)
        path = state_path(state_dir, key, fingerprint=resolved)
        jpath = journal_path(state_dir, key, fingerprint=resolved)
        claim_target = jpath.with_name(f"{jpath.name}.{unique}.compacting")
        try:
            jpath.rename(claim_target)
        except OSError:
            pass  # nothing to compact, or someone else claimed it first

        # Enumerated under the same lock: a peer mid-claim would otherwise be
        # able to add or retire a claim between the rename and this listing.
        #
        # EVERY outstanding claim, not just the one this call made. A compaction
        # whose snapshot write failed deliberately leaves its claim behind (losing
        # it would drop fsynced keys), but nothing then retired it: the loader kept
        # replaying it on top of every later snapshot, so a stale ``c: false`` could
        # undo a completion that succeeded afterwards and force that file to upload
        # again on every single resume. Folding them all in here is what makes
        # preserving them safe.
        try:
            claimed = sorted(jpath.parent.glob(f"{jpath.name}.*.compacting"))
        except OSError:
            claimed = []

        _disk_completed, _disk_stamps, disk_remote = _load_upload_state_unlocked(state_dir, key, resolved)
        merged_remote: set[str] = set(remote or ())
        if remote is not None:
            merged_remote |= disk_remote

        payload: dict[str, Any] = {"uploaded": sorted(completed), **extra}
        if resolved is not None:
            payload["fingerprint"] = resolved
        if stamps is not None:
            payload["stamps"] = {name: list(value) for name, value in stamps.items()}
        if remote is not None:
            payload["remote"] = sorted(merged_remote)

        tmp = path.with_name(f"{path.name}.{unique}.tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2))
            _restrict(tmp)  # chmod BEFORE the rename, so the file is never briefly world-readable
            tmp.replace(path)  # atomic on POSIX
        except OSError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            # Leave the claimed journal exactly where it is. This invocation
            # never absorbed it, so deleting it would discard the only fsync'd
            # record of those keys. The loader replays `*.compacting` leftovers,
            # which makes leaving it both safe and self-healing.
            raise
        # Only once the snapshot that absorbed it is actually in place. Testing
        # `path.exists()` instead was wrong: an *older* snapshot satisfies it,
        # so a failed write deleted a journal nothing had folded in.
        for stale in claimed:
            try:
                stale.unlink(missing_ok=True)
            except OSError:
                pass


def upload_fingerprint(
    base_url: str,
    organization_uid: str | None,
    dataset_name: str,
    api_key: str | None = None,
    storage_root: str | None = None,
    *,
    user_uid: str | None = None,
) -> str:
    """Identity of an upload *destination*, for checkpoint validation.

    Strictly the destination, and nothing about who is talking to it. Every
    input must be something the server folds into the S3 key prefix or the
    ownership of it, so that a change means the recorded files did not go where
    this run is about to send them:

    * ``base_url`` — a different environment entirely.
    * ``organization_uid`` — selects the owning prefix, and is immutable.
    * ``dataset_name`` — the prefix's last segment. Note that callers key state
      by *slug*, which can stay fixed while the name changes.
    * ``user_uid`` — the immutable owner of a personal upload, and what actually
      decides its prefix (``__u__=/<user_uid>``). Preferred whenever known.
    * ``api_key`` — hashed, never stored, and used **only** as the last-resort
      stand-in for ``user_uid`` on a personal upload whose owner could not be
      resolved. It is the wrong identity for the job: rotating a key, or the
      same person using a second key, selects a different checkpoint even
      though the server still writes to the same uid-rooted prefix — so
      previously uploaded files vanish from the remote-key record and a file
      deleted locally in between can no longer be caught before its stale
      object is finalized. Falling back is still better than the alternatives,
      because it errs toward a *fresh* checkpoint rather than one belonging to
      a different account; see ``cached_user_uid``, which makes the fallback
      rare by persisting the uid across rotations.

    ``api_key`` is deliberately excluded once ``organization_uid`` is present.
    An organization's destination is fixed by the uid and the dataset name; the
    credential that gets you there is not part of it. Folding it in meant a key
    rotation mid-transfer — or a second authorized member resuming a colleague's
    multi-hour upload — produced a different fingerprint, hiding every recorded
    remote key. Deleting one of those files locally before the retry then slipped
    past the stale-key check while finalization still indexed the old object from
    the same prefix.

    ``storage_root`` is accepted for compatibility and ignored: both roots are
    now derived from values already here (``__o__=/<org_uid>`` and
    ``__u__=/<user_uid>``), so including it added no information while making
    the fingerprint depend on a fallible lookup.

    Deliberately not the file contents: re-hashing gigabytes to decide whether
    to skip them would cost more than just re-uploading.
    """
    del storage_root  # see docstring
    # Canonical spelling. The server treats uppercase, brace-wrapped and
    # unhyphenated UUIDs as the same organization, so embedding the caller's
    # raw spelling let two valid ways of naming one destination select two
    # different checkpoints — hiding the remote-key evidence of the first.
    organization_uid = _canonical_uuid(organization_uid) if organization_uid else organization_uid
    account = ""
    if not organization_uid:
        if user_uid:
            account = f"u:{user_uid}"
        elif api_key:
            account = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
    return f"{_canonical_base_url(base_url)}|{organization_uid or ''}|{dataset_name}|{account}"


def clear_completed(
    state_dir: Path, key: str, *, fingerprint: str | None = None, expected_batch: str | None = None
) -> None:
    """Retire only this successful operation's checkpoint, never a newer batch.

    Never raises after finalization: filesystem errors retain recovery state.
    Legacy callers omit expected_batch and may clear only an unbound checkpoint.
    """
    try:
        with _resolved_state_lock(state_dir, key, fingerprint) as resolved:
            batch_path = state_path(state_dir, key, fingerprint=resolved).with_suffix(".batch")
            current = str(uuid.UUID(batch_path.read_text().strip())) if batch_path.exists() else None
            expected = str(uuid.UUID(expected_batch)) if expected_batch is not None else None
            if current != expected:
                return
            if current is not None:
                with _state_lock(state_dir / "managed-sources", current, None):
                    shutil.rmtree(_managed_source_directory(state_dir, current), ignore_errors=True)
            for path in (
                state_path(state_dir, key, fingerprint=resolved),
                journal_path(state_dir, key, fingerprint=resolved),
                batch_path,
            ):
                path.unlink(missing_ok=True)
    except (OSError, ValueError, UploadStateError):
        pass


def managed_upload_batch(state_dir: Path, key: str, *, fingerprint: str, resume: bool = True) -> str | None:
    """Persist the provider-binding identity before sending any file bytes.

    A legacy checkpoint must finish with the original protocol because its
    already-uploaded objects do not belong to a managed upload batch.
    """
    _ensure_private_dir(state_dir)
    with _resolved_state_lock(state_dir, key, fingerprint) as resolved:
        path = state_path(state_dir, key, fingerprint=resolved).with_suffix(".batch")
        if path.exists():
            try:
                previous_batch = str(uuid.UUID(path.read_text().strip()))
                if resume:
                    return previous_batch
            except ValueError as exc:
                raise UploadStateError("Upload batch checkpoint is invalid; preserve it for recovery.") from exc
        completed, _stamps, remote = _load_upload_state_unlocked(state_dir, key, resolved)
        if not path.exists() and (completed or remote):
            return None
        if path.exists():
            # Keep the old remote session discoverable for recovery. Restarting
            # locally never aborts its multipart upload or deletes its objects.
            history = path.with_suffix(".retired-batches")
            with history.open("a") as handle:
                _restrict(history)
                handle.write(previous_batch + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            # Invalidate skips before publishing a new batch. A crash leaves
            # either the old recoverable batch or a fresh batch with no skips.
            # Keep every remote key so legacy POST's stale-file guard still works.
            with journal_path(state_dir, key, fingerprint=resolved).open("a") as journal:
                _restrict(Path(journal.name))
                for relative in completed:
                    journal.write(json.dumps({"r": relative, "s": INVALID_STAMP, "c": False, "f": resolved}) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
        batch = str(uuid.uuid4())
        temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        with temporary.open("x") as handle:
            _restrict(temporary)
            handle.write(batch)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        return batch


def _managed_source_directory(state_dir: Path, batch: str) -> Path:
    return state_dir / "managed-sources" / hashlib.sha256(batch.encode()).hexdigest()


def bind_managed_source(state_dir: Path, batch: str, relative: str, stamp: list[int]) -> None:
    """Reject changed files before reusing an immutable managed upload session."""
    root = state_dir / "managed-sources"
    _ensure_private_dir(root)
    # One stable lock inode per batch, rather than one permanent lock per file.
    # Keep it outside the disposable directory so concurrent waiters stay safe.
    with _state_lock(root, batch, None):
        directory = _managed_source_directory(state_dir, batch)
        _ensure_private_dir(directory)
        identity = hashlib.sha256(relative.encode()).hexdigest()
        path = directory / f"{identity}.json"
        if path.exists():
            try:
                previous = json.loads(path.read_text())
            except ValueError as exc:
                raise UploadStateError("Managed source checkpoint is invalid; start a fresh upload batch.") from exc
            if previous != stamp:
                raise UploadStateError("Local file changed since this batch started; start a fresh upload batch.")
            return
        with path.open("x") as handle:
            _restrict(path)
            json.dump(stamp, handle)
            handle.flush()
            os.fsync(handle.fileno())


def bind_managed_batch(state_dir: Path, key: str, *, fingerprint: str, batch: str | None) -> None:
    """Do not let another batch inherit this destination's completed files."""
    _ensure_private_dir(state_dir)
    with _resolved_state_lock(state_dir, key, fingerprint) as resolved:
        path = state_path(state_dir, key, fingerprint=resolved).with_suffix(".batch")
        if path.exists():
            if batch is None or path.read_text().strip() != str(uuid.UUID(batch)):
                raise UploadStateError("Upload batch differs from the saved checkpoint; use its original batch UUID.")
            return
        if batch is None:
            return
        completed, _stamps, remote = _load_upload_state_unlocked(state_dir, key, resolved)
        if completed or remote:
            raise UploadStateError("A new managed batch cannot reuse legacy file checkpoints.")
        with path.open("x") as handle:
            _restrict(path)
            handle.write(str(uuid.UUID(batch)))
            handle.flush()
            os.fsync(handle.fileno())
