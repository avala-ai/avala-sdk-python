"""Datasets resource."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Dict, List, Tuple

from avala._pagination import CursorPage
from avala._uploads import (
    INVALID_STAMP,
    MAX_RETRIES,
    STATE_DIR,
    append_completed,
    bind_managed_batch,
    bind_managed_source,
    cached_user_uid,
    clear_completed,
    file_stamp,
    is_retryable,
    load_completed,
    load_completed_stamps,
    load_remote_keys,
    migrate_state,
    managed_upload_batch,
    remember_user_uid,
    save_completed,
    sleep_backoff,
    stamps_match,
    upload_fingerprint,
    validate_presigned_url,
)
from avala.errors import QuotaExceededError
from avala.resources._base import BaseAsyncResource, BaseSyncResource
from avala.types.dataset import (
    CameraCalibration,
    Dataset,
    DatasetCalibration,
    DatasetFrame,
    DatasetHealth,
    DatasetItem,
    DatasetSequence,
    FrameImage,
    Quat,
    Vec3,
)
from avala.types.manual_upload import AllowedMimes, UploadQuota

_MIN_INTERVAL = 1.0

# Minimum wall-clock gap between checkpoint SNAPSHOT rewrites. A snapshot
# re-serializes the entire completed set, so writing one per file is quadratic
# on large datasets. Throttling it is safe because it is not the durable
# record: every file is appended to the journal first, unthrottled and fsync'd
# (``append_completed``). The snapshot is compaction — see ``_record``.
_CHECKPOINT_INTERVAL = 2.0

# Per-operation bounds for the data-plane POST, as (connect, read, write, pool).
# Not a cap on the whole upload: each bounds how long one step may make no
# progress, so a multi-gigabyte file over a slow link is fine while a half-open
# socket is not. Unbounded waits here defeated both the retry classifier and the
# fail-fast path, since an in-flight request cannot be cancelled.
#
# Kept as plain numbers rather than an ``httpx.Timeout``: this module is
# imported at package-import time and httpx is only needed once bytes move.
_UPLOAD_TIMEOUTS = (30.0, 300.0, 600.0, 60.0)


# Aliased so tests can redirect resume state away from the real home directory
# by monkeypatching this module attribute (see ``tests/test_fleet.py`` for the
# same pattern on the fleet path).
#
# Namespaced under ``datasets/`` because the checkpoint filename is derived from
# a caller-supplied key: a dataset slug that happened to equal a fleet recording
# uid would otherwise have the two paths overwriting and deleting each other's
# state in the shared directory.
_STATE_DIR = STATE_DIR / "datasets"


def gather_local_files(source: str) -> list[tuple[str, str]]:
    """Return ``(local_path, file_path_in_dataset)`` for a file or directory tree.

    Skips the resume-state directory when it falls inside ``source``. Uploading
    from ``~`` (or any other ancestor of ``~/.avala/uploads``) would otherwise
    sweep the checkpoints in as dataset content: it publishes upload metadata
    into the dataset, and worse, makes the run's own checkpoint a source file
    that mutates while workers write to it — so the end-of-run manifest check
    sees it change and aborts an upload that was otherwise fine.

    Prunes the **shared** ``STATE_DIR``, not just this module's ``datasets/``
    child. Fleet recording checkpoints sit directly under the shared parent
    (``resources/fleet/uploads.py``), so excluding only the child still uploads
    them — leaking their session uid, source path and progress into a dataset
    that has nothing to do with the fleet.

    The test fixture in ``tests/conftest.py`` relocates state out of ``tmp_path``
    for exactly this reason; production has the same hazard and needs the same
    exclusion.
    """
    import os
    from pathlib import Path

    root_path = Path(source)
    # ``_STATE_DIR`` is what tests monkeypatch, so derive from it rather than
    # importing STATE_DIR directly — otherwise a redirected test would prune the
    # real home directory and miss the relocated one.
    try:
        state_root = _STATE_DIR.parent.resolve()
    except OSError:  # pragma: no cover - resolve() on an unreadable parent
        state_root = _STATE_DIR.parent

    # Refuse a source that IS the state root, sits inside it, or is one of its
    # files. Pruning below only skips it as a *child* of the walk, so pointing
    # the uploader straight at `~/.avala/uploads` (or a checkpoint file) walked
    # right past the guard and uploaded the checkpoints as dataset content —
    # publishing the API host and upload metadata, and making the files this
    # very run is writing part of its own manifest, which the end-of-run
    # validation then correctly aborts on.
    try:
        resolved = root_path.resolve()
    except OSError:  # pragma: no cover - unreadable parent
        resolved = root_path
    if resolved == state_root or state_root in resolved.parents:
        raise ValueError(
            f"{source} is inside the Avala upload state directory ({state_root}). That directory holds "
            "resume checkpoints, not dataset content — uploading it would publish upload metadata and "
            "include files this run is still writing. Point --source at your data instead."
        )

    if root_path.is_file():
        return [(str(root_path), root_path.name)]
    out: list[tuple[str, str]] = []
    for root, dirs, files in os.walk(root_path):
        # Prune in place so os.walk does not descend into it at all.
        dirs[:] = [d for d in dirs if Path(root, d).resolve(strict=False) != state_root]
        for fname in sorted(files):
            local_path = Path(root) / fname
            # Pruning above only removes state DIRECTORIES from the walk. A file
            # symlink pointing at a checkpoint or journal is listed as an
            # ordinary file, and the upload follows it later — publishing that
            # metadata into the dataset, and racing a journal this same run is
            # still writing. Resolve and skip anything that lands in the state
            # tree.
            try:
                if (
                    local_path.resolve(strict=False) == state_root
                    or state_root in local_path.resolve(strict=False).parents
                ):
                    continue
            except OSError:  # pragma: no cover - unresolvable link
                continue
            out.append((str(local_path), local_path.relative_to(root_path).as_posix()))
    return out


def _build_frame(frames: list[dict[str, Any]], frame_idx: int, sequence_uid: str) -> DatasetFrame:
    if not 0 <= frame_idx < len(frames):
        raise IndexError(f"frame_idx {frame_idx} out of range for sequence {sequence_uid} with {len(frames)} frames")
    raw = frames[frame_idx]
    images_raw = raw.get("images") or []
    images = [FrameImage(**{k: v for k, v in img.items() if k in FrameImage.model_fields}) for img in images_raw]
    device_position = Vec3(**raw["device_position"]) if isinstance(raw.get("device_position"), dict) else None
    device_heading = Quat(**raw["device_heading"]) if isinstance(raw.get("device_heading"), dict) else None
    model = raw.get("model") or raw.get("camera_model")
    return DatasetFrame(
        frame_index=frame_idx,
        key=raw.get("key"),
        model=model,
        camera_model=raw.get("camera_model") or raw.get("model"),
        xi=raw.get("xi"),
        alpha=raw.get("alpha"),
        device_position=device_position,
        device_heading=device_heading,
        images=images,
        raw=raw,
    )


def _build_calibration_from_sequence(sequence: DatasetSequence) -> DatasetCalibration:
    frames = sequence.frames or []
    if not frames:
        return DatasetCalibration(sequence_uid=sequence.uid, cameras=[])
    frame0 = frames[0]
    cameras: list[CameraCalibration] = []
    for img in frame0.get("images") or []:
        position = Vec3(**img["position"]) if isinstance(img.get("position"), dict) else None
        heading = Quat(**img["heading"]) if isinstance(img.get("heading"), dict) else None
        cameras.append(
            CameraCalibration(
                camera_id=img.get("camera") or img.get("camera_id") or img.get("sensor_id"),
                position=position,
                heading=heading,
                width=img.get("width"),
                height=img.get("height"),
                fx=img.get("fx"),
                fy=img.get("fy"),
                cx=img.get("cx"),
                cy=img.get("cy"),
                model=img.get("model") or img.get("camera_model") or frame0.get("model"),
                xi=img.get("xi") if img.get("xi") is not None else frame0.get("xi"),
                alpha=(img.get("alpha") if img.get("alpha") is not None else frame0.get("alpha")),
            )
        )
    return DatasetCalibration(sequence_uid=sequence.uid, cameras=cameras)


def _transfer_payload(*, organization_uid: str | None, owner_username: str | None) -> dict[str, Any]:
    """Build (and validate) a /transfer/ body.

    Checked client-side as well as server-side so the common mistake — passing
    both, or neither — is a clear ``ValueError`` at the call site rather than a
    400 the caller has to decode. The server check is the real one; this is
    ergonomics.
    """
    if (organization_uid is None) == (owner_username is None):
        raise ValueError("Pass exactly one of organization_uid or owner_username.")
    if organization_uid is not None:
        return {"organization_uid": organization_uid}
    return {"owner_username": owner_username}


class Datasets(BaseSyncResource):
    def list(
        self,
        *,
        data_type: str | None = None,
        name: str | None = None,
        status: str | None = None,
        visibility: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> CursorPage[Dataset]:
        """List authorized workspace datasets; visibility filtering grants no access.

        Servers supporting Unlisted accept private, unlisted, or public filters.
        A known Unlisted URL does not add a dataset to this workspace listing.
        """
        params: dict[str, Any] = {}
        if data_type is not None:
            params["data_type"] = data_type
        if name is not None:
            params["name"] = name
        if status is not None:
            params["status"] = status
        if visibility is not None:
            params["visibility"] = visibility
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page("/datasets/", Dataset, params=params or None)

    def get(self, uid: str) -> Dataset:
        data = self._transport.request("GET", f"/datasets/{uid}/")
        return Dataset.model_validate(data)

    def get_by_slug(self, owner: str, slug: str) -> Dataset:
        data = self._transport.request("GET", f"/datasets/{owner}/{slug}/")
        return Dataset.model_validate(data)

    def transfer(
        self,
        owner: str,
        slug: str,
        *,
        organization_uid: str | None = None,
        owner_username: str | None = None,
    ) -> Dataset:
        """Move a dataset to a different owner.

        Exactly one of ``organization_uid`` or ``owner_username`` must be given —
        a dataset is owned by a user XOR an organization, never both.

        Transferring requires the OWNER role on the dataset's current
        organization (an ADMIN may edit a dataset but may not give it away), and
        authority at the destination too. ``owner_username`` may only name
        *yourself*: the API refuses handing a dataset to another account, since
        that would park a tenant's data on a login that never agreed to take it.

        Returns the dataset at its NEW path; its ``owner_name`` is what the
        canonical ``/@<owner>/datasets/<slug>`` URL now uses.
        """
        payload = _transfer_payload(organization_uid=organization_uid, owner_username=owner_username)
        data = self._transport.request("POST", f"/datasets/{owner}/{slug}/transfer/", json=payload)
        return Dataset.model_validate(data)

    def create(
        self,
        *,
        name: str,
        slug: str,
        data_type: str,
        visibility: str = "private",
        create_metadata: bool = True,
        provider_config: dict[str, Any] | None = None,
        owner_name: str | None = None,
        organization_id: int | None = None,
        organization_uid: str | None = None,
        gpu_texture_format: str | None = None,
        metadata: dict[str, Any] | None = None,
        industry: int | None = None,
        license: int | None = None,
    ) -> Dataset:
        payload: dict[str, Any] = {
            "name": name,
            "slug": slug,
            "data_type": data_type,
            "visibility": visibility,
            "create_metadata": create_metadata,
        }
        if provider_config is not None:
            payload["provider_config"] = provider_config
        if owner_name is not None:
            payload["owner_name"] = owner_name
        if organization_id is not None:
            payload["organization_id"] = organization_id
        if organization_uid is not None:
            payload["organization_uid"] = organization_uid
        if gpu_texture_format is not None:
            payload["gpu_texture_format"] = gpu_texture_format
        if metadata is not None:
            payload["metadata"] = metadata
        if industry is not None:
            payload["industry"] = industry
        if license is not None:
            payload["license"] = license
        data = self._transport.request("POST", "/datasets/", json=payload)
        return Dataset.model_validate(data)

    def prepare_manual_upload_batch(
        self,
        dataset_name: str,
        *,
        organization_uid: str | None = None,
        resume: bool = True,
        state_key: str | None = None,
    ) -> str | None:
        """Allocate or recover a batch using the same state_key as upload_files."""
        key = state_key or dataset_name
        return managed_upload_batch(
            _STATE_DIR,
            key,
            fingerprint=self._upload_fingerprint(organization_uid, dataset_name, key),
            resume=resume,
        )

    def create_manual_upload_url(
        self,
        *,
        dataset_name: str,
        file_path_in_dataset: str,
        content_length: int,
        organization_uid: str | None = None,
        dataset_upload_uid: str | None = None,
    ) -> dict[str, Any]:
        """Create a managed upload target, using POST or negotiated v2 PUT/multipart.

        Pass ``organization_uid`` for an organization-owned dataset. The server
        roots the S3 key on the org slug (``orgs/<slug>/…``) instead of the
        caller's username, and it must be passed **identically** here and to
        :meth:`create_from_manual_upload` — presign under one owner and create
        under the other and the dataset's prefix is empty, so it lists no items.
        """
        payload: dict[str, Any] = {
            "dataset_name": dataset_name,
            "file_path_in_dataset": file_path_in_dataset,
            "content_length": content_length,
        }
        if dataset_upload_uid is not None:
            payload.update(upload_protocol_version=2, dataset_upload_uid=dataset_upload_uid)
        if organization_uid is not None:
            payload["organization_uid"] = organization_uid
        result: dict[str, Any] = self._transport.request(
            "POST",
            "/datasets/manual-upload/file-upload-url/",
            json=payload,
        )
        return result

    def _resolve_user_uid(self) -> str | None:
        """The authenticated user's uid, cached on disk across runs.

        Personal uploads root at ``__u__=/<user_uid>/``, so this — not the API
        key — is the identity a personal checkpoint belongs to. The client is
        never told it, hence the lookup; the cache means a rotated key costs one
        request rather than re-sending the whole dataset, and a later transient
        failure costs nothing at all.
        """
        base_url = self._transport.base_url
        api_key = self._transport.api_key
        cached = cached_user_uid(_STATE_DIR, base_url, api_key)
        if cached:
            return cached
        try:
            me = self._transport.request("GET", "/users/me/")
        except Exception:  # noqa: BLE001 - advisory; never block an upload on it
            return None
        uid = me.get("uid") if isinstance(me, dict) else None
        if not uid:
            return None
        remember_user_uid(_STATE_DIR, base_url, api_key, str(uid))
        return str(uid)

    def _migrate_fallback_state(self, dataset_name: str, user_uid: str | None, state_key: str) -> None:
        """Move checkpoint state written under the credential fallback onto the uid.

        A personal run whose ``/users/me/`` lookup failed fingerprints by API-key
        hash; the next run, with the lookup healthy, fingerprints by uid. Same
        destination, two checkpoint files — so the first run's ``remote`` set is
        invisible to the second, and a file deleted in between slips past the
        stale-key guard while finalization still indexes its object.

        This is the recovery case, not just rotation: identity changed merely
        because the lookup started working. Rather than refuse to resume until
        the uid is known (which would make a transient blip cost a full
        re-upload), carry the older state forward the first time the uid
        resolves.
        """
        if not user_uid or not self._transport.api_key:
            return
        fallback = upload_fingerprint(self._transport.base_url, None, dataset_name, self._transport.api_key)
        canonical = upload_fingerprint(
            self._transport.base_url, None, dataset_name, self._transport.api_key, user_uid=user_uid
        )
        if fallback == canonical:
            return
        migrate_state(_STATE_DIR, state_key, from_fingerprint=fallback, to_fingerprint=canonical)

    def _upload_fingerprint(self, organization_uid: str | None, dataset_name: str, state_key: str | None = None) -> str:
        """The destination fingerprint for this upload.

        Single entry point on purpose: every caller that reads, writes or clears
        checkpoint state has to derive the identical value, and the ways to get
        it subtly wrong (omit the org uid, key by slug, hash the credential) have
        each already cost a round of review.
        """
        user_uid = None if organization_uid else self._resolve_user_uid()
        if user_uid:
            # Carry forward anything a previous run wrote while the lookup was
            # down, before any loader consults the new fingerprint.
            self._migrate_fallback_state(dataset_name, user_uid, state_key or dataset_name)
        return upload_fingerprint(
            self._transport.base_url,
            organization_uid,
            dataset_name,
            self._transport.api_key,
            user_uid=user_uid,
        )

    def _resolve_storage_root(self, organization_uid: str | None) -> str | None:
        """The S3 prefix root the server will actually upload under, if knowable.

        Mirrors ``owner_s3_prefix`` server-side
        (``server/apps/dataset/manual_upload_utils.py``): every manual upload
        roots at ``__o__=/<org_uid>/`` or ``__u__=/<user_uid>/``, keyed on the
        owner's **uid**. Both are ``UUIDField(editable=False)``.

        This used to resolve the org *slug* (``orgs/<slug>/``) or the username,
        because those were what the prefix was built from and both are mutable —
        a rename relocated the destination mid-transfer, and a resume that did
        not notice finalized a partial dataset. The server retired that scheme
        (PR #13946), so the hazard is gone at the source: **a rename can no
        longer move anything.**

        Keeping the slug lookup after that flip was worse than useless. It made
        the root depend on a mutable value and a network call that the real
        destination no longer depends on, so a rename — or a transient list
        failure returning ``None`` — would fail the post-upload assertion after
        every byte had already landed correctly, and change the fingerprint so
        the next run re-sent the whole dataset. Deriving from the uid removes
        both false alarms and, for org uploads, the round-trip entirely.
        """
        if organization_uid:
            # Purely local: the caller already holds the immutable key.
            return f"__o__=/{organization_uid}"

        # Personal uploads still need one lookup, since the client is not told
        # its own uid. Unlike the username this cannot change, so the value is
        # stable for the lifetime of the account; a failure here degrades to
        # "unknown" rather than to a wrong answer.
        try:
            me = self._transport.request("GET", "/users/me/")
        except Exception:  # noqa: BLE001 - advisory; never block an upload on it
            return None
        uid = me.get("uid") if isinstance(me, dict) else None
        return f"__u__=/{uid}" if uid else None

    def resolve_storage_root(self, organization_uid: str | None = None) -> str | None:
        """Public wrapper for the current upload destination root.

        Callers that finalize a dataset themselves — the CLI does — need to take
        this before uploading and check it again afterwards.
        """
        return self._resolve_storage_root(organization_uid)

    def assert_storage_root_unchanged(self, expected: str | None, *, organization_uid: str | None = None) -> None:
        """Refuse to finalize if the upload destination moved mid-run.

        Now that both roots key on an immutable uid this cannot fire for an
        org upload, and is retained as a cheap invariant rather than an active
        defense: it costs one dict-free string build, and it fails loudly if a
        future change reintroduces a mutable component to the prefix. That is
        exactly the regression that made this guard necessary the first time.

        Shared rather than inlined because there is more than one path that
        uploads and then finalizes, and a guard on only one of them is how this
        hole stayed open after it was first fixed.
        """
        now = self._resolve_storage_root(organization_uid)
        if now is None or expected is None:
            # "Unknown" is not "changed". Only the personal-upload path can land
            # here, and only when ``/users/me/`` failed on one of the two calls.
            # Aborting on that would reject a completed, correctly-placed upload
            # because of an unrelated blip on an advisory lookup — the bytes are
            # already in the right prefix either way, since the server derives it
            # from the authenticated user rather than from anything sent here.
            return
        if now != expected:
            raise RuntimeError(
                f"the upload destination moved during this run ({expected!r} -> {now!r}). Files uploaded "
                "before the change are under the old prefix and would not be indexed. Re-run to upload "
                "into the new location."
            )

    def manual_upload_quota(self, *, organization_uid: str | None = None) -> UploadQuota:
        """Storage quota for the caller, or for an organization they belong to.

        ``used`` counts open presign reservations as well as settled bytes, so
        an upload still in flight is already reflected.
        """
        params = {"organization_uid": organization_uid} if organization_uid else None
        data = self._transport.request("GET", "/datasets/manual-upload/quota/", params=params)
        return UploadQuota.model_validate(data)

    def manual_upload_allowed_mimes(self) -> AllowedMimes:
        """MIME types the manual-upload path indexes. Advisory, not enforced."""
        data = self._transport.request("GET", "/datasets/manual-upload/allowed-mimes/")
        return AllowedMimes.model_validate(data)

    def create_from_manual_upload(
        self,
        *,
        name: str,
        slug: str,
        data_type: str,
        visibility: str = "private",
        create_metadata: bool = True,
        owner_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        industry: int | None = None,
        license: int | None = None,
        organization_uid: str | None = None,
        dataset_upload_uid: str | None = None,
    ) -> Dataset:
        """Create a dataset from files uploaded with ``create_manual_upload_url``.

        ``organization_uid`` must match what the presign calls used — see
        :meth:`create_manual_upload_url`.
        """
        payload: dict[str, Any] = {
            "name": name,
            "slug": slug,
            "data_type": data_type,
            "visibility": visibility,
            "create_metadata": create_metadata,
        }
        if owner_name is not None:
            payload["owner_name"] = owner_name
        if metadata is not None:
            payload["metadata"] = metadata
        if industry is not None:
            payload["industry"] = industry
        if license is not None:
            payload["license"] = license
        if organization_uid is not None:
            payload["organization_uid"] = organization_uid
        if dataset_upload_uid is not None:
            payload["dataset_upload_uid"] = dataset_upload_uid
        data = self._transport.request("POST", "/datasets/manual-upload/", json=payload)
        return Dataset.model_validate(data)

    def upload_files(
        self,
        *,
        dataset_name: str,
        files: List[Tuple[str, str]],
        workers: int = 8,
        on_progress: Callable[[str, int], None] | None = None,
        on_skipped: Callable[[str], None] | None = None,
        organization_uid: str | None = None,
        resume: bool = True,
        state_key: str | None = None,
        clear_state_on_success: bool = False,
        dataset_upload_uid: str | None = None,
    ) -> int:
        """Upload local files to Avala-managed storage for a (to-be-created) dataset.

        ``files`` is a list of ``(local_path, file_path_in_dataset)``. Returns the
        total bytes uploaded **this call** — files skipped because a previous run
        already confirmed them are not counted again. Finalize the dataset with
        :meth:`create_from_manual_upload` (or use :meth:`create_from_local`).

        ``on_progress(relative, size)`` fires per file transferred *this* call;
        ``on_skipped(relative)`` fires per file the checkpoint or server already completed.
        A progress display that counts only the former against the full manifest
        misreports a successful resume as a near-total failure — 1 of 100 files,
        with 99 already safely uploaded — so anything showing a total needs both.

        Pass ``organization_uid`` to target an organization-owned dataset; it must
        match the value later given to :meth:`create_from_manual_upload`.

        Durability, because this is routinely a multi-hour transfer over a link
        that flaps:

        * Every file is retried with jittered exponential backoff on transport
          errors, timeouts, 408/429 and 5xx. A 4xx that isn't one of those means
          the request is wrong, so it fails immediately rather than burning the
          retry budget.
        * The presign is re-issued **inside** the retry loop. A presigned POST
          expires, so retrying a long-delayed attempt against the original target
          would fail forever.
        * Each confirmed file is recorded under ``~/.avala/uploads/<key>.json``
          (``state_key``, defaulting to ``dataset_name``). A re-run skips what
          already landed; pass ``resume=False`` to force a full re-upload. The
          checkpoint records the destination (API host + organization) and is
          ignored if either differs, so a checkpoint from a failed upload to
          one org can never make an upload to another org skip files.
        * ``clear_state_on_success`` removes the checkpoint once every file has
          landed. It defaults to **False** because this method does not finalize
          anything: the documented flow is ``upload_files`` then
          ``create_from_manual_upload``, and clearing in between means a failed
          create both re-sends the whole payload and — worse — discards the
          record of which objects are already under the prefix, so a file
          deleted before the retry can no longer be caught before finalization
          indexes its stale object.

          Pass ``True`` only when nothing further will be finalized, or call
          ``avala._uploads.clear_completed`` yourself once the dataset exists
          (``create_from_local`` does exactly that).

        The whole run still fails fast: the first unrecoverable error stops new
        work and is re-raised, but the checkpoint means a re-run resumes rather
        than starting over.
        """
        import mimetypes
        import os
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        import httpx

        items: List[Tuple[str, str]] = list(files)
        if not items:
            return 0

        # Two local files cannot share one destination key. They would be
        # presigned to the same object concurrently, so the stored bytes depend
        # on which POST wins — and every piece of resume state (checkpoint
        # entry, stamp, remote key) is keyed by that shared relative path, so
        # nothing downstream can tell them apart. If the two happen to share a
        # size and mtime, even the end-of-run manifest check passes and the
        # nondeterministic winner is finalized silently.
        seen: dict[str, str] = {}
        for local, relative in items:
            if relative in seen:
                first = seen[relative]
                detail = f"{first} and {local}" if first != local else f"{local}, listed twice"
                raise ValueError(
                    f"two entries map to the same destination path {relative!r} ({detail}). "
                    "Uploads are keyed by that path, so they would race for one object; a repeated "
                    "entry also uploads the same bytes twice and leaves the checkpoint unable to "
                    "match the manifest, so the state file is never cleared. Give each file a "
                    "distinct path within the dataset, and list it once."
                )
            seen[relative] = local

        key = state_key or dataset_name
        # No storage root here, deliberately. It used to pin the mutable
        # `orgs/<slug>` / `<username>` the prefix was built from; now that both
        # roots are `__o__=/<org_uid>` and `__u__=/<user_uid>`, each is a pure
        # function of something already in this fingerprint — the org uid, or
        # the API key that determines the user. It would add no information.
        #
        # Including it would actively hurt. The personal root still costs a
        # `/users/me/` request, so a transient failure on either run flips it
        # between `__u__=/<uid>` and None; the fingerprint would then not match
        # and every confirmed file would be re-sent, turning a blip on an
        # advisory lookup into a full re-upload. That is the same failure the
        # org path was just fixed for, reached through the other branch.
        fingerprint = self._upload_fingerprint(organization_uid, dataset_name, key)
        bind_managed_batch(_STATE_DIR, key, fingerprint=fingerprint, batch=dataset_upload_uid)
        # Loaded unconditionally. ``resume`` controls whether confirmed files are
        # SKIPPED; it must not disable the stale-key check below, which is a
        # safety property rather than an optimisation.
        #
        # Dropping the checkpoint on ``resume=False`` hid exactly the case that
        # check exists for: an interrupted run put `a` and `b` in the prefix, the
        # caller deleted `b` locally and re-ran with ``--no-resume``, and with no
        # record of `b` nothing noticed it was gone. The re-run uploads the
        # current manifest, finalization indexes the whole server prefix, and the
        # deleted file is in the dataset anyway — silently, which is the outcome
        # the caller used ``--no-resume`` to avoid.
        recorded = load_completed(_STATE_DIR, key, fingerprint=fingerprint)
        recorded_stamps = load_completed_stamps(_STATE_DIR, key, fingerprint=fingerprint)
        # Every key this destination is known to hold, invalidated ones included
        # — see ``load_remote_keys``. Not the same question as "skippable".
        remote: set[str] = load_remote_keys(_STATE_DIR, key, fingerprint=fingerprint)
        stamps: dict[str, list[int]] = dict(recorded_stamps) if resume else {}
        confirmed: set[str] = set(recorded) if resume else set()

        def _unchanged_since_upload(local_path: str, relative: str) -> bool:
            """Whether the local file still matches what was uploaded.

            A checkpoint records paths, but a path is not the file: regenerate a
            source file between runs and skipping it would leave the previous
            bytes in the dataset with nothing to indicate it. Older checkpoints
            carry no stamps, in which case the recorded path is all we have —
            trust it rather than forcing a full re-upload on upgrade.
            """
            stamp = stamps.get(relative)
            if stamp == INVALID_STAMP:
                # Explicitly invalidated: the source was replaced while its
                # bytes were in flight, so what landed cannot be vouched for.
                return False
            if stamp is None:
                # No stamp at all means a checkpoint written before stamps
                # existed. Trust it rather than forcing a full re-upload on
                # upgrade — distinct from the sentinel above, which is the whole
                # reason invalidation uses a value instead of a deletion.
                return True
            try:
                return stamps_match(stamp, file_stamp(local_path))
            except OSError:
                return False

        # A confirmed file that has vanished from the source is not something
        # this path can quietly proceed through: those bytes are already in the
        # dataset's prefix, the server indexes the whole prefix at finalization,
        # and nothing here can delete a remote key. Continuing would register
        # content the caller no longer has — so say so instead.
        # Checked against ``remote`` — every key that reached the prefix,
        # whatever this call was told to skip and whatever was later invalidated.
        # ``recorded``/``confirmed`` both answer "skippable", which is a strictly
        # smaller set and would miss a stale object nothing can delete.
        missing = remote - {rel for _local, rel in items}
        if missing:
            raise ValueError(
                f"{len(missing)} file(s) recorded as uploaded are no longer in {dataset_name}'s source "
                f"(e.g. {min(missing)}). They are already stored server-side and would be indexed "
                "at finalization. Restore them, or use a new dataset name — note that resume=False "
                "re-sends the files but cannot remove what already landed."
            )

        already = {rel for local, rel in items if rel in confirmed and _unchanged_since_upload(local, rel)}
        if on_skipped:
            # Announced before any transfer starts, so a progress display is
            # seeded with what the checkpoint already covers rather than
            # counting up from zero against the full manifest.
            for _local, relative in items:
                if relative in already:
                    on_skipped(relative)
        # Deliberately NO early return when this is empty. An "everything was
        # already uploaded" run still has to pass the whole-manifest check at
        # the end: `already` was decided by a stat taken moments ago, and a
        # generator can replace or delete one of those files while the
        # `on_skipped` callbacks run. Short-circuiting here skipped exactly that
        # check on exactly the path where nothing else would ever catch it, and
        # then cleared the only evidence of the remote objects.
        #
        # With no pending work the executor below is a no-op, so the run falls
        # through to the flush, the revalidation and the same cleanup, and still
        # returns 0 bytes.
        pending = [item for item in items if item[1] not in already]

        stop = threading.Event()
        uploaded_bytes = 0
        completed: set[str] = set(already)
        completed_lock = threading.Lock()

        last_save = [0.0]

        def _record(relative: str, local_path: str, sent_stamp: list[int] | None) -> None:
            """Persist progress after each file.

            Two writes with different jobs. ``append_completed`` puts one line
            in the journal — O(1), fsync'd, never throttled — and that is the
            durable record. ``save_completed`` rewrites the whole set as a
            snapshot, which is O(N) per call and therefore O(N²) across a run,
            so it is throttled to one write per ``_CHECKPOINT_INTERVAL`` and
            acts purely as compaction.

            Throttling the snapshot alone was not enough. The checkpoint is not
            just a skip-optimisation: it is also what notices that a confirmed
            file was deleted locally before a retry, and finalization indexes the
            whole server prefix regardless. So keys lost in the throttle window
            do not merely cost re-uploads — they let a stale remote object be
            indexed silently. The journal closes that window; the snapshot keeps
            the cost linear.

            ``sent_stamp`` is the file's identity as read *before* the upload.
            Re-stat'ing here instead would record whatever is on disk now: if
            another process atomically replaced the path mid-upload, the open
            handle still streamed the old inode, and stamping the replacement
            would mark bytes as uploaded that never were. Only keep the stamp if
            the path still matches what was sent; otherwise drop it, which makes
            a resume re-upload the file rather than trust it.
            """
            with completed_lock:
                # The PUT succeeded, so the object exists under this key. That
                # stays true even if the verdict below is "do not skip".
                remote.add(relative)
                completed.add(relative)
                # (may be discarded again just below if the file moved under us)
                try:
                    unchanged = sent_stamp is not None and stamps_match(sent_stamp, file_stamp(local_path))
                except OSError:
                    unchanged = False
                if unchanged:
                    stamps[relative] = sent_stamp  # type: ignore[assignment]
                else:
                    # The source changed under the open handle, so what landed
                    # is the old inode's bytes. Record the invalidation but do
                    # NOT count the file as completed: a run where every
                    # transfer "succeeded" would otherwise satisfy the
                    # all-files-done check, clear the checkpoint — deleting the
                    # only invalidation signal — and finalize stale content.
                    stamps[relative] = list(INVALID_STAMP)
                    completed.discard(relative)
                # Durable, O(1), and never throttled: this is the record that
                # survives a kill. The snapshot below is only a compaction of it.
                append_completed(
                    _STATE_DIR,
                    key,
                    relative,
                    stamp=stamps.get(relative),
                    completed=unchanged,
                    fingerprint=fingerprint,
                    expected_batch=dataset_upload_uid,
                    # Loud on failure: this line is the only durable record that
                    # the POST succeeded. Continuing without it means an
                    # interrupted run whose local file is later removed has no
                    # evidence the object exists, and finalization indexes the
                    # stale bytes silently.
                    strict=True,
                )
                now = time.monotonic()
                if unchanged and now - last_save[0] < _CHECKPOINT_INTERVAL:
                    return
                last_save[0] = now
                try:
                    # ``remote`` is not optional here. This compaction deletes
                    # the journal, so any key it omits stops being recorded
                    # anywhere — and the keys it would omit are precisely the
                    # invalidated ones, which are the whole reason the remote set
                    # exists. Unlike the crash cases, this runs on every throttle
                    # tick, so leaving it out loses the evidence routinely rather
                    # than rarely.
                    save_completed(
                        _STATE_DIR,
                        key,
                        completed,
                        fingerprint=fingerprint,
                        expected_batch=dataset_upload_uid,
                        stamps=stamps,
                        remote=remote,
                        dataset_name=dataset_name,
                    )
                except OSError:
                    # An unwritable state dir must not fail the upload itself;
                    # the worst case is that a re-run repeats work.
                    pass

        def _upload_one(item: Tuple[str, str]) -> int:
            local_path, relative = item
            size = os.path.getsize(local_path)
            last_exc: Exception | None = None
            managed_sent = [0]
            managed_sent_lock = threading.Lock()

            def record_sent(size: int) -> None:
                with managed_sent_lock:
                    managed_sent[0] += size

            for attempt in range(MAX_RETRIES):
                if stop.is_set():
                    return 0
                # Read the file's identity BEFORE sending it, and re-read it
                # inside the retry loop so a retry stamps what that attempt
                # actually sent.
                try:
                    sent_stamp: list[int] | None = file_stamp(local_path)
                except OSError:
                    sent_stamp = None
                # Take the length from that SAME stat. Sizing the presign from a
                # stat taken before the first attempt means a file replaced
                # during backoff — a generator finishing its write, exactly the
                # case retries exist to ride out — is signed for the old length:
                # S3 rejects the body against the policy's content-length range,
                # or, when the difference fits the allowance, accepts it while
                # quota and progress report a size that was never sent.
                if sent_stamp is not None:
                    size = sent_stamp[0]
                transferred_bytes = size
                try:
                    info = self.create_manual_upload_url(
                        dataset_name=dataset_name,
                        file_path_in_dataset=relative,
                        content_length=size,
                        organization_uid=organization_uid,
                        **({"dataset_upload_uid": dataset_upload_uid} if dataset_upload_uid is not None else {}),
                    )
                    if dataset_upload_uid is not None and info.get("method") in {"PUT", "MULTIPART"}:
                        from avala._managed_upload import transfer_managed_upload

                        bind_managed_source(_STATE_DIR, dataset_upload_uid, relative, file_stamp(local_path))

                        transfer_managed_upload(
                            local_path,
                            info,
                            request=self._transport.request,
                            timeout=httpx.Timeout(
                                connect=_UPLOAD_TIMEOUTS[0],
                                read=_UPLOAD_TIMEOUTS[1],
                                write=_UPLOAD_TIMEOUTS[2],
                                pool=_UPLOAD_TIMEOUTS[3],
                            ),
                            cancelled=stop.is_set,
                            on_sent=record_sent,
                        )
                        transferred_bytes = managed_sent[0]
                    else:
                        if info.get("method", "POST") != "POST":
                            raise ValueError("Unsupported managed upload method.")
                        validate_presigned_url(info["url"])
                        fields = info["fields"]
                        content_type = (
                            fields.get("Content-Type")
                            or mimetypes.guess_type(local_path)[0]
                            or "application/octet-stream"
                        )
                        if stop.is_set():
                            # A peer failed while this worker was blocked in
                            # presign. Starting a potentially multi-gigabyte POST
                            # now burns bandwidth and quota for a run that is
                            # already known to have failed, and the executor would
                            # wait for it before surfacing the original error.
                            return 0
                        with open(local_path, "rb") as fh:
                            resp = httpx.post(
                                info["url"],
                                data=fields,
                                files={"file": (os.path.basename(local_path), fh, content_type)},
                                # Finite, and generous. `timeout=None` let a half-open
                                # connection block this worker forever: the stall never
                                # reached the retry classifier, and because a request in
                                # flight cannot be interrupted, a peer's permanent
                                # failure could not surface either , the executor waits
                                # for this future no matter what `stop` says. The write
                                # timeout is the one that must be large, since it bounds
                                # inactivity rather than the whole upload.
                                timeout=httpx.Timeout(
                                    connect=_UPLOAD_TIMEOUTS[0],
                                    read=_UPLOAD_TIMEOUTS[1],
                                    write=_UPLOAD_TIMEOUTS[2],
                                    pool=_UPLOAD_TIMEOUTS[3],
                                ),
                            )
                            resp.raise_for_status()
                except Exception as exc:  # broad on purpose — classified just below
                    last_exc = exc
                    # A transport error or timeout is AMBIGUOUS: the provider may
                    # have committed the object and lost the response on the way
                    # back. Record the key as possibly-present before retrying —
                    # `_record` has not run, so nothing else knows about it, and
                    # if this run is interrupted and the file is deleted before
                    # the next one, the stale-key guard would see no evidence
                    # while finalization still lists and indexes the object.
                    #
                    # Deliberately not for an HTTP status response: that is the
                    # provider explicitly refusing, and marking those would raise
                    # false "restore this file" errors on every 4xx.
                    #
                    # Nor for failures that happen BEFORE any byte is sent. If
                    # the connection was never established, or we never got a
                    # pool slot, the provider cannot have committed anything —
                    # recording the key there would make the stale-key guard
                    # demand the user restore a file that was never uploaded, or
                    # pick a new dataset name, over a plain connection refusal.
                    # Only write/read-phase failures are genuinely ambiguous.
                    never_sent = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
                    ambiguous = isinstance(exc, (httpx.TransportError, httpx.TimeoutException)) and not isinstance(
                        exc, never_sent
                    )
                    if ambiguous:
                        with completed_lock:
                            remote.add(relative)
                        # strict, for the same reason `_record` is: this line
                        # is the ONLY evidence that an object may exist under
                        # this key. Suppressing the write and then re-raising
                        # the transport error leaves an ambiguous upload with no
                        # record at all, so a retry after the local file is
                        # removed cannot catch the possibly-committed object.
                        # Losing that beats surfacing the wrong error, so the
                        # persistence failure wins and chains the original.
                        try:
                            append_completed(
                                _STATE_DIR,
                                key,
                                relative,
                                stamp=None,
                                completed=False,
                                fingerprint=fingerprint,
                                expected_batch=dataset_upload_uid,
                                strict=True,
                            )
                        except OSError as persist_error:
                            raise persist_error from exc
                    if not is_retryable(exc) or attempt == MAX_RETRIES - 1:
                        raise
                    # Interruptible: a peer's permanent failure must not wait out
                    # this worker's backoff, which a provider Retry-After can
                    # stretch to minutes.
                    sleep_backoff(attempt, exc, stop)
                    continue
                _record(relative, local_path, sent_stamp)
                if stamps.get(relative) == INVALID_STAMP:
                    raise RuntimeError(
                        f"{relative} was modified while it was being uploaded, so the bytes that "
                        "landed are not the current file. Re-run to upload it again."
                    )
                if transferred_bytes == 0:
                    if on_skipped is not None:
                        on_skipped(relative)
                elif on_progress is not None:
                    on_progress(relative, transferred_bytes)
                return transferred_bytes
            # Unreachable: the loop either returns or raises. Kept so the
            # function has no implicit ``None`` return path.
            raise RuntimeError(f"{relative}: upload failed after {MAX_RETRIES} attempts: {last_exc}")

        first_error: Exception | None = None
        iterator = iter(pending)
        active: Dict[Any, Tuple[str, str]] = {}
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for _ in range(max(1, workers)):
                try:
                    nxt = next(iterator)
                except StopIteration:
                    break
                active[pool.submit(_upload_one, nxt)] = nxt
            while active:
                future = next(as_completed(active))
                active.pop(future)
                try:
                    uploaded_bytes += future.result()
                    if first_error is None and not stop.is_set():
                        try:
                            nxt = next(iterator)
                            active[pool.submit(_upload_one, nxt)] = nxt
                        except StopIteration:
                            pass
                except Exception as exc:  # noqa: BLE001 - fail fast, re-raised below
                    if first_error is None:
                        first_error = exc
                        stop.set()
                        for pending_future in list(active):
                            pending_future.cancel()
        # Flush whatever the throttle in ``_record`` held back. Without this a
        # run that ends inside the interval loses its last writes, and a caller
        # passing ``clear_state_on_success=False`` (the CLI, which finalizes
        # separately) would resume from a checkpoint missing the final files.
        with completed_lock:
            try:
                save_completed(
                    _STATE_DIR,
                    key,
                    completed,
                    fingerprint=fingerprint,
                    expected_batch=dataset_upload_uid,
                    stamps=stamps,
                    remote=remote,
                    dataset_name=dataset_name,
                )
            except OSError:
                pass

        if first_error is not None:
            raise first_error

        # Re-validate the WHOLE manifest immediately before reporting success —
        # every file this run considers complete, however it got there.
        #
        # Each per-file check runs at a different moment and then stops looking:
        # ``already`` is computed once up front, and ``_record``'s check happens
        # the instant a file's POST returns. Both leave the rest of the run
        # unwatched, and they leave different files unwatched, so checking only
        # one set closes only part of the hole:
        #
        #   * a checkpoint-skipped file never enters ``_upload_one`` at all, so
        #     ``_record`` never speaks for it;
        #   * a file uploaded early is cleared by ``_record`` and then rewritten
        #     while later files are still in flight — after its own check, before
        #     this one.
        #
        # Either way ``create_from_local`` would finalize stale remote bytes and
        # report success. The exposure is the full duration of the run, which for
        # the multi-hour transfers resume exists for is exactly when a generator
        # is most likely to still be writing.
        #
        # Deliberately re-stat rather than requeue: those bytes are already in
        # the dataset's prefix under the same key, so re-uploading here would
        # race the very rewrite just detected. Invalidate, keep the checkpoint,
        # and let the next run send the settled file.
        changed_during_run = sorted(
            rel for local, rel in items if rel in completed and not _unchanged_since_upload(local, rel)
        )
        if changed_during_run:
            with completed_lock:
                for relative in changed_during_run:
                    stamps[relative] = list(INVALID_STAMP)
                    completed.discard(relative)
                try:
                    save_completed(
                        _STATE_DIR,
                        key,
                        completed,
                        fingerprint=fingerprint,
                        expected_batch=dataset_upload_uid,
                        stamps=stamps,
                        remote=remote,
                        dataset_name=dataset_name,
                    )
                except OSError:
                    pass
            raise RuntimeError(
                f"{len(changed_during_run)} file(s) counted as uploaded changed on disk during this run "
                f"(e.g. {changed_during_run[0]}), so the stored copy is stale. They have been marked for "
                "re-upload; re-run to send them before creating the dataset."
            )

        # Every file in this dataset is confirmed — the checkpoint has no more
        # work to describe. Only clear it on a fully clean run, and only if the
        # caller isn't going to need it for a later finalize step.
        if clear_state_on_success and len(completed) >= len(items):
            clear_completed(_STATE_DIR, key, fingerprint=fingerprint, expected_batch=dataset_upload_uid)
        return uploaded_bytes

    def create_from_local(
        self,
        *,
        source: str,
        name: str,
        slug: str,
        data_type: str,
        visibility: str = "private",
        create_metadata: bool = True,
        owner_name: str | None = None,
        industry: int | None = None,
        license: int | None = None,
        workers: int = 8,
        on_progress: Callable[[str, int], None] | None = None,
        wait: bool = False,
        wait_timeout: float = 3600.0,
        organization_uid: str | None = None,
        metadata: dict[str, Any] | None = None,
        resume: bool = True,
        check_quota: bool = True,
    ) -> Dataset:
        """Upload a local file or directory and create a dataset from it.

        Convenience wrapper around :meth:`upload_files` + :meth:`create_from_manual_upload`.
        Set ``wait=True`` to block until indexing completes. Pass
        ``organization_uid`` to create the dataset under an organization rather
        than the calling user.

        With ``check_quota=True`` (the default) the payload size is compared
        against the owner's remaining quota before any bytes move, so an
        obviously-too-large upload fails in a second instead of an hour. This is
        advisory only — the server remains the source of truth and still returns
        413 mid-run if the picture changes, which :meth:`upload_files` surfaces
        as :class:`~avala.errors.QuotaExceededError`. Set it to ``False`` to skip
        the extra round-trip.
        """
        import os

        files = gather_local_files(source)
        if not files:
            raise ValueError(f"no files found in {source}")
        # Keyed on the dataset NAME, not the slug. The remote prefix is
        # `<owner-root>/<dataset_name>/`, so the name is what decides where the
        # bytes are; the slug is an independently-unique label that never
        # reaches S3.
        #
        # Keying on the slug broke the one retry that matters here: every file
        # uploads, finalization rejects a colliding slug, and the natural fix is
        # to change only the slug — leaving the remote prefix untouched. That
        # pointed at a fresh checkpoint, so the whole payload was re-sent, and
        # worse, the new file had none of the remote-key evidence, so a source
        # file deleted in the meantime could no longer be caught before its
        # stale object was finalized.
        state_key = name
        # Captured once, up front, and re-checked before finalization below.
        storage_root_before = self._resolve_storage_root(organization_uid)
        if check_quota:
            # Count only what this run will actually send. Bytes confirmed by an
            # earlier run are already reflected in ``quota.used``, so summing
            # every local file would compare the whole payload against the
            # headroom left after part of it was uploaded — refusing to resume
            # exactly the large transfers resume exists for.
            # Must match what ``upload_files`` computes, or this reads an empty
            # set and preflights the whole payload instead of the remainder.
            fingerprint = self._upload_fingerprint(organization_uid, name)
            # Skippable, not merely recorded. A confirmed file whose bytes
            # changed on disk WILL be re-sent by ``upload_files``, so excluding
            # it here understates the payload — and the preflight exists to
            # catch an over-cap upload before it starts, which it cannot do if
            # it measures less than the run will actually send.
            recorded = load_completed(_STATE_DIR, state_key, fingerprint=fingerprint) if resume else set()
            recorded_stamps = load_completed_stamps(_STATE_DIR, state_key, fingerprint=fingerprint) if resume else {}

            def _will_skip(local: str, relative: str) -> bool:
                if relative not in recorded:
                    return False
                stamp = recorded_stamps.get(relative)
                if stamp == INVALID_STAMP:
                    return False
                if stamp is None:
                    return True  # pre-stamp checkpoint; upload_files trusts it too
                try:
                    return stamps_match(stamp, file_stamp(local))
                except OSError:
                    return False

            already = {relative for local, relative in files if _will_skip(local, relative)}
            total = sum(os.path.getsize(local) for local, relative in files if relative not in already)
            quota = None
            try:
                quota = self.manual_upload_quota(organization_uid=organization_uid)
            except Exception:  # noqa: BLE001 - deliberately swallowed; see below
                # This check is advisory. The meter needs the ``datasets.read``
                # scope that a write-only API key legitimately lacks, it doesn't
                # exist on older servers, and it can fail transiently like any
                # other request. None of those is a reason to refuse an upload
                # the server would have accepted: the presign's 413 remains the
                # authority, and the upload path retries on its own. Failing
                # open here is the difference between a nice-to-have and a new
                # availability dependency for every existing caller.
                quota = None
            # Refuse only what can NEVER fit, not merely what doesn't fit right
            # now. ``quota.used`` includes open presign reservations, so a file
            # whose presign succeeded but whose POST failed is counted both in
            # ``used`` and in ``total`` — comparing against ``remaining`` then
            # rejects a resume the server would happily accept, and keeps
            # rejecting it for the 24h until the reservation ages out. Comparing
            # against the hard ``limit`` has no such false positives, still
            # catches the case worth catching in a second instead of an hour,
            # and leaves every marginal call to the server's 413.
            if quota is not None and total > quota.limit:
                raise QuotaExceededError(
                    f"{source} needs {total} bytes, more than this owner's entire {quota.limit}-byte storage limit.",
                    limit=quota.limit,
                    used=quota.used,
                )
        dataset_upload_uid = self.prepare_manual_upload_batch(name, organization_uid=organization_uid, resume=resume)
        self.upload_files(
            dataset_name=name,
            files=files,
            workers=workers,
            on_progress=on_progress,
            organization_uid=organization_uid,
            resume=resume,
            state_key=state_key,
            # Keep the checkpoint until the dataset actually exists. If the
            # create call below fails — or its response is lost — a re-run
            # should resume at finalization, not re-send every byte.
            clear_state_on_success=False,
            dataset_upload_uid=dataset_upload_uid,
        )
        self.assert_storage_root_unchanged(storage_root_before, organization_uid=organization_uid)

        # Finalization is deliberately NOT auto-recovered. A create that
        # committed but whose response was lost is indistinguishable, from here,
        # from a plain name collision with a dataset that already existed —
        # both surface as the same "already exists" validation error. An earlier
        # revision tried to tell them apart by looking the name up, which is
        # unsound twice over: the list filter matches names loosely, and even an
        # exact name+slug hit proves nothing about who created it. Guessing
        # wrong means returning somebody else's dataset and discarding the
        # checkpoint after this run's bytes have already been written into the
        # colliding prefix — far worse than the rare failure it papered over.
        #
        # So the error propagates, and the checkpoint is kept (cleared only on
        # success below), which means a re-run still skips every uploaded file
        # and only retries the finalize. That is the safe half of the recovery;
        # the ambiguous half belongs to a human.
        dataset = self.create_from_manual_upload(
            name=name,
            slug=slug,
            data_type=data_type,
            visibility=visibility,
            create_metadata=create_metadata,
            owner_name=owner_name,
            industry=industry,
            license=license,
            organization_uid=organization_uid,
            metadata=metadata,
            dataset_upload_uid=dataset_upload_uid,
        )
        # The dataset exists, so the checkpoint has nothing left to protect.
        # Same fingerprint the upload used, or this clears a different
        # destination's file (or nothing at all).
        clear_completed(
            _STATE_DIR,
            state_key,
            fingerprint=self._upload_fingerprint(organization_uid, name),
            expected_batch=dataset_upload_uid,
        )
        if wait:
            dataset = self.wait(dataset.uid, status="created", interval=10.0, timeout=wait_timeout)
        return dataset

    def list_items(
        self,
        owner: str,
        slug: str,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> CursorPage[DatasetItem]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page(f"/datasets/{owner}/{slug}/items/", DatasetItem, params=params or None)

    def get_item(self, owner: str, slug: str, item_uid: str) -> DatasetItem:
        data = self._transport.request("GET", f"/datasets/{owner}/{slug}/items/{item_uid}/")
        return DatasetItem.model_validate(data)

    def list_sequences(
        self,
        owner: str,
        slug: str,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> CursorPage[DatasetSequence]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page(
            f"/datasets/{owner}/{slug}/sequences/",
            DatasetSequence,
            params=params or None,
        )

    def get_sequence(self, owner: str, slug: str, sequence_uid: str) -> DatasetSequence:
        data = self._transport.request("GET", f"/datasets/{owner}/{slug}/sequences/{sequence_uid}/")
        return DatasetSequence.model_validate(data)

    def get_frame(self, owner: str, slug: str, sequence_uid: str, frame_idx: int) -> DatasetFrame:
        """Return a single frame's LiDAR JSON metadata.

        Indexes into ``get_sequence().frames`` client-side — the server embeds
        the full frame array on the sequence response, so no extra round-trip
        is needed beyond the sequence fetch.
        """
        sequence = self.get_sequence(owner, slug, sequence_uid)
        return _build_frame(sequence.frames or [], frame_idx, sequence_uid)

    def get_calibration(self, owner: str, slug: str, sequence_uid: str) -> DatasetCalibration:
        """Return a canonicalized rig view for a sequence, derived from frame[0]."""
        sequence = self.get_sequence(owner, slug, sequence_uid)
        return _build_calibration_from_sequence(sequence)

    def get_health(self, owner: str, slug: str) -> DatasetHealth:
        """Return a read-only health snapshot for the dataset.

        Calls ``GET /datasets/<owner>/<slug>/health/`` — intended for
        post-ingest validation (frame counts, indexing status, per-sequence
        calibration presence, S3 prefix, any issues detected).
        """
        data = self._transport.request("GET", f"/datasets/{owner}/{slug}/health/")
        return DatasetHealth.model_validate(data)

    def wait(
        self,
        uid: str,
        *,
        status: str = "created",
        interval: float = 10.0,
        timeout: float = 3600.0,
        _on_poll: Callable[[Dataset], None] | None = None,
    ) -> Dataset:
        """Poll a dataset until it reaches the target status.

        Args:
            uid: The dataset UID to poll.
            status: Target status to wait for (default ``"created"``).
            interval: Seconds between polls (default 10, minimum 1).
            timeout: Maximum seconds to wait before raising ``TimeoutError`` (default 3600).
            _on_poll: Optional callback invoked after each non-terminal poll with the current dataset.

        Returns:
            The Dataset object once it reaches the target status.

        Raises:
            TimeoutError: If the dataset does not reach the target status within *timeout* seconds.
        """
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        if interval < 0:
            raise ValueError("interval must be non-negative")
        interval = max(interval, _MIN_INTERVAL)
        deadline = time.monotonic() + timeout
        while True:
            dataset = self.get(uid)
            if dataset.status == status:
                return dataset
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Dataset {uid} did not reach status '{status}' within {timeout}s (last status: {dataset.status})"
                )
            if _on_poll is not None:
                _on_poll(dataset)
            time.sleep(interval)


class AsyncDatasets(BaseAsyncResource):
    async def list(
        self,
        *,
        data_type: str | None = None,
        name: str | None = None,
        status: str | None = None,
        visibility: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> CursorPage[Dataset]:
        """List authorized workspace datasets; visibility filtering grants no access.

        Servers supporting Unlisted accept private, unlisted, or public filters.
        A known Unlisted URL does not add a dataset to this workspace listing.
        """
        params: dict[str, Any] = {}
        if data_type is not None:
            params["data_type"] = data_type
        if name is not None:
            params["name"] = name
        if status is not None:
            params["status"] = status
        if visibility is not None:
            params["visibility"] = visibility
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page("/datasets/", Dataset, params=params or None)

    async def get(self, uid: str) -> Dataset:
        data = await self._transport.request("GET", f"/datasets/{uid}/")
        return Dataset.model_validate(data)

    async def get_by_slug(self, owner: str, slug: str) -> Dataset:
        data = await self._transport.request("GET", f"/datasets/{owner}/{slug}/")
        return Dataset.model_validate(data)

    async def transfer(
        self,
        owner: str,
        slug: str,
        *,
        organization_uid: str | None = None,
        owner_username: str | None = None,
    ) -> Dataset:
        """Move a dataset to a different owner.

        Exactly one of ``organization_uid`` or ``owner_username`` must be given —
        a dataset is owned by a user XOR an organization, never both.

        Transferring requires the OWNER role on the dataset's current
        organization (an ADMIN may edit a dataset but may not give it away), and
        authority at the destination too. ``owner_username`` may only name
        *yourself*: the API refuses handing a dataset to another account, since
        that would park a tenant's data on a login that never agreed to take it.

        Returns the dataset at its NEW path; its ``owner_name`` is what the
        canonical ``/@<owner>/datasets/<slug>`` URL now uses.
        """
        payload = _transfer_payload(organization_uid=organization_uid, owner_username=owner_username)
        data = await self._transport.request("POST", f"/datasets/{owner}/{slug}/transfer/", json=payload)
        return Dataset.model_validate(data)

    async def create(
        self,
        *,
        name: str,
        slug: str,
        data_type: str,
        visibility: str = "private",
        create_metadata: bool = True,
        provider_config: dict[str, Any] | None = None,
        owner_name: str | None = None,
        organization_id: int | None = None,
        organization_uid: str | None = None,
        gpu_texture_format: str | None = None,
        metadata: dict[str, Any] | None = None,
        industry: int | None = None,
        license: int | None = None,
    ) -> Dataset:
        payload: dict[str, Any] = {
            "name": name,
            "slug": slug,
            "data_type": data_type,
            "visibility": visibility,
            "create_metadata": create_metadata,
        }
        if provider_config is not None:
            payload["provider_config"] = provider_config
        if owner_name is not None:
            payload["owner_name"] = owner_name
        if organization_id is not None:
            payload["organization_id"] = organization_id
        if organization_uid is not None:
            payload["organization_uid"] = organization_uid
        if gpu_texture_format is not None:
            payload["gpu_texture_format"] = gpu_texture_format
        if metadata is not None:
            payload["metadata"] = metadata
        if industry is not None:
            payload["industry"] = industry
        if license is not None:
            payload["license"] = license
        data = await self._transport.request("POST", "/datasets/", json=payload)
        return Dataset.model_validate(data)

    async def create_manual_upload_url(
        self,
        *,
        dataset_name: str,
        file_path_in_dataset: str,
        content_length: int,
        organization_uid: str | None = None,
        dataset_upload_uid: str | None = None,
    ) -> dict[str, Any]:
        """Create a managed upload target, using POST or negotiated v2 PUT/multipart.

        ``organization_uid`` must match what :meth:`create_from_manual_upload`
        is later given — see the sync counterpart for why.
        """
        payload: dict[str, Any] = {
            "dataset_name": dataset_name,
            "file_path_in_dataset": file_path_in_dataset,
            "content_length": content_length,
        }
        if dataset_upload_uid is not None:
            payload.update(upload_protocol_version=2, dataset_upload_uid=dataset_upload_uid)
        if organization_uid is not None:
            payload["organization_uid"] = organization_uid
        result: dict[str, Any] = await self._transport.request(
            "POST",
            "/datasets/manual-upload/file-upload-url/",
            json=payload,
        )
        return result

    async def manual_upload_quota(self, *, organization_uid: str | None = None) -> UploadQuota:
        """Storage quota for the caller, or for an organization they belong to."""
        params = {"organization_uid": organization_uid} if organization_uid else None
        data = await self._transport.request("GET", "/datasets/manual-upload/quota/", params=params)
        return UploadQuota.model_validate(data)

    async def manual_upload_allowed_mimes(self) -> AllowedMimes:
        """MIME types the manual-upload path indexes. Advisory, not enforced."""
        data = await self._transport.request("GET", "/datasets/manual-upload/allowed-mimes/")
        return AllowedMimes.model_validate(data)

    async def create_from_manual_upload(
        self,
        *,
        name: str,
        slug: str,
        data_type: str,
        visibility: str = "private",
        create_metadata: bool = True,
        owner_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        industry: int | None = None,
        license: int | None = None,
        organization_uid: str | None = None,
        dataset_upload_uid: str | None = None,
    ) -> Dataset:
        """Create a dataset from files uploaded with ``create_manual_upload_url``.

        ``organization_uid`` must match what the presign calls used.
        """
        payload: dict[str, Any] = {
            "name": name,
            "slug": slug,
            "data_type": data_type,
            "visibility": visibility,
            "create_metadata": create_metadata,
        }
        if owner_name is not None:
            payload["owner_name"] = owner_name
        if metadata is not None:
            payload["metadata"] = metadata
        if industry is not None:
            payload["industry"] = industry
        if license is not None:
            payload["license"] = license
        if organization_uid is not None:
            payload["organization_uid"] = organization_uid
        if dataset_upload_uid is not None:
            payload["dataset_upload_uid"] = dataset_upload_uid
        data = await self._transport.request("POST", "/datasets/manual-upload/", json=payload)
        return Dataset.model_validate(data)

    async def list_items(
        self,
        owner: str,
        slug: str,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> CursorPage[DatasetItem]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page(
            f"/datasets/{owner}/{slug}/items/", DatasetItem, params=params or None
        )

    async def get_item(self, owner: str, slug: str, item_uid: str) -> DatasetItem:
        data = await self._transport.request("GET", f"/datasets/{owner}/{slug}/items/{item_uid}/")
        return DatasetItem.model_validate(data)

    async def list_sequences(
        self,
        owner: str,
        slug: str,
        *,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> CursorPage[DatasetSequence]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page(
            f"/datasets/{owner}/{slug}/sequences/",
            DatasetSequence,
            params=params or None,
        )

    async def get_sequence(self, owner: str, slug: str, sequence_uid: str) -> DatasetSequence:
        data = await self._transport.request("GET", f"/datasets/{owner}/{slug}/sequences/{sequence_uid}/")
        return DatasetSequence.model_validate(data)

    async def get_frame(self, owner: str, slug: str, sequence_uid: str, frame_idx: int) -> DatasetFrame:
        """Return a single frame's LiDAR JSON metadata (async)."""
        sequence = await self.get_sequence(owner, slug, sequence_uid)
        return _build_frame(sequence.frames or [], frame_idx, sequence_uid)

    async def get_calibration(self, owner: str, slug: str, sequence_uid: str) -> DatasetCalibration:
        """Return a canonicalized rig view for a sequence (async)."""
        sequence = await self.get_sequence(owner, slug, sequence_uid)
        return _build_calibration_from_sequence(sequence)

    async def get_health(self, owner: str, slug: str) -> DatasetHealth:
        """Return a read-only health snapshot for the dataset (async)."""
        data = await self._transport.request("GET", f"/datasets/{owner}/{slug}/health/")
        return DatasetHealth.model_validate(data)

    async def wait(
        self,
        uid: str,
        *,
        status: str = "created",
        interval: float = 10.0,
        timeout: float = 3600.0,
        _on_poll: Callable[[Dataset], None] | None = None,
    ) -> Dataset:
        """Poll a dataset until it reaches the target status.

        Args:
            uid: The dataset UID to poll.
            status: Target status to wait for (default ``"created"``).
            interval: Seconds between polls (default 10, minimum 1).
            timeout: Maximum seconds to wait before raising ``TimeoutError`` (default 3600).
            _on_poll: Optional callback invoked after each non-terminal poll with the current dataset.

        Returns:
            The Dataset object once it reaches the target status.

        Raises:
            TimeoutError: If the dataset does not reach the target status within *timeout* seconds.
        """
        import asyncio

        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        if interval < 0:
            raise ValueError("interval must be non-negative")
        interval = max(interval, _MIN_INTERVAL)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            dataset = await self.get(uid)
            if dataset.status == status:
                return dataset
            if loop.time() >= deadline:
                raise TimeoutError(
                    f"Dataset {uid} did not reach status '{status}' within {timeout}s (last status: {dataset.status})"
                )
            if _on_poll is not None:
                _on_poll(dataset)
            await asyncio.sleep(interval)
