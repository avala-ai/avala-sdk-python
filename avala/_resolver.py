"""Credential-free transport and identity checks for public dataset revisions."""

from __future__ import annotations

import hashlib
import math
import re
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, BinaryIO, Literal, Mapping, Optional, cast
from urllib.parse import parse_qsl, urlsplit

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError as PydanticValidationError,
    field_validator,
    model_validator,
)

from avala._config import DEFAULT_BASE_URL, _normalize_base_url
from avala.errors import (
    DatasetDownloadError,
    DatasetIntegrityError,
    DatasetReferenceError,
    DatasetResolverError,
    DatasetRevisionWithdrawnError,
)

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_UUID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
_OWNER_PATTERN = r"[a-z0-9][a-z0-9_-]{0,254}"
_DATASET_PATTERN = r"[a-z0-9][a-z0-9._-]{0,99}"
_SELECTOR_PATTERN = r"(?:[0-9a-f]{64}|[a-z][a-z0-9._-]{0,62})"
_ORIGIN_NAME_PATTERN = r"^[a-z][a-z0-9._-]{0,62}$"
_ROLE_PATTERN = r"^[a-z][a-z0-9_]{0,31}$"
_CANONICAL_REFERENCE_RE = re.compile(
    rf"^avala://datasets/(?P<dataset_uid>{_UUID_PATTERN[1:-1]})@(?P<revision_sha256>[0-9a-f]{{64}})$"
)
_FRIENDLY_REFERENCE_RE = re.compile(
    rf"^(?P<owner>{_OWNER_PATTERN})/(?P<dataset>{_DATASET_PATTERN})(?:@(?P<selector>{_SELECTOR_PATTERN}))?$"
)
_SELECTOR_RE = re.compile(rf"^{_SELECTOR_PATTERN}$")
_ASCII_DNS_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{1,61}[a-z0-9])$")
_REGION_RE = re.compile(r"^(?!us-gov-)[a-z]{2}(?:-[a-z0-9]+)+-[0-9]$")
_R2_HOST_RE = re.compile(r"^[0-9a-f]{32}(?:\.eu)?\.r2\.cloudflarestorage\.com$")
_AZURE_HOST_RE = re.compile(r"^[a-z0-9]{3,24}\.blob\.core\.windows\.net$")
_PROVIDERS = frozenset({"aws_s3", "gcs", "azure_blob", "cloudflare_r2", "avala_proxy", "avala_edge"})
_EDGE_HOSTS_BY_API_BASE = {
    "https://api.avala.ai/api/v1": "data.avala.ai",
    "https://server.avala.ai/api/v1": "data.avala.ai",
    "https://server.dev.alala.ai/api/v1": "data-development.avala.ai",
}
_MAX_JSON_SAFE_INTEGER = 9_007_199_254_740_991
_MAX_GRANT_TTL_SECONDS = 300
_LOCAL_CLOCK_SKEW_SECONDS = 30
_MIN_GRANT_REMAINING_SECONDS = 5
_REQUEST_TIMEOUT_SECONDS = 30.0
_DOWNLOAD_SPOOL_MEMORY_BYTES = 8 * 1024 * 1024
_MAX_RATE_LIMIT_RETRIES = 3
_MAX_RETRY_AFTER_SECONDS = 300.0


def _contains_url_control(value: str) -> bool:
    return any(ord(character) <= 0x20 or ord(character) == 0x7F for character in value)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _monotonic_time() -> float:
    return time.monotonic()


class _ResolverModel(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True, strict=True)


class ResolverOrigin(_ResolverModel):
    name: str = Field(pattern=_ORIGIN_NAME_PATTERN)
    kind: str
    canonical_ref: str
    external_revision: str


class ResolverManifest(_ResolverModel):
    schema_version: Literal[1]
    sha256: str = Field(pattern=_SHA256_PATTERN)
    object_count: int = Field(ge=0, le=100_000)
    total_size_bytes: int = Field(ge=0, le=_MAX_JSON_SAFE_INTEGER)
    origins: list[ResolverOrigin]
    objects_path: str


class ResolverRights(_ResolverModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    status: Literal["declared"]
    access: Literal["open_download"]
    name: str = Field(min_length=1, max_length=1024)
    url: str = Field(min_length=1, max_length=2048)
    document_sha256: str = Field(pattern=_SHA256_PATTERN)
    attestation_sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        try:
            parsed = urlsplit(value)
            parsed.port
        except ValueError:
            raise ValueError("rights URL must be a valid HTTPS URL") from None
        if (
            parsed.scheme.lower() != "https"
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or _contains_url_control(value)
        ):
            raise ValueError("rights URL must be a valid HTTPS URL")
        return value


class ResolvedRevisionDocument(_ResolverModel):
    schema_version: Literal[1]
    requested_reference: str
    display_reference: str
    canonical_reference: str
    dataset_uid: str = Field(pattern=_UUID_PATTERN)
    revision_uid: str = Field(pattern=_UUID_PATTERN)
    revision_sha256: str = Field(pattern=_SHA256_PATTERN)
    state: Literal["published"]
    manifest: ResolverManifest
    rights: ResolverRights


class ManifestObjectDocument(_ResolverModel):
    object_uid: str = Field(pattern=_UUID_PATTERN)
    logical_path: str
    role: str = Field(pattern=_ROLE_PATTERN)
    ordinal: int = Field(ge=0, le=_MAX_JSON_SAFE_INTEGER)
    # ``Optional`` rather than ``| None`` is intentional: Pydantic evaluates
    # model annotations at runtime, and the base SDK supports Python 3.9
    # without requiring the optional ``eval_type_backport`` package.
    episode_ordinal: Optional[int] = Field(ge=0, le=_MAX_JSON_SAFE_INTEGER)
    content_type: str
    origin: str = Field(pattern=_ORIGIN_NAME_PATTERN)
    size_bytes: int = Field(ge=0, le=_MAX_JSON_SAFE_INTEGER)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    access_path: str

    @field_validator("logical_path")
    @classmethod
    def validate_logical_path(cls, value: str) -> str:
        try:
            encoded = value.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("logical_path must be a canonical POSIX-relative path") from None
        segments = value.split("/")
        if (
            not 1 <= len(encoded) <= 1024
            or unicodedata.normalize("NFC", value) != value
            or value.startswith("/")
            or value.endswith("/")
            or "\\" in value
            or re.search(r"%2f", value, re.IGNORECASE) is not None
            or any(segment in {"", ".", ".."} for segment in segments)
            or any(unicodedata.category(character) == "Cc" for character in value)
        ):
            raise ValueError("logical_path must be a canonical POSIX-relative path")
        return value

    @model_validator(mode="after")
    def validate_episode_grouping(self) -> ManifestObjectDocument:
        if self.role == "episode" and self.episode_ordinal is not None:
            raise ValueError("episode objects cannot reference an episode ordinal")
        return self


class ManifestObjectPageDocument(_ResolverModel):
    schema_version: Literal[1]
    canonical_reference: str
    revision_sha256: str = Field(pattern=_SHA256_PATTERN)
    manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    next_cursor: Optional[str]
    previous_cursor: Optional[str]
    results: list[ManifestObjectDocument] = Field(max_length=100)


class WithdrawnRevisionDocument(_ResolverModel):
    schema_version: Literal[1]
    code: Literal["dataset_revision_withdrawn"]
    canonical_reference: str
    dataset_uid: str = Field(pattern=_UUID_PATTERN)
    revision_sha256: str = Field(pattern=_SHA256_PATTERN)


@dataclass(frozen=True)
class ParsedDatasetReference:
    path: str
    requested_reference: str
    exact_dataset_uid: str | None
    exact_revision_sha256: str | None


@dataclass(frozen=True, repr=False)
class DatasetAccessGrant:
    provider: str
    url: str
    expires_at: datetime
    server_date: datetime
    object_uid: str
    size_bytes: int
    sha256: str
    range_supported: bool
    remaining_at_receipt: float
    received_at_monotonic: float

    @property
    def remaining_seconds(self) -> float:
        elapsed = max(0.0, _monotonic_time() - self.received_at_monotonic)
        return self.remaining_at_receipt - elapsed

    def __repr__(self) -> str:
        return (
            "DatasetAccessGrant("
            f"provider={self.provider!r}, object_uid={self.object_uid!r}, expires_at={self.expires_at!r})"
        )


def _validate_withdrawn_revision(
    document: WithdrawnRevisionDocument,
    *,
    expected_dataset_uid: str | None,
    expected_revision_sha256: str | None,
) -> WithdrawnRevisionDocument:
    canonical = f"avala://datasets/{document.dataset_uid}@{document.revision_sha256}"
    if (
        expected_revision_sha256 is None
        or document.canonical_reference != canonical
        or expected_dataset_uid not in {None, document.dataset_uid}
        or expected_revision_sha256 not in {None, document.revision_sha256}
    ):
        raise DatasetIntegrityError("withdrawal_identity_mismatch")
    return document


def parse_dataset_reference(reference: str, revision: str | None = None) -> ParsedDatasetReference:
    """Parse one unambiguous friendly or canonical public dataset reference."""

    if not isinstance(reference, str) or not reference:
        raise DatasetReferenceError("reference must be a non-empty string")
    if revision is not None and (not isinstance(revision, str) or _SELECTOR_RE.fullmatch(revision) is None):
        raise DatasetReferenceError("revision must be a canonical selector")

    canonical = _CANONICAL_REFERENCE_RE.fullmatch(reference)
    if canonical is not None:
        if revision is not None:
            raise DatasetReferenceError("canonical_reference_with_revision")
        dataset_uid = canonical.group("dataset_uid")
        revision_sha256 = canonical.group("revision_sha256")
        return ParsedDatasetReference(
            path=f"/resolve/uid/{dataset_uid}/{revision_sha256}/",
            requested_reference=reference,
            exact_dataset_uid=dataset_uid,
            exact_revision_sha256=revision_sha256,
        )

    friendly = _FRIENDLY_REFERENCE_RE.fullmatch(reference)
    if friendly is None:
        raise DatasetReferenceError("invalid_dataset_reference")
    embedded_selector = friendly.group("selector")
    if embedded_selector is not None and revision is not None:
        raise DatasetReferenceError("selector_and_revision")
    selector = embedded_selector or revision or "main"
    owner = friendly.group("owner")
    dataset = friendly.group("dataset")
    requested_reference = f"{owner}/{dataset}@{selector}"
    return ParsedDatasetReference(
        path=f"/resolve/{requested_reference}/",
        requested_reference=requested_reference,
        exact_dataset_uid=None,
        exact_revision_sha256=selector if re.fullmatch(_SHA256_PATTERN, selector) else None,
    )


def _normalize_public_base_url(base_url: str) -> str:
    normalized = _normalize_base_url(base_url)
    parsed = urlsplit(normalized)
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or _contains_url_control(normalized)
    ):
        raise ValueError("base_url contains unsupported or unsafe URL components")
    return normalized


def _validate_resolver_path(path: str) -> None:
    if not isinstance(path, str) or not path.startswith("/resolve/"):
        raise DatasetIntegrityError("invalid_resolver_path")
    if any(character in path for character in ("?", "#", "\\")) or _contains_url_control(path):
        raise DatasetIntegrityError("invalid_resolver_path")
    if "//" in path[1:] or any(segment in {".", ".."} for segment in path.split("/")):
        raise DatasetIntegrityError("invalid_resolver_path")
    lowered = path.lower()
    if any(encoded in lowered for encoded in ("%2e", "%2f", "%5c", "%25")) or "://" in lowered:
        raise DatasetIntegrityError("invalid_resolver_path")


def join_api_path(base_url: str, api_relative_path: str) -> str:
    """Join a versioned API root to a resolver-root-relative path exactly once."""

    normalized = _normalize_public_base_url(base_url)
    _validate_resolver_path(api_relative_path)
    return f"{normalized}{api_relative_path}"


def _ascii_hostname(hostname: str | None) -> str | None:
    if hostname is None or not hostname or hostname.endswith("."):
        return None
    try:
        return hostname.encode("idna").decode("ascii").lower()
    except (UnicodeError, ValueError):
        return None


def _is_aws_region(region: str) -> bool:
    return len(region) <= 63 and _REGION_RE.fullmatch(region) is not None


def _is_aws_s3_hostname(hostname: str) -> bool:
    if hostname == "s3.amazonaws.com":
        return True
    dualstack = re.fullmatch(r"s3\.dualstack\.([a-z0-9-]{1,63})\.amazonaws\.com", hostname)
    if dualstack is not None:
        return _is_aws_region(dualstack.group(1))
    direct = re.fullmatch(r"s3[.-]([a-z0-9-]{1,63})\.amazonaws\.com", hostname)
    if direct is not None:
        return _is_aws_region(direct.group(1))

    suffix = ".amazonaws.com"
    if not hostname.endswith(suffix):
        return False
    labels = hostname[: -len(suffix)].split(".")
    if len(labels) < 2 or _ASCII_DNS_LABEL_RE.fullmatch(labels[0]) is None:
        return False
    service = labels[1:]
    if service in (["s3"], ["s3-accelerate"], ["s3-accelerate", "dualstack"]):
        return True
    if len(service) == 2 and service[0] == "s3":
        return _is_aws_region(service[1])
    if len(service) == 3 and service[:2] == ["s3", "dualstack"]:
        return _is_aws_region(service[2])
    if len(service) == 1 and service[0].startswith("s3-"):
        return _is_aws_region(service[0][3:])
    return False


def is_provider_hostname_allowed(
    provider: str,
    hostname: str,
    *,
    api_root_hostname: str | None = None,
) -> bool:
    """Return whether a provider hostname is in the deterministic v1 allowlist."""

    ascii_hostname = _ascii_hostname(hostname)
    if ascii_hostname is None:
        return False
    if provider == "aws_s3":
        return _is_aws_s3_hostname(ascii_hostname)
    if provider == "gcs":
        if ascii_hostname == "storage.googleapis.com":
            return True
        suffix = ".storage.googleapis.com"
        bucket = ascii_hostname[: -len(suffix)] if ascii_hostname.endswith(suffix) else ""
        return _ASCII_DNS_LABEL_RE.fullmatch(bucket) is not None
    if provider == "azure_blob":
        return _AZURE_HOST_RE.fullmatch(ascii_hostname) is not None
    if provider == "cloudflare_r2":
        return _R2_HOST_RE.fullmatch(ascii_hostname) is not None
    if provider == "avala_edge":
        return any(
            ascii_hostname == edge_host and _ascii_hostname(api_root_hostname) == urlsplit(api_base).hostname
            for api_base, edge_host in _EDGE_HOSTS_BY_API_BASE.items()
        )
    if provider == "avala_proxy":
        return ascii_hostname == _ascii_hostname(api_root_hostname)
    return False


def _validate_provider_url(url: str, provider: str, base_url: str) -> None:
    try:
        parsed = urlsplit(url)
        explicit_port = parsed.port
    except (TypeError, ValueError):
        raise DatasetIntegrityError("invalid_access_grant_url") from None
    if (
        parsed.scheme != "https"
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or explicit_port is not None
        or _contains_url_control(url)
    ):
        raise DatasetIntegrityError("invalid_access_grant_url")
    api_root = urlsplit(base_url)
    if not is_provider_hostname_allowed(provider, parsed.hostname, api_root_hostname=api_root.hostname):
        raise DatasetIntegrityError("untrusted_download_host")
    if provider == "avala_proxy":
        download_prefix = f"{api_root.path.rstrip('/')}/resolve/download/"
        lowered_path = parsed.path.lower()
        if (
            not parsed.path.startswith(download_prefix)
            or "\\" in parsed.path
            or any(encoded in lowered_path for encoded in ("%2e", "%2f", "%5c", "%25"))
            or "//" in parsed.path[1:]
            or any(segment in {".", ".."} for segment in parsed.path.split("/"))
        ):
            raise DatasetIntegrityError("untrusted_download_path")


def _validate_edge_grant_url(
    url: str,
    revision: ResolvedRevisionDocument,
    manifest_object: ManifestObjectDocument,
    base_url: str,
) -> None:
    parsed = urlsplit(url)
    expected_prefix = (
        f"/v1/datasets/{revision.dataset_uid}/{revision.revision_sha256}/{revision.manifest.sha256}"
        f"/objects/{manifest_object.object_uid}/"
    )
    if (
        parsed.netloc != _EDGE_HOSTS_BY_API_BASE.get(base_url)
        or not parsed.path.startswith(expected_prefix)
        or re.fullmatch(r"[0-9a-f]{64}", parsed.path[len(expected_prefix) :]) is None
        or "#" in url
    ):
        raise DatasetIntegrityError("access_grant_identity_mismatch")
    if not parsed.query.startswith("grant=") or "&" in parsed.query or len(parsed.query) > 6 + 3 * 4096:
        raise DatasetIntegrityError("invalid_access_grant_url")
    query = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True, max_num_fields=1)
    if len(query) != 1 or query[0][0] != "grant" or re.fullmatch(r"[A-Za-z0-9_.:-]{1,4096}", query[0][1]) is None:
        raise DatasetIntegrityError("invalid_access_grant_url")
    # The edge verifies the signed evidence digest against retained storage
    # proof. The SDK independently binds the resolved identities above and
    # verifies the manifest's full size and SHA-256 after streaming the bytes.


def _parse_model(model: type[_ResolverModel], data: Mapping[str, Any], reason: str) -> _ResolverModel:
    try:
        return model.model_validate(data)
    except PydanticValidationError:
        raise DatasetIntegrityError(reason) from None


def _try_parse_withdrawn_revision(
    data: Mapping[str, Any],
    *,
    expected_dataset_uid: str | None,
    expected_revision_sha256: str | None,
) -> tuple[WithdrawnRevisionDocument | None, str | None]:
    """Parse a tombstone without retaining its potentially untrusted extras."""

    try:
        document = cast(
            WithdrawnRevisionDocument,
            _parse_model(WithdrawnRevisionDocument, data, "invalid_withdrawal"),
        )
        return (
            _validate_withdrawn_revision(
                document,
                expected_dataset_uid=expected_dataset_uid,
                expected_revision_sha256=expected_revision_sha256,
            ),
            None,
        )
    except DatasetIntegrityError as error:
        return None, error.reason


def _validate_resolved_revision(
    document: ResolvedRevisionDocument,
    parsed_reference: ParsedDatasetReference,
) -> ResolvedRevisionDocument:
    canonical = f"avala://datasets/{document.dataset_uid}@{document.revision_sha256}"
    expected_objects_path = f"/resolve/uid/{document.dataset_uid}/{document.revision_sha256}/objects/"
    display_match = _FRIENDLY_REFERENCE_RE.fullmatch(document.display_reference)
    requested_match = _FRIENDLY_REFERENCE_RE.fullmatch(parsed_reference.requested_reference)
    if (
        display_match is None
        or display_match.group("selector") != document.revision_sha256
        or (
            requested_match is not None
            and (
                display_match.group("owner") != requested_match.group("owner")
                or display_match.group("dataset") != requested_match.group("dataset")
            )
        )
    ):
        raise DatasetIntegrityError("resolver_identity_mismatch")
    if (
        document.requested_reference != parsed_reference.requested_reference
        or document.canonical_reference != canonical
        or document.manifest.objects_path != expected_objects_path
        or parsed_reference.exact_dataset_uid not in {None, document.dataset_uid}
        or parsed_reference.exact_revision_sha256 not in {None, document.revision_sha256}
        or not 1 <= len(document.manifest.origins) <= 32
    ):
        raise DatasetIntegrityError("resolver_identity_mismatch")
    _validate_resolver_path(document.manifest.objects_path)
    origin_names = [origin.name for origin in document.manifest.origins]
    try:
        sorted_names = sorted(origin_names, key=lambda name: name.encode("ascii"))
    except UnicodeEncodeError:
        raise DatasetIntegrityError("invalid_manifest_origins") from None
    if origin_names != sorted_names or len(origin_names) != len(set(origin_names)):
        raise DatasetIntegrityError("invalid_manifest_origins")
    return document


def _validate_object_page(
    page: ManifestObjectPageDocument,
    revision: ResolvedRevisionDocument,
    *,
    role: str | None,
    ordinal: int | None,
    ordinals: tuple[int, ...] | None,
    cursor: str | None,
) -> ManifestObjectPageDocument:
    exact_lookup = ordinal is not None or ordinals is not None
    if (
        page.canonical_reference != revision.canonical_reference
        or page.revision_sha256 != revision.revision_sha256
        or page.manifest_sha256 != revision.manifest.sha256
        or (page.next_cursor is not None and (not page.next_cursor or len(page.next_cursor) > 8192))
        or (page.previous_cursor is not None and (not page.previous_cursor or len(page.previous_cursor) > 8192))
        or (ordinal is not None and len(page.results) > 1)
        or (ordinals is not None and len(page.results) > len(ordinals))
        or (exact_lookup and (page.next_cursor is not None or page.previous_cursor is not None))
        or (not exact_lookup and cursor is None and page.previous_cursor is not None)
        or (not exact_lookup and cursor is not None and (page.previous_cursor is None or not page.results))
    ):
        raise DatasetIntegrityError("object_page_identity_mismatch")
    known_origins = {origin.name for origin in revision.manifest.origins}
    for manifest_object in page.results:
        expected_path = f"{revision.manifest.objects_path}{manifest_object.object_uid}/access/"
        if (
            manifest_object.access_path != expected_path
            or manifest_object.origin not in known_origins
            or role not in {None, manifest_object.role}
            or ordinal not in {None, manifest_object.ordinal}
            or (ordinals is not None and manifest_object.ordinal not in ordinals)
        ):
            raise DatasetIntegrityError("object_identity_mismatch")
        _validate_resolver_path(manifest_object.access_path)
    return page


def _parse_utc_datetime(value: Any, reason: str) -> datetime:
    if not isinstance(value, str):
        raise DatasetIntegrityError(reason)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise DatasetIntegrityError(reason) from None
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(None):
        raise DatasetIntegrityError(reason)
    return parsed.astimezone(timezone.utc)


def _parse_server_date(value: str | None) -> datetime:
    if value is None:
        raise DatasetIntegrityError("missing_grant_date")
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        raise DatasetIntegrityError("invalid_grant_date") from None
    if parsed.tzinfo is None:
        raise DatasetIntegrityError("invalid_grant_date")
    return parsed.astimezone(timezone.utc)


def _require_string(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise DatasetIntegrityError("invalid_access_grant")
    return value


def _parse_access_grant(
    data: Mapping[str, Any],
    *,
    date_header: str | None,
    revision: ResolvedRevisionDocument,
    manifest_object: ManifestObjectDocument,
    base_url: str,
) -> DatasetAccessGrant:
    schema_version = data.get("schema_version")
    size_bytes = data.get("size_bytes")
    range_supported = data.get("range_supported")
    provider = _require_string(data, "provider")
    url = _require_string(data, "url")
    if (
        type(schema_version) is not int
        or schema_version != 1
        or type(size_bytes) is not int
        or size_bytes < 0
        or type(range_supported) is not bool
        or provider not in _PROVIDERS
        or _require_string(data, "method") != "GET"
        or _require_string(data, "canonical_reference") != revision.canonical_reference
        or _require_string(data, "dataset_uid") != revision.dataset_uid
        or _require_string(data, "revision_sha256") != revision.revision_sha256
        or _require_string(data, "manifest_sha256") != revision.manifest.sha256
        or _require_string(data, "object_uid") != manifest_object.object_uid
        or size_bytes != manifest_object.size_bytes
        or _require_string(data, "sha256") != manifest_object.sha256
    ):
        raise DatasetIntegrityError("access_grant_identity_mismatch")
    server_date = _parse_server_date(date_header)
    expires_at = _parse_utc_datetime(data.get("expires_at"), "invalid_grant_expiry")
    # V1 never caches a grant: the server Date caps its declared lifetime,
    # while the local receipt clocks prevent replay or processing delay from
    # restoring that full lifetime.
    lifetime = (expires_at - server_date).total_seconds()
    if lifetime <= 0 or lifetime > _MAX_GRANT_TTL_SECONDS:
        raise DatasetIntegrityError("invalid_grant_expiry")
    received_at = _utc_now()
    received_at_monotonic = _monotonic_time()
    response_age = (received_at - server_date).total_seconds()
    if response_age < -_LOCAL_CLOCK_SKEW_SECONDS:
        raise DatasetIntegrityError("invalid_grant_expiry")
    # The skew allowance accepts a Date slightly ahead of the local clock; it
    # must never restore lifetime to a response that is already old locally.
    local_remaining = lifetime - max(0.0, response_age)
    remaining_at_receipt = min(lifetime, local_remaining)
    _validate_provider_url(url, provider, base_url)
    if provider == "avala_edge":
        _validate_edge_grant_url(url, revision, manifest_object, base_url)
    return DatasetAccessGrant(
        provider=provider,
        url=url,
        expires_at=expires_at,
        server_date=server_date,
        object_uid=manifest_object.object_uid,
        size_bytes=size_bytes,
        sha256=manifest_object.sha256,
        range_supported=range_supported,
        remaining_at_receipt=remaining_at_receipt,
        received_at_monotonic=received_at_monotonic,
    )


def _try_parse_access_grant(
    data: Mapping[str, Any],
    *,
    date_header: str | None,
    revision: ResolvedRevisionDocument,
    manifest_object: ManifestObjectDocument,
    base_url: str,
) -> tuple[DatasetAccessGrant | None, str | None]:
    """Parse secret-bearing grant data without propagating its traceback frame."""

    try:
        return (
            _parse_access_grant(
                data,
                date_header=date_header,
                revision=revision,
                manifest_object=manifest_object,
                base_url=base_url,
            ),
            None,
        )
    except DatasetIntegrityError as error:
        return None, error.reason
    except Exception:
        # This is a secret-bearing response boundary. Even an unexpected
        # parser failure must be converted here so its traceback cannot retain
        # the signed URL in ``data`` or a nested validation frame.
        return None, "invalid_access_grant"


def _decode_resolver_json(response: httpx.Response) -> tuple[Any, bool]:
    """Decode a response without propagating an unexpected secret-bearing frame."""

    try:
        return response.json(), False
    except (ValueError, TypeError):
        return None, False
    except Exception:
        return None, True


def _rate_limit_retry_delay(value: str | None, attempt: int) -> float:
    """Return a bounded Retry-After delay with a small exponential fallback."""

    fallback = min(float(2**attempt), 30.0)
    if value is None:
        return fallback
    try:
        delay = float(value)
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return fallback
        if retry_at.tzinfo is None:
            return fallback
        delay = (retry_at.astimezone(timezone.utc) - _utc_now()).total_seconds()
    if not math.isfinite(delay) or delay < 0:
        return fallback
    return min(delay, _MAX_RETRY_AFTER_SECONDS)


class SyncDatasetResolverTransport:
    """Anonymous-only resolver client; it has no credential input or fallback."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL) -> None:
        self.base_url = _normalize_public_base_url(base_url)
        self._client = httpx.Client(
            headers={"Accept": "application/json"},
            timeout=_REQUEST_TIMEOUT_SECONDS,
            follow_redirects=False,
            trust_env=False,
        )

    def close(self) -> None:
        self._client.close()

    def _request(
        self,
        path: str,
        *,
        method: Literal["GET", "POST"] = "GET",
        params: Mapping[str, Any] | None = None,
        expected_dataset_uid: str | None = None,
        expected_revision_sha256: str | None = None,
    ) -> tuple[httpx.Response, Mapping[str, Any]]:
        url = join_api_path(self.base_url, path)
        for rate_limit_attempt in range(_MAX_RATE_LIMIT_RETRIES + 1):
            response: httpx.Response | None = None
            try:
                response = self._client.request(method, url, params=params)
            except httpx.HTTPError:
                pass
            if response is None:
                # Raise after leaving the httpx exception context: a partial JSON
                # body can already contain a signed URL retained by stream frames.
                raise DatasetResolverError("transport_error") from None
            if method == "POST" and response.status_code == 405:
                # Older servers expose only GET on this same authorized grant
                # resource. Never fall back for denials, outages or redirects.
                # Discard the body before recursion/any later exception.
                del response
                return self._request(
                    path,
                    params=params,
                    expected_dataset_uid=expected_dataset_uid,
                    expected_revision_sha256=expected_revision_sha256,
                )
            data, unexpected_decode_error = _decode_resolver_json(response)
            if unexpected_decode_error:
                del data
                del response
                raise DatasetIntegrityError("invalid_resolver_response") from None
            if response.status_code == 429 and rate_limit_attempt < _MAX_RATE_LIMIT_RETRIES:
                delay = _rate_limit_retry_delay(response.headers.get("Retry-After"), rate_limit_attempt)
                del data
                del response
                time.sleep(delay)
                continue
            if response.status_code == 410:
                if not isinstance(data, dict):
                    del data
                    del response
                    raise DatasetResolverError("invalid_withdrawal", 410)
                withdrawn, error_reason = _try_parse_withdrawn_revision(
                    data,
                    expected_dataset_uid=expected_dataset_uid,
                    expected_revision_sha256=expected_revision_sha256,
                )
                del data
                del response
                if withdrawn is None:
                    raise DatasetIntegrityError(error_reason or "invalid_withdrawal")
                canonical_reference = withdrawn.canonical_reference
                dataset_uid = withdrawn.dataset_uid
                revision_sha256 = withdrawn.revision_sha256
                del withdrawn
                raise DatasetRevisionWithdrawnError(
                    canonical_reference,
                    dataset_uid,
                    revision_sha256,
                )
            if not response.is_success:
                status_code = response.status_code
                if status_code == 404:
                    reason = "not_found"
                elif status_code == 429:
                    reason = "rate_limited"
                elif status_code >= 500:
                    reason = "service_unavailable"
                else:
                    reason = "request_rejected"
                del data
                del response
                raise DatasetResolverError(reason, status_code)
            if not isinstance(data, dict):
                del data
                del response
                raise DatasetIntegrityError("invalid_resolver_response")
            return response, cast(Mapping[str, Any], data)
        raise AssertionError("unreachable")

    def resolve(self, parsed_reference: ParsedDatasetReference) -> ResolvedRevisionDocument:
        _, data = self._request(
            parsed_reference.path,
            expected_dataset_uid=parsed_reference.exact_dataset_uid,
            expected_revision_sha256=parsed_reference.exact_revision_sha256,
        )
        document = cast(
            ResolvedRevisionDocument,
            _parse_model(ResolvedRevisionDocument, data, "invalid_resolver_response"),
        )
        return _validate_resolved_revision(document, parsed_reference)

    def list_objects(
        self,
        revision: ResolvedRevisionDocument,
        *,
        role: str | None = None,
        ordinal: int | None = None,
        ordinals: tuple[int, ...] | None = None,
        cursor: str | None = None,
        limit: int = 100,
    ) -> ManifestObjectPageDocument:
        if ordinals is not None and (
            role is None
            or ordinal is not None
            or cursor is not None
            or not 1 <= len(ordinals) <= 100
            or len(ordinals) != len(set(ordinals))
            or any(type(value) is not int or not 0 <= value <= _MAX_JSON_SAFE_INTEGER for value in ordinals)
        ):
            raise ValueError("ordinals must contain 1-100 unique canonical integers for one role")
        params: dict[str, Any] = {} if ordinals is not None else {"limit": limit}
        if role is not None:
            params["role"] = role
        if ordinal is not None:
            params["ordinal"] = ordinal
        if ordinals is not None:
            params["ordinals"] = ordinals
        if cursor is not None:
            params["cursor"] = cursor
        _, data = self._request(
            revision.manifest.objects_path,
            params=params,
            expected_dataset_uid=revision.dataset_uid,
            expected_revision_sha256=revision.revision_sha256,
        )
        page = cast(
            ManifestObjectPageDocument,
            _parse_model(ManifestObjectPageDocument, data, "invalid_object_page"),
        )
        return _validate_object_page(
            page,
            revision,
            role=role,
            ordinal=ordinal,
            ordinals=ordinals,
            cursor=cursor,
        )

    def issue_access_grant(
        self,
        revision: ResolvedRevisionDocument,
        manifest_object: ManifestObjectDocument,
        *,
        explicit_download: bool = False,
    ) -> DatasetAccessGrant:
        response, data = self._request(
            manifest_object.access_path,
            method="POST" if explicit_download else "GET",
            params={"transport": "avala-edge-v1"},
            expected_dataset_uid=revision.dataset_uid,
            expected_revision_sha256=revision.revision_sha256,
        )
        grant, error_reason = _try_parse_access_grant(
            data,
            date_header=response.headers.get("Date"),
            revision=revision,
            manifest_object=manifest_object,
            base_url=self.base_url,
        )
        # Neither the response body nor the httpx response may survive onto a
        # validation exception's traceback locals: both contain the signed URL.
        del data
        del response
        if grant is None:
            raise DatasetIntegrityError(error_reason or "invalid_access_grant")
        return grant


def _download_headers(provider: str) -> dict[str, str]:
    # GCS otherwise performs decompressive transcoding for objects stored with
    # ``Content-Encoding: gzip``. Requesting gzip keeps the signed object's raw
    # compressed bytes intact so size and SHA-256 verify against the manifest.
    return {"Accept-Encoding": "gzip" if provider == "gcs" else "identity"}


def _new_download_spool(*, max_size: int | None = None) -> BinaryIO:
    """Allocate the spool before any signed access grant is requested or retained."""

    return cast(
        BinaryIO,
        tempfile.SpooledTemporaryFile(
            max_size=_DOWNLOAD_SPOOL_MEMORY_BYTES if max_size is None else max_size,
            mode="w+b",
        ),
    )


def download_object_file(content: BinaryIO, grant: DatasetAccessGrant) -> BinaryIO:
    """Download into a verified seekable spool without retaining the object in RAM."""

    provider = grant.provider
    object_uid = grant.object_uid
    expected_size_bytes = grant.size_bytes
    expected_sha256 = grant.sha256
    download_url = grant.url
    del grant
    digest = hashlib.sha256()
    bytes_written = 0
    chunk = b""
    failure_reason: str | None = None
    client: httpx.Client | None = None
    response: httpx.Response | None = None
    try:
        client = httpx.Client(
            headers=_download_headers(provider),
            timeout=_REQUEST_TIMEOUT_SECONDS,
            follow_redirects=False,
            trust_env=False,
        )
        with client:
            with client.stream("GET", download_url) as response:
                if response.status_code != 200:
                    failure_reason = "redirect_rejected" if 300 <= response.status_code < 400 else "provider_response"
                else:
                    for chunk in response.iter_raw():
                        content.write(chunk)
                        bytes_written += len(chunk)
                        digest.update(chunk)
                        if bytes_written > expected_size_bytes:
                            failure_reason = "size_mismatch"
                            break
    except httpx.HTTPError:
        failure_reason = "transport_error"
    except OSError:
        failure_reason = "local_storage_error"
    finally:
        # Telemetry may inspect traceback locals recursively, so clear both the
        # secret string and every httpx object that can retain its request URL.
        download_url = ""
        client = None
        response = None
        chunk = b""
    if failure_reason is not None:
        content.close()
        raise DatasetDownloadError(provider, object_uid, failure_reason)
    if bytes_written != expected_size_bytes:
        content.close()
        raise DatasetDownloadError(provider, object_uid, "size_mismatch")
    if digest.hexdigest() != expected_sha256:
        content.close()
        raise DatasetDownloadError(provider, object_uid, "sha256_mismatch")
    content.seek(0)
    return content


def download_object_bytes(content: BinaryIO, grant: DatasetAccessGrant) -> bytes:
    """Download through the verified spool and return its bytes."""

    try:
        content = download_object_file(content, grant)
    finally:
        del grant
    with content:
        return content.read()


__all__ = [
    "DatasetAccessGrant",
    "ManifestObjectDocument",
    "ManifestObjectPageDocument",
    "ParsedDatasetReference",
    "ResolvedRevisionDocument",
    "SyncDatasetResolverTransport",
    "download_object_bytes",
    "download_object_file",
    "is_provider_hostname_allowed",
    "join_api_path",
    "parse_dataset_reference",
]
