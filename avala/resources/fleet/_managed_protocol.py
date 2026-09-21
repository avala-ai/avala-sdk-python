"""Validate managed Fleet part capabilities against the versioned server grammar."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import parse_qsl, quote, urlsplit
from uuid import UUID

import httpx

from avala._fleet_uploads import SourceFile
from avala.errors import UploadStateError

_PART_SIZE = 64 * 1024**2
_R2_HOST = re.compile(r"[0-9a-f]{32}(?:\.eu)?\.r2\.cloudflarestorage\.com")
_BUCKET = re.compile(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_QUERY_FIELDS = {
    "uploadId",
    "partNumber",
    "X-Amz-Algorithm",
    "X-Amz-Credential",
    "X-Amz-Date",
    "X-Amz-Expires",
    "X-Amz-SignedHeaders",
    "X-Amz-Signature",
}
_GRANT_FIELDS = {"grant_uid", "part_number", "offset", "size_bytes", "url", "headers", "expires_at"}


@dataclass(frozen=True)
class ValidatedGrant:
    url: str = field(repr=False)
    headers: dict[str, str] = field(repr=False)
    pin: str


def _uuid(value: object) -> str:
    if not isinstance(value, str) or str(UUID(value)) != value:
        raise ValueError
    return value


def _signature_query(query: str) -> dict[str, str]:
    pairs = parse_qsl(query, keep_blank_values=True, strict_parsing=True, errors="strict", max_num_fields=16)
    values = dict(pairs)
    if len(values) != len(pairs) or set(values) != _QUERY_FIELDS:
        raise ValueError
    # Match boto's percent encoding, including literal plus signs in upload IDs.
    # Canonical pairs also reject malformed escapes and URL parser ambiguity.
    canonical = "&".join(f"{quote(key, safe='-_.~')}={quote(value, safe='-_.~')}" for key, value in pairs)
    if canonical != query:
        raise ValueError
    return values


def _validate(
    raw: dict[str, Any], *, session_uid: str, file: SourceFile, number: int, pin: str | None
) -> ValidatedGrant:
    if not isinstance(raw, dict) or set(raw) != _GRANT_FIELDS:
        raise ValueError
    _uuid(raw["grant_uid"])
    _uuid(session_uid)
    if (
        type(file.size_bytes) is not int
        or not 0 < file.size_bytes <= 8 * 1024**3
        or type(number) is not int
        or not 1 <= number <= (file.size_bytes + _PART_SIZE - 1) // _PART_SIZE
        or not isinstance(file.path, str)
        or unicodedata.normalize("NFC", file.path) != file.path
        or "\\" in file.path
        or any(ord(char) < 32 or ord(char) == 127 for char in file.path)
        or any(segment in {"", ".", ".."} for segment in file.path.split("/"))
        or file.path.startswith("__derived__/")
        or not file.path.lower().endswith(".mcap")
    ):
        raise ValueError
    offset = (number - 1) * _PART_SIZE
    size = min(_PART_SIZE, file.size_bytes - offset)
    for key, expected in (("part_number", number), ("offset", offset), ("size_bytes", size)):
        if type(raw[key]) is not int or raw[key] != expected:
            raise ValueError
    headers = raw["headers"]
    if (
        not isinstance(headers, dict)
        or len(headers) != 1
        or any(not isinstance(key, str) or key.lower() != "content-length" for key in headers)
        or list(headers.values()) != [str(size)]
    ):
        raise ValueError
    url = raw["url"]
    if (
        not isinstance(url, str)
        or not 1 <= len(url) <= 16_384
        or not url.startswith("https://")
        or "#" in url
        or any(not 33 <= ord(char) <= 126 for char in url)
    ):
        raise ValueError
    parsed = urlsplit(url)
    if parsed.scheme != "https" or _R2_HOST.fullmatch(parsed.netloc) is None or parsed.fragment:
        raise ValueError
    # Status intentionally withholds storage credentials and routing internals.
    # The first grant's owner/account/bucket is trusted from the authenticated API;
    # the pin prevents any later capability from changing that destination.
    segments = parsed.path.split("/")
    if (
        len(segments) < 7
        or segments[0] != ""
        or _BUCKET.fullmatch(segments[1]) is None
        or segments[2] != "__o__%3D"
        or segments[4] != "__cloud__"
    ):
        raise ValueError
    owner_uid = _uuid(segments[3])
    key = f"__o__=/{owner_uid}/__cloud__/{session_uid}/{file.path}"
    expected_path = f"/{segments[1]}/{quote(key, safe='/')}"
    effective_path = httpx.URL(url).raw_path.split(b"?", 1)[0].decode("ascii")
    if parsed.path != expected_path or effective_path != expected_path or len(key.encode("utf-8")) > 1024:
        raise ValueError
    query = _signature_query(parsed.query)
    upload_id = query["uploadId"]
    if (
        query["partNumber"] != str(number)
        or query["X-Amz-Algorithm"] != "AWS4-HMAC-SHA256"
        or query["X-Amz-SignedHeaders"] != "content-length;host"
        or _SHA256.fullmatch(query["X-Amz-Signature"]) is None
        or not 1 <= len(upload_id) <= 1024
        or upload_id != upload_id.strip()
        or not upload_id.isascii()
        or not upload_id.isprintable()
    ):
        raise ValueError
    lifetime = query["X-Amz-Expires"]
    if (
        not lifetime.isascii()
        or not lifetime.isdecimal()
        or lifetime != str(int(lifetime))
        or not 1 <= int(lifetime) <= 300
    ):
        raise ValueError
    signed_at = datetime.strptime(query["X-Amz-Date"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    if query["X-Amz-Date"] != signed_at.strftime("%Y%m%dT%H%M%SZ"):
        raise ValueError
    scope = query["X-Amz-Credential"].split("/")
    if (
        len(scope) != 5
        or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", scope[0])
        or scope[1:] != [signed_at.strftime("%Y%m%d"), "auto", "s3", "aws4_request"]
    ):
        raise ValueError
    if not isinstance(raw["expires_at"], str):
        raise ValueError
    expires_at = datetime.fromisoformat(raw["expires_at"].replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    if (
        expires_at.tzinfo is None
        or expires_at != signed_at + timedelta(seconds=int(lifetime))
        or signed_at > now + timedelta(seconds=30)
        or expires_at <= now
    ):
        raise ValueError
    identity = json.dumps([parsed.netloc, expected_path, upload_id], ensure_ascii=True, separators=(",", ":"))
    observed_pin = hashlib.sha256(identity.encode("ascii")).hexdigest()
    if pin is not None and (
        not isinstance(pin, str) or _SHA256.fullmatch(pin) is None or not hmac.compare_digest(pin, observed_pin)
    ):
        raise ValueError
    return ValidatedGrant(url=url, headers={"Content-Length": str(size)}, pin=observed_pin)


def validate_grant(
    raw: dict[str, Any], *, session_uid: str, file: SourceFile, number: int, pin: str | None = None
) -> ValidatedGrant:
    """Return one validated capability and its secret-free transport binding."""
    try:
        return _validate(raw, session_uid=session_uid, file=file, number=number, pin=pin)
    except (ValueError, TypeError, KeyError, OverflowError, httpx.InvalidURL):
        pass
    # Raise outside the handler so URL parser exceptions cannot leak capabilities
    # through exception chaining even when a caller prints the complete traceback.
    raise UploadStateError("Managed Fleet part capability is invalid. Keep the upload checkpoint and original files.")
