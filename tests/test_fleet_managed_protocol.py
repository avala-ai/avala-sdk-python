"""Exercise the actual Fleet multipart capability grammar before any file PUT."""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

import pytest
from botocore.auth import S3SigV4QueryAuth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials

from avala._fleet_uploads import SourceFile
from avala.errors import UploadStateError
from avala.resources.fleet._managed_protocol import validate_grant

PART_SIZE = 64 * 1024**2
SESSION = "11111111-1111-4111-8111-111111111111"
OWNER = "22222222-2222-4222-8222-222222222222"
FILE = SourceFile("robot + α/clip.MCAP", PART_SIZE + 7, "a" * 64)
ACCOUNT = "0123456789abcdef0123456789abcdef"


def _grant(*, number: int = 1, eu: bool = False, seconds_ago: int = 1) -> dict[str, Any]:
    signed = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(seconds=seconds_ago)
    query = {
        "uploadId": "synthetic+upload/==",
        "partNumber": str(number),
        "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
        "X-Amz-Credential": f"fixture-access/{signed:%Y%m%d}/auto/s3/aws4_request",
        "X-Amz-Date": f"{signed:%Y%m%dT%H%M%SZ}",
        "X-Amz-Expires": "300",
        "X-Amz-SignedHeaders": "content-length;host",
        "X-Amz-Signature": "a" * 64,
    }
    key = f"__o__=/{OWNER}/__cloud__/{SESSION}/{FILE.path}"
    endpoint = f"https://{ACCOUNT}{'.eu' if eu else ''}.r2.cloudflarestorage.com"
    size = min(PART_SIZE, FILE.size_bytes - (number - 1) * PART_SIZE)
    return {
        "grant_uid": "33333333-3333-4333-8333-333333333333",
        "part_number": number,
        "offset": (number - 1) * PART_SIZE,
        "size_bytes": size,
        "url": f"{endpoint}/avala-dev/{quote(key, safe='/')}?{urlencode(query, quote_via=quote)}",
        "headers": {"Content-Length": str(size)},
        "expires_at": (signed + timedelta(seconds=300)).isoformat(),
    }


def _query(raw: dict[str, Any], **changes: str | None) -> None:
    parsed = urlsplit(raw["url"])
    query = dict(parse_qsl(parsed.query))
    for key, value in changes.items():
        if value is None:
            query.pop(key, None)
        else:
            query[key] = value
    raw["url"] = urlunsplit(parsed._replace(query=urlencode(query, quote_via=quote)))


@pytest.mark.parametrize("eu", [False, True])
@pytest.mark.parametrize("number", [1, 2])
def test_valid_grant_preserves_capability_and_hides_it_from_repr(eu: bool, number: int) -> None:
    raw = _grant(number=number, eu=eu)
    result = validate_grant(raw, session_uid=SESSION, file=FILE, number=number)
    assert result.url == raw["url"]
    assert result.headers == raw["headers"]
    assert len(result.pin) == 64
    assert raw["url"] not in repr(result)
    assert "synthetic+upload" not in repr(result)
    assert "fixture-access" not in repr(result)


def test_real_botocore_part_signer_matches_sdk_grammar() -> None:
    raw = _grant(eu=True)
    # Exercise the same offline S3 query signer without constructing a storage
    # client outside the provider boundary or resolving ambient credentials.
    unsigned = urlunsplit(urlsplit(raw["url"])._replace(query=""))
    request = AWSRequest(
        method="PUT",
        url=unsigned,
        params={"uploadId": "synthetic+upload/==", "partNumber": "1"},
        headers=raw["headers"],
    )
    signer = S3SigV4QueryAuth(Credentials("fixture-access", "fixture-signing-material"), "s3", "auto", expires=300)
    signer.add_auth(request)
    raw["url"] = request.url
    signed = dict(parse_qsl(urlsplit(raw["url"]).query))["X-Amz-Date"]
    timestamp = datetime.strptime(signed, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    raw["expires_at"] = (timestamp + timedelta(seconds=300)).isoformat()
    assert validate_grant(raw, session_uid=SESSION, file=FILE, number=1).url == raw["url"]


def test_pin_survives_part_change_signature_refresh_and_query_reordering() -> None:
    first = validate_grant(_grant(), session_uid=SESSION, file=FILE, number=1)
    raw = _grant(number=2, seconds_ago=2)
    _query(raw, **{"X-Amz-Signature": "b" * 64})
    parsed = urlsplit(raw["url"])
    raw["url"] = urlunsplit(parsed._replace(query="&".join(reversed(parsed.query.split("&")))))
    assert validate_grant(raw, session_uid=SESSION, file=FILE, number=2, pin=first.pin).pin == first.pin


@pytest.mark.parametrize("change", ["endpoint", "jurisdiction", "bucket", "owner", "upload_id"])
def test_retained_pin_rejects_destination_or_upload_substitution(change: str) -> None:
    first = validate_grant(_grant(), session_uid=SESSION, file=FILE, number=1)
    raw = _grant()
    if change == "endpoint":
        raw["url"] = raw["url"].replace(ACCOUNT, "f" * 32)
    elif change == "jurisdiction":
        raw["url"] = raw["url"].replace(".r2.", ".eu.r2.")
    elif change == "bucket":
        raw["url"] = raw["url"].replace("/avala-dev/", "/different-bucket/")
    elif change == "owner":
        raw["url"] = raw["url"].replace(OWNER, "44444444-4444-4444-8444-444444444444")
    else:
        _query(raw, uploadId="different-upload")
    with pytest.raises(UploadStateError):
        validate_grant(raw, session_uid=SESSION, file=FILE, number=1, pin=first.pin)


@pytest.mark.parametrize(
    "name,value",
    [
        ("part_number", True),
        ("part_number", 2),
        ("offset", True),
        ("offset", 1),
        ("size_bytes", str(PART_SIZE)),
        ("size_bytes", PART_SIZE - 1),
        ("grant_uid", "not-a-uuid"),
        ("headers", {}),
        ("headers", {"Content-Length": str(PART_SIZE), "content-length": str(PART_SIZE)}),
        ("headers", {"Content-Length": str(PART_SIZE), "Authorization": "synthetic-sensitive-value"}),
        ("headers", {"Content-Length": f"{PART_SIZE}\r\nX-Forged: yes"}),
        ("headers", {"Content-Length": f"0{PART_SIZE}"}),
        ("headers", {"Content-Length": PART_SIZE}),
        ("expires_at", "2026-09-12T00:00:00"),
        ("expires_at", "not-a-date"),
    ],
)
def test_rejects_malformed_part_descriptor(name: str, value: Any) -> None:
    raw = _grant()
    raw[name] = value
    with pytest.raises(UploadStateError):
        validate_grant(raw, session_uid=SESSION, file=FILE, number=1)


@pytest.mark.parametrize("field", list(_grant()))
def test_rejects_missing_required_descriptor_fields(field: str) -> None:
    raw = _grant()
    del raw[field]
    with pytest.raises(UploadStateError):
        validate_grant(raw, session_uid=SESSION, file=FILE, number=1)


@pytest.mark.parametrize(
    "name,value",
    [
        ("uploadId", ""),
        ("uploadId", "a" * 1025),
        ("uploadId", " leading-space"),
        ("uploadId", "line\nbreak"),
        ("partNumber", "2"),
        ("partNumber", "01"),
        ("X-Amz-Algorithm", "AWS4-ECDSA-P256-SHA256"),
        ("X-Amz-Expires", "301"),
        ("X-Amz-Expires", "0"),
        ("X-Amz-Expires", "0300"),
        ("X-Amz-SignedHeaders", "host"),
        ("X-Amz-SignedHeaders", "authorization;content-length;host"),
        ("X-Amz-Signature", "z" * 64),
        ("X-Amz-Signature", None),
        ("X-Amz-Credential", "fixture-access/20260101/us-east-1/s3/aws4_request"),
        ("X-Amz-Credential", "fixture-access/20260101/auto/execute-api/aws4_request"),
        ("X-Amz-Security-Token", "unexpected-capability"),
        ("unexpected", "field"),
    ],
)
def test_rejects_unsupported_signature_queries(name: str, value: str | None) -> None:
    raw = _grant()
    _query(raw, **{name: value})
    with pytest.raises(UploadStateError):
        validate_grant(raw, session_uid=SESSION, file=FILE, number=1)


@pytest.mark.parametrize(
    "old,new",
    [
        ("https://", "http://"),
        ("https://", "https://user:password@"),
        (".com/", ".com:443/"),
        (".com/", ".com.evil.invalid/"),
        (".r2.", ".fedramp.r2."),
        (ACCOUNT, "a" * 31),
        ("__o__%3D/", "__o__=/"),
        ("__o__%3D/", "__u__%3D/"),
        (SESSION, "55555555-5555-4555-8555-555555555555"),
        ("robot%20", "./robot%20"),
        ("robot%20", "folder/../robot%20"),
        ("clip.MCAP", "other.mcap"),
        ("robot%20", "robot "),
        ("%CE%B1", "α"),
        ("%CE%B1", "%ce%b1"),
        ("%2B", "+"),
    ],
)
def test_rejects_changed_or_noncanonical_targets(old: str, new: str) -> None:
    raw = _grant()
    raw["url"] = raw["url"].replace(old, new)
    with pytest.raises(UploadStateError):
        validate_grant(raw, session_uid=SESSION, file=FILE, number=1)


@pytest.mark.parametrize("suffix", ["#fragment", "#", "&partNumber=1", "&X-Amz-Expires=300", "&", "\n"])
def test_rejects_duplicate_or_ambiguous_url_material(suffix: str) -> None:
    raw = _grant()
    raw["url"] += suffix
    with pytest.raises(UploadStateError):
        validate_grant(raw, session_uid=SESSION, file=FILE, number=1)


@pytest.mark.parametrize("seconds_ago", [301, -61])
def test_rejects_expired_or_future_signature(seconds_ago: int) -> None:
    with pytest.raises(UploadStateError):
        validate_grant(_grant(seconds_ago=seconds_ago), session_uid=SESSION, file=FILE, number=1)


def test_requires_expiry_to_match_the_actual_signature_window() -> None:
    raw = _grant()
    raw["expires_at"] = (datetime.now(timezone.utc) + timedelta(seconds=600)).isoformat()
    with pytest.raises(UploadStateError):
        validate_grant(raw, session_uid=SESSION, file=FILE, number=1)


def test_errors_and_repr_do_not_expose_capabilities() -> None:
    raw = _grant()
    _query(raw, **{"X-Amz-Signature": "synthetic-sensitive-signature"})
    with pytest.raises(UploadStateError) as caught:
        validate_grant(raw, session_uid=SESSION, file=FILE, number=1)
    assert "synthetic-sensitive-signature" not in str(caught.value)
    assert "fixture-access" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_validator_does_not_mutate_the_raw_response() -> None:
    raw = _grant()
    original = copy.deepcopy(raw)
    validate_grant(raw, session_uid=SESSION, file=FILE, number=1)
    assert raw == original
