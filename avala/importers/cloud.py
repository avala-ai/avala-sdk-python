"""Import a dataset over an EXISTING S3/GCS bucket — zero-copy, no re-upload.

Avala creates the dataset pointing at your bucket + prefix and indexes the objects
in place (the server lists the bucket and registers each matching file; nothing is
copied or re-uploaded). This is the fast path for "I already have terabytes in a
bucket — make it an Avala dataset."

Credentials are attached to the dataset's provider config:

* **AWS S3** — either static access keys (``access_key_id`` / ``secret_access_key``)
  or keyless cross-account ``role_arn`` (the customer grants Avala's account +
  external-id via an IAM trust policy; see ``client.storage_configs.setup_info()``).
* **GCS** — a service-account JSON blob (``gcs_auth_json``).

``data_type`` is required: the server only indexes objects whose extension matches it
(image / video / lidar / mcap / splat), so a folder of ``.jpg`` must be imported as
``image``. Narrow the scope with ``included_extensions`` / ``ignored_paths``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple, Union

from avala.importers import register_importer

if TYPE_CHECKING:
    from avala._client import Client
    from avala.types.dataset import Dataset

__all__ = ["import_cloud", "parse_cloud_uri"]

_VALID_DATA_TYPES = ("image", "video", "lidar", "mcap", "splat")


def parse_cloud_uri(uri: str) -> Tuple[str, str, str]:
    """Parse ``s3://bucket/prefix`` or ``gs://bucket/prefix`` (also ``gcs://``).

    Returns ``(provider, bucket, prefix)`` where provider is ``aws_s3`` or
    ``gc_storage`` and prefix may be empty.
    """
    for scheme, provider in (("s3://", "aws_s3"), ("gs://", "gc_storage"), ("gcs://", "gc_storage")):
        if uri.startswith(scheme):
            rest = uri[len(scheme) :]
            bucket, _, prefix = rest.partition("/")
            if not bucket:
                raise ValueError(f"missing bucket in cloud URI {uri!r}")
            return provider, bucket, prefix
    raise ValueError(f"unsupported cloud URI {uri!r}; expected s3://… or gs://…")


def _csv(value: Union[str, Sequence[str], None]) -> Optional[str]:
    """Normalize a list or CSV string to a clean CSV string (or None).

    Splits string input on commas and strips each item, so ``"jpg, png"`` becomes
    ``"jpg,png"`` — otherwise the server's extension/path filter would miss objects
    on the whitespace-padded entry.
    """
    if value is None:
        return None
    items = value.split(",") if isinstance(value, str) else value
    joined = ",".join(str(v).strip() for v in items if str(v).strip())
    return joined or None


def import_cloud(
    client: "Client",
    *,
    uri: str,
    name: str,
    slug: str,
    data_type: str,
    region: Optional[str] = None,
    access_key_id: Optional[str] = None,
    secret_access_key: Optional[str] = None,
    role_arn: Optional[str] = None,
    accelerated: bool = False,
    cloudfront_domain: Optional[str] = None,
    cloudfront_public_key_id: Optional[str] = None,
    gcs_auth_json: Optional[str] = None,
    included_extensions: Union[str, Sequence[str], None] = None,
    ignored_paths: Union[str, Sequence[str], None] = None,
    visibility: str = "private",
    create_metadata: bool = True,
    owner_name: Optional[str] = None,
    organization_id: Optional[int] = None,
    organization_uid: Optional[str] = None,
    industry: Optional[int] = None,
    license: Optional[int] = None,
    wait: bool = False,
    wait_timeout: float = 3600.0,
) -> "Dataset":
    """Create a zero-copy Avala dataset over an existing S3/GCS bucket.

    ``uri`` is ``s3://bucket/prefix`` or ``gs://bucket/prefix``. Provide S3 credentials
    (``access_key_id`` + ``secret_access_key``) or a keyless ``role_arn`` for S3, or a
    service-account JSON (``gcs_auth_json``, a JSON string or a path to a ``.json`` file)
    for GCS. With ``wait=True`` the call blocks until the server finishes indexing.
    """
    if data_type not in _VALID_DATA_TYPES:
        raise ValueError(f"invalid data_type {data_type!r}; expected one of {list(_VALID_DATA_TYPES)}")

    # Keyless S3 (IAM role) needs an owning organization: the server resolves the
    # cross-account external id from the dataset's organization, so a personal dataset
    # fails with "s3_external_id is required". Require an org explicitly up front.
    if role_arn and organization_id is None and organization_uid is None:
        raise ValueError(
            "keyless S3 imports (role_arn) require an owning organization — pass "
            "organization_uid (or organization_id). The server reads the cross-account "
            "external id from the organization."
        )

    provider, bucket, prefix = parse_cloud_uri(uri)
    provider_config: Dict[str, Any] = {"provider": provider}

    if provider == "aws_s3":
        provider_config["s3_bucket_name"] = bucket
        provider_config["s3_bucket_prefix"] = prefix
        if not region:
            raise ValueError("region is required for S3 imports (e.g. us-west-2)")
        provider_config["s3_bucket_region"] = region
        provider_config["s3_is_accelerated"] = accelerated
        if role_arn:
            provider_config["s3_auth_method"] = "iam_role"
            provider_config["s3_role_arn"] = role_arn
        elif access_key_id and secret_access_key:
            provider_config["s3_auth_method"] = "access_key"
            provider_config["s3_access_key_id"] = access_key_id
            provider_config["s3_secret_access_key"] = secret_access_key
        else:
            raise ValueError(
                "provide S3 credentials: either role_arn (keyless IAM role) or access_key_id + secret_access_key"
            )
        if cloudfront_domain:
            provider_config["s3_cloudfront_domain"] = cloudfront_domain
        if cloudfront_public_key_id:
            provider_config["s3_cloudfront_public_key_id"] = cloudfront_public_key_id
    else:  # gc_storage
        provider_config["gc_storage_bucket_name"] = bucket
        provider_config["gc_storage_prefix"] = prefix
        if not gcs_auth_json:
            raise ValueError("gcs_auth_json (service-account JSON or path to a .json file) is required for GCS")
        provider_config["gc_storage_auth_json_content"] = _read_gcs_auth(gcs_auth_json)

    included_csv = _csv(included_extensions)
    if included_csv:
        provider_config["included_extensions"] = included_csv
    ignored_csv = _csv(ignored_paths)
    if ignored_csv:
        provider_config["ignored_paths"] = ignored_csv

    dataset = client.datasets.create(
        name=name,
        slug=slug,
        data_type=data_type,
        visibility=visibility,
        create_metadata=create_metadata,
        provider_config=provider_config,
        owner_name=owner_name,
        organization_id=organization_id,
        organization_uid=organization_uid,
        industry=industry,
        license=license,
    )
    if wait:
        client.datasets.wait(dataset.uid, status="created", timeout=wait_timeout)
        dataset = client.datasets.get(dataset.uid)
    return dataset


def _read_gcs_auth(value: str) -> str:
    """Return service-account JSON content. ``value`` may be the JSON itself or a path."""
    import os

    if value.lstrip().startswith("{"):
        return value
    if os.path.isfile(value):
        with open(value, encoding="utf-8") as fh:
            return fh.read()
    raise ValueError(
        f"gcs_auth_json {value!r} is neither inline JSON nor a path to an existing file; "
        "pass the service-account JSON string or a valid .json key file path"
    )


# ``import_cloud`` is dependency-light (no boto3/gcs client — the server does the bucket
# I/O), so importing this for registration is cheap.
register_importer("cloud", import_cloud)
