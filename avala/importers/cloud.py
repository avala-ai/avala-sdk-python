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


def _apply_storage_config(
    client: "Client",
    storage_config_uid: str,
    uri: Optional[str],
    region: Optional[str],
    organization_uid: Optional[str],
) -> tuple[str, str, str, Optional[str], bool]:
    """Resolve a saved storage config into ``(provider, bucket, prefix, region, accelerated)``.

    A storage config is a verified, reusable pointer at a customer bucket. Reusing
    it beats retyping the bucket, region and prefix on every import — that
    retyping is how a dataset ends up registered against the wrong region and
    fails to index with no obvious cause.

    ``uri`` stays optional and, when given, must point into the *same* bucket:
    one config commonly backs many datasets under different prefixes. A URI for
    a different bucket is a mistake, not an override, so it is rejected rather
    than silently winning.
    """
    # Scope the lookup to the org the dataset will belong to. A storage config
    # is org-owned but the read model exposes no owner, so an unscoped fetch
    # lets a caller in orgs A and B point a B-owned dataset at A's bucket —
    # exposing A's objects to B's members where the credentials happen to work,
    # or failing obscurely because IAM external ids are per-organization. The
    # server enforces the scope, so a mismatch 404s rather than being trusted.
    # `organization_uid` is REQUIRED here, not optional. Storage configs are
    # always organization-owned — `StorageConfigViewSet.perform_create` resolves
    # one unconditionally and 403s a caller with no membership, so a personal
    # config does not exist. Allowing the omission therefore did two wrong
    # things at once: the lookup went unscoped across every membership, and the
    # subsequent `datasets.create` received no organization, quietly producing a
    # personally-owned dataset over an organization's bucket. It would not show
    # up for the colleagues it was meant for, and it would stay with the user
    # after they left the org.
    if not organization_uid:
        raise ValueError(
            "organization_uid is required when importing with a saved storage config: configs are "
            "organization-owned, so the dataset must be created under the same organization. Pass "
            "--organization-uid (CLI) or organization_uid= (SDK)."
        )
    org_slug = client.organizations.slug_for_uid(organization_uid)
    if org_slug is None:
        raise ValueError(
            f"could not resolve organization {organization_uid} — you must be a member of the "
            "organization that owns the storage config and the dataset."
        )
    config = client.storage_configs.get(storage_config_uid, organization=org_slug)
    if not config.is_verified:
        raise ValueError(
            f"storage config {config.name!r} has not been verified. "
            f"Run `avala storage-configs test {storage_config_uid}` and fix the access it reports."
        )

    if config.provider == "aws_s3":
        bucket = config.s3_bucket_name or ""
        prefix = config.s3_bucket_prefix or ""
        # The config's region wins. ``--region`` reads from ``AWS_REGION``, so
        # letting the argument take precedence means merely running in a shell
        # with that variable set silently replaces the region the config was
        # *verified* against — registering a dataset that then fails to index,
        # with nothing in the command to explain why.
        resolved_region = config.s3_bucket_region or region
        accelerated = config.s3_is_accelerated
    else:
        bucket = config.gc_storage_bucket_name or ""
        prefix = config.gc_storage_prefix or ""
        resolved_region = region
        accelerated = False
    if not bucket:
        raise ValueError(f"storage config {config.name!r} has no bucket configured")

    if uri:
        uri_provider, uri_bucket, uri_prefix = parse_cloud_uri(uri)
        if uri_provider != config.provider:
            raise ValueError(f"{uri} is a {uri_provider} URI but storage config {config.name!r} is {config.provider}.")
        if uri_bucket != bucket:
            raise ValueError(
                f"{uri} points at bucket {uri_bucket!r}, but storage config {config.name!r} "
                f"is for {bucket!r}. Use a URI inside the config's bucket, or omit it."
            )
        # A URI may only *narrow* the config's prefix, never step outside it.
        # Same-bucket is not the boundary the config represents: a config
        # verified for `deliveries/` says nothing about `finance-exports/`, and
        # silently honouring the latter would import from a location whose
        # access was never checked.
        #
        # Compare and register the prefixes EXACTLY as given — no stripping, no
        # reconstruction. S3 prefixes are opaque lexical byte strings: `/x`,
        # `x`, `x/` and `x//` are four different namespaces, and every
        # normalisation step is a guess about which one the customer meant.
        #
        # Three consecutive review rounds found a different delimiter shape that
        # the strip-compare-reconstruct approach silently rewrote — a leading
        # slash, then a slash-only root, then a doubled trailing slash — each
        # time letting a URI register outside the namespace the config was
        # verified for, because both sides compared equal once stripped. They
        # were three instances of one defect: the comparison discarded exactly
        # the characters that distinguish the namespaces. Comparing raw ends the
        # class rather than the case.
        #
        # A URI that isn't inside the configured prefix is now REJECTED rather
        # than rewritten into one that is. That is the honest answer: the SDK
        # cannot know whether `s3://bucket//deliveries/run` against a config
        # rooted at `deliveries` is a typo or a literal `/deliveries/` key, and
        # guessing wrong indexes the wrong objects silently. The error names
        # both strings so the fix is obvious.
        if prefix and uri_prefix != prefix:
            # A configured prefix that already ends in the delimiter is its own
            # boundary. One that doesn't needs the delimiter appended, or
            # `deliveries` would also admit `deliveries-archive/…` — lexically a
            # match, but a different directory and never verified.
            boundary = prefix if prefix.endswith("/") else f"{prefix}/"
            if not uri_prefix.startswith(boundary):
                raise ValueError(
                    f"{uri} is outside storage config {config.name!r}'s prefix {prefix!r}. "
                    "A URI may narrow the config's prefix, not replace it — note that leading, "
                    "trailing and repeated '/' are all significant in an S3 key."
                )
        prefix = uri_prefix

    return config.provider, bucket, prefix, resolved_region, accelerated


def import_cloud(
    client: "Client",
    *,
    uri: Optional[str] = None,
    name: str,
    slug: str,
    data_type: str,
    storage_config_uid: Optional[str] = None,
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

    Nothing is uploaded: the server indexes the objects where they already are.
    For data that is already in cloud storage this is strictly better than a
    managed upload — no transfer cost, no second copy, and a correction is a
    re-push to the same key rather than a whole new dataset.

    ``uri`` is ``s3://bucket/prefix`` or ``gs://bucket/prefix``. Provide S3
    credentials (``access_key_id`` + ``secret_access_key``) or a keyless
    ``role_arn`` for S3, or a service-account JSON (``gcs_auth_json``, a JSON
    string or a path to a ``.json`` file) for GCS. With ``wait=True`` the call
    blocks until the server finishes indexing.

    ``storage_config_uid`` reuses a saved, verified storage config for the
    bucket, region, prefix and acceleration flag, so those need not be retyped
    per dataset. **Credentials are still required separately**: the server never
    returns them, and it does not return ``s3_role_arn`` either — the
    storage-config read serializer exposes no auth material at all, by design
    (ARNs are treated as sensitive there; see the redaction in
    ``server/apps/dataset/api_storage.py``). Pass ``uri`` alongside it to select
    a narrower prefix inside the same bucket.
    """
    if data_type not in _VALID_DATA_TYPES:
        raise ValueError(f"invalid data_type {data_type!r}; expected one of {list(_VALID_DATA_TYPES)}")
    if not uri and not storage_config_uid:
        raise ValueError("provide a cloud uri (s3://… or gs://…) or a storage_config_uid")

    # Keyless S3 (IAM role) needs an owning organization: the server resolves the
    # cross-account external id from the dataset's organization, so a personal dataset
    # fails with "s3_external_id is required". Require an org explicitly up front.
    if role_arn and organization_id is None and organization_uid is None:
        raise ValueError(
            "keyless S3 imports (role_arn) require an owning organization — pass "
            "organization_uid (or organization_id). The server reads the cross-account "
            "external id from the organization."
        )

    if storage_config_uid and organization_id is not None and organization_uid is None:
        # A storage config is org-owned, and the lookup is scoped by *slug*.
        # There is no id->slug route, so an id-only caller would fall through to
        # an unscoped fetch — exactly the cross-organization case the scoping
        # exists to prevent, just reached by the other parameter. Refuse rather
        # than quietly widen it.
        raise ValueError(
            "storage_config_uid requires organization_uid (organization_id cannot be scoped to a "
            "storage config). Pass the organization's uid."
        )

    if storage_config_uid:
        provider, bucket, prefix, region, accelerated = _apply_storage_config(
            client, storage_config_uid, uri, region, organization_uid
        )
    else:
        assert uri is not None  # guarded above
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
        elif storage_config_uid:
            # Be specific about why a saved config isn't enough on its own —
            # "provide credentials" reads like a bug when the user just
            # supplied a config that demonstrably has working access.
            raise ValueError(
                "a storage config supplies the bucket, region and prefix, but not credentials: "
                "the server never returns them (nor the role ARN). Pass role_arn, or "
                "access_key_id + secret_access_key, alongside --storage-config."
            )
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
    # Deliberately does NOT interpolate ``value``: it is either a
    # service-account private key or a path, and this error reaches terminal and
    # CI logs. A BOM-prefixed or truncated JSON blob fails the `{` test above and
    # would otherwise print the whole credential while reporting the problem.
    raise ValueError(
        "gcs_auth_json is neither inline JSON nor a path to an existing file; "
        "pass the service-account JSON string or a valid .json key file path"
    )


# ``import_cloud`` is dependency-light (no boto3/gcs client — the server does the bucket
# I/O), so importing this for registration is cheap.
register_importer("cloud", import_cloud)
