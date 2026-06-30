"""Regression tests for pentest 2026-04-18 sdks/s3-4.

The HTTP transports re-raise the server's ``detail`` string verbatim
into the exception message and ``body`` attribute. The server commonly
echoes parts of the request payload in 4xx/5xx errors (validation
errors quote the offending field value). Without redaction, secrets
in the request flow into caller logs / Sentry / stdout.

The fix applies a small set of regex redactions to ``message`` and
``body`` before raising — best-effort defense-in-depth.
"""

from __future__ import annotations

from avala._redaction import redact


class TestRedactStrings:
    def test_passes_through_safe_string(self):
        assert redact("Validation failed for field 'name'") == "Validation failed for field 'name'"

    def test_redacts_jwt(self):
        leaky = "Invalid JWT: eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyMSJ9.signature_here_aaa"
        out = redact(leaky)
        assert "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9" not in out
        assert "[redacted]" in out

    def test_redacts_aws_access_key(self):
        leaky = "Provided key 'AKIAIOSFODNN7EXAMPLE' is not valid"
        out = redact(leaky)
        assert "AKIAIOSFODNN7EXAMPLE" not in out
        assert "[redacted]" in out

    def test_redacts_prefix_keyed_secret(self):
        # Generic ``secret_*`` shape (Stripe / Resend / Supabase / Linear
        # use variants on this pattern). Synthetic ASCII values picked
        # so GitHub's secret-scanner doesn't flag the test fixture
        # itself as a real provider key.
        leaky = "Got secret_NotARealKeyJustATestValueXYZ123456 rejected"
        out = redact(leaky)
        assert "NotARealKeyJustATestValueXYZ123456" not in out
        assert "[redacted]" in out

    def test_redacts_modern_hyphenated_provider_keys(self):
        # Bug-bounty: OpenAI ``sk-proj-...`` and Anthropic ``sk-ant-api03-...``
        # are ``-``-separated, so the underscore-keyed pattern missed them and
        # an inference ``config.api_key`` echoed in a ``detail`` string leaked.
        # Synthetic fake values (not real keys).
        for leaky in (
            "Invalid provider config api_key=sk-proj-NotARealKeyJustATestValue0123456789abcdef",
            "bad anthropic key sk-ant-api03-NotARealKeyJustATestValue0123456789abcdef",
        ):
            out = redact(leaky)
            assert "NotARealKeyJustATestValue0123456789abcdef" not in out
            assert "[redacted]" in out

    def test_modern_provider_key_redacted_via_structured_body(self):
        # The same key embedded in a server ``detail`` string inside a body.
        body = {"detail": "bad provider config api_key=sk-proj-NotARealKeyJustATestValue0123456789abcdef"}
        out = redact(body)
        assert "NotARealKeyJustATestValue0123456789abcdef" not in str(out)
        assert "[redacted]" in out["detail"]

    def test_redacts_avala_api_key_shape(self):
        # 40-char hex matches the Avala ``secrets.token_hex(20)`` output.
        leaky = "Auth failed for key 0123456789abcdef0123456789abcdef01234567"
        out = redact(leaky)
        assert "0123456789abcdef0123456789abcdef01234567" not in out
        assert "[redacted]" in out

    def test_redacts_bearer_header(self):
        leaky = "Server got Authorization: Bearer abc123tokendata"
        out = redact(leaky)
        assert "abc123tokendata" not in out
        assert "[redacted]" in out

    def test_redacts_x_avala_api_key_header_echo(self):
        leaky = "X-Avala-Api-Key: my-secret-value rejected by middleware"
        out = redact(leaky)
        assert "my-secret-value" not in out

    def test_redacts_aws_arn(self):
        leaky = "User arn:aws:iam::123456789012:user/customer-bridge lacks permission"
        out = redact(leaky)
        assert "arn:aws:iam::123456789012:user/customer-bridge" not in out
        assert "[redacted]" in out

    def test_redacts_aws_secret_access_key_field_echo(self):
        """Reviewer P1 follow-up on PR #11315: AWS *secret* access keys
        are 40 chars of base64-ish ``[A-Za-z0-9/+=]`` — too generic to
        match standalone. Context-aware redaction kicks in when the
        secret-key value is echoed next to a known field name.

        Synthetic value picked so GitHub's secret scanner doesn't flag
        the test fixture itself as a real AWS secret.
        """
        # snake_case
        leaky = "Got aws_secret_access_key=NotARealKeyThisIsJustATestValueXYZ123 rejected"
        out = redact(leaky)
        assert "NotARealKeyThisIsJustATestValueXYZ123" not in out
        assert "[redacted]" in out

        # camelCase
        leaky = 'config.secretAccessKey: "NotARealKeyThisIsJustATestValueXYZ123"'
        out = redact(leaky)
        assert "NotARealKeyThisIsJustATestValueXYZ123" not in out

        # SCREAMING_SNAKE_CASE
        leaky = "AWS_SECRET_ACCESS_KEY=NotARealKeyThisIsJustATestValueXYZ123"
        out = redact(leaky)
        assert "NotARealKeyThisIsJustATestValueXYZ123" not in out


class TestRedactStructured:
    def test_dict_recurses(self):
        body = {
            "detail": "Bad token: AKIAIOSFODNN7EXAMPLE",
            "code": "validation_failed",
            "field": "credentials",
        }
        out = redact(body)
        assert isinstance(out, dict)
        assert "AKIAIOSFODNN7EXAMPLE" not in out["detail"]
        # Non-secret fields untouched
        assert out["code"] == "validation_failed"
        assert out["field"] == "credentials"

    def test_list_recurses(self):
        body = [
            "fine",
            "Bearer leaky_token_value",
            {"detail": "AKIAIOSFODNN7EXAMPLE bad"},
        ]
        out = redact(body)
        assert out[0] == "fine"
        assert "leaky_token_value" not in out[1]
        assert "AKIAIOSFODNN7EXAMPLE" not in out[2]["detail"]

    def test_tuple_preserves_type(self):
        out = redact(("safe", "Bearer abc"))
        assert isinstance(out, tuple)
        assert out[0] == "safe"
        assert "abc" not in out[1]

    def test_non_string_non_container_passthrough(self):
        for value in (None, 42, 3.14, True, False):
            assert redact(value) is value

    def test_empty_string(self):
        assert redact("") == ""

    def test_empty_dict_and_list(self):
        assert redact({}) == {}
        assert redact([]) == []


class TestStructuredBodyForceRedact:
    """Reviewer P1 (round 2) on PR #11315: structured bodies like
    ``{"aws_secret_access_key": "<40-char value>"}`` lose key context
    during recursive redaction — the standalone 40-char base64-ish
    value is intentionally too generic for the regex set, so the raw
    secret stayed in ``exc.body``. Force-redact any scalar whose dict
    key matches a known sensitive name.
    """

    def test_structured_aws_secret_key_force_redacted(self):
        # The 40-char base64-ish value is intentionally too generic for
        # the regex set. Without key-aware redaction, this value would
        # pass through.
        body = {"aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"}
        out = redact(body)
        assert out["aws_secret_access_key"] == "[redacted]"

    def test_structured_camelcase_secret_key_force_redacted(self):
        body = {"secretAccessKey": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"}
        out = redact(body)
        assert out["secretAccessKey"] == "[redacted]"

    def test_structured_kebab_case_force_redacted(self):
        body = {"AWS-Secret-Access-Key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"}
        out = redact(body)
        assert out["AWS-Secret-Access-Key"] == "[redacted]"

    def test_structured_dotted_key_force_redacted(self):
        body = {"aws.secret.access.key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"}
        out = redact(body)
        assert out["aws.secret.access.key"] == "[redacted]"

    def test_other_sensitive_key_names_force_redacted(self):
        for key in [
            "apiKey",
            "api_key",
            "Authorization",
            "authorization",
            "password",
            "Password",
            "accessToken",
            "refresh_token",
            "client_secret",
            "private_key",
        ]:
            body = {key: "any-secret-value-regardless-of-shape"}
            out = redact(body)
            assert out[key] == "[redacted]", f"Key {key!r} should force-redact"

    def test_nested_structured_body_recursive_force_redact(self):
        """Nested objects under a sensitive key still have all their
        leaf strings force-redacted — defense-in-depth against odd
        shapes."""
        body = {
            "credentials": {
                "secret": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                "id": "AKIAIOSFODNN7EXAMPLE",
            },
        }
        out = redact(body)
        assert out["credentials"]["secret"] == "[redacted]"
        assert out["credentials"]["id"] == "[redacted]"

    def test_non_sensitive_key_with_safe_value_unchanged(self):
        """Non-sensitive keys preserve their values for debugging."""
        body = {"region": "us-east-1", "bucket": "customer-uploads"}
        out = redact(body)
        assert out == {"region": "us-east-1", "bucket": "customer-uploads"}

    def test_numeric_secret_under_sensitive_key_force_redacted(self):
        """Codex P2 follow-up on PR #11315: numeric OTPs
        (``{"token": 123456}``) and short-numeric passwords ARE secrets.
        The previous "non-string scalars pass through" rationale was
        wrong — JSON parsers collapse numeric-string secrets to ints,
        and OTPs are inherently numeric.

        ``None`` survives so callers can distinguish "field not set"
        from "[redacted]" in their own logs.
        """
        body = {"token": 123456, "password": 1234, "client_secret": False}
        out = redact(body)
        assert out["token"] == "[redacted]"
        assert out["password"] == "[redacted]"
        assert out["client_secret"] == "[redacted]"

    def test_none_under_sensitive_key_passes_through(self):
        """``None`` cannot encode a secret and tells the caller "not set".
        Distinguishing this from an actively-redacted value matters for
        debugging."""
        body = {"secret": None, "password": None}
        out = redact(body)
        assert out["secret"] is None
        assert out["password"] is None


class TestRaiseForStatusRedacts:
    """End-to-end: a 400 with an echoed secret in the detail produces an
    exception whose .args[0] message is redacted, not the raw value."""

    def test_validation_error_message_redacted(self, monkeypatch):
        import httpx
        from avala._http import SyncHTTPTransport
        from avala.errors import ValidationError

        # Build a minimal SyncHTTPTransport stub
        transport = SyncHTTPTransport.__new__(SyncHTTPTransport)

        # Construct an httpx.Response with a leaky detail
        leaky = "Invalid value 'AKIAIOSFODNN7EXAMPLE' for field 'aws_access_key'"
        response = httpx.Response(
            status_code=400,
            json={"detail": leaky},
        )

        try:
            transport._raise_for_status(response)
        except ValidationError as exc:
            assert "AKIAIOSFODNN7EXAMPLE" not in str(exc)
            assert "[redacted]" in str(exc)
            # body should also be redacted so callers logging exc.body don't leak
            assert exc.body is not None
            assert "AKIAIOSFODNN7EXAMPLE" not in exc.body["detail"]
        else:
            raise AssertionError("expected ValidationError to be raised")

    def test_structured_detail_does_not_leak_via_str(self):
        """Reviewer P1 (round 3) on PR #11315: when the server returns a
        structured ``detail`` (some DRF serializers do, e.g.
        ``{"detail": {"aws_secret_access_key": "<value>"}}``), the
        previous code assigned the dict to ``message`` and skipped
        redaction (because ``isinstance(message, str)`` was False). The
        dict was then passed into ``ValidationError(message, ...)``,
        and ``str(exc)`` formatted ``self.args`` — leaking the raw
        secret via the exception's string representation.

        Fix: only adopt ``body["detail"]`` as ``message`` when it's a
        string. Otherwise keep the default ``HTTP <status>`` message.
        The body itself is still surfaced (already key-aware redacted)
        so callers can introspect.
        """
        import httpx
        from avala._http import SyncHTTPTransport
        from avala.errors import ValidationError

        transport = SyncHTTPTransport.__new__(SyncHTTPTransport)
        # Synthetic AWS secret shape — generic enough to avoid push-protection
        # but distinctive enough that any leak is detectable.
        secret_value = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        response = httpx.Response(
            status_code=400,
            json={"detail": {"aws_secret_access_key": secret_value}},
        )

        try:
            transport._raise_for_status(response)
        except ValidationError as exc:
            # Core property: ``str(exc)`` must NOT contain the secret.
            assert secret_value not in str(exc), (
                "Structured detail leaked the raw secret through str(exc). "
                "The dict-typed detail must not become the exception message."
            )
            # Default message should be retained when detail isn't a string.
            assert "HTTP 400" in str(exc) or "[redacted]" not in str(exc)
            # Body is still surfaced but key-aware-redacted.
            assert exc.body is not None
            assert secret_value not in str(exc.body)
            assert exc.body["detail"]["aws_secret_access_key"] == "[redacted]"
        else:
            raise AssertionError("expected ValidationError to be raised")

    def test_async_transport_structured_detail_does_not_leak_via_str(self):
        """Mirror of the sync test on ``AsyncHTTPTransport`` —
        ``_raise_for_status`` is called synchronously inside the async
        request path, so the same defense applies."""
        import httpx
        from avala._async_http import AsyncHTTPTransport
        from avala.errors import ValidationError

        transport = AsyncHTTPTransport.__new__(AsyncHTTPTransport)
        secret_value = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        response = httpx.Response(
            status_code=400,
            json={"detail": {"aws_secret_access_key": secret_value}},
        )

        try:
            transport._raise_for_status(response)
        except ValidationError as exc:
            assert secret_value not in str(exc)
            assert exc.body is not None
            assert secret_value not in str(exc.body)
            assert exc.body["detail"]["aws_secret_access_key"] == "[redacted]"
        else:
            raise AssertionError("expected ValidationError to be raised")


class TestCodexFollowUpsPR11449:
    """Regression tests for the gaps Codex devil's-advocate review surfaced
    on PR #11449. Each test names the exact gap it closes.
    """

    def test_aws_session_token_snake_case_force_redacted(self):
        """Codex P1: ``aws_session_token`` normalises to ``awssessiontoken``,
        not ``sessiontoken``. The pre-fix sensitive-key set missed it."""
        body = {"detail": {"aws_session_token": "FwoGZXIvYXdzECoaDExlYWtTZWNyZXRWYWx1ZQ=="}}
        out = redact(body)
        assert isinstance(out, dict)
        assert out["detail"]["aws_session_token"] == "[redacted]"

    def test_aws_session_token_value_pattern_redacted_standalone(self):
        """Codex P1: even when not under the sensitive key (e.g. the value
        is echoed inside a string), the FwoG/IQoJ value-shape regex catches it.
        """
        leaky = "Invalid session token: FwoGZXIvYXdzECoaDExlYWtTZWNyZXRWYWx1ZQ=="
        out = redact(leaky)
        assert "FwoGZXIvYXdzECoaDExlYWtTZWNyZXRWYWx1ZQ" not in out
        assert "[redacted]" in out

    def test_google_oauth_refresh_token_redacted(self):
        """Codex P1: Google OAuth refresh tokens (``1//`` prefix) were not
        in the pre-fix regex set."""
        leaky = "Refresh failed: 1//04hPzLOLXIQEOCgYIARAAGAQSNwF-L9Irbadtoken_x9ZmVx"
        out = redact(leaky)
        assert "1//04hPzLOLXIQEOCgYIARAAGAQSNwF" not in out
        assert "[redacted]" in out

    def test_pem_private_key_block_redacted(self):
        """Codex P1: PEM-encoded private keys in error bodies."""
        leaky = (
            "Invalid private_key:\n"
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEowIBAAKCAQEA+SECRETKEYBYTES+\n"
            "morebase64lines==\n"
            "-----END RSA PRIVATE KEY-----"
        )
        out = redact(leaky)
        assert "SECRETKEYBYTES" not in out
        assert "morebase64lines" not in out
        assert "[redacted]" in out

    def test_private_key_id_field_value_redacted(self):
        """Codex P1: GCP service-account ``private_key_id`` in JSON shape."""
        leaky = '{"type":"service_account","private_key_id":"abc123def456abc123def456abc123def456abc1"}'
        out = redact(leaky)
        assert "abc123def456abc123def456abc123def456abc1" not in out
        assert "[redacted]" in out
        # ``type`` is not sensitive, the rest of the JSON survives.
        assert "service_account" in out

    def test_private_key_id_under_sensitive_key_force_redacted(self):
        """Pair with the regex above: when surfaced as a structured key
        rather than embedded JSON, the key-aware path force-redacts the
        value regardless of shape."""
        body = {"private_key_id": "any-shape-token-Z9"}
        out = redact(body)
        assert isinstance(out, dict)
        assert out["private_key_id"] == "[redacted]"
