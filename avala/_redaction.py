"""Error-detail redaction for HTTP transports (pentest finding ``s3-4``).

The server commonly echoes parts of the request payload in 4xx/5xx
``detail`` strings (validation errors are the canonical case:
``"value 'sk_live_abc123' is not a valid <foo>"``). When the SDK
re-raises that string into an exception, a secret in the request
flows directly into caller logs / Sentry / stdout.

This helper applies a small, conservative set of regex redactions to
any string we surface to the caller. Pattern set is intentionally
narrow — false positives are worse than misses for caller debugging.
"""

from __future__ import annotations

import re
from typing import Any

# Redaction patterns. Each substitutes the matched fragment with
# ``[redacted]``. Order matters only insofar as more specific patterns
# come before more general ones.
_REDACTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Bearer / JWT-style tokens (eyJ...) and dot-separated JWT triples.
    re.compile(r"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"),
    # AWS access key IDs.
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ASIA[0-9A-Z]{16}"),
    # Codex devil's-advocate finding (PR #11449): AWS STS session tokens
    # have a fixed magic prefix (``FwoG`` for AWS, ``IQoJ`` for some
    # GovCloud regions) followed by a long base64 blob. Redact the prefix
    # plus a generous tail. This catches structured-body leaks like
    # ``{"detail": "Invalid session token: FwoGZXIvYXdz..."}`` that the
    # generic 40-char-hex / sk_ patterns miss.
    re.compile(r"\b(?:FwoG|IQoJ)[A-Za-z0-9+/=]{16,}"),
    # Google OAuth refresh tokens (``1//`` prefix). Public format
    # documented in Google Identity docs; not a real prefix-keyed shape
    # the generic ``sk_`` regex catches.
    re.compile(r"\b1//[A-Za-z0-9_\-]{20,}"),
    # PEM-encoded private key blocks (RSA, EC, generic). The body contains
    # base64 plus newlines; the trailing END line ends the block. Redact
    # entire BEGIN…END span so neither the key bytes nor the metadata
    # leaks. Note: response bodies sometimes have ``\\n`` (escaped) instead
    # of literal newlines when JSON-encoded, so allow both.
    re.compile(
        r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED |)PRIVATE KEY-----"
        r"[\s\S]*?-----END (?:RSA |EC |DSA |OPENSSH |ENCRYPTED |)PRIVATE KEY-----"
    ),
    # GCP service-account JSON: presence of ``"type": "service_account"``
    # in a JSON blob is a strong tell. We can't redact the whole JSON safely
    # without parser support, but the ``private_key`` and ``private_key_id``
    # fields (the parts that actually authenticate) are caught by the PEM
    # pattern above and the ``private_key_id`` 40-hex pattern. Add a
    # context-aware rule for ``private_key_id`` so non-hex shapes still get
    # caught when echoed alongside the field name.
    re.compile(r'(?i)"private_key_id"\s*:\s*"[^"]+"'),
    # Reviewer P1 follow-up on PR #11315: AWS *secret* access keys are
    # 40 chars of base64-ish ``[A-Za-z0-9/+=]`` — too generic to match
    # standalone without false-firing on every base64 blob in an error
    # body. Redact context-aware: when the secret-key value is echoed
    # next to a known field name (``aws_secret_access_key``,
    # ``secretAccessKey``, ``AWS_SECRET_ACCESS_KEY``), strip the value.
    re.compile(
        r"(?i)\b(?:aws[_-]?secret[_-]?access[_-]?key|secretaccesskey)"
        r"[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9/+=]{20,}"
    ),
    # Generic 32+ char base64-ish secret tokens (``sk_live_``, ``pk_test_``,
    # ``secret_``, ``key_``, etc.) — the prefix-keyed variant catches
    # Stripe / Resend / Supabase shapes without false-firing on UUIDs.
    re.compile(r"\b(?:sk|pk|secret|key|token)_(?:live|test|prod)?_?[A-Za-z0-9]{16,}"),
    # Modern hyphenated provider keys: OpenAI ``sk-proj-...`` / legacy
    # ``sk-...`` and Anthropic ``sk-ant-api03-...``. The underscore-keyed
    # pattern above misses these because the segments are ``-``-separated.
    # Reported via the bug-bounty program: inference-provider ``config.api_key``
    # secrets echoed in a ``detail`` string leaked through unredacted.
    re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_\-]{20,}"),
    # Avala API key shape: 40 hex chars (per ``apikey._generate_api_key``).
    re.compile(r"\b[a-f0-9]{40}\b"),
    # Authorization: Bearer <token> headers leaked into response bodies.
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9_.\-+/=]+"),
    # ``X-Avala-Api-Key: <value>`` echoed in error payloads.
    re.compile(r"(?i)X-Avala-Api-Key[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9_\-]+"),
    # AWS ARNs (account IDs, resource paths).
    re.compile(r"arn:aws[a-z0-9\-]*:[a-z0-9\-]*:[a-z0-9\-]*:[0-9]*:[A-Za-z0-9_\-./:*]+"),
)
_REDACTED_TOKEN = "[redacted]"


# Reviewer P1 (round 2) follow-up on PR #11315: structured bodies like
# ``{"aws_secret_access_key": "<40-char value>"}`` lose key context during
# recursive redaction — the standalone 40-char base64-ish value is
# intentionally too generic for the regex set, so the raw secret stays
# in ``exc.body``. Force-redact any scalar whose dict key matches a known
# sensitive name. Keys are normalised to lowercase + non-alphanumerics
# stripped so ``aws_secret_access_key``, ``AWS-Secret-Access-Key``,
# ``awsSecretAccessKey``, and ``aws.secret.access.key`` all match the
# same canonical name.
_SENSITIVE_KEY_NAMES: frozenset[str] = frozenset(
    {
        "awssecretaccesskey",
        "secretaccesskey",
        "awsaccesskeyid",
        "accesskeyid",
        # Codex finding (PR #11449): ``aws_session_token`` normalises to
        # ``awssessiontoken``, NOT ``sessiontoken``. The exact-match key
        # set must include the AWS-prefixed variant or
        # ``{"detail": {"aws_session_token": "FwoG..."}}`` slips through.
        "awssessiontoken",
        "apikey",
        "xavalaapikey",
        "authorization",
        "auth",
        "token",
        "accesstoken",
        "refreshtoken",
        "sessiontoken",
        "idtoken",
        "secret",
        "password",
        "passwd",
        "pwd",
        "credentials",
        "privatekey",
        # Google service-account JSON field; sibling to ``privatekey`` and
        # treated the same way (force-redact the whole value when the key
        # name is reached).
        "privatekeyid",
        "clientsecret",
        # OAuth flows: refresh-token field appears as snake/camelCase on
        # different SDKs; ``refreshtoken`` already covers the normalised
        # form, but be explicit about ``refresh_token`` -> ``refreshtoken``
        # round-trip stability.
    }
)


def _normalise_key(key: Any) -> str:
    """Lowercase + alphanumeric-only key name (matches snake_case,
    camelCase, kebab-case, dotted, SCREAMING_SNAKE_CASE)."""
    if not isinstance(key, str):
        return ""
    return "".join(ch for ch in key.lower() if ch.isalnum())


def _is_sensitive_key(key: Any) -> bool:
    return _normalise_key(key) in _SENSITIVE_KEY_NAMES


def redact(value: Any) -> Any:
    """Return ``value`` with secret-shaped fragments redacted.

    - ``str``: each pattern's matches replaced with ``[redacted]``.
    - ``dict``: recurse into values. If the key name matches a known
      sensitive identifier (``aws_secret_access_key``,
      ``apiKey``, ``Authorization``, ``password``, ...), the entire
      scalar value is force-redacted regardless of content. Containers
      under sensitive keys still recurse so nested structure isn't
      flattened, but their leaf strings get force-redacted too.
    - ``list``/``tuple``: recurse into elements.
    - everything else: returned unchanged.

    The function is intentionally non-throwing — redaction is best-effort
    defense-in-depth, not a security control. Falling back to the
    original value on any unexpected type is preferable to raising
    inside an error path.
    """
    if isinstance(value, str):
        out = value
        for pattern in _REDACTION_PATTERNS:
            out = pattern.sub(_REDACTED_TOKEN, out)
        return out
    if isinstance(value, dict):
        return {k: (_force_redact(v) if _is_sensitive_key(k) else redact(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        cleaned = [redact(v) for v in value]
        return type(value)(cleaned) if isinstance(value, tuple) else cleaned
    return value


def _force_redact(value: Any) -> Any:
    """Replace every scalar in ``value`` with ``[redacted]``.

    Used for values reached via a sensitive key — the value is by
    definition the secret, regardless of shape. Recurses into containers
    so nested structure (an unusual but legal shape) doesn't slip a
    secret string through.

    Codex P2 follow-up on PR #11315: previously this passed numeric and
    boolean scalars through on the rationale that they "couldn't encode
    a secret." That misses numeric OTPs (``{"token": 123456}``),
    short-numeric passwords, and any string-typed secret that the JSON
    parser collapsed to ``int``. Treat any non-``None`` scalar reached
    via a sensitive key as the secret. ``None`` survives so callers can
    distinguish "not set" from "[redacted]" in their own logs.
    """
    if isinstance(value, dict):
        return {k: _force_redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        cleaned = [_force_redact(v) for v in value]
        return type(value)(cleaned) if isinstance(value, tuple) else cleaned
    if value is None:
        return None
    return _REDACTED_TOKEN
