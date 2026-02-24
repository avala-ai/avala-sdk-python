# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in the Avala Python SDK, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, email **security@avala.ai** with:

- A description of the vulnerability
- Steps to reproduce
- Affected versions
- Any potential impact

We will acknowledge your report within 2 business days and provide a timeline for a fix.

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest  | Yes       |
| < Latest | No       |

We recommend always using the latest version of the SDK. Install or upgrade with:

```bash
pip install --upgrade avala
```

## Security Best Practices

- Never hardcode API keys in source code. Use environment variables: `AVALA_API_KEY`
- Rotate API keys periodically via the [Avala dashboard](https://control.avala.ai)
- Use scoped API keys with minimal required permissions
- Keep the SDK updated to receive security patches
