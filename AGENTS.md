# Avala Python SDK

## Commands
- Install: `pip install -e ".[dev,cli]"` — the `cli` extra is required; CI installs it and
  `tests/test_cli.py` skips without it, so a `[dev]`-only env silently runs fewer tests.
- Test: `pytest`
- Lint: `ruff check .`
- Type check: `mypy avala/`
- Format: `ruff format .`

CI (`.github/workflows/sdk-python-ci.yml`) runs all four on Python 3.9–3.12. **3.9 is the
floor** (`requires-python = ">=3.9"`), so anything that only parses on a later version
breaks the build even when it passes locally.

### End-to-end tests
`tests/e2e/` runs against a real server and is skipped unless `AVALA_E2E_API_KEY` is set
(`AVALA_E2E_BASE_URL` defaults to `http://localhost:8000/api/v1`). The manual-upload tests
need a second opt-in, `AVALA_E2E_UPLOADS=1`, because the presign endpoint 500s unless the
server has `MANUAL_DATASET_UPLOADS_BUCKET_NAME` configured.

## Architecture
- `avala/_client.py` / `avala/_async_client.py` — Sync/async client classes
- `avala/_http.py` / `avala/_async_http.py` — HTTP transports (httpx)
- `avala/resources/` — API resource classes (datasets, projects, exports, tasks)
- `avala/types/` — Pydantic response models
- `avala/errors.py` — Error hierarchy
- `avala/_pagination.py` — Cursor-based pagination
- `avala/_uploads.py` — shared upload primitives: presigned-host allow-list, retry
  classification/backoff, and resume checkpoints under `~/.avala/uploads/`

## Conventions
- Python 3.9+, use `from __future__ import annotations`
- Type hints on all public functions
- Tests use `respx` for httpx mocking

## Uploading files

Two rules, both learned the hard way (see
[`reports/plans/dataset-upload-path-2026-08.md`](../../reports/plans/dataset-upload-path-2026-08.md)):

1. **`organization_uid` must be identical on the presign and the create call.** The server
   derives the S3 key prefix from org context on both. Mismatch them and the upload
   "succeeds" against `<username>/…` while the dataset is registered over `orgs/<slug>/…`,
   so it lists zero items with no error anywhere.
2. **Don't write a bespoke upload loop.** `client.datasets.upload_files` already retries
   transient failures, re-presigns inside the retry (a presigned POST expires), enforces
   the upload-host allow-list, and checkpoints for resume. The CLI calls it too — keep it
   that way rather than reimplementing it per command.

`tests/conftest.py` redirects the resume checkpoint dir per-test. Any new test that
uploads must keep that isolation, or leftover state makes later runs skip files.
