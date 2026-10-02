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

## Public dataset resolver

- Resolver suites load `tests/fixtures/dataset_resolver_v1_fixtures.json` so they also run in
  the standalone public mirror. Keep this copy byte-identical to the monorepo's canonical
  `contracts/dataset_resolver_v1_fixtures.json`; the parity test checks it when present.
  Only that parity check may skip outside the monorepo, never the substantive resolver suites.
- Allocate the verified download spool before requesting an access grant. Signed URLs must not
  survive in exception contexts, traceback locals, response bodies, or parser frames.
- Resolver requests are anonymously throttled. Sync and async transports must honor bounded
  `Retry-After` delays and retry 429 responses; collection algorithms must also avoid request-per-object scans.
- One resolved dataset owns one manifest-evidence tracker and episode-reference cache shared by
  `objects`, `episodes`, and every `for_role()` view. Separate views must not accept conflicting
  identities or exceed the manifest's count/size bounds in aggregate.
- Provider URL validation mirrors the server's grant grammar, including regional path-style S3
  dualstack hosts. Rights metadata has an exact field set; never expose undeclared response extras.

## Inspect Robots importer (`avala/importers/inspect_robots.py`)

Verified against `inspect-robots` 0.60.0; the module docstring has the full mapping table.

- **It is not an Inspect AI log.** `inspect-robots` does not depend on `inspect-ai` and does not
  write `.eval` archives. It writes its own JSON `EvalLog` (schema version 1) whose shape mirrors
  Inspect AI's. Read it with `inspect_robots.read_eval_log`, never `inspect_ai.log`.
- **One `samples[]` entry is a scene, not a trial.** Trial *i* of a scene is index *i* across the
  parallel lists `epochs`, `termination_reasons`, `operator_judgements`, `trial_metadata`. An
  errored or cancelled trial is recorded but never scored, so its `epochs[i]` is `{}`.
- **Trajectories are side-cars, and the run id exists only in their paths.**
  `trial_metadata[i]["actions"]` is `actions/<run_id>/<scene>-e<epoch>.jsonl`, relative to the
  log's directory; frames (only with `--store-frames`) are `frames/<run_id>/*.npy`.
- **Run provenance rides on the label's `source_metadata`** (`TrialMapping.source_metadata()`):
  `importer`, `importer_version`, `run_id`, `task`, `trial_id`, `log_file` (basename only, never a
  local path), `epoch`, `scene`. The server caps it at 20 flat keys (<= 64 chars) to strings
  (<= 512 chars), finite numbers or bools, no null; `_clean_source_metadata` fits values to that
  rather than failing the write. Scores, termination reason etc. stay in the receipt, `-o json`
  and the `/inspect_robots/trial` MCAP topic, because they are nested.
- **Re-runs must not stack label versions.** `_same_label` compares every field the importer
  sends, `source_metadata` included, except `importer_version` (it names the writer, not the
  label; comparing it would add a version to every trial on each SDK upgrade). Any new field
  added to `outcome_kwargs()` must be added to `_same_label` too, or every re-run writes a new
  version; `test_every_sent_field_takes_part_in_the_rerun_comparison` fails when one is missed.
- **Regenerate the fixture with Inspect Robots, never by hand:**
  `tests/fixtures/inspect_robots/generate_fixture.py` (needs Python 3.10+).
