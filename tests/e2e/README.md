# Importer E2E tests

These run each `avala.importers` adapter against a **real Avala server** and assert the
dataset actually materializes (status `created`, expected `item_count` / `data_type`) —
proving the importers work end to end, not just that they build correct requests (that's
what the mocked unit tests in `tests/test_*_import.py` cover).

The whole suite is **skipped** unless `AVALA_E2E_API_KEY` is set, so it never runs in the
default `pytest` / CI path.

## Run against a local server

```bash
# 1) Start the API (separate shell)
cd server && python manage.py runserver 0.0.0.0:8000

# 2) Install the SDK with all importer extras
cd sdks/python && pip install -e ".[dev,cli,lerobot,rosbag]"

# 3) Point the suite at the local server and run
export AVALA_E2E_API_KEY="avk_..."                       # a key on the local server
export AVALA_E2E_BASE_URL="http://localhost:8000/api/v1" # default
export AVALA_ALLOW_INSECURE_BASE_URL=true                # required for http://localhost
pytest tests/e2e -v
```

`folder` and `rosbag` run with generated local fixtures (no extra setup). `cloud` and
`lerobot` are opt-in:

```bash
# cloud (S3) — a small bucket of images you control
export AVALA_E2E_S3_URI="s3://my-test-bucket/sample-images/"
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_REGION=us-west-2
export AVALA_E2E_S3_DATA_TYPE=image           # optional, default image
export AVALA_E2E_ORG_UID=org_...              # required only for keyless --role-arn

# lerobot — a SMALL public Hugging Face dataset (download can be large/slow)
export AVALA_E2E_LEROBOT_REPO="lerobot/<small_dataset>"
export AVALA_E2E_LEROBOT_EPISODES=0           # optional, default episode 0
```

`test_resources_e2e.py` adds a read-only **list sweep** over every top-level resource
(datasets, projects, tasks, exports, agents, webhooks, storage_configs,
inference_providers, organizations, auto_label_jobs) plus create→get→delete round-trips
for webhooks and agents — no extra setup, runs with just `AVALA_E2E_API_KEY`.

## What each test asserts

| Test | Fixture | Asserts |
|---|---|---|
| `test_e2e_folder_images` | 3 generated JPEGs | dataset `data_type=image`, `item_count >= 3` |
| `test_e2e_rosbag` | generated ROS2 bag, 3 camera frames | `data_type=mcap`, `item_count >= 1` |
| `test_e2e_cloud_s3` | your S3 bucket | created, `item_count >= 1` |
| `test_e2e_lerobot` | small HF dataset | `data_type=mcap`, `item_count >= 1` |

Each uses `import_*(wait=True)`, which blocks until server-side indexing finishes and
returns the refreshed dataset. Tests use unique slugs (`<kind>-e2e-<uuid>`) so repeated
runs don't collide; created datasets are left in place (delete them from the dev server
if you want to tidy up).
