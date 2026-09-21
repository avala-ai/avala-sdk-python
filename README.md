# Avala Python SDK

[![PyPI version](https://img.shields.io/pypi/v/avala)](https://pypi.org/project/avala/)
[![Python](https://img.shields.io/pypi/pyversions/avala)](https://pypi.org/project/avala/)
[![CI](https://github.com/avala-ai/avala-sdk-python/actions/workflows/ci.yml/badge.svg)](https://github.com/avala-ai/avala-sdk-python/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official Python SDK for the [Avala API](https://avala.ai/docs) — the open Physical AI Data Platform. Programmatically manage sensor datasets (MCAP, LiDAR, video), annotation projects, exports, tasks, and fleet devices.

> **Note:** This repository is a read-only mirror. To report bugs or request features, please [open an issue](https://github.com/avala-ai/avala-sdk-python/issues). See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Installation

```bash
pip install avala
```

Requires Python 3.9+.

## Quick Start

```python
from avala import Client  # or: from avala import Avala

client = Client()  # reads AVALA_API_KEY env var

# List datasets
page = client.datasets.list(limit=10)
for dataset in page:
    print(dataset.uid, dataset.name)

# Get a specific dataset
dataset = client.datasets.get("dataset-uid")

# Create an export and wait for completion
export = client.exports.create(project="project-uid")
finished = client.exports.wait(export.uid)  # polls until done
print(finished.download_url)

# List tasks with filters
tasks = client.tasks.list(project="project-uid", status="completed")
```

## Load Public Dataset Revisions

Published Physical AI datasets with open-download rights can be loaded without an Avala account
or API key. References resolve to immutable dataset and manifest digests; object bytes are fetched
lazily and verified against their declared size and SHA-256.

```python
import avala

with avala.load("owner/dataset") as dataset:  # defaults to the "main" revision alias
    episode = dataset.episodes[0]
    with episode.open() as raw_mcap:  # verified seekable spool; large objects stay off-heap
        magic = raw_mcap.read(8)
    print(dataset.canonical_reference, episode.sha256, episode.size_bytes, magic)
```

Pin a reproducible revision with `owner/dataset@<revision-digest>` or its returned
`avala://datasets/<uid>@<revision-digest>` canonical reference. Async code uses
`async with await avala.async_load(...) as dataset:`,
`episode = await dataset.episodes[0]`, and `content = await episode.open()` (close the returned
file after consuming it). Use `read()` only for deliberately bounded, small objects.

### Resolve a revision from the CLI

With the `cli` extra installed, resolve a public, published dataset with open-download rights
without an API key:

```bash
avala datasets resolve owner/dataset
avala --output json datasets resolve owner/dataset > dataset-reference.json
avala datasets resolve owner/dataset --revision release-1
```

The command fetches revision metadata only. It does not list objects, request download grants,
or download dataset contents. JSON includes the canonical and requested references, dataset and
revision UIDs, revision and manifest SHA-256 digests, integer `object_count` and `total_size_bytes`,
and the declared `rights` object. Keep the returned canonical reference for reproducible inputs:

```python
import json
from pathlib import Path

import avala

handoff = json.loads(Path("dataset-reference.json").read_text())
with avala.load(handoff["canonical_reference"]) as dataset:
    assert dataset.manifest_sha256 == handoff["manifest_sha256"]
    print(dataset.canonical_reference, dataset.object_count)
```

An embedded `@alias` or `@digest` also works; do not combine it with `--revision`.
Global `--base-url` or `AVALA_BASE_URL` selects another API root. Pass that same `base_url`
to `avala.load` when consuming a handoff from another environment. Metadata resolution does not
verify object contents or guarantee continued availability of a withdrawn revision.

## Authentication

The client reads your API key from the `AVALA_API_KEY` environment variable by default:

```bash
export AVALA_API_KEY="avk_your_api_key"
```

Or pass it explicitly:

```python
client = Client(api_key="avk_your_api_key")
```

## Async Support

```python
from avala import AsyncClient

async with AsyncClient() as client:
    page = await client.datasets.list()
    for dataset in page:
        print(dataset.name)
```

## Pagination

All `.list()` methods return a `CursorPage` that supports iteration:

```python
page = client.datasets.list(limit=20)

for dataset in page:
    print(dataset.name)

# Manual pagination
if page.has_more:
    next_page = client.datasets.list(cursor=page.next_cursor)
```

## Error Handling

```python
from avala.errors import AvalaError, NotFoundError, RateLimitError, AuthenticationError

try:
    dataset = client.datasets.get("nonexistent")
except NotFoundError:
    print("Dataset not found")
except RateLimitError:
    print("Rate limited")
except AuthenticationError:
    print("Invalid API key")
except AvalaError as e:
    print(f"API error: {e}")
```

## CLI Tool

Install the CLI with one command:

```bash
curl -fsSL https://avala.ai/install.sh | bash
```

Or install directly with pip:

```bash
pip install avala[cli]
```

```bash
avala configure                       # Interactive API key setup + validation
avala status                          # Organization dashboard overview
avala datasets list                   # List datasets
avala projects list                   # List projects
avala exports create --project <uid>  # Create an export
avala exports wait <uid>              # Poll until export completes
avala fleet devices list              # List fleet devices
avala -o json datasets list | jq .    # JSON output for scripting
avala shell-completion zsh >> ~/.zshrc # Enable tab completion
avala --version                       # Show CLI version
```

## Fleet recording uploads

Upload files to an existing recording with the CLI or SDK. Configure your API key
with `avala configure` for the CLI or `AVALA_API_KEY` for the SDK.

```bash
avala fleet recordings upload --recording RECORDING_UID --source ./recording --dry-run
avala fleet recordings upload --recording RECORDING_UID --source ./recording --wait
```

```python
from avala import Client

with Client() as client:
    session = client.fleet.uploads.upload_recording("recording-uid", "./recording")
    print(session.status)
```

### Explicit managed MCAP uploads

For an enrolled organization, add `--managed` to upload MCAP originals to Avala
Cloud. Supply the UUID of an existing recording. This option cannot be combined
with `--storage-config`; the default upload command and customer bucket selection
keep their existing behavior.

```bash
avala fleet recordings upload --managed --recording RECORDING_UID --source ./recording --dry-run
avala fleet recordings upload --managed --recording RECORDING_UID --source ./recording --wait
```

The managed profile accepts 1 to 64 uncompressed `.mcap` files, each larger than
zero and at most 8 GiB, with at most 64 GiB in one manifest. Parts use a fixed
64 MiB size, except the final part. `--workers` accepts 1 to 4 parallel transfers.
The dry-run hashes and validates local files without credentials, API requests or
receipt writes. It does not check server enrollment or remaining storage quota.

Without `--wait`, the command can return successfully with publication submitted
and still pending. With `--wait`, it waits for the exact dataset publication,
up to `--wait-timeout` seconds (0 to 3600, default 3600). A timeout or failed
publication exits nonzero and retains the upload. Published output includes the
dataset and publication UUIDs; it does not claim the Fleet recording is ready.
`--output json` exposes a bounded status summary for scripts.

Keep the original files and receipt under `~/.avala/uploads/`. Repeat the same
command with `--managed`, recording, source, API and credentials to resume.
Do not delete the receipt or switch an existing managed recording to the default
upload command. An unsupported server or unavailable enrollment fails without
falling back to a different protocol.

### Upgrading to 0.10.0

This release adds public dataset revision loading through `avala.load` and
`avala.async_load`, plus the anonymous `avala datasets resolve` metadata command.
Fleet uploads now require retained evidence instead of automatically replacing
uncertain sessions. This changes recovery behavior for existing uploads.

Keep the original files at the same source path and preserve the local upload
receipt under `~/.avala/uploads/`. Resume requires matching source contents, API,
credentials, storage selection, and recording identity. Old, corrupt, or ambiguous
receipts refuse automatic replacement. Finish existing work with its original
inputs when possible; otherwise retain its files and receipt and use a new
recording UID. Do not delete checkpoints to force a restart over the same recording.
The source collector rejects symlinks and special files and excludes SDK checkpoint
files, so a source tree accepted by an older version may now need correction.
Resumes with 1,000 or more pending files are refused because the current server
response cannot prove the complete pending inventory; fresh uploads can exceed
1,000 files.

The default SDK can return `completing` while server processing continues. Its CLI `--wait`
requires both that same upload session to complete and the recording to become
ready, within the wait timeout. Completed receipts are retained. Local SHA-256
checks detect observed source changes; they do not independently verify current
remote contents.

## Available Resources

| Resource | Methods | Description |
|----------|---------|-------------|
| `client.datasets` | `list()`, `get(uid)` | Browse and inspect datasets |
| `client.projects` | `list()`, `get(uid)` | Browse and inspect projects |
| `client.exports` | `list()`, `get(uid)`, `create()`, `wait(uid)` | Create, poll, and manage annotation exports |
| `client.tasks` | `list()`, `get(uid)` | Browse tasks with project/status filters |
| `client.storage_configs` | `list()`, `create()`, `test()`, `delete()` | Manage cloud storage connections |
| `client.agents` | `list()`, `get()`, `create()`, `update()`, `delete()`, `list_executions()`, `test()` | Manage automation agents |
| `client.inference_providers` | `list()`, `get()`, `create()`, `update()`, `delete()`, `test()` | Manage inference providers |
| `client.auto_label_jobs` | `list()`, `get()`, `create()` | Batch auto-labeling jobs |
| `client.quality_targets` | `list()`, `get()`, `create()`, `update()`, `delete()`, `evaluate()` | Project quality targets |
| `client.consensus` | `get_summary()`, `list_scores()`, `compute()`, `get_config()`, `update_config()` | Consensus scoring |
| `client.webhooks` | `list()`, `get()`, `create()`, `update()`, `delete()`, `test()` | Manage webhook subscriptions |
| `client.webhook_deliveries` | `list()`, `get()` | Inspect webhook delivery logs |

## Documentation

- [Python SDK Guide](https://avala.ai/docs/sdks/python)
- [API Reference](https://avala.ai/docs/api-reference/overview)
- [Quickstart](https://avala.ai/docs/getting-started/quickstart)
- [Examples](https://avala.ai/docs/resources/examples)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT - see [LICENSE](LICENSE) for details.

### Managed local upload sessions

`client.datasets.create_from_local(...)` and its CLI path negotiate managed
upload protocol v2. The API selects storage for the organization and pins it to
one dataset upload batch. Existing S3 form POST responses remain supported;
opted-in organizations can receive PUT or fixed-size multipart sessions.

The SDK keeps the batch UUID in its local upload checkpoint, uploads at most two
parts concurrently per file, and asks the API to verify completion before dataset
creation. A retry resumes server-confirmed parts only when the local source stamp
is unchanged. Changed files require a fresh batch. Signed URLs and credentials
are never saved in checkpoints. This is local resume, not verified cross-machine
resume. Existing checkpoints created before v2 finish using the legacy protocol.
`resume=False` (CLI `--no-resume`) starts a fresh batch and transfers every file
again. Previous remote objects are retained, and retired batch IDs stay in the
local recovery state. Successful finalization removes that batch's file stamps;
one stable lock per batch remains for safe concurrent checkpoint access.

Lower-level callers can pass the same `dataset_upload_uid` UUID to
`create_manual_upload_url`, `upload_files`, and `create_from_manual_upload`.
Keep `organization_uid` and the dataset name identical across those calls.
Provider rollout and rollback remain server configuration changes; existing
batch bindings remain valid when the selection flag changes. Bring-your-own
bucket imports are unchanged.
