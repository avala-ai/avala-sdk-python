# Avala Python SDK

[![PyPI version](https://img.shields.io/pypi/v/avala)](https://pypi.org/project/avala/)
[![Python](https://img.shields.io/pypi/pyversions/avala)](https://pypi.org/project/avala/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official Python SDK for the [Avala API](https://docs.avala.ai). Build and manage ML annotation datasets, projects, exports, and tasks programmatically.

## Installation

```bash
pip install avala
```

Requires Python 3.9+.

## Quick Start

```python
from avala import Client

client = Client()  # reads AVALA_API_KEY env var

# List datasets
page = client.datasets.list(limit=10)
for dataset in page:
    print(dataset.uid, dataset.name)

# Get a specific dataset
dataset = client.datasets.get("dataset-uid")

# Create an export
export = client.exports.create(project="project-uid")
print(export.uid, export.status)

# List tasks with filters
tasks = client.tasks.list(project="project-uid", status="completed")
```

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

Install with CLI extras for a command-line interface:

```bash
pip install avala[cli]
```

```bash
avala configure                    # Interactive API key setup
avala datasets list                # List datasets
avala projects list                # List projects
avala exports create <project-uid> # Create an export
avala storage-configs list         # List storage configs
avala agents list                  # List automation agents
avala inference-providers list     # List inference providers
avala auto-label list              # List auto-label jobs
avala quality-targets list -p <uid> # List quality targets
avala consensus summary -p <uid>   # Get consensus summary
avala webhooks list                # List webhook subscriptions
```

## Available Resources

| Resource | Methods | Description |
|----------|---------|-------------|
| `client.datasets` | `list()`, `get(uid)` | Browse and inspect datasets |
| `client.projects` | `list()`, `get(uid)` | Browse and inspect projects |
| `client.exports` | `list()`, `get(uid)`, `create()` | Create and manage annotation exports |
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

- [Python SDK Guide](https://docs.avala.ai/sdks/python)
- [API Reference](https://docs.avala.ai/api-reference/overview)
- [Quickstart](https://docs.avala.ai/getting-started/quickstart)
- [Examples](https://docs.avala.ai/resources/examples)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT - see [LICENSE](LICENSE) for details.
