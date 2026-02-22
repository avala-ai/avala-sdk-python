# Avala Python SDK

Official Python SDK for the [Avala API](https://docs.avala.ai).

## Installation

```bash
pip install avala
```

## Quick Start

```python
from avala import Client

client = Client(api_key="your-api-key")

# List datasets
for dataset in client.datasets.list():
    print(dataset.name)

# Get a specific dataset
dataset = client.datasets.get("dataset-uid")
```

## Async Support

```python
from avala import AsyncClient

async with AsyncClient() as client:
    datasets = await client.datasets.list()
```

## Documentation

Full documentation is available at [docs.avala.ai/sdks/python](https://docs.avala.ai/sdks/python).
