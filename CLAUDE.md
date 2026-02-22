# Avala Python SDK

## Commands
- Install: `pip install -e ".[dev]"`
- Test: `pytest`
- Lint: `ruff check .`
- Type check: `mypy avala/`
- Format: `ruff format .`

## Architecture
- `avala/_client.py` / `avala/_async_client.py` — Sync/async client classes
- `avala/_http.py` / `avala/_async_http.py` — HTTP transports (httpx)
- `avala/resources/` — API resource classes (datasets, projects, exports, tasks)
- `avala/types/` — Pydantic response models
- `avala/errors.py` — Error hierarchy
- `avala/_pagination.py` — Cursor-based pagination

## Conventions
- Python 3.9+, use `from __future__ import annotations`
- Type hints on all public functions
- Tests use `respx` for httpx mocking
