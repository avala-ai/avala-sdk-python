# Contributing

Thank you for your interest in the Avala Python SDK!

## Reporting Bugs & Requesting Features

Please [open an issue](https://github.com/avala-ai/avala-sdk-python/issues) with:

- **Bugs**: Python version, SDK version (`pip show avala`), minimal reproduction steps, and the full error traceback.
- **Feature requests**: A description of the use case and the API you'd like to see.

## About This Repository

This repo is a **read-only mirror** of the SDK source code. The Avala team maintains the code internally and publishes updates here. Pull requests cannot be merged — please use issues for all feedback.

## Running Tests Locally

```bash
pip install -e ".[dev]"
pytest -v
ruff check .
mypy avala/
```
