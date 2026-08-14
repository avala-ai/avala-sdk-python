from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_upload_state(tmp_path_factory, monkeypatch):
    """Keep upload resume state out of the developer's real home directory.

    ``upload_files`` checkpoints confirmed files under ``~/.avala/uploads`` so an
    interrupted transfer can resume. In tests that is actively dangerous: state
    left by one run makes the next run *skip* files, so a passing suite can turn
    red (or worse, green for the wrong reason) depending on what ran before it.
    Point every test at a fresh per-test directory instead.

    Deliberately **not** under the test's own ``tmp_path``: tests upload
    ``tmp_path`` as the source directory, so a state dir nested inside it gets
    walked by ``gather_local_files`` and the checkpoint file is uploaded as if it
    were data — which silently inflates byte counts and breaks quota assertions.
    """
    from avala.resources import datasets as datasets_mod
    from avala.resources.fleet import uploads as fleet_uploads_mod

    state_root = tmp_path_factory.mktemp("upload-state")
    monkeypatch.setattr(datasets_mod, "_STATE_DIR", state_root / "datasets")
    monkeypatch.setattr(fleet_uploads_mod, "_STATE_DIR", state_root / "fleet")


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch):
    """Make retry backoff instant.

    The uploader retries transient failures with exponential backoff capped at
    60s. A test that exercises the failure path would otherwise sit through
    minutes of real sleeping for no added coverage.
    """
    from avala.resources import datasets as datasets_mod

    monkeypatch.setattr(datasets_mod, "sleep_backoff", lambda *args, **kwargs: None)
