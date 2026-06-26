"""End-to-end resource tests against a REAL Avala server.

Complements ``test_importers_e2e.py``: a read-only list sweep over every top-level
resource (proves the SDK ↔ server contract for reads end to end), plus create→get→
delete round-trips for the self-contained resources (proves the write path).

Skipped entirely unless ``AVALA_E2E_API_KEY`` is set. Defaults to a local server
(``http://localhost:8000/api/v1``). See ``tests/e2e/README.md``.
"""

from __future__ import annotations

import os
import uuid

import pytest

E2E_API_KEY = os.environ.get("AVALA_E2E_API_KEY")
E2E_BASE_URL = os.environ.get("AVALA_E2E_BASE_URL", "http://localhost:8000/api/v1")

pytestmark = pytest.mark.skipif(
    not E2E_API_KEY,
    reason="set AVALA_E2E_API_KEY (and AVALA_E2E_BASE_URL) to run resource E2E against a server",
)


@pytest.fixture(scope="module")
def client():
    from avala import Client

    if E2E_BASE_URL.startswith("http://"):
        os.environ.setdefault("AVALA_ALLOW_INSECURE_BASE_URL", "true")
    c = Client(api_key=E2E_API_KEY, base_url=E2E_BASE_URL)
    yield c
    c.close()


def _unique(prefix: str) -> str:
    return f"{prefix}-e2e-{uuid.uuid4().hex[:8]}"


# ── read-only list sweep: every customer-accessible top-level resource ──────────
# (name, callable taking the client) — resources whose list() needs no args and is
# reachable with a customer API key.
#
# NOTE: ``projects.list`` now targets the user-scoped route (GET /users/me/projects/),
# which is filtered to the key owner's accessible projects and permits customers — the
# top-level GET /projects/ is a staff-only admin endpoint (lists every project) and a
# customer key would get 403 there. See avala/resources/projects.py.
_LIST_RESOURCES = [
    ("datasets", lambda c: c.datasets.list(limit=1)),
    ("projects", lambda c: c.projects.list(limit=1)),
    ("tasks", lambda c: c.tasks.list(limit=1)),
    ("exports", lambda c: c.exports.list(limit=1)),
    ("agents", lambda c: c.agents.list(limit=1)),
    ("webhooks", lambda c: c.webhooks.list(limit=1)),
    ("storage_configs", lambda c: c.storage_configs.list()),
    ("inference_providers", lambda c: c.inference_providers.list(limit=1)),
    ("organizations", lambda c: c.organizations.list(limit=1)),
    ("auto_label_jobs", lambda c: c.auto_label_jobs.list(limit=1)),
]


@pytest.mark.parametrize("name,call", _LIST_RESOURCES, ids=[n for n, _ in _LIST_RESOURCES])
def test_e2e_list_sweep(client, name, call):
    page = call(client)
    # CursorPage (or list for storage_configs) — just confirm it returned iterable items.
    items = getattr(page, "items", page)
    assert items is not None
    assert isinstance(list(items), list)


# ── write round-trips: create → get → delete on self-contained resources ────────
def test_e2e_webhook_crud(client):
    wh = client.webhooks.create(
        target_url=f"https://example.com/hook/{uuid.uuid4().hex[:8]}",
        events=["task.completed"],
    )
    try:
        assert wh.uid
        fetched = client.webhooks.get(wh.uid)
        assert fetched.uid == wh.uid
    finally:
        client.webhooks.delete(wh.uid)


def test_e2e_agent_crud(client):
    agent = client.agents.create(name=_unique("agent"), events=["task.completed"])
    try:
        assert agent.uid
        assert client.agents.get(agent.uid).uid == agent.uid
    finally:
        client.agents.delete(agent.uid)
