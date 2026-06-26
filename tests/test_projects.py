"""Tests for Projects resource."""

from __future__ import annotations

import httpx
import respx

from avala import Client

BASE_URL = "https://api.avala.ai/api/v1"

# The SDK lists/gets projects via the user-scoped route (``/users/me/projects/``),
# not the staff-only top-level ``/projects/`` admin endpoint.
PROJECTS_URL = f"{BASE_URL}/users/me/projects/"


@respx.mock
def test_list_projects():
    """Projects.list() returns a CursorPage of Project objects."""
    respx.get(PROJECTS_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "uid": "proj-001",
                        "name": "Project Alpha",
                        "status": "active",
                        "created_at": "2024-01-01T00:00:00Z",
                        "updated_at": "2024-01-02T00:00:00Z",
                    }
                ],
                "next": None,
                "previous": None,
            },
        )
    )
    client = Client(api_key="test-key")
    page = client.projects.list()
    assert len(page.items) == 1
    assert page.items[0].uid == "proj-001"
    assert page.items[0].name == "Project Alpha"
    assert page.has_more is False
    client.close()


@respx.mock
def test_get_project():
    """Projects.get() returns a single Project by uid."""
    uid = "proj-001"
    respx.get(f"{PROJECTS_URL}{uid}/").mock(
        return_value=httpx.Response(
            200,
            json={
                "uid": uid,
                "name": "Project Alpha",
                "status": "active",
            },
        )
    )
    client = Client(api_key="test-key")
    project = client.projects.get(uid)
    assert project.uid == uid
    assert project.name == "Project Alpha"
    assert project.status == "active"
    client.close()


@respx.mock
def test_list_projects_pagination_cursor():
    """Projects.list() passes cursor param and extracts next_cursor from response."""
    respx.get(PROJECTS_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"uid": "proj-002", "name": "Project Beta"},
                ],
                "next": f"{PROJECTS_URL}?cursor=next-page-token",
                "previous": f"{PROJECTS_URL}?cursor=prev-page-token",
            },
        )
    )
    client = Client(api_key="test-key")
    page = client.projects.list(cursor="some-cursor", limit=10)
    assert page.has_more is True
    assert page.next_cursor == "next-page-token"
    assert page.previous_cursor == "prev-page-token"
    client.close()
