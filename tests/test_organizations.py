"""Tests for Organizations resource."""

from __future__ import annotations

import json

import httpx
import respx

from avala import Client

BASE_URL = "https://api.avala.ai/api/v1"


@respx.mock
def test_list_organizations():
    """Organizations.list() returns a CursorPage of Organization objects."""
    respx.get(f"{BASE_URL}/organizations/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "uid": "org-001",
                        "name": "Test Org",
                        "slug": "test-org",
                        "is_verified": True,
                        "is_active": True,
                        "member_count": 5,
                    }
                ],
                "next": None,
                "previous": None,
            },
        )
    )
    client = Client(api_key="test-key")
    page = client.organizations.list()
    assert len(page.items) == 1
    assert page.items[0].name == "Test Org"
    assert page.items[0].slug == "test-org"
    assert page.items[0].is_verified is True
    assert page.items[0].member_count == 5
    assert page.has_more is False
    client.close()


@respx.mock
def test_get_organization():
    """Organizations.get() returns a single Organization by slug."""
    slug = "test-org"
    respx.get(f"{BASE_URL}/organizations/{slug}/").mock(
        return_value=httpx.Response(
            200,
            json={
                "uid": "org-001",
                "name": "Test Org",
                "slug": slug,
                "description": "A test organization",
                "is_active": True,
            },
        )
    )
    client = Client(api_key="test-key")
    org = client.organizations.get(slug)
    assert org.uid == "org-001"
    assert org.slug == slug
    assert org.description == "A test organization"
    client.close()


@respx.mock
def test_create_organization():
    """Organizations.create() sends correct payload and returns Organization."""
    route = respx.post(f"{BASE_URL}/organizations/").mock(
        return_value=httpx.Response(
            201,
            json={
                "uid": "org-new-001",
                "name": "New Org",
                "slug": "new-org",
                "visibility": "private",
            },
        )
    )
    client = Client(api_key="test-key")
    org = client.organizations.create(name="New Org", visibility="private")
    assert org.uid == "org-new-001"
    assert org.name == "New Org"
    request_body = json.loads(route.calls.last.request.content)
    assert request_body["name"] == "New Org"
    assert request_body["visibility"] == "private"
    client.close()


@respx.mock
def test_delete_organization():
    """Organizations.delete() sends DELETE request."""
    slug = "test-org"
    respx.delete(f"{BASE_URL}/organizations/{slug}/").mock(return_value=httpx.Response(204))
    client = Client(api_key="test-key")
    client.organizations.delete(slug)
    client.close()


@respx.mock
def test_list_members():
    """Organizations.list_members() returns a CursorPage of OrganizationMember objects."""
    slug = "test-org"
    respx.get(f"{BASE_URL}/organizations/{slug}/members/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "user_uid": "user-001",
                        "username": "alice",
                        "email": "alice@example.com",
                        "full_name": "Alice Smith",
                        "picture": None,
                        "role": "admin",
                        "created_at": "2026-01-01T00:00:00Z",
                    }
                ],
                "next": None,
                "previous": None,
            },
        )
    )
    client = Client(api_key="test-key")
    page = client.organizations.list_members(slug)
    assert len(page.items) == 1
    assert page.items[0].user_uid == "user-001"
    assert page.items[0].username == "alice"
    assert page.items[0].email == "alice@example.com"
    assert page.items[0].full_name == "Alice Smith"
    assert page.items[0].role == "admin"
    client.close()


@respx.mock
def test_list_teams():
    """Organizations.list_teams() returns a CursorPage of Team objects."""
    slug = "test-org"
    respx.get(f"{BASE_URL}/organizations/{slug}/teams/").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "uid": "team-001",
                        "name": "Engineering",
                        "slug": "engineering",
                        "description": "Engineering team",
                        "member_count": 3,
                    }
                ],
                "next": None,
                "previous": None,
            },
        )
    )
    client = Client(api_key="test-key")
    page = client.organizations.list_teams(slug)
    assert len(page.items) == 1
    assert page.items[0].uid == "team-001"
    assert page.items[0].name == "Engineering"
    assert page.items[0].slug == "engineering"
    assert page.items[0].member_count == 3
    client.close()


@respx.mock
def test_get_team_uses_team_slug():
    """Organizations.get_team() uses team_slug (not team_uid) in the URL path."""
    slug = "test-org"
    team_slug = "engineering"
    route = respx.get(f"{BASE_URL}/organizations/{slug}/teams/{team_slug}/").mock(
        return_value=httpx.Response(
            200,
            json={
                "uid": "team-001",
                "name": "Engineering",
                "slug": team_slug,
                "description": "Engineering team",
                "member_count": 3,
            },
        )
    )
    client = Client(api_key="test-key")
    team = client.organizations.get_team(slug, team_slug)
    assert team.slug == team_slug
    assert team.name == "Engineering"
    assert route.called
    client.close()


@respx.mock
def test_leave_path_no_members_segment():
    """Organizations.leave() hits /organizations/{slug}/leave/ (no /members/)."""
    slug = "test-org"
    route = respx.post(f"{BASE_URL}/organizations/{slug}/leave/").mock(return_value=httpx.Response(204))
    client = Client(api_key="test-key")
    client.organizations.leave(slug)
    assert route.called
    client.close()


@respx.mock
def test_transfer_ownership_path_no_members_segment():
    """Organizations.transfer_ownership() hits /organizations/{slug}/transfer-ownership/ (no /members/)."""
    slug = "test-org"
    route = respx.post(f"{BASE_URL}/organizations/{slug}/transfer-ownership/").mock(return_value=httpx.Response(204))
    client = Client(api_key="test-key")
    client.organizations.transfer_ownership(slug, new_owner_uid="user-uid-001")
    assert route.called
    request_body = json.loads(route.calls.last.request.content)
    assert request_body["new_owner_uid"] == "user-uid-001"
    client.close()
