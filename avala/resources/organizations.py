"""Organizations resource."""

from __future__ import annotations

from typing import Any, Dict

from avala._pagination import CursorPage
from avala.resources._base import BaseAsyncResource, BaseSyncResource
from avala.types.organization import Invitation, Organization, OrganizationMember, Team, TeamMember


class Organizations(BaseSyncResource):
    # ── Core CRUD ────────────────────────────────────────────

    def list(self, *, limit: int | None = None, cursor: str | None = None) -> CursorPage[Organization]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page("/organizations/", Organization, params=params or None)

    def get(self, slug: str) -> Organization:
        data = self._transport.request("GET", f"/organizations/{slug}/")
        return Organization.model_validate(data)

    def create(
        self,
        *,
        name: str,
        description: str | None = None,
        logo: str | None = None,
        website: str | None = None,
        industry: str | None = None,
        email: str | None = None,
        phone: str | None = None,
        visibility: str | None = None,
    ) -> Organization:
        payload: dict[str, Any] = {"name": name}
        if description is not None:
            payload["description"] = description
        if logo is not None:
            payload["logo"] = logo
        if website is not None:
            payload["website"] = website
        if industry is not None:
            payload["industry"] = industry
        if email is not None:
            payload["email"] = email
        if phone is not None:
            payload["phone"] = phone
        if visibility is not None:
            payload["visibility"] = visibility
        data = self._transport.request("POST", "/organizations/", json=payload)
        return Organization.model_validate(data)

    def update(
        self,
        slug: str,
        *,
        name: str | None = None,
        description: str | None = None,
        logo: str | None = None,
        website: str | None = None,
        handle: str | None = None,
        industry: str | None = None,
        email: str | None = None,
        phone: str | None = None,
        visibility: str | None = None,
    ) -> Organization:
        payload: dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if logo is not None:
            payload["logo"] = logo
        if website is not None:
            payload["website"] = website
        if handle is not None:
            payload["handle"] = handle
        if industry is not None:
            payload["industry"] = industry
        if email is not None:
            payload["email"] = email
        if phone is not None:
            payload["phone"] = phone
        if visibility is not None:
            payload["visibility"] = visibility
        data = self._transport.request("PATCH", f"/organizations/{slug}/", json=payload)
        return Organization.model_validate(data)

    def delete(self, slug: str) -> None:
        self._transport.request("DELETE", f"/organizations/{slug}/")

    def industry_choices(self) -> Dict[str, Any]:
        data = self._transport.request("GET", "/organizations/industry-choices/")
        return data  # type: ignore[no-any-return]

    # ── Members ──────────────────────────────────────────────

    def list_members(
        self,
        slug: str,
        *,
        search: str | None = None,
        role: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> CursorPage[OrganizationMember]:
        params: dict[str, Any] = {}
        if search is not None:
            params["search"] = search
        if role is not None:
            params["role"] = role
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page(
            f"/organizations/{slug}/members/", OrganizationMember, params=params or None
        )

    def remove_member(self, slug: str, user_uid: str) -> None:
        self._transport.request("DELETE", f"/organizations/{slug}/members/{user_uid}/")

    def update_member_role(self, slug: str, user_uid: str, *, role: str) -> None:
        self._transport.request("PATCH", f"/organizations/{slug}/members/{user_uid}/role/", json={"role": role})

    def leave(self, slug: str) -> None:
        self._transport.request("POST", f"/organizations/{slug}/leave/")

    def transfer_ownership(self, slug: str, *, new_owner_uid: str) -> None:
        self._transport.request(
            "POST", f"/organizations/{slug}/transfer-ownership/", json={"new_owner_uid": new_owner_uid}
        )

    # ── Invitations ──────────────────────────────────────────

    def list_invitations(
        self, slug: str, *, limit: int | None = None, cursor: str | None = None
    ) -> CursorPage[Invitation]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page(f"/organizations/{slug}/invitations/", Invitation, params=params or None)

    def create_invitation(self, slug: str, *, email: str, role: str | None = None) -> Invitation:
        payload: dict[str, Any] = {"email": email}
        if role is not None:
            payload["role"] = role
        data = self._transport.request("POST", f"/organizations/{slug}/invitations/", json=payload)
        return Invitation.model_validate(data)

    def resend_invitation(self, slug: str, invitation_uid: str) -> None:
        self._transport.request("POST", f"/organizations/{slug}/invitations/{invitation_uid}/resend/")

    def cancel_invitation(self, slug: str, invitation_uid: str) -> None:
        self._transport.request("POST", f"/organizations/{slug}/invitations/{invitation_uid}/cancel/")

    # ── Teams ────────────────────────────────────────────────

    def list_teams(self, slug: str, *, limit: int | None = None, cursor: str | None = None) -> CursorPage[Team]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page(f"/organizations/{slug}/teams/", Team, params=params or None)

    def create_team(self, slug: str, *, name: str, description: str | None = None, color: str | None = None) -> Team:
        payload: dict[str, Any] = {"name": name}
        if description is not None:
            payload["description"] = description
        if color is not None:
            payload["color"] = color
        data = self._transport.request("POST", f"/organizations/{slug}/teams/", json=payload)
        return Team.model_validate(data)

    def get_team(self, slug: str, team_slug: str) -> Team:
        data = self._transport.request("GET", f"/organizations/{slug}/teams/{team_slug}/")
        return Team.model_validate(data)

    def update_team(
        self,
        slug: str,
        team_slug: str,
        *,
        name: str | None = None,
        description: str | None = None,
        color: str | None = None,
    ) -> Team:
        payload: dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if color is not None:
            payload["color"] = color
        data = self._transport.request("PATCH", f"/organizations/{slug}/teams/{team_slug}/", json=payload)
        return Team.model_validate(data)

    def delete_team(self, slug: str, team_slug: str) -> None:
        self._transport.request("DELETE", f"/organizations/{slug}/teams/{team_slug}/")

    # ── Team Members ─────────────────────────────────────────

    def list_team_members(
        self, slug: str, team_slug: str, *, limit: int | None = None, cursor: str | None = None
    ) -> CursorPage[TeamMember]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return self._transport.request_page(
            f"/organizations/{slug}/teams/{team_slug}/members/", TeamMember, params=params or None
        )

    def add_team_member(self, slug: str, team_slug: str, *, user_uid: str, role: str | None = None) -> TeamMember:
        payload: dict[str, Any] = {"user_uid": user_uid}
        if role is not None:
            payload["role"] = role
        data = self._transport.request("POST", f"/organizations/{slug}/teams/{team_slug}/members/", json=payload)
        return TeamMember.model_validate(data)

    def remove_team_member(self, slug: str, team_slug: str, user_uid: str) -> None:
        self._transport.request("DELETE", f"/organizations/{slug}/teams/{team_slug}/members/{user_uid}/")

    def update_team_member_role(self, slug: str, team_slug: str, user_uid: str, *, role: str) -> None:
        self._transport.request(
            "PATCH", f"/organizations/{slug}/teams/{team_slug}/members/{user_uid}/role/", json={"role": role}
        )


class AsyncOrganizations(BaseAsyncResource):
    # ── Core CRUD ────────────────────────────────────────────

    async def list(self, *, limit: int | None = None, cursor: str | None = None) -> CursorPage[Organization]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page("/organizations/", Organization, params=params or None)

    async def get(self, slug: str) -> Organization:
        data = await self._transport.request("GET", f"/organizations/{slug}/")
        return Organization.model_validate(data)

    async def create(
        self,
        *,
        name: str,
        description: str | None = None,
        logo: str | None = None,
        website: str | None = None,
        industry: str | None = None,
        email: str | None = None,
        phone: str | None = None,
        visibility: str | None = None,
    ) -> Organization:
        payload: dict[str, Any] = {"name": name}
        if description is not None:
            payload["description"] = description
        if logo is not None:
            payload["logo"] = logo
        if website is not None:
            payload["website"] = website
        if industry is not None:
            payload["industry"] = industry
        if email is not None:
            payload["email"] = email
        if phone is not None:
            payload["phone"] = phone
        if visibility is not None:
            payload["visibility"] = visibility
        data = await self._transport.request("POST", "/organizations/", json=payload)
        return Organization.model_validate(data)

    async def update(
        self,
        slug: str,
        *,
        name: str | None = None,
        description: str | None = None,
        logo: str | None = None,
        website: str | None = None,
        handle: str | None = None,
        industry: str | None = None,
        email: str | None = None,
        phone: str | None = None,
        visibility: str | None = None,
    ) -> Organization:
        payload: dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if logo is not None:
            payload["logo"] = logo
        if website is not None:
            payload["website"] = website
        if handle is not None:
            payload["handle"] = handle
        if industry is not None:
            payload["industry"] = industry
        if email is not None:
            payload["email"] = email
        if phone is not None:
            payload["phone"] = phone
        if visibility is not None:
            payload["visibility"] = visibility
        data = await self._transport.request("PATCH", f"/organizations/{slug}/", json=payload)
        return Organization.model_validate(data)

    async def delete(self, slug: str) -> None:
        await self._transport.request("DELETE", f"/organizations/{slug}/")

    async def industry_choices(self) -> Dict[str, Any]:
        data = await self._transport.request("GET", "/organizations/industry-choices/")
        return data  # type: ignore[no-any-return]

    # ── Members ──────────────────────────────────────────────

    async def list_members(
        self,
        slug: str,
        *,
        search: str | None = None,
        role: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> CursorPage[OrganizationMember]:
        params: dict[str, Any] = {}
        if search is not None:
            params["search"] = search
        if role is not None:
            params["role"] = role
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page(
            f"/organizations/{slug}/members/", OrganizationMember, params=params or None
        )

    async def remove_member(self, slug: str, user_uid: str) -> None:
        await self._transport.request("DELETE", f"/organizations/{slug}/members/{user_uid}/")

    async def update_member_role(self, slug: str, user_uid: str, *, role: str) -> None:
        await self._transport.request("PATCH", f"/organizations/{slug}/members/{user_uid}/role/", json={"role": role})

    async def leave(self, slug: str) -> None:
        await self._transport.request("POST", f"/organizations/{slug}/leave/")

    async def transfer_ownership(self, slug: str, *, new_owner_uid: str) -> None:
        await self._transport.request(
            "POST", f"/organizations/{slug}/transfer-ownership/", json={"new_owner_uid": new_owner_uid}
        )

    # ── Invitations ──────────────────────────────────────────

    async def list_invitations(
        self, slug: str, *, limit: int | None = None, cursor: str | None = None
    ) -> CursorPage[Invitation]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page(
            f"/organizations/{slug}/invitations/", Invitation, params=params or None
        )

    async def create_invitation(self, slug: str, *, email: str, role: str | None = None) -> Invitation:
        payload: dict[str, Any] = {"email": email}
        if role is not None:
            payload["role"] = role
        data = await self._transport.request("POST", f"/organizations/{slug}/invitations/", json=payload)
        return Invitation.model_validate(data)

    async def resend_invitation(self, slug: str, invitation_uid: str) -> None:
        await self._transport.request("POST", f"/organizations/{slug}/invitations/{invitation_uid}/resend/")

    async def cancel_invitation(self, slug: str, invitation_uid: str) -> None:
        await self._transport.request("POST", f"/organizations/{slug}/invitations/{invitation_uid}/cancel/")

    # ── Teams ────────────────────────────────────────────────

    async def list_teams(self, slug: str, *, limit: int | None = None, cursor: str | None = None) -> CursorPage[Team]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page(f"/organizations/{slug}/teams/", Team, params=params or None)

    async def create_team(
        self, slug: str, *, name: str, description: str | None = None, color: str | None = None
    ) -> Team:
        payload: dict[str, Any] = {"name": name}
        if description is not None:
            payload["description"] = description
        if color is not None:
            payload["color"] = color
        data = await self._transport.request("POST", f"/organizations/{slug}/teams/", json=payload)
        return Team.model_validate(data)

    async def get_team(self, slug: str, team_slug: str) -> Team:
        data = await self._transport.request("GET", f"/organizations/{slug}/teams/{team_slug}/")
        return Team.model_validate(data)

    async def update_team(
        self,
        slug: str,
        team_slug: str,
        *,
        name: str | None = None,
        description: str | None = None,
        color: str | None = None,
    ) -> Team:
        payload: dict[str, Any] = {}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if color is not None:
            payload["color"] = color
        data = await self._transport.request("PATCH", f"/organizations/{slug}/teams/{team_slug}/", json=payload)
        return Team.model_validate(data)

    async def delete_team(self, slug: str, team_slug: str) -> None:
        await self._transport.request("DELETE", f"/organizations/{slug}/teams/{team_slug}/")

    # ── Team Members ─────────────────────────────────────────

    async def list_team_members(
        self, slug: str, team_slug: str, *, limit: int | None = None, cursor: str | None = None
    ) -> CursorPage[TeamMember]:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if cursor is not None:
            params["cursor"] = cursor
        return await self._transport.request_page(
            f"/organizations/{slug}/teams/{team_slug}/members/", TeamMember, params=params or None
        )

    async def add_team_member(self, slug: str, team_slug: str, *, user_uid: str, role: str | None = None) -> TeamMember:
        payload: dict[str, Any] = {"user_uid": user_uid}
        if role is not None:
            payload["role"] = role
        data = await self._transport.request("POST", f"/organizations/{slug}/teams/{team_slug}/members/", json=payload)
        return TeamMember.model_validate(data)

    async def remove_team_member(self, slug: str, team_slug: str, user_uid: str) -> None:
        await self._transport.request("DELETE", f"/organizations/{slug}/teams/{team_slug}/members/{user_uid}/")

    async def update_team_member_role(self, slug: str, team_slug: str, user_uid: str, *, role: str) -> None:
        await self._transport.request(
            "PATCH", f"/organizations/{slug}/teams/{team_slug}/members/{user_uid}/role/", json={"role": role}
        )
