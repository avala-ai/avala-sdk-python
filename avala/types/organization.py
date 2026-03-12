from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class Organization(BaseModel):
    uid: str
    name: str
    slug: str
    handle: Optional[str] = None
    description: Optional[str] = None
    logo: Optional[str] = None
    website: Optional[str] = None
    industry: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    visibility: Optional[str] = None
    plan: Optional[str] = None
    is_verified: bool = False
    is_active: bool = True
    member_count: Optional[int] = None
    team_count: Optional[int] = None
    dataset_count: Optional[int] = None
    project_count: Optional[int] = None
    slice_count: Optional[int] = None
    role: Optional[str] = None
    joined_at: Optional[datetime] = None
    slug_edits_remaining: Optional[int] = None
    allowed_domains: Optional[List[str]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class OrganizationMember(BaseModel):
    user_uid: str
    username: Optional[str] = None
    email: Optional[str] = None
    full_name: Optional[str] = None
    picture: Optional[str] = None
    role: Optional[str] = None
    created_at: Optional[datetime] = None


class Invitation(BaseModel):
    uid: str
    organization_name: Optional[str] = None
    organization_slug: Optional[str] = None
    invited_email: str
    invited_by_username: Optional[str] = None
    role: Optional[str] = None
    status: Optional[str] = None
    expires_at: Optional[datetime] = None
    is_expired: Optional[bool] = None
    accept_url: Optional[str] = None
    copy_link: Optional[str] = None
    created_at: Optional[datetime] = None


class Team(BaseModel):
    uid: str
    name: str
    slug: Optional[str] = None
    description: Optional[str] = None
    color: Optional[str] = None
    organization_uid: Optional[str] = None
    organization_name: Optional[str] = None
    organization_slug: Optional[str] = None
    member_count: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class TeamMember(BaseModel):
    user_uid: str
    username: Optional[str] = None
    email: Optional[str] = None
    full_name: Optional[str] = None
    picture: Optional[str] = None
    role: Optional[str] = None
    created_at: Optional[datetime] = None
