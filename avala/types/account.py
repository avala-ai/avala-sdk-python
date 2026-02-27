"""Account-related Pydantic models."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class SignupUser(BaseModel):
    uid: str
    username: str
    email: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    in_waitlist: bool = True


class SignupResponse(BaseModel):
    user: SignupUser
    api_key: str
