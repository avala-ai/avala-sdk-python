from __future__ import annotations

import pytest

from avala.types.organization import Organization


@pytest.mark.parametrize("value", [None, "", "profile-value"])
@pytest.mark.parametrize(
    "field", ["readme", "organization_type", "x_url", "hugging_face_url", "github_url", "linkedin_url"]
)
def test_optional_profile_metadata_round_trips(field: str, value: str | None) -> None:
    payload = {"uid": "org-example", "name": "Example", "slug": "example", field: value}
    assert Organization.model_validate(payload).model_dump()[field] == value


def test_legacy_organization_payload_has_no_inferred_metadata() -> None:
    organization = Organization(uid="org-example", name="Example University", slug="example")
    for field in ("readme", "organization_type", "x_url", "hugging_face_url", "github_url", "linkedin_url"):
        assert organization.model_dump()[field] is None
