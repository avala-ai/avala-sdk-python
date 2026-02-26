"""
API contract tests: verify SDK types match the API contract.

These tests load contracts/api_contracts.json and verify that every field
listed in the contract is present in the corresponding SDK Pydantic model.
If a Django serializer field is added/removed, the contract file is updated
(via contracts/verify_server.py), and these tests will fail until the SDK
type is updated to match.

This prevents the class of bugs where SDK types silently miss or misname
fields from the Django API.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Set, cast

import pytest

from avala.types.annotation_issue import AnnotationIssue, AnnotationIssueMetrics, AnnotationIssueToolDetail
from avala.types.dataset import DatasetItem, DatasetSequence
from avala.types.organization import Invitation, Organization, OrganizationMember, Team, TeamMember
from avala.types.slice import Slice, SliceItem

# ── Load contract ───────────────────────────────────────────────

CONTRACT_PATH = Path(__file__).resolve().parent.parent.parent.parent / "contracts" / "api_contracts.json"


@pytest.fixture(scope="module")
def contract() -> Dict[str, Any]:
    with open(CONTRACT_PATH) as f:
        return cast(Dict[str, Any], json.load(f))


# ── Type → Pydantic model mapping ──────────────────────────────

SDK_TYPE_MAP = {
    "Organization": Organization,
    "OrganizationMember": OrganizationMember,
    "Invitation": Invitation,
    "Team": Team,
    "TeamMember": TeamMember,
    "Slice": Slice,
    "SliceItem": SliceItem,
    "DatasetItem": DatasetItem,
    "DatasetSequence": DatasetSequence,
    "AnnotationIssue": AnnotationIssue,
    "AnnotationIssueMetrics": AnnotationIssueMetrics,
    "AnnotationIssueToolDetail": AnnotationIssueToolDetail,
}


def get_pydantic_fields(model_cls: Any) -> Set[str]:
    """Get field names from a Pydantic model."""
    return set(model_cls.model_fields.keys())


# ── SDK transport method → response shape mapping ───────────────

# Maps SDK method patterns to the transport method they use.
# These are verified by inspecting the source code of each resource class.
RESPONSE_SHAPE_MAP: Dict[str, str] = {}


def _build_response_shape_map() -> None:
    """Build a map of endpoint → response shape from actual SDK source code."""
    import inspect

    from avala.resources.annotation_issues import AnnotationIssues
    from avala.resources.datasets import Datasets
    from avala.resources.organizations import Organizations
    from avala.resources.slices import Slices

    resource_map = {
        "organizations": Organizations,
        "slices": Slices,
        "datasets": Datasets,
        "annotation_issues": AnnotationIssues,
    }

    for resource_name, resource_cls in resource_map.items():
        for method_name in dir(resource_cls):
            if method_name.startswith("_"):
                continue
            method = getattr(resource_cls, method_name)
            if not callable(method):
                continue

            try:
                source = inspect.getsource(method)
            except (TypeError, OSError):
                continue

            endpoint_key = f"{resource_name}.{method_name}"

            if "request_page(" in source:
                RESPONSE_SHAPE_MAP[endpoint_key] = "paginated"
            elif "request_list(" in source:
                RESPONSE_SHAPE_MAP[endpoint_key] = "array"
            elif re.search(r'request\(\s*"DELETE"', source):
                RESPONSE_SHAPE_MAP[endpoint_key] = "void"
            elif re.search(r'request\(\s*"(POST|PATCH)"', source) and "model_validate" not in source:
                RESPONSE_SHAPE_MAP[endpoint_key] = "void"
            elif "model_validate(" in source or "requestSingle" in source:
                RESPONSE_SHAPE_MAP[endpoint_key] = "single"
            elif "requestCreate" in source:
                RESPONSE_SHAPE_MAP[endpoint_key] = "single"


_build_response_shape_map()


# ── Tests ───────────────────────────────────────────────────────


class TestSDKTypesCoverContractFields:
    """Verify SDK Pydantic models have all fields from the API contract."""

    @pytest.mark.parametrize(
        "type_name",
        [
            "Organization",
            "OrganizationMember",
            "Invitation",
            "Team",
            "TeamMember",
            "Slice",
            "SliceItem",
            "DatasetItem",
            "DatasetSequence",
            "AnnotationIssue",
            "AnnotationIssueMetrics",
            "AnnotationIssueToolDetail",
        ],
    )
    def test_sdk_type_covers_all_serializer_fields(self, contract: Dict[str, Any], type_name: str) -> None:
        """SDK type must have a field for every field in the contract serializer(s)."""
        type_def = contract["types"][type_name]
        sdk_model = SDK_TYPE_MAP[type_name]
        sdk_fields = get_pydantic_fields(sdk_model)

        # Collect all fields from all serializer variants
        all_api_fields: Set[str] = set()
        for key in ("fields", "list_fields", "detail_fields", "api_key_list_fields"):
            if key in type_def:
                all_api_fields.update(type_def[key])

        missing = all_api_fields - sdk_fields
        if missing:
            pytest.fail(
                f"{type_name} SDK type is missing fields from API contract: {sorted(missing)}. "
                f"Add these fields to avala/types/ or update contracts/api_contracts.json."
            )

    @pytest.mark.parametrize(
        "type_name",
        [
            "OrganizationMember",
            "Invitation",
            "TeamMember",
            "AnnotationIssue",
            "AnnotationIssueMetrics",
            "AnnotationIssueToolDetail",
        ],
    )
    def test_sdk_type_exact_match_for_flat_types(self, contract: Dict[str, Any], type_name: str) -> None:
        """For types with a single serializer, SDK fields should exactly match contract fields."""
        type_def = contract["types"][type_name]
        if "fields" not in type_def:
            pytest.skip("Type has multiple serializer variants")

        sdk_model = SDK_TYPE_MAP[type_name]
        sdk_fields = get_pydantic_fields(sdk_model)
        contract_fields = set(type_def["fields"])

        extra_in_sdk = sdk_fields - contract_fields
        if extra_in_sdk:
            pytest.fail(
                f"{type_name} SDK type has fields not in the API contract: {sorted(extra_in_sdk)}. "
                f"These fields may not exist in the Django serializer. "
                f"Remove them from avala/types/ or add them to contracts/api_contracts.json."
            )


class TestResponseShapesMatchContract:
    """Verify SDK methods use the correct transport method for each endpoint's response shape."""

    @pytest.mark.parametrize(
        "endpoint_key",
        [
            "organizations.list",
            "organizations.list_members",
            "organizations.list_invitations",
            "organizations.list_teams",
            "organizations.list_team_members",
            "slices.list",
            "slices.list_items",
            "datasets.list_items",
            "datasets.list_sequences",
            "annotation_issues.list_by_sequence",
            "annotation_issues.list_by_dataset",
            "annotation_issues.list_tools",
        ],
    )
    def test_list_endpoint_uses_correct_transport(self, contract: Dict[str, Any], endpoint_key: str) -> None:
        """SDK must use request_page for paginated endpoints and request_list for array endpoints."""
        endpoint = contract["endpoints"][endpoint_key]
        expected_shape = endpoint["response_shape"]

        actual_shape = RESPONSE_SHAPE_MAP.get(endpoint_key)
        if actual_shape is None:
            pytest.fail(
                f"Could not determine response shape for SDK method '{endpoint_key}'. "
                f"Method may not exist or uses an unrecognized transport pattern."
            )

        assert actual_shape == expected_shape, (
            f"SDK method '{endpoint_key}' uses '{actual_shape}' transport "
            f"but contract says '{expected_shape}'. "
            f"This will cause a runtime crash! "
            f"Use request_page() for paginated, request_list() for array endpoints."
        )

    @pytest.mark.parametrize(
        "endpoint_key",
        [
            "organizations.delete",
            "organizations.remove_member",
            "organizations.update_member_role",
            "organizations.leave",
            "organizations.transfer_ownership",
            "organizations.resend_invitation",
            "organizations.cancel_invitation",
            "organizations.delete_team",
            "organizations.remove_team_member",
            "organizations.update_team_member_role",
            "annotation_issues.delete",
        ],
    )
    def test_void_endpoint_uses_correct_transport(self, contract: Dict[str, Any], endpoint_key: str) -> None:
        """SDK methods for void endpoints must not try to parse response as a model."""
        endpoint = contract["endpoints"][endpoint_key]
        expected_shape = endpoint["response_shape"]

        actual_shape = RESPONSE_SHAPE_MAP.get(endpoint_key)
        assert actual_shape == expected_shape, (
            f"SDK method '{endpoint_key}' uses '{actual_shape}' transport but contract says '{expected_shape}'."
        )

    @pytest.mark.parametrize(
        "endpoint_key",
        [
            "datasets.create",
            "organizations.update_team",
            "organizations.add_team_member",
        ],
    )
    def test_single_endpoint_uses_correct_transport(self, contract: Dict[str, Any], endpoint_key: str) -> None:
        """SDK methods for single-object endpoints must parse and validate model responses."""
        endpoint = contract["endpoints"][endpoint_key]
        expected_shape = endpoint["response_shape"]

        actual_shape = RESPONSE_SHAPE_MAP.get(endpoint_key)
        assert actual_shape == expected_shape, (
            f"SDK method '{endpoint_key}' uses '{actual_shape}' transport but contract says '{expected_shape}'."
        )
