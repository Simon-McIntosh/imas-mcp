"""Candidate filtering and deletion backstop for protected identities."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from imas_codex.standard_names import graph_ops, protection


def _graph_with_protected_parent() -> MagicMock:
    gc = MagicMock()

    def query(cypher: str, **params: Any) -> list[dict[str, Any]]:
        if "RETURN DISTINCT parent.id AS parent_id" in cypher:
            return [
                {"parent_id": "ion_power_density"},
                {"parent_id": "unspent_structural_parent"},
            ]
        if "MERGE (cost)-[:FOR_STANDARD_NAME]->(sn)" in cypher:
            return [{"linked": 1}]
        if "OPTIONAL MATCH (cost:LLMCost)-[:FOR_STANDARD_NAME]->(sn)" in cypher:
            assert params["names"] in (
                ["ion_power_density"],
                ["ion_power_density", "unspent_structural_parent"],
            )
            return [
                {
                    "id": "ion_power_density",
                    "reasons": ["recorded_llm_spend_usd=0.25"],
                    "recorded_spend": 0.25,
                }
            ]
        raise AssertionError(f"unexpected query: {cypher}")

    gc.query.side_effect = query
    return gc


def test_structural_candidate_selector_excludes_protected_identity() -> None:
    gc = _graph_with_protected_parent()

    assert graph_ops._query_derived_parents_for_admission_cleanup(gc) == [
        "unspent_structural_parent"
    ]


def test_direct_structural_delete_still_refuses_protected_identity() -> None:
    gc = _graph_with_protected_parent()

    with pytest.raises(
        graph_ops.DerivedParentCleanupRefusal,
        match=r"ion_power_density.*recorded_llm_spend_usd=0.25",
    ) as raised:
        graph_ops._delete_derived_parent_nodes(gc, ["ion_power_density"])

    assert isinstance(raised.value.__cause__, protection.ProtectedDeletionError)
    assert not any(
        "DETACH DELETE sn" in call.args[0] for call in gc.query.call_args_list
    )
