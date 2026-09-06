"""A rename moves an identity; it must not silently unpublish its document.

``persist_refined_name`` mints the successor of every renamed StandardName and
copies the predecessor's description onto it. Documentation is the other half
of the same prose and was dropped, so a re-rendering of an already-documented
quantity landed as an undocumented node with its docs axis reset — the text
was written about a meaning that had not changed.

These tests pin both halves of the rule. Documentation and the whole docs
lifecycle travel when the two spellings parse to the same grammar IR, and they
do not travel otherwise: a name that says something else must not inherit an
accepted document asserting what nobody wrote about it.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

_GC_PATH = "imas_codex.standard_names.graph_ops.GraphClient"

# Measured live pairs. The re-rendered spellings are from the cohort a
# rendering-rule migration minted; the meaning-changing ones are refinements
# that also rewrote the description.
RERENDERED = [
    ("maximum_of_energy_flux_at_limiter", "energy_flux_maximum_at_limiter"),
    ("accumulated_neutral_count_at_wall", "neutral_count_accumulated_at_wall"),
    (
        "wave_current_of_antenna_strap_amplitude",
        "wave_current_amplitude_of_antenna_strap",
    ),
]
REMEANT = [
    ("beta", "toroidal_plasma_beta"),
    ("etendue", "viewing_etendue_of_spectrometer_channel"),
    ("psi_star", "difference_of_magnetic_flux_and_toroidal_magnetic_flux"),
]


def _capture_persist(old_name: str, new_name: str) -> tuple[str, dict[str, Any]]:
    """Run the persist against a mocked transaction; return its Cypher + params.

    The preflight cannot commit against a mock, so it refuses — the statement
    it issued and the parameters bound to it are what these tests read.
    """
    from imas_codex.standard_names.graph_ops import (
        RefinedNamePersistenceRefusal,
        persist_refined_name,
    )

    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)
    tx = MagicMock()
    tx.closed = False
    tx.run.return_value = []
    session = MagicMock()
    session.begin_transaction = MagicMock(return_value=tx)

    @contextmanager
    def _session_ctx():
        yield session

    gc.session = _session_ctx

    with patch(_GC_PATH, return_value=gc):
        with pytest.raises(RefinedNamePersistenceRefusal):
            persist_refined_name(
                old_name=old_name,
                new_name=new_name,
                description="unchanged prose about one quantity",
            )

    call = tx.run.call_args_list[0]
    return " ".join(call.args[0].split()), call.kwargs


class TestTheGrammarDecidesWhetherTheMeaningMoved:
    """IR equality is the test for a rename that changed only the spelling."""

    @pytest.mark.parametrize(("old", "new"), RERENDERED)
    def test_a_re_rendering_preserves_meaning(self, old: str, new: str):
        from imas_codex.standard_names.graph_ops import rename_preserves_meaning

        assert rename_preserves_meaning(old, new) is True

    @pytest.mark.parametrize(("old", "new"), REMEANT)
    def test_a_different_quantity_does_not(self, old: str, new: str):
        from imas_codex.standard_names.graph_ops import rename_preserves_meaning

        assert rename_preserves_meaning(old, new) is False

    def test_an_unparseable_spelling_answers_no(self):
        """Absence of evidence is not evidence of sameness."""
        from imas_codex.standard_names.graph_ops import rename_preserves_meaning

        assert rename_preserves_meaning("beta", "!!! not a name !!!") is False
        assert rename_preserves_meaning("!!! not a name !!!", "beta") is False


class TestTheSuccessorInheritsTheDocumentation:
    """A re-rendering carries the text and the docs lifecycle with it."""

    def test_documentation_is_read_from_the_predecessor(self):
        cypher, _ = _capture_persist(*RERENDERED[0])
        assert "new.documentation = CASE WHEN $meaning_preserved" in cypher
        assert "THEN old.documentation" in cypher

    def test_the_docs_stage_is_the_predecessors_not_pending(self):
        cypher, _ = _capture_persist(*RERENDERED[0])
        assert (
            "new.docs_stage = CASE WHEN $meaning_preserved "
            "AND coalesce(old.documentation, '') <> '' "
            "THEN coalesce(old.docs_stage, 'pending') ELSE 'pending' END" in cypher
        )

    @pytest.mark.parametrize(
        "field", ["docs_chain_length", "docs_model", "docs_generated_at"]
    )
    def test_the_docs_provenance_travels_with_the_text(self, field: str):
        """Text without its provenance would misreport who wrote it and when."""
        cypher, _ = _capture_persist(*RERENDERED[0])
        assert f"new.{field} = CASE WHEN $meaning_preserved" in cypher

    @pytest.mark.parametrize(("old", "new"), RERENDERED)
    def test_the_bound_parameter_is_true_for_a_re_rendering(self, old: str, new: str):
        _, params = _capture_persist(old, new)
        assert params["meaning_preserved"] is True


class TestAChangedMeaningStartsTheDocsAxisOver:
    """The negative half: no accepted document may follow a new meaning."""

    @pytest.mark.parametrize(("old", "new"), REMEANT)
    def test_the_bound_parameter_is_false_for_a_new_meaning(self, old: str, new: str):
        _, params = _capture_persist(old, new)
        assert params["meaning_preserved"] is False

    def test_the_false_branch_writes_pending_and_no_text(self):
        """Every docs field's ELSE branch must be the undocumented state."""
        cypher, _ = _capture_persist(*REMEANT[0])
        for fragment in (
            "new.docs_stage = CASE WHEN $meaning_preserved "
            "AND coalesce(old.documentation, '') <> '' "
            "THEN coalesce(old.docs_stage, 'pending') ELSE 'pending' END",
            "new.documentation = CASE WHEN $meaning_preserved "
            "THEN old.documentation ELSE null END",
            "new.docs_model = CASE WHEN $meaning_preserved "
            "THEN old.docs_model ELSE null END",
            "new.docs_generated_at = CASE WHEN $meaning_preserved "
            "THEN old.docs_generated_at ELSE null END",
            "new.docs_chain_length = CASE WHEN $meaning_preserved "
            "THEN coalesce(old.docs_chain_length, 0) ELSE 0 END",
        ):
            assert fragment in cypher

    def test_an_undocumented_predecessor_leaves_the_stage_pending(self):
        """Carrying an 'accepted' stage over empty text would claim a document
        that does not exist, so the stage guard reads the text and not the
        stage."""
        cypher, _ = _capture_persist(*RERENDERED[0])
        assert "coalesce(old.documentation, '') <> ''" in cypher
