Coordinator verification of n-sgr-does-the-edits-flag-scope-the-run, base 69f24046b

CONFIRMED — the primary answer
  graph_ops.py:15322  if edits_only: scope_where += " AND coalesce(sn.edit_status,'') = 'open'"
                      => the flag DOES scope the pool claims.                        EXACT
  loop.py:1284        if skip_global_maintenance and not (scope_run_id or drain_scope_id): raise
                      => edits_only is absent from that tuple, so --edits alone
                         cannot bypass global maintenance.                            EXACT
  live graph, edit_status='open' by stage:
                      drafted 105, reviewed 48, accepted 25, exhausted 22,
                      superseded 13, total 213                                        EXACT (all five)
  So the claim half is scoped and the maintenance half is not. That resolves the
  2026-09-05 "whole pool machinery" observation as the maintenance half.

NOT CONFIRMED — the secondary causal claim
  The report states at line 72 that "the validation raises _NonCanonicalParseError
  ('name is not canonical: flat segment order renders ...')".
  - _NonCanonicalParseError does not exist anywhere in imas_codex/.        FABRICATED SYMBOL
  - the real raise is a bare ValueError at grammar_adapter.py:123, message
    "name {name!r} is not canonical; ordered grammar renders {canonical!r}"  PARAPHRASED
  - every parse_canonical_name call in workers.py, the review module, is
    wrapped in try/except catching ValueError/ParseError/Exception
    (lines 492, 530, 1351, 3476), so a non-canonical name does NOT
    propagate from there.                                                  CONTRADICTS
  - an unguarded raise DOES exist, at models.py:498 and models.py:568
    (to_ir -> parse_canonical_name), so the mechanism is plausible; that the
    REVIEW path reaches those sites is not established.                    PLAUSIBLE

NOT VERIFIED BY COORDINATOR
  the 49-of-105 strict-non-canonical partition (28/19/2). Confirming it needs the
  strict parser, which is a test execution rather than an audit. Substring counts
  over the 105 give 37 containing flux_surface_averaged and 17 containing
  accumulat, which measure something different and neither confirm nor refute 49.
  The report's word "trailing" is wrong: the tokens sit mid-name, between base and
  locus, e.g. argon_density_flux_surface_averaged_at_plasma_boundary.

VERDICT: primary deliverable CONFIRMED and valuable; cause PLAUSIBLE, not established. EXIT=0
