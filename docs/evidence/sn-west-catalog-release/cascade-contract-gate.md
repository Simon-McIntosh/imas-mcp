# Cascade report-vs-persistence gate logs (preserved from /tmp, which is ephemeral)

## gate_baseline
```
........................................................................ [ 43%]
........................................................................ [ 87%]
.....................                                                    [100%]
=============================== warnings summary ===============================
../../../../imas-codex/.venv/lib/python3.12/site-packages/_pytest/config/__init__.py:1434
  /home/ITER/mcintos/Code/imas-codex/.venv/lib/python3.12/site-packages/_pytest/config/__init__.py:1434: PytestConfigWarning: Unknown config option: cache_dir
  
    self._warn_or_fail_if_strict(f"Unknown config option: {key}\n")

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
============================= slowest 10 durations =============================
0.37s setup    tests/standard_names/test_cascade_atomicity.py::TestCascadeAtomicity::test_locus_children_remain_untouched_when_root_accepts
0.25s call     tests/standard_names/test_edit_parity.py::TestPipelineGateBaseline::test_gate_quarantines_bad_unit
0.21s call     tests/standard_names/test_edit_engine.py::TestRenameEligibility::test_eligible_stage_rename_applies[accepted]
0.08s call     tests/standard_names/test_cascade_atomicity.py::TestCascadeAtomicity::test_locus_children_remain_untouched_when_root_accepts
0.06s call     tests/standard_names/test_edit_scope.py::TestFamilyMapping::test_leaf_family_scope_never_lifts_across_locus_parent
0.05s call     tests/standard_names/test_rename_cascade.py::TestOldRootCascadeRecovery::test_change_ledger_failure_rolls_back_complete_cascade
0.05s call     tests/standard_names/test_rename_cascade.py::TestOldRootCascadeRecovery::test_fallback_collision_fails_before_write
0.04s setup    tests/standard_names/test_rename_cascade.py::TestOldRootCascadeRecovery::test_fallback_guard_failures[successor_updates1-superseded-True-not a rename edit]
0.03s call     tests/standard_names/test_rename_cascade.py::TestAuditLog::test_dry_run_writes_audit_lines
0.03s call     tests/standard_names/test_rename_cascade.py::TestOldRootCascadeRecovery::test_apply_excludes_root_and_reconciles_renamed_descendant
165 passed, 1 warning in 10.35s
```

## gate_final
```
........................................................................ [ 41%]
........................................................................ [ 83%]
.............................                                            [100%]
=============================== warnings summary ===============================
../../../../imas-codex/.venv/lib/python3.12/site-packages/_pytest/config/__init__.py:1434
  /home/ITER/mcintos/Code/imas-codex/.venv/lib/python3.12/site-packages/_pytest/config/__init__.py:1434: PytestConfigWarning: Unknown config option: cache_dir
  
    self._warn_or_fail_if_strict(f"Unknown config option: {key}\n")

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
============================= slowest 10 durations =============================
0.85s setup    tests/standard_names/test_cascade_atomicity.py::TestCascadeAtomicity::test_locus_children_remain_untouched_when_root_accepts
0.22s call     tests/standard_names/test_edit_parity.py::TestPipelineGateBaseline::test_gate_quarantines_bad_unit
0.08s call     tests/standard_names/test_cascade_atomicity.py::TestCascadeAtomicity::test_locus_children_remain_untouched_when_root_accepts
0.06s call     tests/standard_names/test_edit_scope.py::TestFamilyMapping::test_leaf_family_scope_never_lifts_across_locus_parent
0.05s call     tests/standard_names/test_edit_engine.py::TestRenameEligibility::test_eligible_stage_rename_applies[accepted]
0.03s call     tests/standard_names/test_rename_cascade.py::TestOldRootCascadeRecovery::test_apply_excludes_root_and_reconciles_renamed_descendant
0.03s call     tests/standard_names/test_rename_cascade.py::TestDeepChainPlanConstruction::test_linear_three_level_chain_plans_without_unreachable
0.03s call     tests/standard_names/test_rename_cascade.py::TestOldRootCascadeRecovery::test_change_ledger_failure_rolls_back_complete_cascade
0.03s call     tests/standard_names/test_edit_engine.py::TestRenameCascadeProtections::test_opt_in_flags_recorded_on_successor
0.03s call     tests/standard_names/test_rename_cascade.py::TestOldRootCascadeRecovery::test_reconciliation_failure_rolls_back_identity_and_topology
173 passed, 1 warning in 11.92s
```

## newtests_final
```
........                                                                 [100%]
=============================== warnings summary ===============================
../../../../imas-codex/.venv/lib/python3.12/site-packages/_pytest/config/__init__.py:1434
  /home/ITER/mcintos/Code/imas-codex/.venv/lib/python3.12/site-packages/_pytest/config/__init__.py:1434: PytestConfigWarning: Unknown config option: cache_dir
  
    self._warn_or_fail_if_strict(f"Unknown config option: {key}\n")

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
============================= slowest 10 durations =============================
0.59s setup    tests/standard_names/test_cascade_descendant_reporting.py::TestUnacceptedRootDefersDescendants::test_planning_reports_deferral_and_writes_nothing
0.08s call     tests/standard_names/test_cascade_descendant_reporting.py::TestUnacceptedRootDefersDescendants::test_planning_reports_deferral_and_writes_nothing

(8 durations < 0.005s hidden.  Use -vv to show these durations.)
```

## revert_final
```
.F..FFFF                                                                 [100%]
=================================== FAILURES ===================================
_ TestUnacceptedRootDefersDescendants.test_apply_request_on_unaccepted_root_reports_deferred _
tests/standard_names/test_cascade_descendant_reporting.py:146: in test_apply_request_on_unaccepted_root_reports_deferred
    assert result.dry_run is True
E   assert False is True
E    +  where False = CascadeResult(old_name='temperature', new_name='density', renamed=[], skipped=[], conflicts=["successor 'density' is not accepted", "successor 'density' edit is not applied"], total_descendants=0, dry_run=False).dry_run
____ TestNonAppliedPathsReportTrue.test_successor_mismatch_is_not_an_apply _____
tests/standard_names/test_cascade_descendant_reporting.py:224: in test_successor_mismatch_is_not_an_apply
    assert result.dry_run is True
E   assert False is True
E    +  where False = CascadeResult(old_name='temperature', new_name='other', renamed=[], skipped=[], conflicts=["successor_id 'density' does not match new_root 'other'"], total_descendants=0, dry_run=False).dry_run
_____ TestNonAppliedPathsReportTrue.test_unknown_successor_is_not_an_apply _____
tests/standard_names/test_cascade_descendant_reporting.py:236: in test_unknown_successor_is_not_an_apply
    assert result.dry_run is True
E   assert False is True
E    +  where False = CascadeResult(old_name='temperature', new_name='missing', renamed=[], skipped=[], conflicts=["successor 'missing' not found in graph"], total_descendants=0, dry_run=False).dry_run
___ TestNonAppliedPathsReportTrue.test_colliding_rename_root_is_not_an_apply ___
tests/standard_names/test_cascade_descendant_reporting.py:250: in test_colliding_rename_root_is_not_an_apply
    assert result.dry_run is True
E   assert False is True
E    +  where False = CascadeResult(old_name='temperature', new_name='source_rate', renamed=[], skipped=[], conflicts=["destination StandardName 'source_rate' already exists (collision with rename root)"], total_descendants=0, dry_run=False).dry_run
________ TestNonAppliedPathsReportTrue.test_noop_rename_is_not_an_apply ________
tests/standard_names/test_cascade_descendant_reporting.py:263: in test_noop_rename_is_not_an_apply
    assert result.dry_run is True
E   AssertionError: assert False is True
E    +  where False = CascadeResult(old_name='temperature', new_name='temperature', renamed=[], skipped=[], conflicts=['old_name == new_name (no-op rename)'], total_descendants=0, dry_run=False).dry_run
=============================== warnings summary ===============================
../../../../imas-codex/.venv/lib/python3.12/site-packages/_pytest/config/__init__.py:1434
  /home/ITER/mcintos/Code/imas-codex/.venv/lib/python3.12/site-packages/_pytest/config/__init__.py:1434: PytestConfigWarning: Unknown config option: cache_dir
  
    self._warn_or_fail_if_strict(f"Unknown config option: {key}\n")

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
============================= slowest 10 durations =============================
0.48s setup    tests/standard_names/test_cascade_descendant_reporting.py::TestUnacceptedRootDefersDescendants::test_planning_reports_deferral_and_writes_nothing
0.08s call     tests/standard_names/test_cascade_descendant_reporting.py::TestUnacceptedRootDefersDescendants::test_planning_reports_deferral_and_writes_nothing

(8 durations < 0.005s hidden.  Use -vv to show these durations.)
=========================== short test summary info ============================
FAILED tests/standard_names/test_cascade_descendant_reporting.py::TestUnacceptedRootDefersDescendants::test_apply_request_on_unaccepted_root_reports_deferred
FAILED tests/standard_names/test_cascade_descendant_reporting.py::TestNonAppliedPathsReportTrue::test_successor_mismatch_is_not_an_apply
FAILED tests/standard_names/test_cascade_descendant_reporting.py::TestNonAppliedPathsReportTrue::test_unknown_successor_is_not_an_apply
FAILED tests/standard_names/test_cascade_descendant_reporting.py::TestNonAppliedPathsReportTrue::test_colliding_rename_root_is_not_an_apply
FAILED tests/standard_names/test_cascade_descendant_reporting.py::TestNonAppliedPathsReportTrue::test_noop_rename_is_not_an_apply
```

