# Complete standard-names suite baseline

## Verdict

The standard-names suite baseline at revision
`3d7faa0d61299b7502240268a161555dec7ccfe4` is a completed run with
**118 failed, 7,119 passed, 11 skipped, 322 deselected, and 34 warnings in
316.75 seconds**. The complete failure list contains all 118 identities and
continues through modules sorting after `p`, including `test_release_verify`,
`test_sn_approve`, `test_supersession_source_migration`, and
`test_unit_overrides`.

Nothing was truncated. The three 18-failure executions and both 118-failure
executions all reached `[100%]`. Their missing final counts line was caused by
effective double-quiet output: `pyproject.toml` already supplies `-q`, and each
recorded command added another `-q`, producing `-qq`. The completed run omitted
the command-line `-q` and pytest emitted its own final counts line.

The 100 added failures have a separate, exact cause. Commit
`51f9f9131c1a0620dacd37bbaee46f3d5411fb0e` added
`test_release_import_locality.py`; its second test deletes every loaded
`imas_codex.standard_names.*` module twice and reimports the package without
restoring the original module objects. Tests already collected after it retain
functions from the original modules, while their string-based patches resolve
against the replacement modules. Excluding this one contaminating module turns
exactly those 100 failures back into passes.

## Completed run

The suite ran once on the `all_debug` partition with the repository's shared
environment. `TMPDIR=/tmp` was requested in the `srun` environment and exported
again after the login shell initialized. The compute-node `uv` shell function
supplied the sole `--no-sync`; the command passed no explicit `--no-sync`.

```text
srun --export=ALL,TMPDIR=/tmp --partition=all_debug --time=00:59:00 \
  --cpus-per-task=4 --mem=32G \
  bash -lc 'export TMPDIR=/tmp; cd <worktree>; \
    UV_NO_SYNC=1 \
    UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv \
    PYTHONPATH=$PWD \
    uv run pytest -p no:cacheprovider tests/standard_names/'
```

The durable log is:

`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260908T060914409633-n-sli-the-suite-base-is-a-complete-run/complete-standard-names.log`

Its terminal pytest line is:

```text
118 failed, 7119 passed, 11 skipped, 322 deselected, 34 warnings in 316.75s (0:05:16)
```

The log is 358,973 bytes, contains 118 `FAILED` lines, and exits 1 because the
suite has failures. The following `srun: error ... Exited with exit code 1`
line is Slurm reporting pytest's non-zero exit; it is not evidence that the
stream was interrupted.

## Why the counts line was absent

All four supplied logs reach pytest's terminal reporting sequence: warnings,
the slowest-duration table, and `short test summary info`. More decisively, all
four reach `[100%]`, as does the new completed run. There is no collection
error, per-test timeout, faulthandler termination, xdist worker crash, session
abort, or output truncation.

The counts-line mechanism is pytest verbosity:

- `pyproject.toml` supplies one `-q` in `[tool.pytest.ini_options].addopts`.
- The four earlier suite commands supplied another command-line `-q`.
- Effective `-qq` retains the short failure-ID summary but suppresses pytest's
  final aggregate counts line.
- The completed run retained only the configured `-q`; the final counts line
  appeared without any other output-capture change.

The operator isolated this configuration effect on
`test_release_verify.py`: neutralized `addopts`, configured `-q`, and configured
`-q --durations=10` each emitted final statistics, while configured
`-q --durations=10` plus a command-line `-q` exited 0 without a statistics
line. The repository's documented `pytest tests/standard_names/ -q` form
therefore suppresses the very completion evidence it is intended to collect.

The byte evidence independently rules out a hard capture cap. The earlier
118-failure log is 359,400 bytes and has no counts line. The completed log is
smaller at 358,973 bytes, yet contains the same 118 failure IDs plus the counts
line. The two 118-ID sequences compare equal. A 50 KB cap also cannot explain a
359,400-byte artifact.

The `TMPDIR` diagnostic was a real but independent environment defect. The
login shell exported an unwritable `/run/user/39486` path; Slurm reported the
failure and fell back to `/tmp`, after which `.bashrc` re-exported the bad path
inside the compute-node shell. The operator fixed the source configuration by
exporting `/tmp` in the SLURM branch and verified on `all_debug` that the result
is `/tmp` and writable. That defect neither explains the missing counts line
nor the 100-failure delta.

| Evidence log | Bytes | `FAILED` ids | Pytest counts line | Observation |
|---|---:|---:|---|---|
| `after-tests-standard-names.log` | 50,180 | 18 | absent | effective `-qq`; reaches short summary |
| `after-standard-names.log` | 50,577 | 18 | absent | effective `-qq`; reaches short summary |
| `scratch/after_suite.log` | 50,191 | 18 | absent | effective `-qq`; reaches short summary |
| `/home/ITER/mcintos/after_suite_20260908.log` | 359,400 | 118 | absent | effective `-qq`; all 118 IDs present |
| `complete-standard-names.log` | 358,973 | 118 | present | configured single `-q`; completed base |

## Why 18 failures became 118

The alphabetical boundary is real, but it is a test-order side effect rather
than an execution boundary. The three 18-failure logs predate commit
`51f9f9131c1a0620dacd37bbaee46f3d5411fb0e`. The 118-failure log and this
completed base include its new `test_release_import_locality.py`, which sorts
immediately before `test_release_verify.py`.

`test_both_import_orders_load` removes all keys containing
`imas_codex.standard_names` from `sys.modules`, imports the package, removes
them again, and imports it a second time. Pytest had already collected later
test modules, so their function objects still close over the original module
globals. Patches expressed as import strings subsequently resolve through the
replacement entries in `sys.modules`. For example,
`test_release_verify._patch_gc` patches the replacement
`imas_codex.standard_names.graph_ops.GraphClient`, while the collected release
function executes against the original `graph_ops` globals. The intended mock
does not intercept construction, and the default-tier live-graph guard refuses
the resulting `GraphClient` call. This exact refusal appears in all 26
`test_release_verify.py` failures.

Three discriminators establish the complete effect:

| Selection | Result | Meaning |
|---|---:|---|
| `test_release_verify.py` alone | 30 passed | the file is green in an uncontaminated interpreter |
| `test_release_import_locality.py` then `test_release_verify.py` | 26 failed, 6 passed | the preceding module reproduces all 26 release-verification failures |
| full suite excluding `test_release_import_locality.py` | 18 failed, 7,217 passed, 11 skipped, 322 deselected | all 100 extra failures disappear |

The complete run selects 7,570 outcomes: 118 failed, 7,119 passed, 11 skipped,
and 322 deselected. Excluding the two passing import-locality tests selects
7,568: 18 failed, 7,217 passed, 11 skipped, and 322 deselected. After accounting
for those two excluded tests, exactly 100 shared outcomes change from failure
to pass. Thus the known 18 are the underlying failure set for the shared
selection, and the other 100 are suite-order contamination introduced by the
new test module. No test execution stops at `p`.

The discriminator logs are durable at:

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260908T060914409633-n-sli-the-suite-base-is-a-complete-run/release-verify-isolated.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260908T060914409633-n-sli-the-suite-base-is-a-complete-run/import-locality-contamination.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260908T060914409633-n-sli-the-suite-base-is-a-complete-run/without-import-locality.log`

## Failure census

| Module | Failures |
|---|---:|
| `test_release_verify.py` | 26 |
| `test_supersession_source_migration.py` | 13 |
| `test_unit_overrides.py` | 13 |
| `test_sn_approve.py` | 10 |
| `test_audits.py` | 8 |
| `test_review_dimension_void.py` | 8 |
| `test_seed_all_domains.py` | 8 |
| `test_docs_review_eligibility.py` | 4 |
| `test_supersession_catalog_guard.py` | 3 |
| `test_writer_resilience.py` | 3 |
| Four modules with two failures each | 8 |
| Fourteen modules with one failure each | 14 |
| **Total** | **118** |

## Complete failure list

```text
tests/standard_names/test_audits.py::TestOperatorUnitConsistency::test_equal_dimension_ratios_accept_dimensionless_unit[ratio_of_ion_average_temperature_to_volume_averaged_ion_average_temperature-1]
tests/standard_names/test_audits.py::TestOperatorUnitConsistency::test_equal_dimension_ratios_accept_dimensionless_unit[ratio_of_ion_average_temperature_to_volume_averaged_ion_average_temperature--]
tests/standard_names/test_audits.py::TestOperatorUnitConsistency::test_equal_dimension_ratios_accept_dimensionless_unit[ratio_of_ion_average_temperature_to_volume_averaged_ion_average_temperature-none]
tests/standard_names/test_audits.py::TestCanonicalLocusCheck::test_every_registered_locus_is_never_flagged[reflectometer_cutoff_position]
tests/standard_names/test_audits.py::TestCanonicalLocusCheck::test_every_registered_locus_is_never_flagged[charge_exchange_channel]
tests/standard_names/test_audits.py::TestCanonicalLocusCheck::test_every_registered_locus_is_never_flagged[thomson_scattering_channel_position]
tests/standard_names/test_audits.py::TestCanonicalLocusCheck::test_every_registered_locus_is_never_flagged[langmuir_probe_position]
tests/standard_names/test_audits.py::TestCanonicalLocusCheck::test_every_registered_locus_is_never_flagged[calorimetry_component]
tests/standard_names/test_compose_prompt_examples_round_trip.py::test_endorsed_prompt_name_round_trips[toroidal_surface_integrated_current_density]
tests/standard_names/test_compose_prompt_examples_round_trip.py::test_operator_projection_forms_round_trip[toroidal_surface_integrated_current_density]
tests/standard_names/test_derivation.py::test_time_averaged_of_maximum_of_temperature
tests/standard_names/test_docs_review_eligibility.py::test_winning_methods_are_derived_from_schema
tests/standard_names/test_docs_review_eligibility.py::test_export_gate_and_population_use_shared_traversal
tests/standard_names/test_docs_review_eligibility.py::test_pending_count_and_claim_use_the_same_atomic_predicate
tests/standard_names/test_docs_review_eligibility.py::test_stranded_promotion_uses_shared_traversal
tests/standard_names/test_edit_prompt_injection.py::test_no_edit_render_matches_golden
tests/standard_names/test_error_siblings.py::TestReconcileErrorSiblings::test_reconcile_orphans_error_siblings
tests/standard_names/test_parent_admission.py::TestFilterAdmissibleParentsShadowVeto::test_new_single_batch_projection_is_source_probed_and_preserved
tests/standard_names/test_release_verify.py::test_generate_name_correct_token
tests/standard_names/test_release_verify.py::test_sn_correct_token[enrich]
tests/standard_names/test_release_verify.py::test_sn_correct_token[review_names]
tests/standard_names/test_release_verify.py::test_sn_correct_token[review_docs]
tests/standard_names/test_release_verify.py::test_generate_name_wrong_token_noop
tests/standard_names/test_release_verify.py::test_sn_wrong_token_noop[enrich]
tests/standard_names/test_release_verify.py::test_sn_wrong_token_noop[review_names]
tests/standard_names/test_release_verify.py::test_sn_wrong_token_noop[review_docs]
tests/standard_names/test_release_verify.py::test_generate_name_after_orphan_sweep_noop
tests/standard_names/test_release_verify.py::test_sn_after_orphan_sweep_noop[enrich]
tests/standard_names/test_release_verify.py::test_sn_after_orphan_sweep_noop[review_names]
tests/standard_names/test_release_verify.py::test_sn_after_orphan_sweep_noop[review_docs]
tests/standard_names/test_release_verify.py::test_sn_stage_mismatch_noop[enrich]
tests/standard_names/test_release_verify.py::test_sn_stage_mismatch_noop[review_names]
tests/standard_names/test_release_verify.py::test_sn_stage_mismatch_noop[review_docs]
tests/standard_names/test_release_verify.py::test_generate_name_failed_release
tests/standard_names/test_release_verify.py::test_sn_failed_release_reverts_stage[enrich_failed]
tests/standard_names/test_release_verify.py::test_sn_failed_release_reverts_stage[review_names_failed]
tests/standard_names/test_release_verify.py::test_sn_failed_release_reverts_stage[review_docs_failed]
tests/standard_names/test_release_verify.py::test_sn_failed_release_wrong_token_noop[enrich_failed]
tests/standard_names/test_release_verify.py::test_sn_failed_release_wrong_token_noop[review_names_failed]
tests/standard_names/test_release_verify.py::test_sn_failed_release_wrong_token_noop[review_docs_failed]
tests/standard_names/test_release_verify.py::test_sn_batch_partial_release[enrich]
tests/standard_names/test_release_verify.py::test_sn_batch_partial_release[review_names]
tests/standard_names/test_release_verify.py::test_sn_batch_partial_release[review_docs]
tests/standard_names/test_release_verify.py::test_generate_name_batch_partial_release
tests/standard_names/test_rescore_descendant_acceptance.py::test_rescore_accepts_parent_without_touching_accepted_child
tests/standard_names/test_rescore_descendant_acceptance.py::test_rename_acceptance_still_refuses_conflicting_descendant
tests/standard_names/test_review_artifact_schema.py::test_frozen_artifact_rejects_incomplete_source_accounting
tests/standard_names/test_review_cost.py::test_write_reviews_forwards_cost_and_tokens
tests/standard_names/test_review_dimension_void.py::test_surviving_dimension_scores_are_byte_identical
tests/standard_names/test_review_dimension_void.py::test_review_score_equals_mean_of_surviving_dimensions
tests/standard_names/test_review_dimension_void.py::test_voided_score_survives_in_the_canonical_projection
tests/standard_names/test_review_dimension_void.py::test_void_writes_a_change_ledger_record
tests/standard_names/test_review_dimension_void.py::test_the_reason_is_the_callers_and_the_signature_is_an_observation
tests/standard_names/test_review_dimension_void.py::test_signature_is_captured_at_void_time_when_the_caller_omits_it
tests/standard_names/test_review_dimension_void.py::test_update_review_aggregates_follows_the_void
tests/standard_names/test_review_dimension_void.py::test_second_void_of_the_same_dimension_leaves_one_record
tests/standard_names/test_seed_all_domains.py::TestMixedDomainSkipped::test_mixed_not_seeded
tests/standard_names/test_seed_all_domains.py::TestMixedDomainSkipped::test_non_mixed_domains_are_seeded
tests/standard_names/test_seed_all_domains.py::TestMixedDomainSkipped::test_total_excludes_mixed
tests/standard_names/test_seed_all_domains.py::TestMixedDomainSkipped::test_only_mixed_returns_zero
tests/standard_names/test_seed_all_domains.py::TestMaxSourcesCap::test_stops_after_cap_reached
tests/standard_names/test_seed_all_domains.py::TestMaxSourcesCap::test_no_cap_seeds_all_domains
tests/standard_names/test_seed_all_domains.py::TestMaxSourcesCap::test_cap_exact_boundary
tests/standard_names/test_seed_all_domains.py::TestSourceForwarding::test_source_dd_forwarded
tests/standard_names/test_sn_approve.py::TestRunApprovalPassingReviewPath::test_docs_edit_attached_like_sn_edit_with_reason
tests/standard_names/test_sn_approve.py::TestRunApprovalPassingReviewPath::test_docs_score_at_threshold_stages_for_quorum
tests/standard_names/test_sn_approve.py::test_approval_fold_alone_controls_catalog_status_and_undo_reverses_it
tests/standard_names/test_sn_approve.py::test_reviewed_edit_refused_by_catalog_guard_is_not_reported_as_approved
tests/standard_names/test_sn_approve.py::TestRunApprovalContestPath::test_low_score_contests_not_accepted_not_refined
tests/standard_names/test_sn_approve.py::TestRunApprovalContestPath::test_full_review_runs_but_refine_never_invoked
tests/standard_names/test_sn_approve.py::TestRunApprovalNameEdit::test_name_edit_routes_through_rename_mode
tests/standard_names/test_sn_approve.py::TestRunApprovalEdgeCases::test_unmatched_id_is_reported_without_attaching
tests/standard_names/test_sn_approve.py::TestRunApprovalEdgeCases::test_blocked_edit_is_recorded_and_not_scored
tests/standard_names/test_sn_approve.py::TestRunApprovalEdgeCases::test_dry_run_attaches_nothing
tests/standard_names/test_sn_approve_contested_provenance.py::test_contested_reviewer_edit_retains_merged_pr_provenance
tests/standard_names/test_sn_approve_reviewer_base.py::test_additive_pr_detects_edit_against_cut_time_catalog
tests/standard_names/test_sn_approve_tag.py::TestApprovedCatalogMaterialization::test_contested_entry_is_removed_from_main_by_a_correction_commit
tests/standard_names/test_sn_edit_kind.py::test_rename_derives_successor_kind_instead_of_copying_predecessor
tests/standard_names/test_sn_resolve_materializes.py::test_override_materializes_reviewer_wording_and_pr_provenance
tests/standard_names/test_source_linking.py::TestGraphOpsHelpers::test_mark_sources_composed_cypher_invariants
tests/standard_names/test_source_linking.py::TestGraphOpsHelpers::test_mark_sources_attached_cypher_invariants
tests/standard_names/test_stranded_promotion.py::TestQuorumShortfallIsHonoured::test_parked_name_survives_a_later_maintenance_pass
tests/standard_names/test_structural_authority_replay.py::test_replay_names_entailing_children_and_refuses_ungrounded_parent
tests/standard_names/test_supersession.py::TestSupersedePriorSourceNamesRecording::test_records_live_stage
tests/standard_names/test_supersession.py::TestPersistRefinedNameRecording::test_records_published_signal
tests/standard_names/test_supersession_catalog_guard.py::test_predecessor_selection_excludes_published_catalog_content
tests/standard_names/test_supersession_catalog_guard.py::test_predecessor_selection_still_excludes_terminal_stages
tests/standard_names/test_supersession_catalog_guard.py::test_structural_parents_stay_ineligible
tests/standard_names/test_supersession_source_migration.py::TestUnjudgedSourcesStayBehind::test_only_the_judged_set_is_retargeted
tests/standard_names/test_supersession_source_migration.py::TestUnjudgedSourcesStayBehind::test_predecessor_is_not_superseded_while_sources_remain
tests/standard_names/test_supersession_source_migration.py::TestUnjudgedSourcesStayBehind::test_predecessor_path_projection_is_rebuilt
tests/standard_names/test_supersession_source_migration.py::TestUnjudgedSourcesStayBehind::test_refusal_is_recorded_in_the_change_ledger
tests/standard_names/test_supersession_source_migration.py::TestJudgedSourcesMigrate::test_recomposed_source_carries_across
tests/standard_names/test_supersession_source_migration.py::TestJudgedSourcesMigrate::test_sources_already_on_the_successor_travel_with_it
tests/standard_names/test_supersession_source_migration.py::TestFailClosed::test_missing_successor_binding_for_the_recomposed_source_refuses
tests/standard_names/test_supersession_source_migration.py::TestFailClosed::test_guard_rejection_still_rolls_back_the_refused_pass
tests/standard_names/test_supersession_source_migration.py::TestPreflightContract::test_judged_set_is_defined_by_a_successor_binding
tests/standard_names/test_supersession_source_migration.py::TestPreflightContract::test_unjudged_sources_are_enumerated_not_dropped
tests/standard_names/test_supersession_source_migration.py::TestPreflightContract::test_recomposed_source_is_resolved_through_its_dd_edge
tests/standard_names/test_supersession_source_migration.py::TestPreflightContract::test_the_dd_edge_alone_does_not_single_out_the_recomposed_source
tests/standard_names/test_supersession_source_migration.py::TestPreflightContract::test_protected_predecessors_stay_excluded
tests/standard_names/test_unit_overrides.py::TestResolveUnitOverrides::test_e_on_ionisation_potential_becomes_ev
tests/standard_names/test_unit_overrides.py::TestResolveUnitOverrides::test_unit_vector_m_overridden_to_dimensionless[bolometer/camera/channel/aperture/x1_unit_vector/x]
tests/standard_names/test_unit_overrides.py::TestResolveUnitOverrides::test_unit_vector_m_overridden_to_dimensionless[mse/channel/detector/x2_unit_vector/y]
tests/standard_names/test_unit_overrides.py::TestResolveUnitOverrides::test_unit_vector_m_overridden_to_dimensionless[reflectometer_fluctuation/channel/antennas_orientation/antenna_detection/x1_unit_vector/z]
tests/standard_names/test_unit_overrides.py::TestResolveUnitOverrides::test_unit_vector_m_overridden_to_dimensionless[spi/injector/shatter_cone/unit_vector_major/x]
tests/standard_names/test_unit_overrides.py::TestResolveUnitOverrides::test_unit_vector_m_overridden_to_dimensionless[nbi/unit/source/x3_unit_vector/y]
tests/standard_names/test_unit_overrides.py::TestResolveUnitOverrides::test_direction_vector_m_overridden_to_dimensionless[camera_ir/channel/camera/direction/x]
tests/standard_names/test_unit_overrides.py::TestResolveUnitOverrides::test_direction_vector_m_overridden_to_dimensionless[ec_launchers/mirror/movement/direction/y]
tests/standard_names/test_unit_overrides.py::TestResolveUnitOverrides::test_direction_vector_m_overridden_to_dimensionless[operational_instrumentation/sensor/direction/z]
tests/standard_names/test_unit_overrides.py::TestResolveUnitOverrides::test_direction_vector_m_overridden_to_dimensionless[spi/injector/shatter_cone/direction/x]
tests/standard_names/test_unit_overrides.py::TestResolveUnitOverrides::test_direction_vector_m_overridden_to_dimensionless[operational_instrumentation/sensor/direction_second/y]
tests/standard_names/test_unit_overrides.py::TestResolveUnitOverrides::test_direction_vector_m_overridden_to_dimensionless[camera_ir/channel/camera/up/z]
tests/standard_names/test_unit_overrides.py::TestResolveUnitOverrides::test_direction_vector_m_overridden_to_dimensionless[spi/injector/shatter_cone/injection_direction/x]
tests/standard_names/test_vocab_consumers_see_operators.py::TestTokenFilterSeesOperators::test_degrades_to_empty_without_the_grammar
tests/standard_names/test_writer_resilience.py::test_write_single_timeout_then_succeed
tests/standard_names/test_writer_resilience.py::test_heartbeat_fires_debug_when_idle
tests/standard_names/test_writer_resilience.py::test_heartbeat_fires_info_when_pending
```

## Scope boundary

This report establishes the completed 118-failure baseline, the output-format
mechanism, and the single-test cause of the 100-failure delta. Repairing
`test_release_import_locality.py` is outside this node's exclusive write path.
The remaining 18 failures also span production and test files outside this
scope and require separately scoped ownership.
