# Cascade report-vs-persistence contract — before/after evidence

The defect class: a subtree rename plans and prints its descendant renames,
then persists them only if the root itself reaches `accepted` — and the
surface said one thing while the graph stored another. Measured 2026-09-03 on
`emissivity_due_to_fusion` (exhausted at 0.825), whose children
`deuterium_deuterium_emissivity_due_to_fusion` and
`deuterium_tritium_emissivity_due_to_fusion` were reported as planned and
remained unchanged and still flagged.

The repair (landed in earlier work on this path) made the report honest and
kept the write path unchanged: the discriminator is `CascadeResult.dry_run`,
`True` on every return that wrote nothing, `False` only where the renames in
`renamed` were persisted. This document records the before/after output for
both halves of the contract plus the linear three-level chain plan-construction
case, with the test names that hold each. Reproduced and captured 2026-09-10
with the in-memory `FakeGraph` (no live graph needed); the full before/after
runs appear verbatim below.

## 1. Exhausted root (root does NOT reach `accepted`) — defer, never claim

Measured shape: the root has exhausted (rotation cap spent) so it will not
accept here. The subtree rename plans the two descendants, but the plan is
deferred work awaiting the root's acceptance — nothing is written.

Test that holds this case:
`tests/standard_names/test_cascade_atomicity.py::TestReportMatchesPersistence::test_unaccepted_root_defers_descendants_and_writes_nothing`.

```
graph BEFORE (ids present):
['deuterium_deuterium_emissivity_due_to_fusion', 'deuterium_tritium_emissivity_due_to_fusion', 'emissivity_due_to_fusion']

CascadeResult (subtree rename planned over the exhausted root):
dry_run = True
conflicts = []
renamed  = [{'from': 'emissivity_due_to_fusion', 'to': 'source_rate'},
            {'from': 'deuterium_deuterium_emissivity_due_to_fusion', 'to': 'deuterium_deuterium_source_rate'},
            {'from': 'deuterium_tritium_emissivity_due_to_fusion', 'to': 'deuterium_tritium_source_rate'}]

graph AFTER (ids present):
['deuterium_deuterium_emissivity_due_to_fusion', 'deuterium_tritium_emissivity_due_to_fusion', 'emissivity_due_to_fusion']
descendants unchanged: True
successor ids NOT created: True
```

The descending rows are carried (the operator must see the consequence of the
edit they propose) but `dry_run=True` is the proof they are a plan, not a
result. The CLI rendering of those rows names the deferral explicitly:

```
Cascade (deferred descendant renames — not yet applied):
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Current                                      ┃ Becomes on acceptance           ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ deuterium_deuterium_emissivity_due_to_fusion │ deuterium_deuterium_source_rate │
│ deuterium_tritium_emissivity_due_to_fusion   │ deuterium_tritium_source_rate   │
└──────────────────────────────────────────────┴─────────────────────────────────┘
  awaiting source_rate reaching accepted; these ids are unchanged until then, and stay unchanged if it is withheld or exhausted
```

Tests holding the deferral message:
`test_cascade_atomicity.py::TestReportMatchesPersistence::test_deferral_message_names_what_it_waits_on` and (the renderer pins) `tests/standard_names/test_rename_cascade.py::TestDeferredCascadeIsReportedAsDeferred::test_render_names_the_deferral_and_what_it_waits_on`.

## 2. Accepted root (root reaches `accepted`) — the printed rows are persisted

The same two descendants, once the root's successor reaches `accepted` during
an `imas-codex sn edit` subtree acceptance, are renamed in the same atomic
write and the result reports `dry_run=False`.

Test holding this case:
`tests/standard_names/test_cascade_atomicity.py::TestReportMatchesPersistence::test_accepted_root_persists_exactly_what_it_reports`
and the acceptance-gate test `TestCascadeAtomicity::test_clean_cascade_applies_on_acceptance`.

```
graph BEFORE:
['density', 'electron_temperature', 'ion_temperature', 'temperature']
name_stage after acceptance = accepted
graph AFTER:
['density', 'electron_density', 'ion_density', 'temperature']
descendants persisted: True
```

## 3. Strictly linear three-level chain — plan construction, not a topology fault

A second failure in the same builder emitted `unreachable in cascade — no
parent in plan` on a strictly linear, non-branching three-level chain. The
real cause was a middle node leaving the plan deliberately at a semantic
boundary or safety refusal, which the builder misclassified as a topology
fault; a child below a stopped ancestor is reachable and belongs in `skipped`
with the stopping ancestor's reason, not in `conflicts`. A fully provable
linear chain plans through its whole depth.

Tests holding this case:
`tests/standard_names/test_cascade_atomicity.py::TestLinearChainPlanConstruction::{test_linear_three_level_chain_plans_without_unreachable, test_subtree_stopped_at_a_boundary_is_skipped_not_a_fault}` and (the earlier pins) `tests/standard_names/test_rename_cascade.py::TestDeepChainPlanConstruction`.

```
linear chain (temperature ← ion_temperature ← core_ion_temperature):
conflicts = []
renamed  = [{'from': 'temperature', 'to': 'temperature_of_plasma_boundary'},
            {'from': 'ion_temperature', 'to': 'ion_temperature_of_plasma_boundary'},
            {'from': 'core_ion_temperature', 'to': 'core_ion_temperature_of_plasma_boundary'}]
no 'unreachable in cascade' conflict: True
```

## 4. Live exposure during the 2026-09-10 tail drain (refine_name)

The drain ran 18 `refine_name` persistence operations (verified from
`~/.local/share/imas-codex/logs/sn_sn-compose.log`, one per
`persist_refined_name:` line, 2026-09-10 06:21–06:40Z):

| # | Predecessor (refined) | Successor (minted) | chain |
|---|---|---|---|
| 1 | line_integrated_opacity | line_integrated_opacity_at_ece_channel_emission_position | 1 |
| 2 | response_function_of_detector | neutron_response_function_of_neutron_detector | 1 |
| 3 | effective_coefficient_of_wall_material_due_to_sputtering | effective_incident_neutral_coefficient_of_wall_material_due_to_sputtering | 1 |
| 4 | atomic_count_of_pellet_injector | gas_atomic_count_of_pellet_injector | 1 |
| 5 | accumulated_thermal_energy | accumulated_thermal_coolant_absorbed_energy_of_calorimetry_component | 1 |
| 6 | alpha_angle_of_passive_loop | alpha_angle_of_passive_loop_element | 1 |
| 7 | second_local_tangential_width_of_neutron_detector | second_local_tangential_width_of_diagnostic_aperture | 1 |
| 8 | maximum_of_neutron_flux | neutron_flux_maximum_at_first_wall | 1 |
| 9 | field_line_fraction | field_line_breakdown_fraction | 1 |
| 10 | line_integrated_opacity_at_ece_channel_emission_position | opacity_at_ece_channel_emission_position | 2 |
| 11 | acceleration | acceleration_of_passive_structure | 1 |
| 12 | volumetric_fraction_of_passive_structure | ratio_of_volume_of_wall_material_to_volume_of_passive_structure | 1 |
| 13 | accumulated_thermal_coolant_absorbed_energy_of_calorimetry_component | accumulated_coolant_absorbed_energy_of_calorimetry_component | 2 |
| 14 | strain | strain_of_strain_gauge | 1 |
| 15 | diameter_of_line_of_sight | viewing_diameter_of_line_of_sight | 1 |
| 16 | poloidal_linear_magnetic_field_of_magnetic_field_probe | poloidal_linear_calibration_reference_magnetic_field_at_measurement_position | 1 |
| 17 | curvature_of_electron_cyclotron_beam | wave_curvature_of_electron_cyclotron_beam | 1 |
| 18 | field_line_breakdown_fraction | ratio_of_field_line_count_to_total_field_line_count | 2 |

Findings:

- **The false-`planned` half is inactive in the drain.** The drain's pool path
  contains no cascade planning or printing at all: zero `cascade` / `descendant`
  log lines across the whole run, and the `refine_name` pool refines single
  identities as plain name rotations (no `edit_scope`). Nothing was reported as
  planned that was not persisted, because nothing about descendants was reported.
- **The persistence half is structurally exposed, silently.** Both cascade gates —
  preflight (`graph_ops.persist_reviewed_name`, requires
  `edit_scope in ("family", "subtree")`) and the acceptance apply — never fire
  for a pipeline-refined successor, because the pool path never sets that scope.
  So whether or not a refined identity is a parent, its `HAS_PARENT` descendants
  (if any) keep ids derived from the old parent spelling: the refine re-points
  the children's edges at the successor but no descendant rename follows, not
  even when the successor later reaches `accepted`. Three of the 18 successors
  reached `accepted` during the drain
  (`neutron_response_function_of_neutron_detector` @0.988,
  `alpha_angle_of_passive_loop_element` @1.0,
  `opacity_at_ece_channel_emission_position` @1.0, chain 2); several others
  reached `reviewed` and then exhausted (e.g.
  `gas_atomic_count_of_pellet_injector`,
  `field_line_breakdown_fraction` @0.5).
- **Which of the 18 are parents with descendants is not establishable this
  session.** The live graph host (`iter`) is unreachable (ssh connection
  refused), and the local exports are Neo4j admin dumps requiring a server, so
  the `HAS_PARENT` topology of these 18 identities could not be read. Repair of
  any such rows is out of this node's scope; the bounded read to close the gap
  is one query over the 18 predecessors counting their live `HAS_PARENT`
  descendants, to run when the graph tunnel is back.

## Tests that hold this document

Run 10 in `tests/standard_names/test_cascade_atomicity.py` (all pass, exit 0,
2026-09-10): the five acceptance-gate tests
(`TestCascadeAtomicity`) plus `TestReportMatchesPersistence`
(3 tests: deferred-and-unwritten, persisted-when-accepted, deferral-message)
plus `TestLinearChainPlanConstruction` (2 tests: full linear plan, boundary
stop is a skip). Supporting pins live in
`tests/standard_names/test_cascade_descendant_reporting.py`
(report-vs-persistence contract over `dry_run`) and
`tests/standard_names/test_rename_cascade.py`
(deep-chain plan construction + deferral rendering).
