# Cause of the unsourced chain-cap identities: which write path minted them

## Verdict

Every unsourced `chain_length >= 3` identity with no source-bearing ancestor

was born through one of two root-minting paths, established per identity from

the live graph and the code, not inferred:

| Root-minting path | Distinct identities | Live discriminator | Code anchor |
|---|---:|---|---|
| Legacy generated-name persistence | **48** | At least one chain root has `generated_at`, no `imported_at`, and a generation model. | `graph_ops.write_standard_names` (currently `graph_ops.py:5125`) — pre-ledger writer; its bare `MERGE` persists a name without creating the authoritative `PRODUCED_NAME` edge. |
| Removed bulk catalog-import fold | **36** | Every root carries `origin='catalog_edit'` and the single bulk-write fingerprint `created_at=2026-07-04T21:20:38.632Z`, `imported_at=2026-07-04T21:21:17.079Z`, null `generated_at`, null model. | `catalog_import._write_import_entries`, removed by `4b949931` (2026-07-09); the parent revision's `MERGE (sn:StandardName {id: b.id})` wrote no `StandardNameSource` and no `PRODUCED_NAME`. |
| **Total** | **84** | Identity-collapsed live census, 2026-09-07. | — |

Zero roots carry `origin='derived'`; zero carry `consolidated_at`; none match any

other origin. There is no mixed-root identity in the set.

The composer does not explain the roots. The current generate writer

(`persist_generated_name_winners`, `graph_ops.py:7119`) requires claimed

`source_id`/`source_types` and its contract returns "the finalized

`StandardNameSource` identities" — it cannot emit an unbound name. The roots

therefore entered through the two pre-authoritative writers above, and the

chain cap was reached because refinement propagated an already-unsourced

predecessor: `persist_refined_name` collected the predecessor's (empty) source

cohort and, up to 2026-09-01, accepted the empty no-op.

## Live census (2026-09-07, HEAD `69f24046`)

All figures are read-only queries against the live graph through

`GraphClient`; no graph mutation was attempted.

| Measure | Live result |
|---|---:|
| `StandardName` | 5,108 |
| `chain_length >= 3` | 168 |
| unsourced by absent `PRODUCED_NAME` | 93 |
| sourced | 75 |
| non-empty denormalized `source_paths` | 81 |

Identity-collapsed ancestry partition of the **93** unsourced:

| Disposition | Distinct identities |
|---|---:|
| No source-bearing ancestor anywhere | **84** |
| — of which no `REFINED_FROM` predecessor exists | 1 (`toroidal_diamagnetic_magnetic_flux_at_flux_surface`) |
| Source-bearing ancestor at some depth | **9** |
| — of which immediate-predecessor repair available | 5 |
| — of which source exists only at greater depth | 4 |
| **Total unsourced** | **93** |

The 9 with sourced ancestry:

- Immediate-predecessor repairable (5): `absorbed_plasma_heating_power`,

  `inverse_of_spectral_surface_curvature_of_optical_element`,

  `normalized_toroidal_plasma_beta`, `parallel_neutral_state_convection_velocity`,

  `radial_plasma_momentum_source`.
- Greater-depth only (4): `neutral_internal_state_atomic_power_density_due_to_collisions`,

  `surface_thickness_of_cryostat`,

  `total_momentum_flux_normalized_due_to_perturbed_parallel_vector_potential`,

  `vertical_coordinate_of_plasma_filament`.

## Growth since the recorded census

The recorded 2026-09-01 ancestry census (`unsourced-ancestry.md`) measured 78

unsourced = 72 no-ancestor (37 legacy + 35 catalog) + 4 immediate + 2 deeper +

1 no-predecessor, on 126 chain-cap. The live census is larger and its

membership moved:

- 168 chain-cap (was 126), 93 unsourced (was 78).
- 84 no-ancestor (was 72): legacy **48** (was 37), catalog **36** (was 35).
- Ancestry-bearing 9 (was 6): 5 immediate (was 4, adds `normalized_toroidal_plasma_beta`),

  4 deeper (was 2; `poloidal_magnetic_field_of_magnetic_field_probe` is no longer

  in the set, `surface_thickness_of_cryostat`,

  `total_momentum_flux_normalized_due_to_perturbed_parallel_vector_potential` and

  `vertical_coordinate_of_plasma_filament` are).

The growth is concentrated in the legacy bucket (+11) and includes

chain-cap terminals minted **after** the empty-cohort fix merged — see "Live

minting route" below. The catalog fold cannot mint after 2026-07-09 (the

writer is removed); its +1 is chain growth on older roots, not a new fold.

## Root-level detail

79 distinct roots anchor the 84 identities (chain branching: some roots anchor

more than one chain-cap descendant, and one identity has two roots).

- **47 legacy-generated roots**: 30 carry `origin` null, 17 carry

  `origin='pipeline'`; models span `hosted_vllm/deepseek-v4-flash` (27),

  `openrouter/openai/gpt-5.5` (15), `openrouter/deepseek/deepseek-v4-flash` (10),

  `openrouter/anthropic/claude-opus-4.8` (1); creation dates span the pre-ledger

  generation period from 2026-06-17. Their writer predates the authoritative

  `StandardNameSource -[:PRODUCED_NAME]-> StandardName` ledger: it MERGEd the

  name and wrote the historical `(:IMASNode)-[:HAS_STANDARD_NAME]->(sn)`

  projection, so a generated root carried its source-bearing *input* but no

  ledger edge the current census recognizes.
- **32 catalog-fold roots**: all share the exact `2026-07-04T21:20:38.632Z` /

  `21:21:17.079Z` created/imported fingerprint — one bulk import event on

  2026-07-04, written by the fold path that was removed on 2026-07-09. The fold

  ignored catalog `sources:` and created neither a `StandardNameSource` nor a

  `PRODUCED_NAME`, by its own contract.

## Live minting route (the second question)

**persist_refined_name no longer accepts the empty-cohort no-op for automated

refinement — the refusal has landed and is in HEAD.**

- `graph_ops.persist_refined_name` computes the predecessor's authoritative

  source cohort in its atomic preflight; when the cohort is observed and empty

  on an **automated** refine (no `edit_mode`), it raises

  `RefinedNamePersistenceRefusalReason.AUTHORITATIVE_SOURCE_COHORT_EMPTY` and

  rolls the whole transaction back before minting the successor. The successor

  is never created.
- The change (`9c4403dba` "require refine source provenance") merged to main

  via `5c6c015fb` at **2026-09-01 19:17:55 +02:00**, and is enshrined by

  `test_refine_source_cohort_gate.py::test_empty_authoritative_source_cohort_refuses_successor`.
- A later carve-out (`8ee947b04`, 2026-09-01 21:35 +02:00) deliberately

  preserves **source-less human edits**: the refusal is gated on

  `edit_mode` being unset, and `_allow_empty_noop=(not authoritative_cohort_observed or bool(edit_mode))`.

  The governed `sn edit` rename path (`edit.py`, the sole caller that passes

  `edit_mode`) may still persist an unsourced successor — that is the designed

  exception because a human is explicitly steering.

**Empirical check — the graph shows the fix did not take effect in the runs

that minted successors between 2026-09-01 21:49Z and 2026-09-04 13:54Z.**

Nine automated refine successors (`origin='pipeline'`, `edit_mode` null,

no `run_id`, `created_at === generated_at`, empty source cohort, no

change-ledger record) were minted **after** the refusal merged, e.g.

`accumulated_thermal_coolant_absorbed_energy` (09-02 13:19Z),

`volume_averaged_linear_thermal_electron_decay_time_due_to_disruption`

(09-04 13:54Z), `net_electron_power_density` (09-01 21:49Z). Their node

fingerprint is exactly `persist_refined_name`'s ON-CREATE stamp

(`origin='pipeline'`, `created_at=generated_at=datetime()`, `refine_reason`

populated), so they were written by that function — a `persist_refined_name`

whose empty-cohort no-op was still unconditional, i.e. a **pre-fix runtime

snapshot**. The code at HEAD refuses; a run executing HEAD cannot produce

these nodes. This is the documented environment-lag phenomenon (the pipeline's

editable install can resolve to a tree behind the merged main) rather than the

guard failing.

**Since 2026-09-04 13:54Z, the only unsourced mints in these chains carry

`edit_mode='rename'`, `model='sn-edit'`** (09-05 and 09-07, e.g.

`coolant_absorbed_energy_accumulated_of_plasma_facing_component`,

`flux_surface_averaged_parallel_electric_field_at_separatrix`) — the governed

edit carve-out working as designed, not the automated route.

**Verdict on the second question.** The path that this node's premise feared —

an automated refine silently no-oping an empty cohort and minting an

ungrounded successor — is **no longer live at HEAD**: it now refuses and

rolls back before minting, and the refusal is tested. Two residuals remain:

1. A **runtime-environment lag** can still execute the pre-fix snapshot and

   reproduce the defect despite the merged guard (measured: post-merge mints).

   The guard only helps once the running environment executes HEAD code.
2. The **governed edit carve-out** still permits a source-less successor under

   `edit_mode` by design. Whether that should also refuse is a product

   judgement, because a successor born unsourced carries no signal that it is

   ungrounded regardless of who steered it — the current carve-out trades that

   risk against the legitimate case of a governed edit of an

   already-unsourced name, which has no cohort to migrate anyway.

## Reproduction predicates

Population (unsourced chain-cap with no source-bearing ancestor):

```cypher
MATCH (sn:StandardName)
WHERE sn.chain_length >= 3
  AND NOT EXISTS { MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(sn) }
  AND NOT EXISTS {
    MATCH (sn)-[:REFINED_FROM*1..]->(ancestor:StandardName)
    WHERE EXISTS { MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(ancestor) }
  }
```

Roots per identity were selected by walking `REFINED_FROM*0..` and retaining

nodes with no outgoing `REFINED_FROM`; each root joined to `origin`,

`generated_at`, `imported_at`, `model`, `created_at`, `consolidated_at`, and

incoming/outgoing `HAS_PARENT`. Counts were collapsed per identity (84);

root-level counts (79) are reported separately. All queries were read-only.

## Appendix — identity manifests

**48 legacy-generated roots (no sourced ancestor):**

`accumulated_coolant_absorbed_energy_of_plasma_facing_component`, `beat_length`,

`convected_heat_flux_coefficient`,

`coolant_absorbed_energy_accumulated_of_plasma_facing_component`,

`cumulative_inside_flux_surface_ion_charge_state_source_rate`,

`cumulative_inside_flux_surface_ion_heating_power`,

`deposited_energy_accumulated_of_plasma_facing_component`,

`difference_of_radial_coordinate_and_radial_coordinate_of_outboard_midplane_separatrix`,

`doppler_beat_frequency`, `electron_temperature_peaking_factor`,

`flux_surface_normal_surface_integrated_net_energy_flux_at_plasma_boundary`,

`left_hand_circularly_polarized_wave_fraction`,

`logarithm_of_spectral_signal_to_noise_ratio_of_spectrometer_channel`,

`mean_ion_state_ionisation_potential`,

`net_coefficient_due_to_neoclassical_tearing_mode`,

`neutral_beam_particle_fraction_of_beamlet_group`,

`neutral_hydrogenic_isotope_fraction`, `neutron_flux_at_line_of_sight`,

`parallel_total_pressure_over_edge_region`, `particle_probability`,

`perpendicular_normalized_gyrocenter_heat_perturbed_flux_of_gyrokinetic_eigenmode`,

`plasma_pulse_duration`, `poloidal_accumulated_magnetic_flux_due_to_resistive_dissipation`,

`poloidal_cross_sectional_area_of_plasma_boundary`,

`poloidal_magnetic_flux_perturbed_at_measurement_position_due_to_wave_particle_interaction`,

`poloidal_parity_of_gyrokinetic_eigenmode`,

`poloidal_suprathermal_electron_angle_perturbed_at_measurement_position`,

`radiated_power_at_wall_due_to_surface_emission`,

`ratio_of_ion_average_temperature_to_volume_averaged_ion_average_temperature`,

`ratio_of_neutral_species_gas_count_to_total_gas_count`,

`root_mean_square_of_fluctuating_floating_electrostatic_potential`,

`spectral_bremsstrahlung_radiance`, `spectral_bremsstrahlung_rate`,

`steady_state_total_plasma_absorbed_power`,

`thermal_plasma_field_aligned_power_over_halo_region_due_to_conductive_losses`,

`toroidal_diamagnetic_magnetic_flux_at_flux_surface`,

`toroidal_net_plasma_torque_of_neoclassical_tearing_mode`,

`toroidal_offset_at_measurement_position`, `toroidal_sonic_mach_number`,

`toroidal_total_momentum`, `total_co_passing_pressure`,

`total_current_due_to_fusion_born_alpha`,

`total_launched_wave_power_of_electron_cyclotron_launcher`,

`total_neutral_particle_source_rate_at_wall_due_to_convection`,

`velocity_of_pellet_magnitude`,

`volume_averaged_linear_thermal_electron_decay_time_due_to_disruption`,

`volume_averaged_lithium_fraction`, `wall_gap_of_antenna_strap`

**36 catalog-fold roots (no sourced ancestor):**

`absorbed_coolant_power_of_plant_component_port`,

`curvature_inverse_of_arc_of_circle_center`,

`deposited_power_at_divertor_target`, `energy_flux_at_control_surface`,

`ethylene_count_cumulative_due_to_gas_injection`,

`flux_surface_averaged_parallel_electric_field_at_separatrix`,

`flux_surface_normal_momentum_convection_velocity`,

`flux_surface_normal_neutral_energy_diffusion_coefficient`,

`front_surface_area_of_langmuir_probe`,

`inverse_of_tangential_curvature_of_optical_element`,

`ion_temperature_at_outboard_midplane_separatrix`,

`ion_upper_bound_charge_number`, `lithium_volume_of_breeder_blanket`,

`molecular_gas_count_due_to_pellet_injection`, `net_forward_power_of_wave_beam`,

`net_plasma_power_density`,

`neutral_species_kinetic_energy_flux_at_wall_due_to_surface_emission`,

`non_axisymmetric_current_of_conductor`, `normal_distance_of_antenna_strap`,

`normal_width_of_plasma_filament`,

`parallel_electric_field_flux_surface_averaged_at_separatrix`,

`plasma_electrostatic_potential_at_outboard_midplane`,

`plasma_electrostatic_potential_at_wall`,

`power_over_scrape_off_layer_due_to_radiation`,

`radial_offset_of_lower_hybrid_antenna`,

`root_mean_square_of_spectral_width_of_spectrometer_channel`,

`root_mean_square_spectral_width_of_spectrometer_channel`,

`spectral_width_root_mean_square_of_spectrometer_channel`,

`tangential_curvature_inverse_of_optical_element`,

`tendency_of_runaway_electron_density`, `total_electron_power_density`,

`total_incident_thermal_power`, `total_launched_power_due_to_ion_cyclotron_heating`,

`total_particle_flux_at_divertor_target_due_to_recycling`,

`wave_critical_ordinary_mode_frequency`, `wave_magnetic_field_amplitude`
