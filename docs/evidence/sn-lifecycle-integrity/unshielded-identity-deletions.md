# Unshielded identity deletions and classification

## Scope
- Causal window checked in live graph: `2026-09-08T11:55:49.265Z` to `2026-09-08T11:57:36Z`
- Canonical 78-name set used in this report is the intersection of the 2,096-name `reconcile_catalog_edit_origin` repair window and the `remove_derived_parent` cleanup window.
- Query checks were kept under the login-node-local exception (`titan` tunnel unavailable; all queries executed with `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD` and `GraphClient`).

## Findings
- All 78 deleted names below share this pattern in live graph:
  - `StandardName` node absent
  - live `StandardNameReview` rows: `0` for each identity
  - live `StandardNameSource` rows with `produced_sn_id` matching the identity: `0`
  - one `remove_derived_parent` event in the cleanup window with `origin: pipeline_cleanup` and reason `structural derived parent no longer satisfies lifecycle admission`.
  - names are physical/diagnostic quantities, not an obvious scaffolding-only family.

| name | verdict | publishability | live_review_rows | live_source_link | classification_note |
|---|---|---|---|---|---|
| absorbed_power_of_neutral_beam_injector | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| absorbed_power_of_plant_system | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| angle_of_electron_cyclotron_launcher_mirror | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| argon_density_at_pedestal_top | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| atomic_count_of_pellet | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| atomic_fraction_of_neutron_detector_converter | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| atomic_mass_of_wall_material | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| beryllium_density_at_pedestal_top | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| boron_density_at_pedestal_top | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| bulk_plasma_velocity_due_to_diamagnetic_drift | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| carbon_density_at_pedestal_top | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| count_of_pellet | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| critical_electric_field | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| current_density_due_to_viscosity | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| density_at_pedestal_top | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| density_of_pellet | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| deuterium_density_at_pedestal_top | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| deuterium_deuterium_neutron_flux | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| deuterium_deuterium_neutron_flux_due_to_beam_thermal_fusion | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| deuterium_deuterium_neutron_source_rate_due_to_thermal_fusion | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| deuterium_tritium_density_at_pedestal_top | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| deuterium_tritium_neutron_flux_due_to_beam_thermal_fusion | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| effective_charge_at_pedestal_top | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| electron_density_at_pedestal_top | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| electron_power_density_due_to_collisions | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| energy_convection_velocity | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| energy_flux_at_wall | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| flux_due_to_diamagnetic_drift | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| gyrocenter_pressure | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| helium_3_density_at_pedestal_top | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| helium_4_density_at_pedestal_top | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| hydrogen_density_at_pedestal_top | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| ion_momentum | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| ion_power_density | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| ion_state_energy_flux | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| ion_state_momentum_flux | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| iron_density_at_pedestal_top | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| krypton_density_at_pedestal_top | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| launched_power_of_electron_cyclotron_launcher | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| lithium_density_at_pedestal_top | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| mass_of_wall_material | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| motional_stark_photon_radiance_at_spectral_line | RESTORE | critical | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| neon_density_at_pedestal_top | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| nitrogen_density_at_pedestal_top | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| normalized_gyrocenter_perturbed_pressure | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| normalized_perturbed_vector_potential | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| oxygen_density_at_pedestal_top | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| parallel_normalized_perturbed_vector_potential | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| parity_of_gyrokinetic_eigenmode | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| particle_flux_at_wall | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| particle_flux_at_wall_due_to_recombination | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| perturbed_gyrocenter_pressure | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| perturbed_particle_pressure | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| perturbed_plasma_mass_density | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| perturbed_plasma_pressure | RESTORE | critical | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| perturbed_plasma_temperature | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| perturbed_pressure | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| plasma_current_due_to_ohmic_induction | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| plasma_energy | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| poloidal_angle | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| poloidal_momentum_flux_limiter_coefficient | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| power_at_inner_divertor_target | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| power_at_outer_divertor_target | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| power_at_wall_due_to_recombination | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| power_due_to_fusion | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| power_due_to_radiation | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| radial_momentum_flux_limiter_coefficient | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| safety_factor_at_pedestal_top | RESTORE | critical | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| spectral_etendue_of_spectrometer_channel | RESTORE | critical | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| target_atomic_fraction_of_neutron_detector_converter | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| temperature_at_midplane | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| toroidal_neutral_momentum_flux_limiter_coefficient | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| total_ion_density_at_pedestal_top | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| tritium_density_at_pedestal_top | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| tritium_tritium_neutron_flux | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| tungsten_density_at_pedestal_top | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| voltage_of_neutron_detector | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |
| xenon_density_at_pedestal_top | RESTORE | high | 0 | 0 | Has reconcile-cause + cleanup delete trail; no source/review payload preserved in delete event |

## WEST review cut check
- WEST review manifest sampled: `imas_codex/standard_names/manifests/reviews/v0.10.0rc1+west-task-2e.sn_names.yaml` (214 names).
- WEST overlap with this 78-name set: `0` entries.
- Independently known WEST-cut name deleted by same pass without its own reconcile event in this window:
  - `etendue_of_spectrometer_channel`
  - `StandardName` exists: false
  - `StandardNameReview` rows: `0`
  - `StandardNameSource` rows with `produced_sn_id` to name: `0`
  - `reconcile_catalog_edit_origin` rows: `0`
  - `remove_derived_parent` origin: `pipeline_cleanup` at `2026-09-08T11:57:36`

## Restoration route and publishability rationale
- Restoration is not possible from delete events alone (`StandardNameChange` rows only contain metadata), so restoration requires regeneration from source lineage (DD/source re-ingest + compose path) rather than mutation replay.
- Ranking used:
  - `critical`: identities with prior `unchanged_ratification` evidence (`catalog_promotion`) indicating explicit external review lineage: `motional_stark_photon_radiance_at_spectral_line`, `perturbed_plasma_pressure`, `safety_factor_at_pedestal_top`, `spectral_etendue_of_spectrometer_channel`.
  - `high`: all remaining identities; no live review scaffolding survived and no live source link remained, so publishability depends on successful regeneration.
- Aggregate verdict: all 78 identities are classified `RESTORE`.

## Graph codepath note
- `StandardNameChange` rows in the cleanup phase carry only `from_name`, `to_name`, `operation`, `origin`, `reason`, `changed_at` and omit full node payload; this caused complete forward information loss for deleted node content in this pass.
