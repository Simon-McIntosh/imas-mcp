# Reference-edge materialisation

## Outcome

At `2026-09-09T08:16:31.775Z`, the runtime census found 2,949
`StandardName` nodes carrying a non-empty `links` scalar and 6,684 scalar link
entries. The 2,948-name and 6,682-link figures measured on 2026-09-08 are a
prior reading, not a fixed population authority.

All 6,557 entries whose target currently exists were materialised as
`(StandardName)-[:REFERENCES]->(StandardName)` relationships at
`2026-09-09T08:16:32.099Z`. The other 127 entries remain only in the scalar and
are named individually in the exclusion ledger below. No placeholder target
was created, no missing target was adjudicated, and no scalar entry was
dropped.

| Measure | Pre-write runtime value | After-write value | Verdict |
|---|---:|---:|---|
| Standard names with non-empty `links` | 2,949 | 2,949 | equal |
| Scalar link entries: full parity target | 6,684 | 6,684 | equal |
| Resolvable scalar link entries | 6,557 | 6,557 | equal |
| `REFERENCES` relationships | 0 | 6,557 | equal to every resolvable scalar pair |
| Unresolved scalar links | 127 | 127 | preserved in the exclusion ledger |
| Source names out of resolvable-pair parity | not measured | 0 | pass |
| `REFERENCES` edges not backed by the current scalar | 0 | 0 | pass |
| Unsupported prefixes | 0 | 0 | pass |
| Empty target identifiers | 0 | 0 | pass |
| Duplicate source/link pairs | 0 | 0 | pass |

The after-write edge count is therefore 6,557 of the current full-parity target
of 6,684, or 98.10%. This is complete materialisation of the resolvable cohort,
not full relationship parity. The DERIVE beat remains forbidden until all
runtime scalar entries resolve and the relationship count reaches the runtime
full-parity target.

The scalar census was captured immediately before and after materialisation as
2,949 ordered rows of `id` plus `links`. Both snapshots have SHA-256
`d6bbac9c09de54722359df245cab5df135fe391b20dcd5e9815feb77be7beeb2`,
so no scalar value was modified by this node. The DERIVE beat was not performed.

## Measurement boundary

The census used the schema-declared `StandardName.id`, `StandardName.links`,
and `StandardName.references` fields and the authored relationship direction.
It ran against graph `codex` through the ordinary `GraphClient`, which resolved
`bolt://98dci4-gpu-0002:7687`. Every query began from `StandardName`, bounded
its work to the populated `links` cohort, returned aggregates or at most the
complete 127-row unresolved set, and completed in at most 0.341 seconds. The
login-node placement was required because the live graph connection is not
available to a compute allocation.

This is the gate for this node only. No product test suite was run here; merged
verification belongs to the separately dispatched test node.

## Unresolved links

Every scalar entry has the supported `name:` prefix and a non-empty target id.
Each row below is unresolved for the same explicit reason: no live
`StandardName` node has the target id.

| # | Source standard name | Absent target id | Reason |
|---:|---|---|---|
| 1 | `argon_density_at_plasma_boundary` | `argon_density_at_pedestal` | no StandardName node has the target id |
| 2 | `bulk_plasma_velocity_due_to_diamagnetic_drift_magnitude` | `bulk_plasma_velocity_due_to_diamagnetic_drift` | no StandardName node has the target id |
| 3 | `coolant_temperature` | `temperature_at_inlet` | no StandardName node has the target id |
| 4 | `coolant_temperature` | `temperature_at_outlet` | no StandardName node has the target id |
| 5 | `critical_momentum_due_to_avalanche` | `critical_electric_field` | no StandardName node has the target id |
| 6 | `current_due_to_ohmic_induction` | `plasma_current_due_to_ohmic_induction` | no StandardName node has the target id |
| 7 | `deuterium_deuterium_neutron_flux_due_to_beam_beam_fusion` | `neutron_flux_due_to_beam_beam_fusion` | no StandardName node has the target id |
| 8 | `deuterium_tritium_neutron_flux_due_to_beam_beam_fusion` | `neutron_flux_due_to_beam_beam_fusion` | no StandardName node has the target id |
| 9 | `effective_charge_at_plasma_boundary` | `charge_at_plasma_boundary` | no StandardName node has the target id |
| 10 | `efficiency_of_spectrometer_channel` | `spectral_etendue_of_spectrometer_channel` | no StandardName node has the target id |
| 11 | `electron_energy_flux_at_wall` | `energy_flux_at_wall` | no StandardName node has the target id |
| 12 | `electron_particle_flux_at_wall` | `particle_flux_at_wall` | no StandardName node has the target id |
| 13 | `electron_pressure_at_pedestal_top` | `electron_density_at_pedestal_top` | no StandardName node has the target id |
| 14 | `electron_temperature_at_midplane` | `temperature_at_midplane` | no StandardName node has the target id |
| 15 | `energy_density` | `plasma_energy` | no StandardName node has the target id |
| 16 | `fast_electron_source_rate_due_to_hot_tail` | `electron_source_rate_due_to_hot_tail` | no StandardName node has the target id |
| 17 | `fast_neutral_internal_state_number_density` | `neutral_internal_state_number_density` | no StandardName node has the target id |
| 18 | `flux_due_to_beam_beam_fusion` | `neutron_flux_due_to_beam_beam_fusion` | no StandardName node has the target id |
| 19 | `flux_due_to_recombination` | `particle_flux_at_wall_due_to_recombination` | no StandardName node has the target id |
| 20 | `flux_surface_averaged_carbon_density` | `ratio_of_carbon_density_to_electron_density` | no StandardName node has the target id |
| 21 | `flux_surface_averaged_deuterium_tritium_density` | `flux_surface_averaged_total_ion_density` | no StandardName node has the target id |
| 22 | `fraction_of_neutron_detector_converter` | `atomic_fraction_of_neutron_detector_converter` | no StandardName node has the target id |
| 23 | `gradient_of_radial_electron_density` | `radial_electron_density` | no StandardName node has the target id |
| 24 | `ion_average_temperature_at_magnetic_axis` | `average_temperature_at_magnetic_axis` | no StandardName node has the target id |
| 25 | `ion_particle_flux_at_wall` | `particle_flux_at_wall` | no StandardName node has the target id |
| 26 | `ion_state_kinetic_energy_flux_at_wall_due_to_surface_emission` | `energy_flux_at_wall` | no StandardName node has the target id |
| 27 | `ion_state_momentum_convection_velocity` | `radial_plasma_effective_momentum_convection_velocity` | no StandardName node has the target id |
| 28 | `ion_state_particle_flux` | `ion_state_energy_flux` | no StandardName node has the target id |
| 29 | `ion_state_particle_flux_at_wall` | `particle_flux_at_wall` | no StandardName node has the target id |
| 30 | `ion_state_temperature` | `edge_ion_average_temperature` | no StandardName node has the target id |
| 31 | `ion_temperature` | `ion_temperature_at_wall` | no StandardName node has the target id |
| 32 | `ion_temperature_at_midplane` | `temperature_at_midplane` | no StandardName node has the target id |
| 33 | `maximum_of_energy_flux_at_first_wall` | `energy_flux_at_wall` | no StandardName node has the target id |
| 34 | `maximum_of_energy_flux_at_limiter` | `energy_flux_at_wall` | no StandardName node has the target id |
| 35 | `maximum_power_at_inner_divertor_target` | `power_at_inner_divertor_target` | no StandardName node has the target id |
| 36 | `maximum_power_at_outer_divertor_target` | `power_at_outer_divertor_target` | no StandardName node has the target id |
| 37 | `neon_prefill_count` | `xenon_prefill_count` | no StandardName node has the target id |
| 38 | `net_absorbed_power_of_plant_system` | `absorbed_power_of_plant_system` | no StandardName node has the target id |
| 39 | `neutral_energy_flux_at_wall` | `energy_flux_at_wall` | no StandardName node has the target id |
| 40 | `neutral_internal_state_momentum_flux_limiter_coefficient_over_edge_region` | `internal_state_momentum_flux_limiter_coefficient_over_edge_region` | no StandardName node has the target id |
| 41 | `neutral_particle_flux_at_wall` | `particle_flux_at_wall` | no StandardName node has the target id |
| 42 | `neutral_power_at_wall_due_to_recombination` | `power_at_wall_due_to_recombination` | no StandardName node has the target id |
| 43 | `neutral_state_energy_convection_velocity` | `energy_convection_velocity` | no StandardName node has the target id |
| 44 | `neutral_state_energy_flux` | `ion_state_energy_flux` | no StandardName node has the target id |
| 45 | `neutral_state_energy_flux_at_wall` | `energy_flux_at_wall` | no StandardName node has the target id |
| 46 | `neutral_state_energy_flux_due_to_recombination` | `energy_flux_due_to_recombination` | no StandardName node has the target id |
| 47 | `neutral_state_particle_flux_at_wall` | `particle_flux_at_wall` | no StandardName node has the target id |
| 48 | `neutron_flux` | `deuterium_deuterium_neutron_flux` | no StandardName node has the target id |
| 49 | `neutron_flux` | `tritium_tritium_neutron_flux` | no StandardName node has the target id |
| 50 | `neutron_rate_of_neutron_detector` | `rate_of_neutron_detector` | no StandardName node has the target id |
| 51 | `neutron_source_rate_due_to_beam_beam_fusion` | `neutron_flux_due_to_beam_beam_fusion` | no StandardName node has the target id |
| 52 | `normalized_atomic_count_of_pellet` | `atomic_count_of_pellet` | no StandardName node has the target id |
| 53 | `normalized_atomic_count_of_pellet` | `count_of_pellet` | no StandardName node has the target id |
| 54 | `normalized_perturbed_density_imaginary_part` | `normalized_perturbed_density` | no StandardName node has the target id |
| 55 | `normalized_perturbed_pressure` | `perturbed_pressure` | no StandardName node has the target id |
| 56 | `normalized_total_particle_perturbed_pressure_of_gyrokinetic_eigenmode` | `normalized_gyrocenter_perturbed_pressure` | no StandardName node has the target id |
| 57 | `nuclear_power_density_of_breeder_blanket_module` | `power_density_of_breeder_blanket_module` | no StandardName node has the target id |
| 58 | `outer_atomic_count_of_pellet` | `atomic_count_of_pellet` | no StandardName node has the target id |
| 59 | `oxygen_density_at_limiter` | `ion_density_at_limiter` | no StandardName node has the target id |
| 60 | `parallel_effective_neutral_internal_state_velocity_due_to_diamagnetic_drift` | `effective_neutral_internal_state_velocity_due_to_diamagnetic_drift` | no StandardName node has the target id |
| 61 | `parallel_momentum_flux_due_to_perturbed_parallel_vector_potential` | `parallel_momentum_flux_due_to_perturbed_parallel_magnetic_field` | no StandardName node has the target id |
| 62 | `parallel_momentum_flux_due_to_perturbed_parallel_vector_potential` | `perpendicular_momentum_flux_due_to_perturbed_parallel_vector_potential` | no StandardName node has the target id |
| 63 | `parallel_normalized_perturbed_vector_potential_amplitude` | `parallel_normalized_perturbed_vector_potential` | no StandardName node has the target id |
| 64 | `peak_voltage_of_neutron_detector` | `requested_upper_voltage_of_neutron_detector` | no StandardName node has the target id |
| 65 | `peak_voltage_of_neutron_detector` | `voltage_of_neutron_detector` | no StandardName node has the target id |
| 66 | `perpendicular_momentum_flux_due_to_perturbed_parallel_magnetic_field` | `momentum_flux_due_to_perturbed_parallel_magnetic_field` | no StandardName node has the target id |
| 67 | `perpendicular_momentum_flux_due_to_perturbed_parallel_magnetic_field` | `parallel_momentum_flux_due_to_perturbed_parallel_magnetic_field` | no StandardName node has the target id |
| 68 | `perpendicular_normalized_gyrocenter_perturbed_pressure` | `normalized_gyrocenter_perturbed_pressure` | no StandardName node has the target id |
| 69 | `perturbed_vector_potential` | `normalized_perturbed_vector_potential` | no StandardName node has the target id |
| 70 | `perturbed_vector_potential` | `parallel_normalized_perturbed_vector_potential` | no StandardName node has the target id |
| 71 | `plasma_velocity_due_to_diamagnetic_drift` | `bulk_plasma_velocity_due_to_diamagnetic_drift` | no StandardName node has the target id |
| 72 | `poloidal_current_density_due_to_collisions` | `current_density_due_to_collisions` | no StandardName node has the target id |
| 73 | `poloidal_current_density_due_to_viscosity` | `current_density_due_to_viscosity` | no StandardName node has the target id |
| 74 | `poloidal_ion_momentum` | `poloidal_momentum` | no StandardName node has the target id |
| 75 | `poloidal_ion_state_energy_diffusivity` | `poloidal_ion_state_energy_diffusion_coefficient` | no StandardName node has the target id |
| 76 | `poloidal_ion_state_momentum` | `poloidal_momentum` | no StandardName node has the target id |
| 77 | `poloidal_ion_state_momentum_flux` | `ion_state_momentum_flux` | no StandardName node has the target id |
| 78 | `poloidal_ion_state_velocity_due_to_e_cross_b_drift` | `parallel_ion_state_velocity_due_to_e_cross_b_drift` | no StandardName node has the target id |
| 79 | `poloidal_neutral_state_momentum` | `poloidal_momentum` | no StandardName node has the target id |
| 80 | `power_at_wall_due_to_conduction` | `energy_flux_at_wall` | no StandardName node has the target id |
| 81 | `power_of_divertor_due_to_fusion` | `power_due_to_fusion` | no StandardName node has the target id |
| 82 | `power_of_divertor_due_to_radiation` | `power_due_to_radiation` | no StandardName node has the target id |
| 83 | `power_of_neutral_beam_injector` | `absorbed_power_of_neutral_beam_injector` | no StandardName node has the target id |
| 84 | `radial_centroid_of_electron_cyclotron_launcher_mirror` | `angle_of_electron_cyclotron_launcher_mirror` | no StandardName node has the target id |
| 85 | `radial_current_density_due_to_viscosity` | `current_density_due_to_viscosity` | no StandardName node has the target id |
| 86 | `radial_derivative_of_poloidal_ion_state_velocity` | `second_radial_derivative_of_poloidal_ion_state_velocity` | no StandardName node has the target id |
| 87 | `radial_effective_total_ion_energy_convection_velocity` | `radial_ion_momentum_effective_convection_velocity` | no StandardName node has the target id |
| 88 | `radial_energy_convection_velocity` | `energy_convection_velocity` | no StandardName node has the target id |
| 89 | `radial_ion_state_momentum_flux` | `ion_state_momentum_flux` | no StandardName node has the target id |
| 90 | `radial_neutral_state_energy_diffusion_coefficient` | `radial_ion_state_energy_diffusion_coefficient` | no StandardName node has the target id |
| 91 | `radial_neutral_state_momentum_convection_velocity` | `radial_neutral_species_momentum_convection_velocity` | no StandardName node has the target id |
| 92 | `radial_plasma_momentum_source` | `plasma_momentum_source` | no StandardName node has the target id |
| 93 | `radiance_at_spectral_line` | `motional_stark_radiance_at_spectral_line` | no StandardName node has the target id |
| 94 | `ratio_of_coolant_mass_to_time` | `coolant_mass` | no StandardName node has the target id |
| 95 | `reference_calibration_wavelength_of_spectrometer_channel` | `wavelength_of_spectrometer_channel` | no StandardName node has the target id |
| 96 | `second_radial_derivative_of_poloidal_ion_velocity` | `second_radial_derivative_of_poloidal_ion_state_velocity` | no StandardName node has the target id |
| 97 | `second_radial_derivative_of_toroidal_ion_state_velocity` | `toroidal_ion_state_velocity` | no StandardName node has the target id |
| 98 | `source_rate_due_to_thermal_fusion` | `neutron_power` | no StandardName node has the target id |
| 99 | `thermal_electron_power` | `absorbed_power` | no StandardName node has the target id |
| 100 | `time_derivative_of_electron_temperature` | `time_derivative_of_ion_state_temperature` | no StandardName node has the target id |
| 101 | `time_derivative_of_electron_temperature` | `time_derivative_of_ion_temperature` | no StandardName node has the target id |
| 102 | `time_derivative_of_total_electron_density` | `time_derivative_of_total_ion_density` | no StandardName node has the target id |
| 103 | `time_derivative_of_total_ion_state_density` | `time_derivative_of_fast_ion_state_density` | no StandardName node has the target id |
| 104 | `time_derivative_of_total_ion_state_density` | `time_derivative_of_total_ion_density` | no StandardName node has the target id |
| 105 | `toroidal_beryllium_velocity_at_plasma_boundary` | `toroidal_beryllium_velocity_at_pedestal` | no StandardName node has the target id |
| 106 | `toroidal_co_passing_thermal_ion_state_torque_density_due_to_collisions` | `co_passing_thermal_ion_state_torque_density_due_to_collisions` | no StandardName node has the target id |
| 107 | `toroidal_ion_momentum` | `ion_momentum` | no StandardName node has the target id |
| 108 | `toroidal_ion_state_momentum_flux_limiter_coefficient` | `poloidal_ion_state_momentum_coefficient` | no StandardName node has the target id |
| 109 | `toroidal_ion_state_momentum_flux_limiter_coefficient` | `toroidal_neutral_momentum_coefficient` | no StandardName node has the target id |
| 110 | `toroidal_neutral_state_momentum_flux_limiter_coefficient` | `neutral_state_momentum_coefficient` | no StandardName node has the target id |
| 111 | `toroidal_neutral_state_momentum_flux_limiter_coefficient` | `parallel_neutral_state_momentum_coefficient` | no StandardName node has the target id |
| 112 | `toroidal_neutral_state_momentum_flux_limiter_coefficient` | `poloidal_neutral_state_momentum_coefficient` | no StandardName node has the target id |
| 113 | `toroidal_neutral_state_momentum_flux_limiter_coefficient` | `toroidal_neutral_momentum_coefficient` | no StandardName node has the target id |
| 114 | `toroidal_neutral_state_velocity_due_to_diamagnetic_drift` | `effective_neutral_internal_state_velocity_due_to_diamagnetic_drift` | no StandardName node has the target id |
| 115 | `toroidal_trapped_thermal_ion_state_torque_density_due_to_collisions` | `toroidal_trapped_fast_ion_state_torque_density_due_to_collisions` | no StandardName node has the target id |
| 116 | `total_plasma_energy` | `plasma_energy` | no StandardName node has the target id |
| 117 | `total_power_of_neutral_beam_injector` | `absorbed_power_of_neutral_beam_injector` | no StandardName node has the target id |
| 118 | `total_power_of_plant_system` | `absorbed_power_of_plant_system` | no StandardName node has the target id |
| 119 | `tritium_tritium_neutron_source_rate_due_to_thermal_fusion` | `deuterium_deuterium_neutron_source_rate_due_to_thermal_fusion` | no StandardName node has the target id |
| 120 | `velocity_due_to_diamagnetic_drift` | `bulk_plasma_velocity_due_to_diamagnetic_drift` | no StandardName node has the target id |
| 121 | `vertical_coordinate_of_divertor_target` | `vertical_coordinate_of_inner_divertor_target` | no StandardName node has the target id |
| 122 | `vertical_coordinate_of_divertor_target` | `vertical_coordinate_of_outer_divertor_target` | no StandardName node has the target id |
| 123 | `vertical_ion_state_momentum_flux` | `ion_state_momentum_flux` | no StandardName node has the target id |
| 124 | `vertical_neutral_state_momentum_convection_velocity` | `vertical_neutral_momentum_convection_velocity` | no StandardName node has the target id |
| 125 | `x1_coordinate_of_electron_cyclotron_launcher_mirror` | `angle_of_electron_cyclotron_launcher_mirror` | no StandardName node has the target id |
| 126 | `x2_coordinate_of_electron_cyclotron_launcher_mirror` | `angle_of_electron_cyclotron_launcher_mirror` | no StandardName node has the target id |
| 127 | `xenon_density_at_internal_transport_barrier` | `ion_density_at_internal_transport_barrier` | no StandardName node has the target id |

The canonical tab-separated `source_id`, `target_id`, and `reason` projection
of these 127 rows has SHA-256
`5fd2ebe0c955197797d5d40c588427c7159f5a4fa338e32327fea2575b879ea3`.

## Remaining gate

Later identity restorations may cause members of this exclusion ledger to
resolve. Re-running the same idempotent materialisation will add those now-valid
edges without changing any scalar. DERIVE stays closed until a fresh runtime
census reports zero unresolved links and the `REFERENCES` count equals that
same census's scalar-entry count. This run's full-parity target at
`2026-09-09T08:16:31.775Z` is 6,684, not the 6,682 recorded by the prior
reading.
