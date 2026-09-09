# Reference-edge materialisation

## Outcome

The materialisation is **blocked and was not executed**. The live pre-write
census does not match the fixed input population: it found 2,949
`StandardName` nodes carrying a non-empty `links` scalar and 6,684 link entries,
where the required population is respectively 2,948 and 6,682. In addition,
127 link entries name a target for which no `StandardName` node exists. Only
6,557 entries can therefore be represented by the declared
`(StandardName)-[:REFERENCES]->(StandardName)` relationship without inventing
placeholder identities.

Creating those placeholders would expand the mutation beyond materialising a
relationship between existing identities. Writing only the 6,557 resolvable
relationships would silently drop the 127 unresolved links and fail the
required equality. Neither mutation was performed.

| Measure | Required | Pre-write live value | Final live value | Verdict |
|---|---:|---:|---:|---|
| Standard names with non-empty `links` | 2,948 | 2,949 | 2,949 | blocked: required equality not met |
| Scalar link entries | 6,682 | 6,684 | 6,684 | blocked: required equality not met |
| `REFERENCES` relationships | 6,682 | 0 | 0 | blocked: no write performed |
| Resolvable scalar link entries | 6,682 | 6,557 | 6,557 | blocked: 127 targets absent |
| Unsupported prefixes | 0 | 0 | 0 | pass |
| Empty target identifiers | 0 | 0 | 0 | pass |
| Duplicate source/link pairs | 0 | 0 | 0 | pass |

The scalar census was captured before and after the refused mutation as 2,949
ordered rows of `id` plus `links`. Both snapshots have SHA-256
`d6bbac9c09de54722359df245cab5df135fe391b20dcd5e9815feb77be7beeb2`,
so no scalar value was modified by this node. The DERIVE beat was not
performed.

## Measurement boundary

The census used the schema-declared `StandardName.id`, `StandardName.links`,
and `StandardName.references` fields and the authored relationship direction.
It ran against graph `codex` through the ordinary `GraphClient`, which resolved
`bolt://98dci4-gpu-0002:7687`. Every query began from `StandardName`, bounded
its work to the populated `links` cohort, returned aggregates or at most the
complete 127-row unresolved set, and completed in at most 0.099 seconds. The
login-node placement was required because the live graph connection is not
available to a compute allocation.

This is the gate for this node only. No product test suite was run here; merged
verification belongs to the separately dispatched test node.

## Unresolved links

Every scalar entry has the supported `name:` prefix and a non-empty target id.
Each row below is unresolved for the same explicit reason: no live
`StandardName` node has the target id.

| # | Source standard name | Scalar link | Reason |
|---:|---|---|---|
| 1 | `argon_density_at_plasma_boundary` | `name:argon_density_at_pedestal` | no StandardName node has the target id |
| 2 | `bulk_plasma_velocity_due_to_diamagnetic_drift_magnitude` | `name:bulk_plasma_velocity_due_to_diamagnetic_drift` | no StandardName node has the target id |
| 3 | `coolant_temperature` | `name:temperature_at_inlet` | no StandardName node has the target id |
| 4 | `coolant_temperature` | `name:temperature_at_outlet` | no StandardName node has the target id |
| 5 | `critical_momentum_due_to_avalanche` | `name:critical_electric_field` | no StandardName node has the target id |
| 6 | `current_due_to_ohmic_induction` | `name:plasma_current_due_to_ohmic_induction` | no StandardName node has the target id |
| 7 | `deuterium_deuterium_neutron_flux_due_to_beam_beam_fusion` | `name:neutron_flux_due_to_beam_beam_fusion` | no StandardName node has the target id |
| 8 | `deuterium_tritium_neutron_flux_due_to_beam_beam_fusion` | `name:neutron_flux_due_to_beam_beam_fusion` | no StandardName node has the target id |
| 9 | `effective_charge_at_plasma_boundary` | `name:charge_at_plasma_boundary` | no StandardName node has the target id |
| 10 | `efficiency_of_spectrometer_channel` | `name:spectral_etendue_of_spectrometer_channel` | no StandardName node has the target id |
| 11 | `electron_energy_flux_at_wall` | `name:energy_flux_at_wall` | no StandardName node has the target id |
| 12 | `electron_particle_flux_at_wall` | `name:particle_flux_at_wall` | no StandardName node has the target id |
| 13 | `electron_pressure_at_pedestal_top` | `name:electron_density_at_pedestal_top` | no StandardName node has the target id |
| 14 | `electron_temperature_at_midplane` | `name:temperature_at_midplane` | no StandardName node has the target id |
| 15 | `energy_density` | `name:plasma_energy` | no StandardName node has the target id |
| 16 | `fast_electron_source_rate_due_to_hot_tail` | `name:electron_source_rate_due_to_hot_tail` | no StandardName node has the target id |
| 17 | `fast_neutral_internal_state_number_density` | `name:neutral_internal_state_number_density` | no StandardName node has the target id |
| 18 | `flux_due_to_beam_beam_fusion` | `name:neutron_flux_due_to_beam_beam_fusion` | no StandardName node has the target id |
| 19 | `flux_due_to_recombination` | `name:particle_flux_at_wall_due_to_recombination` | no StandardName node has the target id |
| 20 | `flux_surface_averaged_carbon_density` | `name:ratio_of_carbon_density_to_electron_density` | no StandardName node has the target id |
| 21 | `flux_surface_averaged_deuterium_tritium_density` | `name:flux_surface_averaged_total_ion_density` | no StandardName node has the target id |
| 22 | `fraction_of_neutron_detector_converter` | `name:atomic_fraction_of_neutron_detector_converter` | no StandardName node has the target id |
| 23 | `gradient_of_radial_electron_density` | `name:radial_electron_density` | no StandardName node has the target id |
| 24 | `ion_average_temperature_at_magnetic_axis` | `name:average_temperature_at_magnetic_axis` | no StandardName node has the target id |
| 25 | `ion_particle_flux_at_wall` | `name:particle_flux_at_wall` | no StandardName node has the target id |
| 26 | `ion_state_kinetic_energy_flux_at_wall_due_to_surface_emission` | `name:energy_flux_at_wall` | no StandardName node has the target id |
| 27 | `ion_state_momentum_convection_velocity` | `name:radial_plasma_effective_momentum_convection_velocity` | no StandardName node has the target id |
| 28 | `ion_state_particle_flux` | `name:ion_state_energy_flux` | no StandardName node has the target id |
| 29 | `ion_state_particle_flux_at_wall` | `name:particle_flux_at_wall` | no StandardName node has the target id |
| 30 | `ion_state_temperature` | `name:edge_ion_average_temperature` | no StandardName node has the target id |
| 31 | `ion_temperature` | `name:ion_temperature_at_wall` | no StandardName node has the target id |
| 32 | `ion_temperature_at_midplane` | `name:temperature_at_midplane` | no StandardName node has the target id |
| 33 | `maximum_of_energy_flux_at_first_wall` | `name:energy_flux_at_wall` | no StandardName node has the target id |
| 34 | `maximum_of_energy_flux_at_limiter` | `name:energy_flux_at_wall` | no StandardName node has the target id |
| 35 | `maximum_power_at_inner_divertor_target` | `name:power_at_inner_divertor_target` | no StandardName node has the target id |
| 36 | `maximum_power_at_outer_divertor_target` | `name:power_at_outer_divertor_target` | no StandardName node has the target id |
| 37 | `neon_prefill_count` | `name:xenon_prefill_count` | no StandardName node has the target id |
| 38 | `net_absorbed_power_of_plant_system` | `name:absorbed_power_of_plant_system` | no StandardName node has the target id |
| 39 | `neutral_energy_flux_at_wall` | `name:energy_flux_at_wall` | no StandardName node has the target id |
| 40 | `neutral_internal_state_momentum_flux_limiter_coefficient_over_edge_region` | `name:internal_state_momentum_flux_limiter_coefficient_over_edge_region` | no StandardName node has the target id |
| 41 | `neutral_particle_flux_at_wall` | `name:particle_flux_at_wall` | no StandardName node has the target id |
| 42 | `neutral_power_at_wall_due_to_recombination` | `name:power_at_wall_due_to_recombination` | no StandardName node has the target id |
| 43 | `neutral_state_energy_convection_velocity` | `name:energy_convection_velocity` | no StandardName node has the target id |
| 44 | `neutral_state_energy_flux` | `name:ion_state_energy_flux` | no StandardName node has the target id |
| 45 | `neutral_state_energy_flux_at_wall` | `name:energy_flux_at_wall` | no StandardName node has the target id |
| 46 | `neutral_state_energy_flux_due_to_recombination` | `name:energy_flux_due_to_recombination` | no StandardName node has the target id |
| 47 | `neutral_state_particle_flux_at_wall` | `name:particle_flux_at_wall` | no StandardName node has the target id |
| 48 | `neutron_flux` | `name:deuterium_deuterium_neutron_flux` | no StandardName node has the target id |
| 49 | `neutron_flux` | `name:tritium_tritium_neutron_flux` | no StandardName node has the target id |
| 50 | `neutron_rate_of_neutron_detector` | `name:rate_of_neutron_detector` | no StandardName node has the target id |
| 51 | `neutron_source_rate_due_to_beam_beam_fusion` | `name:neutron_flux_due_to_beam_beam_fusion` | no StandardName node has the target id |
| 52 | `normalized_atomic_count_of_pellet` | `name:atomic_count_of_pellet` | no StandardName node has the target id |
| 53 | `normalized_atomic_count_of_pellet` | `name:count_of_pellet` | no StandardName node has the target id |
| 54 | `normalized_perturbed_density_imaginary_part` | `name:normalized_perturbed_density` | no StandardName node has the target id |
| 55 | `normalized_perturbed_pressure` | `name:perturbed_pressure` | no StandardName node has the target id |
| 56 | `normalized_total_particle_perturbed_pressure_of_gyrokinetic_eigenmode` | `name:normalized_gyrocenter_perturbed_pressure` | no StandardName node has the target id |
| 57 | `nuclear_power_density_of_breeder_blanket_module` | `name:power_density_of_breeder_blanket_module` | no StandardName node has the target id |
| 58 | `outer_atomic_count_of_pellet` | `name:atomic_count_of_pellet` | no StandardName node has the target id |
| 59 | `oxygen_density_at_limiter` | `name:ion_density_at_limiter` | no StandardName node has the target id |
| 60 | `parallel_effective_neutral_internal_state_velocity_due_to_diamagnetic_drift` | `name:effective_neutral_internal_state_velocity_due_to_diamagnetic_drift` | no StandardName node has the target id |
| 61 | `parallel_momentum_flux_due_to_perturbed_parallel_vector_potential` | `name:parallel_momentum_flux_due_to_perturbed_parallel_magnetic_field` | no StandardName node has the target id |
| 62 | `parallel_momentum_flux_due_to_perturbed_parallel_vector_potential` | `name:perpendicular_momentum_flux_due_to_perturbed_parallel_vector_potential` | no StandardName node has the target id |
| 63 | `parallel_normalized_perturbed_vector_potential_amplitude` | `name:parallel_normalized_perturbed_vector_potential` | no StandardName node has the target id |
| 64 | `peak_voltage_of_neutron_detector` | `name:requested_upper_voltage_of_neutron_detector` | no StandardName node has the target id |
| 65 | `peak_voltage_of_neutron_detector` | `name:voltage_of_neutron_detector` | no StandardName node has the target id |
| 66 | `perpendicular_momentum_flux_due_to_perturbed_parallel_magnetic_field` | `name:momentum_flux_due_to_perturbed_parallel_magnetic_field` | no StandardName node has the target id |
| 67 | `perpendicular_momentum_flux_due_to_perturbed_parallel_magnetic_field` | `name:parallel_momentum_flux_due_to_perturbed_parallel_magnetic_field` | no StandardName node has the target id |
| 68 | `perpendicular_normalized_gyrocenter_perturbed_pressure` | `name:normalized_gyrocenter_perturbed_pressure` | no StandardName node has the target id |
| 69 | `perturbed_vector_potential` | `name:normalized_perturbed_vector_potential` | no StandardName node has the target id |
| 70 | `perturbed_vector_potential` | `name:parallel_normalized_perturbed_vector_potential` | no StandardName node has the target id |
| 71 | `plasma_velocity_due_to_diamagnetic_drift` | `name:bulk_plasma_velocity_due_to_diamagnetic_drift` | no StandardName node has the target id |
| 72 | `poloidal_current_density_due_to_collisions` | `name:current_density_due_to_collisions` | no StandardName node has the target id |
| 73 | `poloidal_current_density_due_to_viscosity` | `name:current_density_due_to_viscosity` | no StandardName node has the target id |
| 74 | `poloidal_ion_momentum` | `name:poloidal_momentum` | no StandardName node has the target id |
| 75 | `poloidal_ion_state_energy_diffusivity` | `name:poloidal_ion_state_energy_diffusion_coefficient` | no StandardName node has the target id |
| 76 | `poloidal_ion_state_momentum` | `name:poloidal_momentum` | no StandardName node has the target id |
| 77 | `poloidal_ion_state_momentum_flux` | `name:ion_state_momentum_flux` | no StandardName node has the target id |
| 78 | `poloidal_ion_state_velocity_due_to_e_cross_b_drift` | `name:parallel_ion_state_velocity_due_to_e_cross_b_drift` | no StandardName node has the target id |
| 79 | `poloidal_neutral_state_momentum` | `name:poloidal_momentum` | no StandardName node has the target id |
| 80 | `power_at_wall_due_to_conduction` | `name:energy_flux_at_wall` | no StandardName node has the target id |
| 81 | `power_of_divertor_due_to_fusion` | `name:power_due_to_fusion` | no StandardName node has the target id |
| 82 | `power_of_divertor_due_to_radiation` | `name:power_due_to_radiation` | no StandardName node has the target id |
| 83 | `power_of_neutral_beam_injector` | `name:absorbed_power_of_neutral_beam_injector` | no StandardName node has the target id |
| 84 | `radial_centroid_of_electron_cyclotron_launcher_mirror` | `name:angle_of_electron_cyclotron_launcher_mirror` | no StandardName node has the target id |
| 85 | `radial_current_density_due_to_viscosity` | `name:current_density_due_to_viscosity` | no StandardName node has the target id |
| 86 | `radial_derivative_of_poloidal_ion_state_velocity` | `name:second_radial_derivative_of_poloidal_ion_state_velocity` | no StandardName node has the target id |
| 87 | `radial_effective_total_ion_energy_convection_velocity` | `name:radial_ion_momentum_effective_convection_velocity` | no StandardName node has the target id |
| 88 | `radial_energy_convection_velocity` | `name:energy_convection_velocity` | no StandardName node has the target id |
| 89 | `radial_ion_state_momentum_flux` | `name:ion_state_momentum_flux` | no StandardName node has the target id |
| 90 | `radial_neutral_state_energy_diffusion_coefficient` | `name:radial_ion_state_energy_diffusion_coefficient` | no StandardName node has the target id |
| 91 | `radial_neutral_state_momentum_convection_velocity` | `name:radial_neutral_species_momentum_convection_velocity` | no StandardName node has the target id |
| 92 | `radial_plasma_momentum_source` | `name:plasma_momentum_source` | no StandardName node has the target id |
| 93 | `radiance_at_spectral_line` | `name:motional_stark_radiance_at_spectral_line` | no StandardName node has the target id |
| 94 | `ratio_of_coolant_mass_to_time` | `name:coolant_mass` | no StandardName node has the target id |
| 95 | `reference_calibration_wavelength_of_spectrometer_channel` | `name:wavelength_of_spectrometer_channel` | no StandardName node has the target id |
| 96 | `second_radial_derivative_of_poloidal_ion_velocity` | `name:second_radial_derivative_of_poloidal_ion_state_velocity` | no StandardName node has the target id |
| 97 | `second_radial_derivative_of_toroidal_ion_state_velocity` | `name:toroidal_ion_state_velocity` | no StandardName node has the target id |
| 98 | `source_rate_due_to_thermal_fusion` | `name:neutron_power` | no StandardName node has the target id |
| 99 | `thermal_electron_power` | `name:absorbed_power` | no StandardName node has the target id |
| 100 | `time_derivative_of_electron_temperature` | `name:time_derivative_of_ion_state_temperature` | no StandardName node has the target id |
| 101 | `time_derivative_of_electron_temperature` | `name:time_derivative_of_ion_temperature` | no StandardName node has the target id |
| 102 | `time_derivative_of_total_electron_density` | `name:time_derivative_of_total_ion_density` | no StandardName node has the target id |
| 103 | `time_derivative_of_total_ion_state_density` | `name:time_derivative_of_fast_ion_state_density` | no StandardName node has the target id |
| 104 | `time_derivative_of_total_ion_state_density` | `name:time_derivative_of_total_ion_density` | no StandardName node has the target id |
| 105 | `toroidal_beryllium_velocity_at_plasma_boundary` | `name:toroidal_beryllium_velocity_at_pedestal` | no StandardName node has the target id |
| 106 | `toroidal_co_passing_thermal_ion_state_torque_density_due_to_collisions` | `name:co_passing_thermal_ion_state_torque_density_due_to_collisions` | no StandardName node has the target id |
| 107 | `toroidal_ion_momentum` | `name:ion_momentum` | no StandardName node has the target id |
| 108 | `toroidal_ion_state_momentum_flux_limiter_coefficient` | `name:poloidal_ion_state_momentum_coefficient` | no StandardName node has the target id |
| 109 | `toroidal_ion_state_momentum_flux_limiter_coefficient` | `name:toroidal_neutral_momentum_coefficient` | no StandardName node has the target id |
| 110 | `toroidal_neutral_state_momentum_flux_limiter_coefficient` | `name:neutral_state_momentum_coefficient` | no StandardName node has the target id |
| 111 | `toroidal_neutral_state_momentum_flux_limiter_coefficient` | `name:parallel_neutral_state_momentum_coefficient` | no StandardName node has the target id |
| 112 | `toroidal_neutral_state_momentum_flux_limiter_coefficient` | `name:poloidal_neutral_state_momentum_coefficient` | no StandardName node has the target id |
| 113 | `toroidal_neutral_state_momentum_flux_limiter_coefficient` | `name:toroidal_neutral_momentum_coefficient` | no StandardName node has the target id |
| 114 | `toroidal_neutral_state_velocity_due_to_diamagnetic_drift` | `name:effective_neutral_internal_state_velocity_due_to_diamagnetic_drift` | no StandardName node has the target id |
| 115 | `toroidal_trapped_thermal_ion_state_torque_density_due_to_collisions` | `name:toroidal_trapped_fast_ion_state_torque_density_due_to_collisions` | no StandardName node has the target id |
| 116 | `total_plasma_energy` | `name:plasma_energy` | no StandardName node has the target id |
| 117 | `total_power_of_neutral_beam_injector` | `name:absorbed_power_of_neutral_beam_injector` | no StandardName node has the target id |
| 118 | `total_power_of_plant_system` | `name:absorbed_power_of_plant_system` | no StandardName node has the target id |
| 119 | `tritium_tritium_neutron_source_rate_due_to_thermal_fusion` | `name:deuterium_deuterium_neutron_source_rate_due_to_thermal_fusion` | no StandardName node has the target id |
| 120 | `velocity_due_to_diamagnetic_drift` | `name:bulk_plasma_velocity_due_to_diamagnetic_drift` | no StandardName node has the target id |
| 121 | `vertical_coordinate_of_divertor_target` | `name:vertical_coordinate_of_inner_divertor_target` | no StandardName node has the target id |
| 122 | `vertical_coordinate_of_divertor_target` | `name:vertical_coordinate_of_outer_divertor_target` | no StandardName node has the target id |
| 123 | `vertical_ion_state_momentum_flux` | `name:ion_state_momentum_flux` | no StandardName node has the target id |
| 124 | `vertical_neutral_state_momentum_convection_velocity` | `name:vertical_neutral_momentum_convection_velocity` | no StandardName node has the target id |
| 125 | `x1_coordinate_of_electron_cyclotron_launcher_mirror` | `name:angle_of_electron_cyclotron_launcher_mirror` | no StandardName node has the target id |
| 126 | `x2_coordinate_of_electron_cyclotron_launcher_mirror` | `name:angle_of_electron_cyclotron_launcher_mirror` | no StandardName node has the target id |
| 127 | `xenon_density_at_internal_transport_barrier` | `name:ion_density_at_internal_transport_barrier` | no StandardName node has the target id |

## Required resolution

Before this materialisation can run, the live input population must be
reconciled with the fixed authority: either restore or otherwise adjudicate all
127 missing targets, and resolve why the live scalar cohort is one name and two
links larger than the required 2,948/6,682 census; or explicitly revise the
required counts and target policy. Only after that decision can an all-or-none
materialisation prove count equality without altering `links`.
