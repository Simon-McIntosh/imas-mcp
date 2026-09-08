# The 93 standard names the structural cleanup removed on 2026-09-08

Derived from `StandardNameChange` rows with `operation='remove_derived_parent'` and
`origin='pipeline_cleanup'` in the window `11:57:28Z`–`11:57:36Z`. Recorded here so no worker
has to re-derive the cohort, and so the 15 names that carry no disposition are visible as a set
rather than as an arithmetic remainder.

| | count |
|---|---|
| removed in the pass | 93 |
| reclassified by the origin repair 100 s earlier | 78 |
| dispositioned RESTORE by the census | 67 |
| **carrying no disposition at all** | **15** |

The census covered the 78 the origin repair had reclassified, because that was the cohort the
brief named. The remaining 15 were removed in the same pass by the same mechanism and nobody
has classified them. Nine carry recorded spend; one of them, `etendue_of_spectrometer_channel`,
sits in the published WEST review cut.

## The 15 with no disposition

| standard name | apportioned spend (USD) |
|---|---|
| `etendue_of_spectrometer_channel` | 0.84 |
| `flux_at_wall_due_to_eddy_current` | 0.45 |
| `charge_at_pedestal_top` | 0.34 |
| `flux_at_wall` | 0.28 |
| `ion_density_at_pedestal_top` | 0.11 |
| `width_of_spectrometer_channel` | 0.09 |
| `permeability_of_ferritic_element` | 0.08 |
| `ion_charge_state_power` | 0.07 |
| `flux_at_wall_due_to_pumping` | 0.01 |
| `angle_of_optical_element` | 0.00 |
| `diffusion_coefficient_due_to_diffusion` | 0.00 |
| `factor_of_spectrometer_channel` | 0.00 |
| `flux_at_first_wall` | 0.00 |
| `flux_due_to_pumping` | 0.00 |
| `plasma_heating_power` | 0.00 |

## The full cohort

| standard name | origin-reconciled | census verdict | apportioned spend (USD) |
|---|---|---|---|
| `absorbed_power_of_neutral_beam_injector` | yes | RESTORE | 0.18 |
| `absorbed_power_of_plant_system` | yes | RESTORE | 0.75 |
| `angle_of_electron_cyclotron_launcher_mirror` | yes | RESTORE | 0.00 |
| `angle_of_optical_element` | no | **none** | 0.00 |
| `argon_density_at_pedestal_top` | yes | RESTORE | 1.41 |
| `atomic_count_of_pellet` | yes | RESTORE | 0.11 |
| `atomic_fraction_of_neutron_detector_converter` | yes | RESTORE | 0.19 |
| `atomic_mass_of_wall_material` | yes | RESTORE | 0.10 |
| `beryllium_density_at_pedestal_top` | yes | RESTORE | 0.26 |
| `boron_density_at_pedestal_top` | yes | RESTORE | 0.62 |
| `bulk_plasma_velocity_due_to_diamagnetic_drift` | yes | RESTORE | 0.31 |
| `carbon_density_at_pedestal_top` | yes | RESTORE | 0.36 |
| `charge_at_pedestal_top` | no | **none** | 0.34 |
| `count_of_pellet` | yes | removed | 0.00 |
| `critical_electric_field` | yes | RESTORE | 0.23 |
| `current_density_due_to_viscosity` | yes | RESTORE | 0.47 |
| `density_at_pedestal_top` | yes | RESTORE | 1.62 |
| `density_of_pellet` | yes | RESTORE | 0.15 |
| `deuterium_density_at_pedestal_top` | yes | RESTORE | 0.64 |
| `deuterium_deuterium_neutron_flux` | yes | RESTORE | 0.30 |
| `deuterium_deuterium_neutron_flux_due_to_beam_thermal_fusion` | yes | RESTORE | 0.33 |
| `deuterium_deuterium_neutron_source_rate_due_to_thermal_fusion` | yes | RESTORE | 0.34 |
| `deuterium_tritium_density_at_pedestal_top` | yes | RESTORE | 2.03 |
| `deuterium_tritium_neutron_flux_due_to_beam_thermal_fusion` | yes | RESTORE | 0.80 |
| `diffusion_coefficient_due_to_diffusion` | no | **none** | 0.00 |
| `effective_charge_at_pedestal_top` | yes | RESTORE | 0.39 |
| `electron_density_at_pedestal_top` | yes | RESTORE | 0.46 |
| `electron_power_density_due_to_collisions` | yes | RESTORE | 0.25 |
| `energy_convection_velocity` | yes | removed | 0.00 |
| `energy_flux_at_wall` | yes | RESTORE | 0.78 |
| `etendue_of_spectrometer_channel` | no | **none** | 0.84 |
| `factor_of_spectrometer_channel` | no | **none** | 0.00 |
| `flux_at_first_wall` | no | **none** | 0.00 |
| `flux_at_wall` | no | **none** | 0.28 |
| `flux_at_wall_due_to_eddy_current` | no | **none** | 0.45 |
| `flux_at_wall_due_to_pumping` | no | **none** | 0.01 |
| `flux_due_to_diamagnetic_drift` | yes | RESTORE | 1.08 |
| `flux_due_to_pumping` | no | **none** | 0.00 |
| `gyrocenter_pressure` | yes | RESTORE | 0.84 |
| `helium_3_density_at_pedestal_top` | yes | RESTORE | 0.35 |
| `helium_4_density_at_pedestal_top` | yes | RESTORE | 0.25 |
| `hydrogen_density_at_pedestal_top` | yes | RESTORE | 0.29 |
| `ion_charge_state_power` | no | **none** | 0.07 |
| `ion_density_at_pedestal_top` | no | **none** | 0.11 |
| `ion_momentum` | yes | removed | 0.00 |
| `ion_power_density` | yes | RESTORE | 0.80 |
| `ion_state_energy_flux` | yes | removed | 0.00 |
| `ion_state_momentum_flux` | yes | removed | 0.00 |
| `iron_density_at_pedestal_top` | yes | RESTORE | 0.35 |
| `krypton_density_at_pedestal_top` | yes | RESTORE | 0.59 |
| `launched_power_of_electron_cyclotron_launcher` | yes | removed | 0.00 |
| `lithium_density_at_pedestal_top` | yes | RESTORE | 0.29 |
| `mass_of_wall_material` | yes | RESTORE | 0.39 |
| `motional_stark_photon_radiance_at_spectral_line` | yes | RESTORE | 0.19 |
| `neon_density_at_pedestal_top` | yes | RESTORE | 0.43 |
| `nitrogen_density_at_pedestal_top` | yes | RESTORE | 1.30 |
| `normalized_gyrocenter_perturbed_pressure` | yes | removed | 0.00 |
| `normalized_perturbed_vector_potential` | yes | RESTORE | 0.03 |
| `oxygen_density_at_pedestal_top` | yes | RESTORE | 0.98 |
| `parallel_normalized_perturbed_vector_potential` | yes | RESTORE | 0.17 |
| `parity_of_gyrokinetic_eigenmode` | yes | RESTORE | 0.15 |
| `particle_flux_at_wall` | yes | removed | 0.00 |
| `particle_flux_at_wall_due_to_recombination` | yes | RESTORE | 0.12 |
| `permeability_of_ferritic_element` | no | **none** | 0.08 |
| `perturbed_gyrocenter_pressure` | yes | RESTORE | 1.58 |
| `perturbed_particle_pressure` | yes | removed | 0.00 |
| `perturbed_plasma_mass_density` | yes | RESTORE | 0.05 |
| `perturbed_plasma_pressure` | yes | RESTORE | 0.63 |
| `perturbed_plasma_temperature` | yes | RESTORE | 0.06 |
| `perturbed_pressure` | yes | RESTORE | 1.23 |
| `plasma_current_due_to_ohmic_induction` | yes | RESTORE | 0.58 |
| `plasma_energy` | yes | RESTORE | 0.37 |
| `plasma_heating_power` | no | **none** | 0.00 |
| `poloidal_angle` | yes | RESTORE | 0.31 |
| `poloidal_momentum_flux_limiter_coefficient` | yes | RESTORE | 1.55 |
| `power_at_inner_divertor_target` | yes | RESTORE | 0.44 |
| `power_at_outer_divertor_target` | yes | RESTORE | 0.60 |
| `power_at_wall_due_to_recombination` | yes | removed | 0.00 |
| `power_due_to_fusion` | yes | RESTORE | 0.48 |
| `power_due_to_radiation` | yes | RESTORE | 0.63 |
| `radial_momentum_flux_limiter_coefficient` | yes | RESTORE | 0.99 |
| `safety_factor_at_pedestal_top` | yes | RESTORE | 0.52 |
| `spectral_etendue_of_spectrometer_channel` | yes | RESTORE | 4.03 |
| `target_atomic_fraction_of_neutron_detector_converter` | yes | RESTORE | 0.50 |
| `temperature_at_midplane` | yes | RESTORE | 0.29 |
| `toroidal_neutral_momentum_flux_limiter_coefficient` | yes | RESTORE | 1.19 |
| `total_ion_density_at_pedestal_top` | yes | RESTORE | 0.74 |
| `tritium_density_at_pedestal_top` | yes | RESTORE | 1.26 |
| `tritium_tritium_neutron_flux` | yes | RESTORE | 0.00 |
| `tungsten_density_at_pedestal_top` | yes | RESTORE | 0.38 |
| `voltage_of_neutron_detector` | yes | removed | 0.00 |
| `width_of_spectrometer_channel` | no | **none** | 0.09 |
| `xenon_density_at_pedestal_top` | yes | RESTORE | 0.68 |

Spend is apportioned: a single `LLMCost` row may name several standard names, so its cost is
divided among them. In six of the seven phases each row names exactly one name, so apportioning
moves the cohort total by sixteen cents against a naive row sum.
