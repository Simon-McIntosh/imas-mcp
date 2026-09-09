# Disposition of unobserved validation verdicts

## Verdict

The population moved from the earlier 100 to **99** live, nonterminal rows
with `validation_status` set and `validated_at` null. The already-observed
target is the one-row difference. The status breakdown has not otherwise
moved: **18 pending, 38 quarantined, and 43 valid**.

The exact 43 valid IDs were passed to `drain_validation_for_ids`, the sanctioned
deterministic audit path. It writes a new verdict and `validated_at` together;
it does not manufacture an observation time. The drain observed 35 rows at
`2026-09-09T14:29:47.169Z`: 26 remained valid and nine were re-quarantined.
Eight valid rows could not be claimed because they have no description, which
is the audit's deliberate precondition.

The final live re-read finds **64** still unobserved nonterminal rows: 18
pending, 38 legacy quarantined, and eight descriptionless valid rows. Each is
named below with a disposition. No remaining row is represented as observed
when no admissible audit took place.

All graph work ran on the login node because the Neo4j tunnel is login-node-
local. The reads were bounded respectively to the live 99-row population, the
exact 43 valid IDs, and the resulting 64-row population; each completed in
under ten seconds.

## Exact valid-row outcomes

| Identity | Result | Observation / disposition |
| --- | --- | --- |
| `angle_of_optical_element` | quarantined | Observed `2026-09-09T14:29:47.169Z`; retain genuine quarantine. |
| `argon_density_at_pedestal_top` | valid | Observed `2026-09-09T14:29:47.169Z`. |
| `beryllium_density_at_pedestal_top` | valid | Observed `2026-09-09T14:29:47.169Z`. |
| `boron_density_at_pedestal_top` | valid | Observed `2026-09-09T14:29:47.169Z`. |
| `carbon_density_at_pedestal_top` | valid | Observed `2026-09-09T14:29:47.169Z`. |
| `density_at_pedestal_top` | valid | Observed `2026-09-09T14:29:47.169Z`. |
| `deuterium_density_at_pedestal_top` | valid | Observed `2026-09-09T14:29:47.169Z`. |
| `deuterium_tritium_density_at_pedestal_top` | valid | Observed `2026-09-09T14:29:47.169Z`. |
| `diffusion_coefficient_due_to_diffusion` | quarantined | Observed `2026-09-09T14:29:47.169Z`; retain genuine quarantine. |
| `electron_density_at_pedestal_top` | valid | Observed `2026-09-09T14:29:47.169Z`. |
| `electron_power_density_due_to_collisions` | valid | Observed `2026-09-09T14:29:47.169Z`. |
| `factor_of_spectrometer_channel` | quarantined | Observed `2026-09-09T14:29:47.169Z`; retain genuine quarantine. |
| `flux_at_first_wall` | quarantined | Observed `2026-09-09T14:29:47.169Z`; retain genuine quarantine. |
| `flux_at_wall_due_to_eddy_current` | quarantined | Observed `2026-09-09T14:29:47.169Z`; retain genuine quarantine. |
| `flux_at_wall_due_to_pumping` | quarantined | Observed `2026-09-09T14:29:47.169Z`; retain genuine quarantine. |
| `flux_at_wall_due_to_recombination` | valid | Observed `2026-09-09T14:29:47.169Z`. |
| `flux_due_to_pumping` | valid | Observed `2026-09-09T14:29:47.169Z`. |
| `helium_3_density_at_pedestal_top` | valid | Observed `2026-09-09T14:29:47.169Z`. |
| `helium_4_density_at_pedestal_top` | valid | Observed `2026-09-09T14:29:47.169Z`. |
| `hydrogen_density_at_pedestal_top` | valid | Observed `2026-09-09T14:29:47.169Z`. |
| `ion_charge_state_power` | valid | Observed `2026-09-09T14:29:47.169Z`. |
| `ion_power_density` | valid | Observed `2026-09-09T14:29:47.169Z`. |
| `iron_density_at_pedestal_top` | valid | Observed `2026-09-09T14:29:47.169Z`. |
| `krypton_density_at_pedestal_top` | valid | Observed `2026-09-09T14:29:47.169Z`. |
| `line_averaged_neon_density` | valid, unobserved | No description; materialise it, then drain this exact ID. |
| `lithium_density_at_pedestal_top` | valid | Observed `2026-09-09T14:29:47.169Z`. |
| `neon_density_at_pedestal_top` | valid | Observed `2026-09-09T14:29:47.169Z`. |
| `nitrogen_density_at_pedestal_top` | valid | Observed `2026-09-09T14:29:47.169Z`. |
| `oxygen_density_at_pedestal_top` | valid | Observed `2026-09-09T14:29:47.169Z`. |
| `permeability_of_ferritic_element` | quarantined | Observed `2026-09-09T14:29:47.169Z`; retain genuine quarantine. |
| `plasma_heating_power` | quarantined | Observed `2026-09-09T14:29:47.169Z`; retain genuine quarantine. |
| `poloidal_ion_state_momentum_diffusion_coefficient` | valid, unobserved | No description; materialise it, then drain this exact ID. |
| `radial_coordinate_of_reflector` | valid, unobserved | No description; materialise it, then drain this exact ID. |
| `radius_of_soft_xray_detector` | valid, unobserved | No description; materialise it, then drain this exact ID. |
| `ratio_of_diamagnetic_vorticity_to_major_radius` | valid, unobserved | No description; materialise it, then drain this exact ID. |
| `time` | valid | Observed `2026-09-09T14:29:47.169Z`. |
| `toroidal_coordinate_of_spectrometer` | valid, unobserved | No description; materialise it, then drain this exact ID. |
| `toroidal_tritium_velocity` | valid, unobserved | No description; materialise it, then drain this exact ID. |
| `tritium_density_at_pedestal_top` | valid | Observed `2026-09-09T14:29:47.169Z`. |
| `tungsten_density_at_pedestal_top` | valid | Observed `2026-09-09T14:29:47.169Z`. |
| `vertical_coordinate_of_reflector` | valid, unobserved | No description; materialise it, then drain this exact ID. |
| `width_of_spectrometer_channel` | quarantined | Observed `2026-09-09T14:29:47.169Z`; retain genuine quarantine. |
| `xenon_density_at_pedestal_top` | valid | Observed `2026-09-09T14:29:47.169Z`. |

The eight descriptionless rows' DD bindings, in table order, are
`summary/line_average/n_i/neon/value`,
`plasma_transport/model/ggd/ion/state/momentum/d_pol/values`,
`spectrometer_x_ray_crystal/channel/reflector/sphere_centre/r`,
`soft_x_rays/channel/detector/radius`,
`plasma_profiles/ggd/vorticity_over_r/diamagnetic`,
`spectrometer_visible/channel/geometry_matrix/interpolated/phi`,
`summary/local/pedestal/velocity_phi/tritium/value`, and
`spectrometer_x_ray_crystal/channel/reflector/sphere_centre/z`.

## Export effect and cost

**20 of the original 43 valid rows became fully export-eligible** after their
new validation observation joined their existing name, documentation-review,
and quorum evidence:

`argon_density_at_pedestal_top`, `beryllium_density_at_pedestal_top`,
`boron_density_at_pedestal_top`, `carbon_density_at_pedestal_top`,
`deuterium_density_at_pedestal_top`, `deuterium_tritium_density_at_pedestal_top`,
`electron_density_at_pedestal_top`, `helium_3_density_at_pedestal_top`,
`helium_4_density_at_pedestal_top`, `hydrogen_density_at_pedestal_top`,
`iron_density_at_pedestal_top`, `krypton_density_at_pedestal_top`,
`lithium_density_at_pedestal_top`, `neon_density_at_pedestal_top`,
`nitrogen_density_at_pedestal_top`, `oxygen_density_at_pedestal_top`, `time`,
`tritium_density_at_pedestal_top`, `tungsten_density_at_pedestal_top`, and
`xenon_density_at_pedestal_top`.

Total USD spent: **$0.00**. The sanctioned validation drain invokes no model.

## Remaining rows: dispositions rather than false observations

### Pending rows: 18

Each pending row is nonterminal, not a completed validation verdict. It stays
unstamped until ordinary candidate validation has a description to inspect.

| Identity | Current stage | Disposition |
| --- | --- | --- |
| `accumulated_total_gas_count` | drafted | Await ordinary candidate validation. |
| `connection_length` | drafted | Await ordinary candidate validation. |
| `distance_of_antenna_strap` | drafted | Await ordinary candidate validation. |
| `gap_at_closest_wall_point` | drafted | Await ordinary candidate validation. |
| `hard_xray_emissivity` | drafted | Await ordinary candidate validation. |
| `ion_state_vibrational_level` | drafted | Await ordinary candidate validation. |
| `neutral_pressure` | drafted | Await ordinary candidate validation. |
| `outer_hard_xray_half_width` | drafted | Await ordinary candidate validation. |
| `poloidal_straight_field_line_angle` | drafted | Await ordinary candidate validation. |
| `radial_coordinate_of_plasma_boundary_gap_reference_point` | drafted | Await ordinary candidate validation. |
| `radial_coordinate_of_reflectometer_antenna` | drafted | Await ordinary candidate validation. |
| `radial_distance_at_midplane` | drafted | Await ordinary candidate validation. |
| `surface_temperature` | drafted | Await ordinary candidate validation. |
| `thickness_of_cryostat` | drafted | Await ordinary candidate validation. |
| `toroidal_width_of_antenna_strap` | drafted | Await ordinary candidate validation. |
| `tritium_tritium_neutron_source_rate_due_to_thermal_fusion` | drafted | Await ordinary candidate validation. |
| `vertical_coordinate_of_dr_dz_zero_point` | drafted | Await ordinary candidate validation. |
| `vertical_outline_of_vacuum_vessel` | drafted | Await ordinary candidate validation. |

### Legacy quarantines: 38 rows

The quarantine scalar alone is not proof that an audit was observed. The issue
below is retained evidence where present; a missing issue is a stated
requirement for a future scoped audit, not permission to stamp the row.

| Identity | Stage | Disposition reason |
| --- | --- | --- |
| `count_at_detector_pixel` | accepted | Description implies a rate without a rate marker; preserve quarantine. |
| `cumulative_inside_flux_surface_torque` | accepted | `cumulative_` conflicts with the required suffix form; preserve quarantine. |
| `diamagnetic_current_density` | accepted | No issue snapshot; queue a scoped audit, do not stamp. |
| `energy_density` | accepted | No issue snapshot; queue a scoped audit, do not stamp. |
| `fast_ion_photon_radiance_of_spectral_line_due_to_charge_exchange` | drafted | Canonical locus requires `at_spectral_line`; preserve quarantine. |
| `field_aligned_surface_tilt_angle_of_langmuir_probe` | reviewed | Bare `field` needs a physical qualifier; preserve quarantine. |
| `flux_surface_average_magnetic_field_magnitude` | pending | Grammar round-trip failed; preserve quarantine. |
| `flux_surface_normal_ion_momentum_flux_due_to_diamagnetic_drift` | drafted | Invalid coordinate vocabulary and repeated `flux`; preserve quarantine. |
| `flux_surface_normal_non_axisymmetric_vacuum_magnetic_field_fourier_coefficient_at_control_surface` | drafted | Invalid coordinate vocabulary and repeated `surface`; preserve quarantine. |
| `ion_pressure` | accepted | No issue snapshot; queue a scoped audit, do not stamp. |
| `length_of_antenna_strap` | accepted | No issue snapshot; queue a scoped audit, do not stamp. |
| `magnetic_field_magnitude` | accepted | No issue snapshot; queue a scoped audit, do not stamp. |
| `major_radius` | accepted | No issue snapshot; queue a scoped audit, do not stamp. |
| `particle_count` | accepted | No issue snapshot; queue a scoped audit, do not stamp. |
| `perpendicular_normalized_perturbed_pressure` | accepted | No issue snapshot; queue a scoped audit, do not stamp. |
| `phase_of_electron_cyclotron_beam` | drafted | Canonical form is `electron_cyclotron_beam_phase`; preserve quarantine. |
| `poloidal_magnetic_flux_perturbed_at_ece_channel_emission_position_due_to_wave_particle_interaction` | drafted | Invalid coordinate vocabulary and unit/description mismatch; preserve quarantine. |
| `poloidal_momentum_neutral_internal_state_flux_limiter_coefficient` | reviewed | Grammar round-trip failed; preserve quarantine. |
| `power_of_beam_tracing_beam` | accepted | Repeated `beam` token; preserve quarantine. |
| `power_of_lower_hybrid_antenna` | accepted | No issue snapshot; queue a scoped audit, do not stamp. |
| `radial_outline_of_flux_surface` | accepted | Semantics and canonical locus audit fail; preserve quarantine. |
| `radius_of_plasma_filament` | accepted | No issue snapshot; queue a scoped audit, do not stamp. |
| `radius_of_poloidal_field_coil` | accepted | No issue snapshot; queue a scoped audit, do not stamp. |
| `rotation_frequency_due_to_e_cross_b_drift` | accepted | Mathematical symbol lacks a definition sentence; preserve quarantine. |
| `source_rate_due_to_injection` | accepted | No issue snapshot; queue a scoped audit, do not stamp. |
| `spectral_signal_to_noise_ratio_logarithm_of_spectrometer_channel` | drafted | Logarithm prefix duplicates the logarithmic `dB` unit; preserve quarantine. |
| `thermal_radiative_power_of_divertor_target` | reviewed | Canonical locus requires `at_divertor_target`; preserve quarantine. |
| `time_derivative_of_electron_density` | accepted | No issue snapshot; queue a scoped audit, do not stamp. |
| `toroidal_angle_of_active_limiter_point` | accepted | Dimensionless unit conflicts with an angle; preserve quarantine. |
| `toroidal_coordinate_of_launching_position` | drafted | Canonical locus requires `at_launching_position`; preserve quarantine. |
| `toroidal_coordinate_of_pellet_path` | drafted | Canonical locus requires `at_pellet_path`; preserve quarantine. |
| `toroidal_cumulative_inside_flux_surface_torque` | drafted | `cumulative_` conflicts with the required suffix form; preserve quarantine. |
| `toroidal_current_density` | accepted | No issue snapshot; queue a scoped audit, do not stamp. |
| `total_current_density` | accepted | No issue snapshot; queue a scoped audit, do not stamp. |
| `total_plasma_momentum_field_aligned_convection_velocity` | drafted | Bare `field` needs a physical qualifier; preserve quarantine. |
| `tungsten_density` | accepted | No issue snapshot; queue a scoped audit, do not stamp. |
| `vertical_coordinate_of_outlet_due_to_gas_injection` | reviewed | Canonical locus requires `at_outlet`; preserve quarantine. |
| `volume_integrated_runaway_electron_density` | accepted | `integrated_` conflicts with the required suffix form; preserve quarantine. |

## Follow-up

The validation writer now stamps future persisted verdicts, so this class will
not be recreated through that writer. Remaining work is explicit: materialise
descriptions for the eight valid-but-descriptionless rows and drain exactly
those IDs; audit the 38 legacy quarantines through an evidence-producing scoped
path; and allow the 18 pending candidates to reach ordinary validation. No
signed manifest apply or broad pipeline was run.
