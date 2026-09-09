# Produced-name mirror adjudication

This is a read-only adjudication record. It creates or removes neither a
`PRODUCED_NAME` edge nor a `produced_sn_id` scalar, and it runs no pipeline.

![Verdicts for every multi-target produced-name source](/imas-codex/figures/scalar-edge-authority/produced-name-verdicts.svg)

## Fresh graph measurement

The live, bounded graph census found **9,920** `StandardNameSource` candidates,
of which **5,477** carry `produced_sn_id`. It found **100** rows with more than
one `PRODUCED_NAME` target and **117** produced-name mirror divergences. Those
values are unchanged from the earlier reported figures of 100 and 117; they are
reported here as a fresh measurement rather than treated as a gate constant.

The divergence shape is:

| Mirror shape | Count | Interpretation |
|---|---:|---|
| Multiple edge targets while the scalar is one of them | 100 | The source has an ambiguous relationship set; these are adjudicated below. |
| Scalar names a real target not named by its single edge | 17 | Both identities exist but differ; outside this node's multi-target cohort. |
| Scalar target absent because the target identity is missing | 0 | No missing-scalar-target case in the 117 rows. |
| Scalar has no edge at all | 0 | No scalar-only case in the 117 rows. |

Each query was read-only, bounded to `StandardNameSource` rows and the named
`PRODUCED_NAME` relation, and completed under the ten-second login-node limit.
The graph was queried from the login node because its Neo4j tunnel is
login-node-local.

## Verdict method and totals

For a DD source, the verdict compares the source's own `dd_path` vocabulary to
the candidate identities and descriptions. A path token such as `ion`,
`neutral`, `phi`, `source`, `diffusivity`, or a named geometry object must be
present in the selected candidate's meaning; a generic, wrong-species,
wrong-direction, wrong-object, or wrong-quantity candidate is not selected.
Candidate frequency and review stage are not a rule. Derived rows have neither
`dd_path` nor `signal`, so their source label alone is insufficient under this
node's evidence rule and they remain unadjudicated.

| Verdict class | Count | Consequence for a later repair |
|---|---:|---|
| `scalar-intended` | 66 | Keep the scalar's identity when repairing excess edge targets. |
| `edge-intended` | 8 | Do not use the scalar as authority; retain the named edge target when repairing. |
| `unadjudicated` | 26 | Do not repair either representation until the listed additional evidence exists. |

The rows below record `source id`, scalar identity, and every current edge
identity. `S` means the scalar is the adjudicated intended identity; `E:<id>`
means that named edge is intended; `?` is deliberately not a verdict.

## Scalar intended (66)

| Source id | `produced_sn_id` | All `PRODUCED_NAME` targets | Verdict and path/description evidence |
|---|---|---|---|
| `dd:core_transport/model/profiles_1d/electrons/particles/v` | `radial_electron_convection_velocity` | `electron_convection_velocity`; `radial_electron_convection_velocity` | S — `electrons/particles/v` and the selected description both specify radial electron particle convection. |
| `dd:core_transport/model/profiles_1d/ion/momentum/toroidal/v` | `toroidal_ion_momentum_convection_velocity` | `toroidal_momentum_convection_velocity`; `toroidal_ion_momentum_convection_velocity` | S — path supplies ion and toroidal qualifiers present only in the selected description. |
| `dd:distributions/distribution/global_quantities/collisions/ion/state/torque_fast_phi` | `toroidal_fast_ion_charge_state_torque_due_to_collisions` | `fast_ion_charge_state_torque_due_to_collisions`; `toroidal_fast_ion_charge_state_torque_due_to_collisions` | S — `torque_fast_phi` requires the toroidal collision-torque component. |
| `dd:divertors/divertor/power_recombination_plasma` | `plasma_power_of_divertor_due_to_recombination` | `ion_power_due_to_recombination`; `plasma_power_of_divertor_due_to_recombination` | S — path names divertor plasma recombination power, matching the selected description. |
| `dd:edge_profiles/ggd/j_diamagnetic/phi` | `toroidal_current_density_due_to_diamagnetic_drift` | `toroidal_diamagnetic_current_density`; `toroidal_current_density_due_to_diamagnetic_drift` | S — `j_diamagnetic/phi` selects toroidal current due to diamagnetic drift. |
| `dd:edge_profiles/ggd/j_heat_viscosity/diamagnetic` | `diamagnetic_current_density_due_to_heat_viscosity` | `diamagnetic_heat_current_density_due_to_viscosity`; `diamagnetic_current_density_due_to_heat_viscosity` | S — path contains heat viscosity and diamagnetic current. |
| `dd:edge_profiles/ggd/j_heat_viscosity/phi` | `toroidal_current_density_due_to_heat_viscosity` | `toroidal_current_density_due_to_viscosity`; `toroidal_current_density_due_to_heat_viscosity` | S — selected description retains both heat viscosity and toroidal direction. |
| `dd:edge_profiles/ggd/j_heat_viscosity/poloidal` | `poloidal_current_density_due_to_heat_viscosity` | `poloidal_heat_current_density_due_to_viscosity`; `poloidal_current_density_due_to_heat_viscosity` | S — selected description retains heat viscosity and poloidal direction. |
| `dd:edge_profiles/ggd/j_parallel_viscosity/phi` | `toroidal_current_density_due_to_parallel_viscosity` | `toroidal_current_density_due_to_viscosity`; `toroidal_current_density_due_to_parallel_viscosity` | S — `parallel_viscosity/phi` matches the selected parallel-viscosity toroidal description. |
| `dd:edge_profiles/ggd/j_perpendicular_viscosity/phi` | `toroidal_current_density_due_to_perpendicular_viscosity` | `toroidal_current_density_due_to_viscosity`; `toroidal_current_density_due_to_perpendicular_viscosity` | S — `perpendicular_viscosity/phi` matches only the selected description. |
| `dd:edge_profiles/ggd/vorticity_over_r/z` | `ratio_of_vertical_vorticity_to_major_radius` | `ratio_of_vertical_vorticity_to_major_radius`; `ratio_of_plasma_vorticity_to_major_radius` | S — `z` selects the vertical-vorticity description rather than toroidal vorticity. |
| `dd:edge_sources/source/ggd/ion/state/energy/values` | `ion_charge_state_power_density` | `ion_charge_state_power_density`; `ion_state_power_density` | S — source/ion/state/energy matches the selected charge-state energy-transfer description. |
| `dd:edge_sources/source/ggd/neutral/momentum/parallel` | `parallel_neutral_momentum_source` | `parallel_neutral_momentum_source`; `parallel_momentum_flux` | S — path says neutral momentum source, not momentum flux. |
| `dd:edge_sources/source/ggd/neutral/momentum/phi` | `toroidal_neutral_momentum_source` | `toroidal_neutral_momentum_source`; `toroidal_momentum_flux` | S — path says neutral momentum source with `phi`. |
| `dd:edge_transport/model/ggd/ion/momentum/v/phi` | `toroidal_ion_velocity` | `toroidal_momentum_convection_velocity`; `toroidal_ion_velocity` | S — path's `v/phi` is bulk velocity, as the selected description states, not a convection coefficient. |
| `dd:edge_transport/model/ggd/ion/state/momentum/d/phi` | `toroidal_ion_charge_state_momentum_diffusivity` | `toroidal_momentum_diffusivity`; `toroidal_ion_charge_state_momentum_diffusivity` | S — ion state plus `d/phi` requires charge-state toroidal diffusivity. |
| `dd:edge_transport/model/ggd/ion/state/momentum/v_radial/values` | `radial_ion_charge_state_momentum_convection_velocity` | `radial_ion_state_effective_momentum_convection_velocity`; `radial_ion_charge_state_momentum_convection_velocity` | S — selected description exactly matches radial ion charge-state momentum convection. |
| `dd:edge_transport/model/ggd/momentum/flux_limiter/diamagnetic` | `diamagnetic_momentum_flux_limiter_coefficient` | `momentum_coefficient_due_to_diamagnetic_drift`; `diamagnetic_momentum_flux_limiter_coefficient` | S — path calls out a flux limiter, retained by only the selected description. |
| `dd:edge_transport/model/ggd/neutral/energy/d_pol/values` | `poloidal_neutral_energy_diffusivity` | `poloidal_diffusivity`; `poloidal_neutral_energy_diffusivity` | S — neutral energy and poloidal diffusivity are all represented only by the selected candidate. |
| `dd:edge_transport/model/ggd/neutral/momentum/v/phi` | `toroidal_neutral_momentum_convection_velocity` | `toroidal_momentum_convection_velocity`; `toroidal_neutral_momentum_convection_velocity` | S — path specifies neutral momentum convection and `phi`. |
| `dd:edge_transport/model/ggd/neutral/state/momentum/v/r` | `radial_neutral_internal_state_momentum_convection_velocity` | `radial_neutral_internal_state_momentum_convection_velocity`; `radial_neutral_state_momentum_convection_velocity` | S — selected description explicitly covers radial state-resolved neutral momentum convection. |
| `dd:ferritic/object/centroid/x` | `x_coordinate_of_ferritic_element_centroid` | `x_coordinate_of_ferritic_insert`; `x_coordinate_of_ferritic_element_centroid` | S — `object/centroid/x` matches the element-centroid description. |
| `dd:ferritic/object/centroid/z` | `z_coordinate_of_ferritic_element_centroid` | `z_coordinate_of_ferritic_element_centroid`; `vertical_coordinate_of_diagnostic_component_centre` | S — ferritic centroid is neither a diagnostic component nor an arc point. |
| `dd:gyrokinetics_local/species/potential_energy_gradient_norm` | `derivative_of_normalized_effective_particle_energy_with_respect_to_poloidal_angle` | `gradient_of_effective_potential`; `derivative_of_normalized_effective_particle_energy_with_respect_to_poloidal_angle` | S — normalized potential-energy gradient requires the selected derivative description. |
| `dd:hard_x_rays/emissivity_profile_1d/half_width_internal` | `inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width` | `lower_bound_hard_xray_peak_width`; `normalized_toroidal_hard_xray_peak_lower_bound_width`; `inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width` | S — `half_width_internal` matches the inward normalized-flux peak width. |
| `dd:iron_core/segment/geometry/oblique/alpha` | `alpha_angle_of_iron_core_segment` | `alpha_angle`; `alpha_angle_of_iron_core_segment` | S — selected description is the iron-core segment oblique alpha angle. |
| `dd:pellets/time_slice/pellet/path_geometry/first_point/z` | `vertical_coordinate_of_pellet_path_point` | `vertical_line_of_sight`; `vertical_coordinate_of_pellet_path_point` | S — path is pellet geometry, not a line-of-sight endpoint. |
| `dd:pf_active/coil/element/geometry/oblique/z` | `vertical_coordinate_of_coil_conductor_element` | `vertical_coordinate_of_poloidal_field_coil`; `vertical_coordinate_of_coil_conductor_element` | S — selected description retains conductor element geometry. |
| `dd:pf_active/coil/element/geometry/outline/z` | `vertical_outline` | `vertical_outline`; `vertical_coordinate_of_poloidal_field_coil` | S — outline points, not an arc start, are the source's object. |
| `dd:pf_active/coil/geometry/rectangle/z` | `vertical_coordinate_of_conductor_cross_section` | `vertical_coordinate_of_poloidal_field_coil`; `vertical_coordinate_of_conductor_cross_section` | S — rectangle geometry selects the conductor cross-section centre. |
| `dd:pf_passive/loop/element/geometry/annulus/radius_inner` | `inner_radius_of_passive_loop` | `inner_radius_of_passive_structure`; `inner_radius_of_passive_loop` | S — path names a passive-loop annulus inner radius. |
| `dd:plasma_profiles/ggd/ion/state/velocity_diamagnetic/diamagnetic` | `perpendicular_ion_charge_state_velocity_due_to_diamagnetic_drift` | `perpendicular_ion_charge_state_velocity_due_to_diamagnetic_drift`; `ion_state_diamagnetic_velocity_due_to_diamagnetic_drift` | S — selected description retains the field-normal diamagnetic component. |
| `dd:plasma_profiles/ggd/j_heat_viscosity/phi` | `toroidal_current_density_due_to_heat_viscosity` | `toroidal_current_density_due_to_viscosity`; `toroidal_current_density_due_to_heat_viscosity` | S — `heat_viscosity/phi` is explicit. |
| `dd:plasma_profiles/ggd/mass_density/values` | `total_plasma_mass_density` | `mass_density`; `total_plasma_mass_density` | S — selected description identifies total plasma mass density. |
| `dd:plasma_profiles/ggd/neutral/state/velocity_diamagnetic/r` | `radial_neutral_state_velocity_due_to_diamagnetic_drift` | `radial_neutral_state_velocity_due_to_diamagnetic_drift`; `neutral_state_velocity_due_to_diamagnetic_drift` | S — `r` requires the radial component. |
| `dd:plasma_profiles/profiles_1d/ion/state/z_average` | `volume_averaged_ion_charge_state_average_charge_number` | `volume_averaged_ion_charge_state_average_charge_number`; `ion_state_average_charge_number` | S — selected description supplies the volume average represented by the profile source. |
| `dd:plasma_sources/source/ggd/ion/momentum/phi` | `toroidal_ion_torque_density` | `toroidal_momentum_flux`; `toroidal_ion_torque_density` | S — a source term is torque density, not a flux. |
| `dd:plasma_sources/source/ggd/ion/state/energy/values` | `ion_charge_state_power_density` | `ion_charge_state_power_density`; `ion_state_power_density` | S — ion-state energy source matches selected charge-state power density. |
| `dd:plasma_sources/source/ggd/momentum/phi` | `toroidal_torque_density` | `toroidal_momentum_flux`; `toroidal_torque_density` | S — source term plus `phi` selects torque density. |
| `dd:plasma_sources/source/ggd/neutral/momentum/phi` | `toroidal_neutral_torque_density` | `toroidal_momentum_flux`; `toroidal_neutral_torque_density` | S — neutral source plus `phi` selects neutral torque density. |
| `dd:plasma_sources/source/ggd/neutral/state/momentum/phi` | `toroidal_neutral_internal_state_torque_density` | `neutral_internal_state_torque_density`; `toroidal_neutral_internal_state_torque_density` | S — selected description preserves the toroidal component. |
| `dd:plasma_sources/source/profiles_1d/ion/momentum/radial` | `radial_ion_momentum_source` | `radial_ion_momentum`; `radial_ion_momentum_source` | S — the source path requires the momentum-source, not stored momentum. |
| `dd:plasma_transport/model/ggd/electrons/energy/v_parallel/values` | `parallel_electron_energy_convection_velocity` | `parallel_convection_velocity`; `parallel_electron_energy_convection_velocity` | S — selected description supplies electron energy and field-aligned transport. |
| `dd:plasma_transport/model/ggd/electrons/particles/v_pol/values` | `poloidal_electron_convection_velocity` | `poloidal_convection_velocity`; `poloidal_electron_convection_velocity` | S — selected description supplies electron particle and poloidal qualifiers. |
| `dd:plasma_transport/model/ggd/ion/energy/d/values` | `ion_energy_diffusivity` | `total_ion_energy_diffusivity`; `ion_energy_diffusivity` | S — selected description matches the ion-energy diffusion coefficient. |
| `dd:plasma_transport/model/ggd/ion/momentum/v/phi` | `toroidal_ion_velocity` | `toroidal_momentum_convection_velocity`; `toroidal_ion_velocity` | S — path's velocity field matches bulk ion velocity, not convection coefficient. |
| `dd:plasma_transport/model/ggd/ion/state/energy/v_pol/values` | `poloidal_ion_charge_state_energy_convection_velocity` | `ion_charge_state_energy_convection_velocity`; `poloidal_ion_charge_state_energy_convection_velocity` | S — `state/energy/v_pol` adds the poloidal charge-state qualifier. |
| `dd:plasma_transport/model/ggd/ion/state/particles/d_parallel/values` | `parallel_ion_charge_state_diffusivity` | `ion_state_diffusivity`; `parallel_ion_charge_state_diffusivity` | S — selected description names parallel charge-state particle diffusivity. |
| `dd:plasma_transport/model/ggd/momentum/v/phi` | `toroidal_center_of_mass_velocity` | `toroidal_momentum_convection_velocity`; `toroidal_center_of_mass_velocity` | S — selected description names barycentric velocity, not a convection coefficient. |
| `dd:plasma_transport/model/ggd/neutral/state/momentum/d/phi` | `toroidal_neutral_internal_state_momentum_diffusion_coefficient` | `toroidal_momentum_diffusivity`; `toroidal_neutral_internal_state_momentum_diffusion_coefficient` | S — neutral state, diffusion and `phi` are all selected qualifiers. |
| `dd:plasma_transport/model/ggd/neutral/state/momentum/d/poloidal` | `poloidal_neutral_internal_state_momentum_diffusion_coefficient` | `poloidal_neutral_momentum_diffusion_coefficient`; `poloidal_momentum_diffusivity`; `poloidal_neutral_internal_state_momentum_diffusion_coefficient` | S — selected candidate is the only one retaining neutral internal state and poloidal direction. |
| `dd:plasma_transport/model/ggd/total_ion_energy/d_parallel/values` | `parallel_ion_energy_diffusivity` | `parallel_total_ion_energy_diffusivity`; `parallel_ion_energy_diffusivity` | S — selected description matches ion energy transported in parallel. |
| `dd:plasma_transport/model/profiles_1d/ion/energy/d` | `ion_energy_diffusivity` | `total_ion_energy_diffusivity`; `ion_energy_diffusivity` | S — selected description matches ion-energy diffusivity. |
| `dd:reflectometer_fluctuation/channel/antenna_detection_static/outline/x1` | `first_local_tangential_coordinate_of_reflectometer_antenna` | `x1_coordinate_of_diagnostic_aperture`; `first_local_tangential_coordinate_of_reflectometer_antenna` | S — selected description names reflectometer-antenna local X1 outline coordinate. |
| `dd:reflectometer_fluctuation/channel/antenna_emission_static/outline/x1` | `first_local_tangential_coordinate_of_reflectometer_antenna` | `x1_coordinate_of_diagnostic_aperture`; `first_local_tangential_coordinate_of_reflectometer_antenna` | S — selected description names reflectometer-antenna local X1 outline coordinate. |
| `dd:reflectometer_profile/channel/line_of_sight_emission/first_point/z` | `vertical_coordinate_of_line_of_sight` | `vertical_coordinate_of_line_of_sight`; `vertical_coordinate_of_diagnostic_component_centre` | S — line-of-sight geometry selects the first candidate. |
| `dd:soft_x_rays/channel/filter_window/outline/x1` | `first_local_tangential_coordinate_of_filter` | `x1_coordinate_of_diagnostic_aperture`; `first_local_tangential_coordinate_of_filter` | S — selected description is filter-boundary local X1. |
| `dd:spectrometer_uv/channel/detector/outline/x1` | `first_local_tangential_coordinate_of_optical_element` | `x1_coordinate_of_diagnostic_aperture`; `first_local_tangential_coordinate_of_optical_element` | S — selected description is optical-element local X1. |
| `dd:summary/gas_injection_prefill/helium_4/value` | `accumulated_helium_4_count_due_to_gas_injection` | `accumulated_helium_4_prefill_count_due_to_gas_injection`; `accumulated_helium_4_count_due_to_gas_injection` | S — selected description is the helium-4 gas-injection accumulation. |
| `dd:summary/local/separatrix_average/n_i` | `flux_surface_averaged_ion_density_at_plasma_boundary` | `density_at_separatrix`; `flux_surface_averaged_ion_density_at_plasma_boundary` | S — selected description retains flux-surface average at the boundary. |
| `dd:summary/local/separatrix_average/velocity_phi/beryllium/value` | `flux_surface_averaged_toroidal_beryllium_velocity_at_plasma_boundary` | `toroidal_flux_surface_averaged_beryllium_velocity_at_separatrix`; `flux_surface_averaged_toroidal_beryllium_velocity_at_plasma_boundary` | S — selected description retains species, toroidal, and flux-surface qualifiers. |
| `dd:summary/local/separatrix_average/velocity_phi/helium_4/value` | `flux_surface_averaged_toroidal_helium_4_velocity_at_plasma_boundary` | `toroidal_helium_4_velocity_at_separatrix`; `flux_surface_averaged_toroidal_helium_4_velocity_at_plasma_boundary` | S — selected description retains helium-4 and flux-surface averaging. |
| `dd:summary/local/separatrix_average/velocity_phi/krypton/value` | `flux_surface_averaged_toroidal_krypton_velocity_at_plasma_boundary` | `toroidal_krypton_velocity_at_separatrix`; `flux_surface_averaged_toroidal_krypton_velocity_at_plasma_boundary` | S — selected description retains krypton and flux-surface averaging. |
| `dd:summary/local/separatrix_average/velocity_phi/lithium/value` | `flux_surface_averaged_toroidal_lithium_velocity_at_plasma_boundary` | `toroidal_flux_surface_averaged_lithium_velocity_at_separatrix`; `flux_surface_averaged_toroidal_lithium_velocity_at_plasma_boundary` | S — selected description retains lithium and boundary average. |
| `dd:summary/time_breakdown/value` | `breakdown_initial_time` | `plasma_breakdown_time`; `breakdown_initial_time` | S — selected description is the discharge-breakdown timestamp. |
| `dd:thomson_scattering/channel/position/z` | `vertical_coordinate_of_measurement_position` | `vertical_coordinate_of_measurement_position`; `vertical_coordinate_of_diagnostic_component_centre` | S — path is a measurement position, not a component centre. |

## Edge intended (8)

| Source id | `produced_sn_id` | All `PRODUCED_NAME` targets | Verdict and path/description evidence |
|---|---|---|---|
| `dd:core_transport/model/profiles_1d/ion/momentum/diamagnetic/v` | `diamagnetic_momentum_convection_velocity` | `ion_diamagnetic_momentum_convection_velocity`; `diamagnetic_momentum_convection_velocity` | E:`ion_diamagnetic_momentum_convection_velocity` — source says ion momentum; only this description is ion-specific. |
| `dd:core_transport/model/profiles_1d/ion/particles/v` | `radial_electron_convection_velocity` | `ion_convection_velocity`; `radial_electron_convection_velocity` | E:`ion_convection_velocity` — ion-particle path contradicts the scalar's electron/radial description. |
| `dd:core_transport/model/profiles_1d/neutral/particles/v` | `radial_electron_convection_velocity` | `neutral_convection_velocity`; `radial_electron_convection_velocity` | E:`neutral_convection_velocity` — neutral-particle path contradicts the scalar's electron description. |
| `dd:edge_profiles/ggd/vorticity_over_r/phi` | `ratio_of_vorticity_to_major_radius` | `ratio_of_vorticity_to_major_radius`; `ratio_of_plasma_vorticity_to_major_radius` | E:`ratio_of_plasma_vorticity_to_major_radius` — `phi` selects the toroidal-component description. |
| `dd:plasma_profiles/ggd/vorticity_over_r/phi` | `ratio_of_vorticity_to_major_radius` | `ratio_of_vorticity_to_major_radius`; `ratio_of_plasma_vorticity_to_major_radius` | E:`ratio_of_plasma_vorticity_to_major_radius` — `phi` selects the toroidal-component description. |
| `dd:spectrometer_x_ray_crystal/channel/camera/camera_dimensions` | `extent_of_soft_xray_detector` | `extent_of_soft_xray_detector`; `total_size_of_camera` | E:`total_size_of_camera` — camera dimensions match the X-ray-camera size description, not a soft-X-ray detector. |
| `dd:spi/injector/shatter_cone/direction/x` | `first_measurement_direction_unit_vector_of_shatter_cone` | `first_measurement_direction_unit_vector_of_shatter_cone`; `x_direction_unit_vector_of_shatter_cone` | E:`x_direction_unit_vector_of_shatter_cone` — `direction/x` is an x component, not a whole first direction vector. |
| `dd:spi/injector/shatter_cone/direction/z` | `z_major_axis_unit_vector_of_shatter_cone` | `z_major_axis_unit_vector_of_shatter_cone`; `z_direction_unit_vector_of_shatter_cone` | E:`z_direction_unit_vector_of_shatter_cone` — `direction/z` selects the direction-vector component; no `major_axis` token occurs in the source path. |

## Unadjudicated (26)

| Source id | `produced_sn_id` | All `PRODUCED_NAME` targets | Why no verdict; additional evidence required |
|---|---|---|---|
| `dd:ec_launchers/launcher/beam/phase/curvature` | `wave_curvature_of_beam_tracing_beam` | `wave_curvature_of_beam_tracing_beam`; `wave_curvature_of_wave_beam` | Both descriptions name phase-ellipse curvature; DD field semantics or a source-unit/owner description must distinguish beam-tracing from wave-beam vocabulary. |
| `dd:edge_profiles/ggd/vorticity_over_r/parallel` | `ratio_of_vorticity_to_major_radius` | `ratio_of_vorticity_to_major_radius`; `ratio_of_plasma_vorticity_to_major_radius` | Neither description establishes whether `parallel` is a generic or toroidal component; DD semantic metadata is needed. |
| `dd:edge_profiles/ggd/vorticity_over_r/poloidal` | `ratio_of_vorticity_to_major_radius` | `ratio_of_vorticity_to_major_radius`; `ratio_of_plasma_vorticity_to_major_radius` | The edge is toroidal while the scalar is generic; DD component semantics are needed. |
| `dd:edge_profiles/ggd/vorticity_over_r/r` | `ratio_of_vorticity_to_major_radius` | `ratio_of_vorticity_to_major_radius`; `ratio_of_plasma_vorticity_to_major_radius` | The edge is toroidal while the scalar is generic; DD component semantics are needed. |
| `dd:edge_profiles/ggd/vorticity_over_r/radial` | `ratio_of_vorticity_to_major_radius` | `ratio_of_vorticity_to_major_radius`; `ratio_of_plasma_vorticity_to_major_radius` | The edge is toroidal while the scalar is generic; DD component semantics are needed. |
| `dd:ic_antennas/antenna/module/strap/current` | `root_mean_square_wave_current_of_antenna_strap` | `peak_wave_current_of_antenna_strap_amplitude`; `root_mean_square_wave_current_of_antenna_strap` | The path says only current; DD field documentation or units/convention must establish RMS versus peak amplitude. |
| `dd:plasma_profiles/ggd/vorticity_over_r/parallel` | `ratio_of_vorticity_to_major_radius` | `ratio_of_vorticity_to_major_radius`; `ratio_of_plasma_vorticity_to_major_radius` | Neither description establishes whether `parallel` is generic or toroidal; DD semantic metadata is needed. |
| `dd:plasma_profiles/ggd/vorticity_over_r/poloidal` | `ratio_of_vorticity_to_major_radius` | `ratio_of_vorticity_to_major_radius`; `ratio_of_plasma_vorticity_to_major_radius` | The edge is toroidal while the scalar is generic; DD component semantics are needed. |
| `dd:plasma_profiles/ggd/vorticity_over_r/r` | `ratio_of_vorticity_to_major_radius` | `ratio_of_vorticity_to_major_radius`; `ratio_of_plasma_vorticity_to_major_radius` | The edge is toroidal while the scalar is generic; DD component semantics are needed. |
| `dd:plasma_profiles/ggd/vorticity_over_r/radial` | `ratio_of_vorticity_to_major_radius` | `ratio_of_vorticity_to_major_radius`; `ratio_of_plasma_vorticity_to_major_radius` | The edge is toroidal while the scalar is generic; DD component semantics are needed. |
| `dd:plasma_transport/model/ggd/neutral/state/momentum/flux/radial` | `radial_neutral_internal_state_momentum_flux` | `radial_neutral_internal_state_momentum_flux`; `radial_neutral_state_momentum_flux` | Both accepted descriptions say radial state-resolved neutral momentum flux; the DD field's state-kind convention is required to distinguish `internal_state` from `state`. |
| `derived:conductivity` | `plasma_electrical_conductivity` | `conductivity`; `plasma_electrical_conductivity` | No `dd_path` or signal exists; a derivation receipt or parent relation is required. |
| `derived:fast_ion_state_pressure` | `fast_ion_charge_state_pressure` | `fast_ion_state_pressure`; `fast_ion_charge_state_pressure` | No `dd_path` or signal exists; a derivation receipt or parent relation is required. |
| `derived:fast_neutral_state_pressure` | `parallel_fast_neutral_internal_state_pressure` | `fast_neutral_state_pressure`; `parallel_fast_neutral_internal_state_pressure` | No `dd_path` or signal exists; a derivation receipt or parent relation is required. |
| `derived:flux_surface_averaged_metric` | `flux_surface_averaged_metric` | `flux_surface_averaged_metric`; `flux_surface_normal_contravariant_flux_surface_averaged_metric` | No `dd_path` or signal exists; a derivation receipt or parent relation is required. |
| `derived:ion_state_momentum` | `ion_charge_state_momentum_source` | `ion_state_momentum`; `ion_charge_state_momentum_source` | No `dd_path` or signal exists; a derivation receipt or parent relation is required. |
| `derived:ion_state_momentum_convection_velocity` | `ion_charge_state_momentum_convection_velocity` | `ion_state_momentum_convection_velocity`; `ion_charge_state_momentum_convection_velocity` | No `dd_path` or signal exists; a derivation receipt or parent relation is required. |
| `derived:ion_state_momentum_diffusion_coefficient` | `ion_charge_state_momentum_diffusivity` | `ion_state_momentum_diffusion_coefficient`; `ion_charge_state_momentum_diffusivity` | No `dd_path` or signal exists; a derivation receipt or parent relation is required. |
| `derived:ion_state_momentum_flux_limiter_coefficient` | `ion_charge_state_momentum_flux_limiter_coefficient` | `ion_state_momentum_flux_limiter_coefficient`; `ion_charge_state_momentum_flux_limiter_coefficient` | No `dd_path` or signal exists; a derivation receipt or parent relation is required. |
| `derived:neutral_fraction` | `ratio_of_neutral_state_density_to_total_hydrogenic_density` | `neutral_fraction`; `ratio_of_neutral_state_density_to_total_hydrogenic_density` | No `dd_path` or signal exists; a derivation receipt or parent relation is required. |
| `derived:neutral_state_density` | `neutral_internal_state_density` | `neutral_state_density`; `neutral_internal_state_density` | No `dd_path` or signal exists; a derivation receipt or parent relation is required. |
| `derived:neutral_state_momentum_convection_velocity` | `effective_neutral_internal_state_momentum_velocity_due_to_convection` | `neutral_state_momentum_convection_velocity`; `effective_neutral_internal_state_momentum_velocity_due_to_convection` | No `dd_path` or signal exists; a derivation receipt or parent relation is required. |
| `derived:neutral_state_momentum_diffusion_coefficient` | `neutral_internal_state_momentum_diffusion_coefficient` | `neutral_state_momentum_diffusion_coefficient`; `neutral_internal_state_momentum_diffusion_coefficient` | No `dd_path` or signal exists; a derivation receipt or parent relation is required. |
| `derived:neutral_state_momentum_diffusivity` | `toroidal_neutral_internal_state_momentum_diffusion_coefficient` | `neutral_state_momentum_diffusivity`; `toroidal_neutral_internal_state_momentum_diffusion_coefficient` | No `dd_path` or signal exists; a derivation receipt or parent relation is required. |
| `derived:neutral_state_momentum_flux_limiter_coefficient` | `neutral_internal_state_momentum_flux_limiter_coefficient_over_edge_region` | `neutral_state_momentum_flux_limiter_coefficient`; `neutral_internal_state_momentum_flux_limiter_coefficient_over_edge_region` | No `dd_path` or signal exists; a derivation receipt or parent relation is required. |
| `derived:radius_of_ferritic_element` | `radial_coordinate_of_ferritic_element_centroid` | `radial_coordinate_of_ferritic_element_centroid`; `radius_of_ferritic_element` | No `dd_path` or signal exists; a derivation receipt or parent relation is required. |

## Repair boundary

This document is an adjudication input only. A repair may remove the extra edge
only for the 66 scalar-intended and 8 edge-intended rows after its own fresh
recheck; it must leave all 26 unadjudicated rows unchanged. The 17 one-edge
divergences are not included here because this node is fenced to the
multi-target cohort, but the fresh count records that they remain.
