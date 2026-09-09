# The earlier deletions: what 399 still-absent identities cost, and whether any was published or ratified

Census of the identities this delete path removed **before 2026-09-08** that still remain
absent, with their apportioned LLM spend and a per-identity publication verdict. The verdict is
established from **catalog cut membership** and **`unchanged_ratification` records**, never from
provenance labels. Snapshot of the live graph: **2026-09-09T08:22Z**.

## Result

The delete path (`StandardNameChange` with `operation='remove_derived_parent'`) removed
**539 distinct identities all time**. Of these, **458** were removed before
`2026-09-08T00:00:00Z` and **399 remain absent** at the census snapshot; 59 have been
re-created and are present now.

| Measure | Value |
|---|---|
| Identities removed before 2026-09-08, still absent | **399** |
| — carrying recorded spend | 188 |
| — with zero recorded spend | 211 |
| Apportioned LLM spend on the 399 (six decimal places) | **$57.392665** |
| Cost-row attributions | 1,115 |
| PUBLISHED (member of the issued catalog cut) | **0** |
| RATIFIED (carries an `unchanged_ratification` record) | **3** |
| NEITHER | 396 |

No identity this path removed before 2026-09-08 was **published**: none of the 458 appears in
the only issued catalog cut, the 208-`emitted_identities` array of
`v0.4.0rc6+west-task-2e:.export_report.json`. Three were **ratified** and still carry that
loss: `time`, `plasma_beta` and
`flux_surface_averaged_current_density_due_to_wave_driven_current_drive`, each with three
`unchanged_ratification` records from Catalog PR 3 (2026-09-01/02), each deleted days later.
Six further ratified identities were removed on 2026-09-05 and have since been re-created
(present now, outside this cohort); they are reported as context in the ratification section.

The verdict column varies (RATIFIED versus NEITHER), so the cohort is not a uniform
classification.

## The boundary

Cohort = `DISTINCT to_name` of `StandardNameChange` rows with
`operation='remove_derived_parent'` and `changed_at < datetime('2026-09-08T00:00:00Z')`,
minus identities with a live `StandardName {id: to_name}` node at the snapshot. Absence is
matched on `StandardName.id` (the identity itself), not on a display name.

The pre-09-08 history spans **2026-07-29T13:43Z to 2026-09-07T06:58Z**, with per-day distinct
removal counts: 07-29: 20, 07-30: 13, 07-31: 11, 08-01: 7, 08-08: 9, 08-09: 8, 08-11: 16,
**08-16: 349**, 08-19: 14, 08-20: 11, 08-21: 9, 08-22: 11, 08-25: 34, 08-26: 14, 09-01: 20,
09-02: 18, 09-03: 22, 09-04: 17, 09-05: 35, 09-06: 24, 09-07: 12. The 08-16 single-day cohort
of 349 (its own 406 change rows) is the bulk of the pre-09-08 loss; it includes the 
mislabelled-population families this plan has elsewhere dispositioned.

Twelve of the 458 were removed again on 2026-09-08; eleven of those are present now, and one —
`ion_density_at_pedestal_top` — is absent and so is counted in this cohort. Identities removed
only on 2026-09-08 (the 11:57Z incident pass) are **not** in this cohort; they are the subject
of the other rows of this plan.

## Reconciliation against the all-time figures

The plan records $100.47 of apportioned spend over 491 still-absent identities at
2026-09-08T19:00Z, of which $43.12 was the 11:57Z pass. The census snapshot, taken after the
first restores and regenerations landed, re-derives the same two numbers with the same formula;

| Population | Identities | Apportioned spend (six dp) |
|---|---:|---:|
| Earlier removals, still absent (this census) | 399 | $57.392665 |
| 2026-09-08 pass removals, still absent | 80 | $41.276933 |
| minus `ion_density_at_pedestal_top` (removed on both sides; counted, absent, in both rows) | −1 | −$0.108893 |
| **All-time still absent, census snapshot** | **478** | **$98.560704** |

The plan's all-time figure of $100.47 (491 identities) exceeds the census snapshot by
**$1.909296 across 13 identities**: those identities were absent at 2026-09-08T19:00Z and have
since been re-created (61 all-time removed identities are present now, against 48 then). The
earlier remainder implied by the plan — $100.47 − $43.12 = $57.35 — sits within
$0.04 of this census's $57.392665; the difference is the plan's two-decimal rounding of both
headline figures plus the drift above.

Two subtleties keep the reconciliation arithmetic honest. First, one identity,
`ion_density_at_pedestal_top`, was removed both before and on 2026-09-08 and is absent now, so
it belongs to both cohort rows and must be subtracted once (apportioned spend $0.108893, two
cost rows). Second, the all-time and per-cohort totals are all computed over the **same
snapshot**, so they decompose exactly: 57.392665 + 41.276933 − 0.108893 = 98.560704 (six
decimal places, matching to $10^-6$).

## What it cost

The 399 still-absent earlier identities carry **$57.392665** of apportioned LLM spend across
1,115 cost-row attributions. The phase decomposition shows the same shape the incident pass
showed: the loss is almost entirely documentation, which the archive recovers at zero LLM cost.

| Phase | Apportioned spend | Cost rows |
|---|---:|---:|
| `generate_docs` | $22.134433 | 297 |
| `review_docs` | $20.513855 | 499 |
| `refine_docs` | $14.460490 | 128 |
| `enrich_parents` | $0.132244 | 170 |
| `refine_name` | $0.106001 | 1 |
| `generate_name` | $0.045642 | 20 |
| **Total** | **$57.392665** | **1,115** |

Documentation phases carry $57.108778 of the $57.392665 (99.5%); name composition, review and
refinement combined carry $0.151643, and enrichment $0.132244. The ten largest individual
losses: `thermal_electron_torque_density_due_to_coulomb_collisions_with_electrons`
$1.771703, `ion_density_at_magnetic_axis` $1.307780, `minimum_magnetic_field` $1.248088,
`mode_width` $1.052413, `trapped_thermal_electron_torque_density_due_to_coulomb_collisions_with_electrons`
$0.991549, `electron_torque_density_due_to_coulomb_collisions_with_electrons` $0.856055,
`helium_4_count_due_to_gas_injection` $0.855869, `power_of_wave_beam` $0.818809,
`internal_state_energy_flux_at_wall` $0.801005, `perpendicular_wave_vector` $0.744741.

## Verdicts

### PUBLISHED — none

The repository records exactly one issued catalog cut: the 208-element `emitted_identities`
array of `v0.4.0rc6+west-task-2e:.export_report.json` at tag `v0.4.0rc6+west-task-2e` in the
catalog checkout (`all_gates_passed: true`, 208 of 226 candidates exported, all 208
`quantity` role). No other commit or tag in that repository carries an `.export_report.json`,
so no earlier cut exists to have named any earlier deletion. The intersection of that cut with
the 458 earlier-removed identities is **empty**: zero of the 399 still-absent (and zero of the
458 regardless of presence) was ever named by a catalog cut. The single cut member this reaper
has ever taken, `etendue_of_spectrometer_channel`, was removed in the 2026-09-08 pass, not
before it, and is therefore outside this cohort.

### RATIFIED — three, all still absent

RATIFIED means the identity carries at least one `StandardNameChange` row with
`operation='unchanged_ratification'` (the promoted review outcome that records
`unchanged_ratification` — a name re-issued without content change by a catalog PR). The
change node survives the deletion's `DETACH DELETE` as an orphan, so the ratification record
is durable even though the identity's `HAS_INTERNAL_CHANGE` edge is severed.

All three ratified-and-still-absent identities were ratified in **Catalog PR 3** — the reason
string on every one of their records is *"Catalog PR 3 recorded the unchanged_ratification
editorial outcome."* — over 2026-09-01T21:10Z / 23:45Z / 09-02T00:55Z (three review rounds),
and were deleted 09-03, 09-05 and 09-06 respectively:

| Identity | Ratification records | Ratified (PR 3) | Deleted | Apportioned spend | Verdict |
|---|---|---:|---|---|---:|---|
| `time` | 3 | 2026-09-01/02 | 2026-09-03T08:00:50Z | $0.612605 | RATIFIED |
| `plasma_beta` | 3 | 2026-09-01/02 | 2026-09-06T15:27:10Z | $0.589199 | RATIFIED |
| `flux_surface_averaged_current_density_due_to_wave_driven_current_drive` | 3 | 2026-09-01/02 | 2026-09-05T14:13:16Z | $0.512452 | RATIFIED |

Six further earlier-removed identities carry the same Catalog PR 3 ratification but are
**present now** — removed by the same 2026-09-05T14:13Z pass and re-created since — so they
fall outside the still-absent cohort yet answer the plan's question about ratification loss:
`magnetic_field_at_pedestal_top_low_field_side` ($1.445158, 3 records),
`spectral_signal_to_noise_ratio_of_spectrometer_channel` ($1.250575, 3),
`ion_density_at_plasma_boundary` ($0.858704, 3), `length_of_interferometer_beam` ($0.654113,
3), `voltage_of_ion_cyclotron_heating_antenna` ($0.422601, 2),
`wave_current_of_antenna_strap` ($0.279112, 3). In total, nine of the 458 earlier-removed
identities were ratified at deletion time: three retain that loss, six have been recovered by
re-creation.

This does not reproduce the plan's note that *"four are already known to carry
`unchanged_ratification` records."* That figure predates this census and is not reproduced by
it under either reading (still-absent: three; any earlier-removal: nine). The difference does
not change any verdict: none of the earlier deletions was published, and the three ratified
losses are identified by identity, record and date above.

### NEITHER — 396

The remaining 396 still-absent earlier identities carry no catalog-cut membership and no
`unchanged_ratification` record. Of these, 211 carry zero recorded spend. NEITHER is a verdict
about publication and ratification only; it is not an assertion that the identity was free of
investment — 185 of the 396 NEITHER rows still carry spend (their spend appears in the total
above).

## The full cohort (399 absent identities)

Spend is apportioned to six decimal places with the canonical per-cost-row formula; a small
non-zero charge therefore never displays as zero. `Rat` is the count of
`unchanged_ratification` records; every row's verdict is PUBLISHED, RATIFIED or NEITHER.
Sorted by spend descending then identity.

| identity | apportioned spend (USD) | cost rows | Rat | verdict |
|---|---|---:|---:|---|
| `thermal_electron_torque_density_due_to_coulomb_collisions_with_electrons` | 1.771703 | 24 | 0 | NEITHER |
| `ion_density_at_magnetic_axis` | 1.307780 | 16 | 0 | NEITHER |
| `minimum_magnetic_field` | 1.248088 | 26 | 0 | NEITHER |
| `mode_width` | 1.052413 | 12 | 0 | NEITHER |
| `trapped_thermal_electron_torque_density_due_to_coulomb_collisions_with_electrons` | 0.991549 | 18 | 0 | NEITHER |
| `electron_torque_density_due_to_coulomb_collisions_with_electrons` | 0.856055 | 12 | 0 | NEITHER |
| `helium_4_count_due_to_gas_injection` | 0.855869 | 10 | 0 | NEITHER |
| `power_of_wave_beam` | 0.818809 | 12 | 0 | NEITHER |
| `internal_state_energy_flux_at_wall` | 0.801005 | 7 | 0 | NEITHER |
| `perpendicular_wave_vector` | 0.744741 | 13 | 0 | NEITHER |
| `thermal_electron_energy_flux` | 0.733296 | 11 | 0 | NEITHER |
| `per_toroidal_mode_current_density_due_to_wave_driven_current_drive` | 0.717436 | 12 | 0 | NEITHER |
| `radial_electron_density` | 0.664028 | 10 | 0 | NEITHER |
| `ion_density_at_limiter` | 0.638192 | 9 | 0 | NEITHER |
| `coordinate` | 0.613031 | 10 | 0 | NEITHER |
| `time` | 0.612605 | 22 | 3 | RATIFIED |
| `effective_neutral_internal_state_velocity_due_to_diamagnetic_drift` | 0.606836 | 10 | 0 | NEITHER |
| `plasma_beta` | 0.589199 | 11 | 3 | RATIFIED |
| `average_temperature_at_magnetic_axis` | 0.577440 | 8 | 0 | NEITHER |
| `bremsstrahlung_count_at_detector_pixel` | 0.549897 | 9 | 0 | NEITHER |
| `perturbed_magnetic_field` | 0.544754 | 8 | 0 | NEITHER |
| `flux_surface_averaged_current_density_due_to_wave_driven_current_drive` | 0.512452 | 11 | 3 | RATIFIED |
| `neutral_species_energy_convection_velocity` | 0.508351 | 9 | 0 | NEITHER |
| `momentum_velocity_due_to_convection` | 0.504272 | 7 | 0 | NEITHER |
| `trapped_fast_electron_torque_density_due_to_collisions` | 0.503645 | 12 | 0 | NEITHER |
| `density_at_outboard_midplane` | 0.489340 | 9 | 0 | NEITHER |
| `power_density_of_breeder_blanket_module` | 0.485952 | 8 | 0 | NEITHER |
| `ion_density_at_divertor_target` | 0.474359 | 6 | 0 | NEITHER |
| `average_temperature_at_pedestal_top` | 0.470325 | 6 | 0 | NEITHER |
| `plasma_momentum_diffusion_coefficient` | 0.467726 | 11 | 0 | NEITHER |
| `momentum_due_to_avalanche` | 0.466349 | 6 | 0 | NEITHER |
| `hydrogen_prefill_count` | 0.462545 | 7 | 0 | NEITHER |
| `line_averaged_plasma_velocity` | 0.460592 | 7 | 0 | NEITHER |
| `perturbed_current_density` | 0.448932 | 9 | 0 | NEITHER |
| `charge_at_internal_transport_barrier` | 0.448143 | 8 | 0 | NEITHER |
| `plasma_momentum_source` | 0.445735 | 8 | 0 | NEITHER |
| `particle_power_density_due_to_collisions` | 0.438702 | 7 | 0 | NEITHER |
| `area_of_filter` | 0.435408 | 8 | 0 | NEITHER |
| `power_of_breeder_blanket` | 0.432741 | 8 | 0 | NEITHER |
| `volume_averaged_runaway_electron_current_density` | 0.430569 | 9 | 0 | NEITHER |
| `temperature_at_outboard_midplane` | 0.427637 | 8 | 0 | NEITHER |
| `flux_limiter_coefficient_over_edge_region` | 0.417394 | 5 | 0 | NEITHER |
| `pressure_at_plasma_boundary` | 0.413396 | 8 | 0 | NEITHER |
| `heating_power_of_breeder_blanket_layer` | 0.407897 | 7 | 0 | NEITHER |
| `average_temperature_at_internal_transport_barrier` | 0.406447 | 8 | 0 | NEITHER |
| `wavelength_of_spectrometer_channel` | 0.401868 | 7 | 0 | NEITHER |
| `hydrogen_count_due_to_gas_injection` | 0.397789 | 6 | 0 | NEITHER |
| `ion_density_at_internal_transport_barrier` | 0.385516 | 6 | 0 | NEITHER |
| `thermal_power_of_breeder_blanket_module` | 0.384317 | 6 | 0 | NEITHER |
| `charge_at_limiter` | 0.383246 | 5 | 0 | NEITHER |
| `heating_power_of_breeder_blanket` | 0.375853 | 7 | 0 | NEITHER |
| `particle_power_density_due_to_thermalization` | 0.372479 | 7 | 0 | NEITHER |
| `temperature_at_plasma_boundary` | 0.364900 | 8 | 0 | NEITHER |
| `factor_of_visible_camera` | 0.364542 | 5 | 0 | NEITHER |
| `charge_at_magnetic_axis` | 0.359272 | 7 | 0 | NEITHER |
| `radiance_of_soft_xray_detector` | 0.351135 | 5 | 0 | NEITHER |
| `per_toroidal_mode_electric_field` | 0.351109 | 8 | 0 | NEITHER |
| `ion_decay_length_over_scrape_off_layer` | 0.343358 | 6 | 0 | NEITHER |
| `electron_source_rate_due_to_dreicer` | 0.339654 | 5 | 0 | NEITHER |
| `radius_of_plasma_boundary` | 0.337335 | 7 | 0 | NEITHER |
| `flux_due_to_radiation` | 0.335160 | 9 | 0 | NEITHER |
| `internal_state_energy_convection_velocity` | 0.334437 | 5 | 0 | NEITHER |
| `energy_of_plant_component_port` | 0.328116 | 5 | 0 | NEITHER |
| `particle_flux_due_to_perturbed_parallel_vector_potential` | 0.324990 | 5 | 0 | NEITHER |
| `wave_vector` | 0.324186 | 9 | 0 | NEITHER |
| `flux_of_correction_coil` | 0.321387 | 5 | 0 | NEITHER |
| `electron_source_rate_due_to_hot_tail` | 0.318995 | 5 | 0 | NEITHER |
| `pressure_of_mass_spectrometer_channel` | 0.316147 | 7 | 0 | NEITHER |
| `ion_charge_state_absorbed_wave_power` | 0.315582 | 6 | 0 | NEITHER |
| `power_of_limiter_tile` | 0.315568 | 7 | 0 | NEITHER |
| `neutral_internal_state_momentum_convected_velocity` | 0.312614 | 10 | 0 | NEITHER |
| `ion_power_due_to_charge_exchange` | 0.311544 | 6 | 0 | NEITHER |
| `internal_state_momentum_source` | 0.306478 | 6 | 0 | NEITHER |
| `flux_of_divertor_due_to_recycling` | 0.305210 | 6 | 0 | NEITHER |
| `factor_of_spectrometer` | 0.297942 | 8 | 0 | NEITHER |
| `normalized_momentum_flux_due_to_e_cross_b_drift` | 0.297577 | 7 | 0 | NEITHER |
| `temperature_at_outlet` | 0.297514 | 7 | 0 | NEITHER |
| `collisionality_at_pedestal_top` | 0.295872 | 6 | 0 | NEITHER |
| `temperature_at_internal_transport_barrier` | 0.294792 | 7 | 0 | NEITHER |
| `temperature_at_limiter` | 0.294752 | 6 | 0 | NEITHER |
| `power_density_of_limiter_tile` | 0.289346 | 4 | 0 | NEITHER |
| `carbon_count_due_to_gas_injection` | 0.288637 | 6 | 0 | NEITHER |
| `parallel_normalized_perturbed_current_density` | 0.287953 | 7 | 0 | NEITHER |
| `current_density_due_to_collisions` | 0.287351 | 6 | 0 | NEITHER |
| `frequency_at_measurement_position` | 0.287267 | 5 | 0 | NEITHER |
| `internal_state_momentum_diffusion_coefficient` | 0.285197 | 5 | 0 | NEITHER |
| `co_passing_fast_electron_torque_density_due_to_collisions` | 0.281358 | 8 | 0 | NEITHER |
| `internal_state_momentum_flux_limiter_coefficient_over_edge_region` | 0.281224 | 3 | 0 | NEITHER |
| `fraction_of_breeder_blanket_layer` | 0.276364 | 5 | 0 | NEITHER |
| `rate_of_neutron_detector` | 0.270298 | 5 | 0 | NEITHER |
| `temperature_at_pedestal_top` | 0.268206 | 6 | 0 | NEITHER |
| `period_of_fiber_optic_current_sensor` | 0.266156 | 6 | 0 | NEITHER |
| `source_rate_due_to_dreicer` | 0.265916 | 3 | 0 | NEITHER |
| `power_of_plant_component_port` | 0.264420 | 5 | 0 | NEITHER |
| `charge_at_divertor_target` | 0.263390 | 6 | 0 | NEITHER |
| `electron_collisionality_at_pedestal_top` | 0.260726 | 3 | 0 | NEITHER |
| `photon_radiance_due_to_charge_exchange` | 0.259195 | 7 | 0 | NEITHER |
| `refractive_index_of_optical_element` | 0.256512 | 5 | 0 | NEITHER |
| `energy_at_launching_position` | 0.253608 | 5 | 0 | NEITHER |
| `normalized_perturbed_magnetic_field` | 0.253463 | 8 | 0 | NEITHER |
| `internal_state_velocity_due_to_diamagnetic_drift` | 0.242688 | 5 | 0 | NEITHER |
| `normalized_perturbed_current_density` | 0.240329 | 7 | 0 | NEITHER |
| `neutron_flux_of_correction_coil` | 0.238676 | 5 | 0 | NEITHER |
| `flux_of_neutron_detector` | 0.237938 | 4 | 0 | NEITHER |
| `flux_at_wall_due_to_radiation` | 0.237736 | 5 | 0 | NEITHER |
| `ion_power_at_inner_divertor_target` | 0.237250 | 5 | 0 | NEITHER |
| `electron_energy_convected_velocity` | 0.233508 | 6 | 0 | NEITHER |
| `power_density_of_breeder_blanket_layer` | 0.231505 | 5 | 0 | NEITHER |
| `internal_state_particle_flux_at_wall_due_to_recombination` | 0.226637 | 5 | 0 | NEITHER |
| `radiance_due_to_charge_exchange` | 0.220731 | 6 | 0 | NEITHER |
| `critical_temperature_at_pedestal_top` | 0.214610 | 3 | 0 | NEITHER |
| `calibration_factor_of_spectrometer` | 0.213720 | 5 | 0 | NEITHER |
| `angle_of_iron_core_segment` | 0.208506 | 4 | 0 | NEITHER |
| `electron_source_rate_due_to_compton_scattering` | 0.208350 | 3 | 0 | NEITHER |
| `flux_due_to_recycling` | 0.207890 | 3 | 0 | NEITHER |
| `flux_of_spectrometer_channel` | 0.206097 | 4 | 0 | NEITHER |
| `temperature_at_inlet` | 0.201186 | 5 | 0 | NEITHER |
| `thermal_power_at_outlet` | 0.196845 | 4 | 0 | NEITHER |
| `current_of_mass_spectrometer_channel` | 0.190891 | 4 | 0 | NEITHER |
| `decay_time_due_to_disruption` | 0.190607 | 5 | 0 | NEITHER |
| `internal_state_velocity_due_to_convection` | 0.187901 | 3 | 0 | NEITHER |
| `source_rate_due_to_hot_tail` | 0.187301 | 3 | 0 | NEITHER |
| `charge_at_plasma_boundary` | 0.186978 | 5 | 0 | NEITHER |
| `source_rate_due_to_compton_scattering` | 0.183358 | 3 | 0 | NEITHER |
| `internal_state_momentum_velocity_due_to_convection` | 0.181905 | 2 | 0 | NEITHER |
| `average_temperature_at_limiter` | 0.179477 | 3 | 0 | NEITHER |
| `power_density_due_to_thermalization` | 0.179455 | 4 | 0 | NEITHER |
| `particle_flux_of_divertor_due_to_recycling` | 0.177491 | 3 | 0 | NEITHER |
| `heating_power_of_breeder_blanket_shield` | 0.176941 | 3 | 0 | NEITHER |
| `momentum_flux_limiter_coefficient_over_edge_region` | 0.174168 | 5 | 0 | NEITHER |
| `flux_due_to_perturbed_parallel_vector_potential` | 0.170351 | 3 | 0 | NEITHER |
| `photon_radiance_of_spectral_line_due_to_charge_exchange` | 0.169435 | 5 | 0 | NEITHER |
| `wavelength_of_camera` | 0.169385 | 3 | 0 | NEITHER |
| `neutron_flux_of_toroidal_field_coil` | 0.168286 | 3 | 0 | NEITHER |
| `fluence_at_divertor_target` | 0.164230 | 3 | 0 | NEITHER |
| `heat_diffusivity` | 0.160288 | 3 | 0 | NEITHER |
| `area_of_optical_element` | 0.159194 | 3 | 0 | NEITHER |
| `absorptivity_of_filter` | 0.159050 | 3 | 0 | NEITHER |
| `power_over_scrape_off_layer` | 0.157626 | 3 | 0 | NEITHER |
| `internal_state_density` | 0.154983 | 5 | 0 | NEITHER |
| `energy_of_neutron_detector` | 0.147930 | 3 | 0 | NEITHER |
| `fraction_of_wave_beam` | 0.146414 | 3 | 0 | NEITHER |
| `thermal_power_at_inlet` | 0.145429 | 3 | 0 | NEITHER |
| `power_due_to_charge_exchange` | 0.141050 | 2 | 0 | NEITHER |
| `power_at_inlet` | 0.140610 | 3 | 0 | NEITHER |
| `normalized_wave_vector` | 0.139795 | 7 | 0 | NEITHER |
| `radius_of_correction_coil` | 0.137565 | 6 | 0 | NEITHER |
| `xenon_prefill_count` | 0.136436 | 3 | 0 | NEITHER |
| `flux_of_toroidal_field_coil` | 0.132956 | 2 | 0 | NEITHER |
| `fast_electron_torque_due_to_collisions` | 0.132911 | 7 | 0 | NEITHER |
| `count_due_to_gas_injection` | 0.123608 | 4 | 0 | NEITHER |
| `area_of_divertor_tile` | 0.122607 | 1 | 0 | NEITHER |
| `particle_power_due_to_collisions` | 0.118432 | 3 | 0 | NEITHER |
| `length_of_ferritic_element` | 0.115522 | 5 | 0 | NEITHER |
| `ion_density_at_pedestal_top` | 0.108893 | 2 | 0 | NEITHER |
| `neutron_flux_due_to_fusion_reactions` | 0.104939 | 4 | 0 | NEITHER |
| `density_at_launching_position` | 0.104317 | 3 | 0 | NEITHER |
| `peak_wave_current_of_antenna_strap` | 0.099229 | 4 | 0 | NEITHER |
| `internal_state_velocity_due_to_e_cross_b_drift` | 0.096733 | 3 | 0 | NEITHER |
| `internal_state_particle_flux` | 0.096046 | 3 | 0 | NEITHER |
| `total_neutral_momentum_diffusivity` | 0.095014 | 4 | 0 | NEITHER |
| `efficiency_of_filter` | 0.092302 | 4 | 0 | NEITHER |
| `electron_power_due_to_collisions` | 0.086269 | 3 | 0 | NEITHER |
| `impurity_ion_velocity` | 0.084335 | 4 | 0 | NEITHER |
| `internal_state_heat_diffusivity` | 0.078954 | 3 | 0 | NEITHER |
| `internal_state_particle_flux_at_wall` | 0.077479 | 3 | 0 | NEITHER |
| `internal_state_energy_diffusion_coefficient` | 0.077206 | 3 | 0 | NEITHER |
| `area_of_neutron_detector` | 0.076647 | 3 | 0 | NEITHER |
| `kinetic_energy_flux_at_wall` | 0.072289 | 3 | 0 | NEITHER |
| `flux_due_to_e_cross_b_drift` | 0.072262 | 4 | 0 | NEITHER |
| `density_over_scrape_off_layer` | 0.071116 | 3 | 0 | NEITHER |
| `internal_state_energy_flux` | 0.069724 | 3 | 0 | NEITHER |
| `internal_state_velocity` | 0.067103 | 3 | 0 | NEITHER |
| `total_thermal_electron_energy_flux` | 0.066501 | 3 | 0 | NEITHER |
| `power_at_outlet` | 0.066406 | 3 | 0 | NEITHER |
| `pressure_at_cooling_circuit_inlet` | 0.064991 | 3 | 0 | NEITHER |
| `radius_of_toroidal_field_coil` | 0.059747 | 3 | 0 | NEITHER |
| `angle_of_pellet_injector` | 0.056618 | 4 | 0 | NEITHER |
| `line_integrated_impurity_ion_velocity` | 0.053859 | 3 | 0 | NEITHER |
| `energy_convected_velocity` | 0.053320 | 4 | 0 | NEITHER |
| `magnetic_flux_due_to_resistive_flux_consumption` | 0.052601 | 3 | 0 | NEITHER |
| `convected_velocity` | 0.051605 | 3 | 0 | NEITHER |
| `flux_due_to_fusion_reactions` | 0.049704 | 4 | 0 | NEITHER |
| `phase_of_antenna_strap` | 0.004446 | 2 | 0 | NEITHER |
| `density` | 0.003298 | 2 | 0 | NEITHER |
| `length_of_conductor_cross_section` | 0.001240 | 1 | 0 | NEITHER |
| `deuterated_methane_prefill_count` | 0.000712 | 1 | 0 | NEITHER |
| `weight_of_flux_loop` | 0.000533 | 1 | 0 | NEITHER |
| `absorbed_energy_of_plasma_facing_component` | 0.000000 | 0 | 0 | NEITHER |
| `absorbed_power` | 0.000000 | 0 | 0 | NEITHER |
| `absorbed_power_at_wall` | 0.000000 | 0 | 0 | NEITHER |
| `alpha_parameter_at_pedestal` | 0.000000 | 0 | 0 | NEITHER |
| `angle_at_constraint_position` | 0.000000 | 0 | 0 | NEITHER |
| `angle_of_shatter_cone` | 0.000000 | 0 | 0 | NEITHER |
| `area_of_divertor` | 0.000000 | 0 | 0 | NEITHER |
| `area_of_divertor_target` | 0.000000 | 0 | 0 | NEITHER |
| `average_temperature_at_midplane` | 0.000000 | 0 | 0 | NEITHER |
| `average_temperature_at_post_sawtooth_crash` | 0.000000 | 0 | 0 | NEITHER |
| `beam_tilt_angle_of_neutral_beam_injector` | 0.000000 | 0 | 0 | NEITHER |
| `bulk_electron_temperature_at_last_closed_flux_surface` | 0.000000 | 0 | 0 | NEITHER |
| `calibration_coefficient_of_spectrometer_channel` | 0.000000 | 0 | 0 | NEITHER |
| `calibration_factor_at_line_of_sight` | 0.000000 | 0 | 0 | NEITHER |
| `calibration_wavelength_of_spectrometer_channel` | 0.000000 | 0 | 0 | NEITHER |
| `change_in_rotation_frequency_due_to_e_cross_b_drift` | 0.000000 | 0 | 0 | NEITHER |
| `co_passing_particle_density` | 0.000000 | 0 | 0 | NEITHER |
| `coefficient_of_spectrometer_channel` | 0.000000 | 0 | 0 | NEITHER |
| `coolant_absorbed_energy_of_plasma_facing_component` | 0.000000 | 0 | 0 | NEITHER |
| `coolant_absorbed_power_at_wall` | 0.000000 | 0 | 0 | NEITHER |
| `coolant_mass` | 0.000000 | 0 | 0 | NEITHER |
| `count_at_midplane_due_to_gas_injection` | 0.000000 | 0 | 0 | NEITHER |
| `count_at_pellet_path_due_to_pellet_injection` | 0.000000 | 0 | 0 | NEITHER |
| `count_at_wall` | 0.000000 | 0 | 0 | NEITHER |
| `count_due_to_pellet_injection` | 0.000000 | 0 | 0 | NEITHER |
| `count_of_ion_state` | 0.000000 | 0 | 0 | NEITHER |
| `critical_alpha_parameter` | 0.000000 | 0 | 0 | NEITHER |
| `critical_energy` | 0.000000 | 0 | 0 | NEITHER |
| `critical_momentum_due_to_hot_tail` | 0.000000 | 0 | 0 | NEITHER |
| `current_density_flux_surface_averaged_due_to_wave_driven_current_drive` | 0.000000 | 0 | 0 | NEITHER |
| `decay_time_at_magnetic_axis_due_to_disruption` | 0.000000 | 0 | 0 | NEITHER |
| `density_at_pellet_path` | 0.000000 | 0 | 0 | NEITHER |
| `density_of_isotope` | 0.000000 | 0 | 0 | NEITHER |
| `derivative_with_respect_to_poloidal_magnetic_flux_coordinate_of_poloidal_current_function` | 0.000000 | 0 | 0 | NEITHER |
| `deuterated_methane_count_due_to_gas_injection` | 0.000000 | 0 | 0 | NEITHER |
| `diamagnetic_momentum_damping_rate` | 0.000000 | 0 | 0 | NEITHER |
| `diamagnetic_momentum_source` | 0.000000 | 1 | 0 | NEITHER |
| `diamagnetic_vorticity` | 0.000000 | 0 | 0 | NEITHER |
| `difference_of_total_hydrogenic_density_and_neutral_state_density` | 0.000000 | 0 | 0 | NEITHER |
| `difference_of_total_neutral_density_and_neutral_density_of_isotope` | 0.000000 | 0 | 0 | NEITHER |
| `diffusivity_due_to_diamagnetic_drift` | 0.000000 | 0 | 0 | NEITHER |
| `efficiency_of_hard_xray_detector` | 0.000000 | 0 | 0 | NEITHER |
| `efficiency_of_soft_xray_detector` | 0.000000 | 0 | 0 | NEITHER |
| `efficiency_of_thomson_scattering_detector` | 0.000000 | 0 | 0 | NEITHER |
| `electron_larmor_radius_at_pedestal_top_high_field_side` | 0.000000 | 0 | 0 | NEITHER |
| `electron_pressure_at_post_sawtooth_crash` | 0.000000 | 0 | 0 | NEITHER |
| `electron_temperature_at_last_closed_flux_surface` | 0.000000 | 0 | 0 | NEITHER |
| `ellipticity_of_polarimeter_beam` | 0.000000 | 0 | 0 | NEITHER |
| `energy_decay_time_due_to_disruption` | 0.000000 | 0 | 0 | NEITHER |
| `energy_due_to_ohmic_dissipation` | 0.000000 | 0 | 0 | NEITHER |
| `energy_flux_due_to_e_cross_b_drift` | 0.000000 | 0 | 0 | NEITHER |
| `energy_flux_due_to_perturbed_parallel_magnetic_field` | 0.000000 | 0 | 0 | NEITHER |
| `energy_flux_due_to_perturbed_parallel_vector_potential` | 0.000000 | 0 | 0 | NEITHER |
| `energy_of_plasma_facing_component` | 0.000000 | 0 | 0 | NEITHER |
| `energy_over_halo_region_due_to_ohmic_dissipation` | 0.000000 | 0 | 0 | NEITHER |
| `ethylene_count_due_to_gas_injection` | 0.000000 | 0 | 0 | NEITHER |
| `external_peak_half_width` | 0.000000 | 0 | 0 | NEITHER |
| `factor_at_line_of_sight` | 0.000000 | 0 | 0 | NEITHER |
| `fast_electron_torque_density_volume_integrated_due_to_collisions` | 0.000000 | 0 | 0 | NEITHER |
| `fast_neutral_beam_motional_stark_wavelength` | 0.000000 | 0 | 0 | NEITHER |
| `fast_neutral_beam_reference_wavelength_of_spectral_line` | 0.000000 | 0 | 0 | NEITHER |
| `fast_particle_count` | 0.000000 | 0 | 0 | NEITHER |
| `field_aligned_convection_velocity` | 0.000000 | 0 | 0 | NEITHER |
| `flow_at_outlet` | 0.000000 | 0 | 0 | NEITHER |
| `fluctuating_electron_density` | 0.000000 | 0 | 0 | NEITHER |
| `flux_due_to_perturbed_parallel_magnetic_field` | 0.000000 | 0 | 0 | NEITHER |
| `fraction_at_divertor_target` | 0.000000 | 0 | 0 | NEITHER |
| `fraction_of_beamlet_group` | 0.000000 | 0 | 0 | NEITHER |
| `frequency_of_gyrokinetic_eigenmode` | 0.000000 | 0 | 0 | NEITHER |
| `frequency_of_wave_beam` | 0.000000 | 0 | 0 | NEITHER |
| `gas_count` | 0.000000 | 0 | 0 | NEITHER |
| `gas_count_at_midplane_due_to_gas_injection` | 0.000000 | 0 | 0 | NEITHER |
| `gas_source_rate_at_midplane_due_to_gas_injection` | 0.000000 | 0 | 0 | NEITHER |
| `gas_source_rate_of_vacuum_vessel_due_to_gas_injection` | 0.000000 | 0 | 0 | NEITHER |
| `growth_rate_of_gyrokinetic_eigenmode` | 0.000000 | 0 | 0 | NEITHER |
| `gyrocenter_perturbed_current_density` | 0.000000 | 0 | 0 | NEITHER |
| `gyrocenter_perturbed_density` | 0.000000 | 0 | 0 | NEITHER |
| `gyrocenter_perturbed_pressure` | 0.000000 | 0 | 0 | NEITHER |
| `heating_power_of_breeder_blanket_module` | 0.000000 | 0 | 0 | NEITHER |
| `helium_3_count_due_to_gas_injection` | 0.000000 | 0 | 0 | NEITHER |
| `helium_3_prefill_count_due_to_gas_injection` | 0.000000 | 0 | 0 | NEITHER |
| `incident_power_of_divertor` | 0.000000 | 0 | 0 | NEITHER |
| `initial_current` | 0.000000 | 0 | 0 | NEITHER |
| `initial_time` | 0.000000 | 0 | 0 | NEITHER |
| `initial_vacuum_poloidal_current_function` | 0.000000 | 0 | 0 | NEITHER |
| `internal_state_energy_diffusivity` | 0.000000 | 0 | 0 | NEITHER |
| `internal_state_flux_limiter_coefficient` | 0.000000 | 0 | 0 | NEITHER |
| `internal_state_momentum_convected_velocity` | 0.000000 | 0 | 0 | NEITHER |
| `internal_state_momentum_convection_velocity` | 0.000000 | 0 | 0 | NEITHER |
| `internal_state_momentum_diffusivity` | 0.000000 | 0 | 0 | NEITHER |
| `internal_state_momentum_flux` | 0.000000 | 0 | 0 | NEITHER |
| `internal_state_momentum_flux_limiter_coefficient` | 0.000000 | 0 | 0 | NEITHER |
| `internal_state_torque_density` | 0.000000 | 0 | 0 | NEITHER |
| `ion_critical_energy` | 0.000000 | 0 | 0 | NEITHER |
| `ion_heat_flux` | 0.000000 | 0 | 0 | NEITHER |
| `ion_particle_volumetric_source_rate` | 0.000000 | 0 | 0 | NEITHER |
| `ion_radiance_of_spectral_line_due_to_charge_exchange` | 0.000000 | 0 | 0 | NEITHER |
| `larmor_radius_at_pedestal_top_high_field_side` | 0.000000 | 0 | 0 | NEITHER |
| `line_averaged_total_hydrogenic_density` | 0.000000 | 0 | 0 | NEITHER |
| `linear_growth_rate_of_gyrokinetic_eigenmode` | 0.000000 | 0 | 0 | NEITHER |
| `lithium_prefill_count_due_to_gas_injection` | 0.000000 | 0 | 0 | NEITHER |
| `lower_gas_source_rate_of_vacuum_vessel_due_to_gas_injection` | 0.000000 | 0 | 0 | NEITHER |
| `maximum_energy_of_neutron_detector` | 0.000000 | 0 | 0 | NEITHER |
| `methane_carbon_13_count_due_to_gas_injection` | 0.000000 | 0 | 0 | NEITHER |
| `momentum_convected_velocity` | 0.000000 | 0 | 0 | NEITHER |
| `momentum_diffusivity_due_to_diamagnetic_drift` | 0.000000 | 0 | 0 | NEITHER |
| `momentum_field_aligned_convection_velocity` | 0.000000 | 0 | 0 | NEITHER |
| `momentum_flux_due_to_perturbed_parallel_magnetic_field` | 0.000000 | 0 | 0 | NEITHER |
| `momentum_neutral_internal_state_flux_limiter_coefficient` | 0.000000 | 0 | 0 | NEITHER |
| `neutral_beam_doppler_wavelength` | 0.000000 | 0 | 0 | NEITHER |
| `neutral_beam_motional_stark_wavelength` | 0.000000 | 0 | 0 | NEITHER |
| `neutral_beam_reference_wavelength_of_spectral_line` | 0.000000 | 0 | 0 | NEITHER |
| `neutral_count_at_wall` | 0.000000 | 0 | 0 | NEITHER |
| `neutral_density_of_isotope` | 0.000000 | 0 | 0 | NEITHER |
| `neutral_internal_state_flux_limiter_coefficient` | 0.000000 | 0 | 0 | NEITHER |
| `neutral_source_rate_due_to_gas_injection` | 0.000000 | 0 | 0 | NEITHER |
| `neutral_species_count` | 0.000000 | 0 | 0 | NEITHER |
| `neutral_species_flow_at_outlet` | 0.000000 | 0 | 0 | NEITHER |
| `neutral_state_momentum_velocity_due_to_convection` | 0.000000 | 0 | 0 | NEITHER |
| `normalized_pressure_at_flux_surface` | 0.000000 | 0 | 0 | NEITHER |
| `number_density_of_pellet` | 0.000000 | 0 | 0 | NEITHER |
| `opacity_at_ece_channel_emission_position` | 0.000000 | 0 | 0 | NEITHER |
| `oxygen_count_due_to_gas_injection` | 0.000000 | 0 | 0 | NEITHER |
| `oxygen_prefill_count_due_to_gas_injection` | 0.000000 | 0 | 0 | NEITHER |
| `parameter_at_pedestal` | 0.000000 | 0 | 0 | NEITHER |
| `particle_count_at_pellet_path_due_to_pellet_injection` | 0.000000 | 0 | 0 | NEITHER |
| `particle_count_due_to_gas_injection` | 0.000000 | 0 | 0 | NEITHER |
| `particle_flux_due_to_e_cross_b_drift` | 0.000000 | 0 | 0 | NEITHER |
| `particle_flux_due_to_perturbed_parallel_magnetic_field` | 0.000000 | 0 | 0 | NEITHER |
| `particle_fraction_of_beamlet_group` | 0.000000 | 0 | 0 | NEITHER |
| `particle_perturbed_current_density` | 0.000000 | 0 | 0 | NEITHER |
| `particle_perturbed_energy` | 0.000000 | 0 | 0 | NEITHER |
| `particle_perturbed_pressure` | 0.000000 | 0 | 0 | NEITHER |
| `particle_perturbed_pressure_of_gyrokinetic_eigenmode` | 0.000000 | 0 | 0 | NEITHER |
| `particle_power_due_to_thermalization` | 0.000000 | 0 | 0 | NEITHER |
| `particle_reference_temperature` | 0.000000 | 0 | 0 | NEITHER |
| `particle_simulated_count` | 0.000000 | 0 | 0 | NEITHER |
| `particle_source_rate_at_wall_due_to_gas_injection` | 0.000000 | 0 | 0 | NEITHER |
| `particle_source_rate_due_to_neutral_beam_shinethrough` | 0.000000 | 0 | 0 | NEITHER |
| `particle_volumetric_source_rate` | 0.000000 | 0 | 0 | NEITHER |
| `peak_half_width` | 0.000000 | 0 | 0 | NEITHER |
| `peak_upper_bound_width` | 0.000000 | 0 | 0 | NEITHER |
| `perturbed_density` | 0.000000 | 0 | 0 | NEITHER |
| `perturbed_energy` | 0.000000 | 0 | 0 | NEITHER |
| `perturbed_pressure_of_gyrokinetic_eigenmode` | 0.000000 | 0 | 0 | NEITHER |
| `phase_of_fiber_optic_current_sensor` | 0.000000 | 0 | 0 | NEITHER |
| `phase_of_wave_beam` | 0.000000 | 0 | 0 | NEITHER |
| `plasma_initial_current` | 0.000000 | 0 | 0 | NEITHER |
| `plasma_momentum_field_aligned_convection_velocity` | 0.000000 | 0 | 0 | NEITHER |
| `plasma_radiated_power` | 0.000000 | 0 | 0 | NEITHER |
| `plasma_stored_energy` | 0.000000 | 0 | 0 | NEITHER |
| `plasma_upper_bound_current` | 0.000000 | 0 | 0 | NEITHER |
| `power_at_divertor_target_due_to_halo_current` | 0.000000 | 0 | 0 | NEITHER |
| `power_at_separatrix` | 0.000000 | 0 | 0 | NEITHER |
| `power_at_wall` | 0.000000 | 0 | 0 | NEITHER |
| `power_density_due_to_conductive_losses` | 0.000000 | 0 | 0 | NEITHER |
| `power_due_to_halo_current` | 0.000000 | 0 | 0 | NEITHER |
| `power_of_divertor_target` | 0.000000 | 0 | 0 | NEITHER |
| `power_of_electron_cyclotron_launcher` | 0.000000 | 0 | 0 | NEITHER |
| `power_of_lower_hybrid_antenna_row` | 0.000000 | 0 | 0 | NEITHER |
| `power_over_core_region` | 0.000000 | 0 | 0 | NEITHER |
| `prefill_count_due_to_gas_injection` | 0.000000 | 0 | 0 | NEITHER |
| `pressure_at_flux_surface` | 0.000000 | 0 | 0 | NEITHER |
| `pressure_at_post_sawtooth_crash` | 0.000000 | 0 | 0 | NEITHER |
| `pressure_of_gyrokinetic_eigenmode` | 0.000000 | 0 | 0 | NEITHER |
| `radiated_energy_due_to_impurity_radiation` | 0.000000 | 0 | 0 | NEITHER |
| `radiative_power_of_divertor_target` | 0.000000 | 0 | 0 | NEITHER |
| `reference_temperature` | 0.000000 | 0 | 0 | NEITHER |
| `reference_wavelength_of_spectral_line` | 0.000000 | 0 | 0 | NEITHER |
| `runaway_electron_critical_momentum_due_to_avalanche` | 0.000000 | 0 | 0 | NEITHER |
| `simulated_count` | 0.000000 | 0 | 0 | NEITHER |
| `source_rate_at_midplane_due_to_gas_injection` | 0.000000 | 0 | 0 | NEITHER |
| `source_rate_at_pellet_path_due_to_pellet_injection` | 0.000000 | 0 | 0 | NEITHER |
| `source_rate_at_wall_due_to_gas_injection` | 0.000000 | 0 | 0 | NEITHER |
| `source_rate_due_to_neutral_beam_shinethrough` | 0.000000 | 0 | 0 | NEITHER |
| `source_rate_due_to_pellet_injection` | 0.000000 | 0 | 0 | NEITHER |
| `source_rate_of_vacuum_vessel_due_to_gas_injection` | 0.000000 | 0 | 0 | NEITHER |
| `spectral_wave_opacity_at_ece_channel_emission_position` | 0.000000 | 2 | 0 | NEITHER |
| `spun_twist_phase_of_fiber_optic_current_sensor` | 0.000000 | 0 | 0 | NEITHER |
| `square_toroidal_flux_radius_gradient` | 0.000000 | 0 | 0 | NEITHER |
| `surface_curvature_of_optical_element` | 0.000000 | 0 | 0 | NEITHER |
| `surface_tilt_angle_of_langmuir_probe` | 0.000000 | 0 | 0 | NEITHER |
| `tangential_curvature_of_optical_element` | 0.000000 | 0 | 0 | NEITHER |
| `temperature_at_last_closed_flux_surface` | 0.000000 | 0 | 0 | NEITHER |
| `temperature_at_post_sawtooth_crash` | 0.000000 | 0 | 0 | NEITHER |
| `temperature_at_separatrix` | 0.000000 | 0 | 0 | NEITHER |
| `thermal_coolant_absorbed_power_at_wall` | 0.000000 | 0 | 0 | NEITHER |
| `thermal_energy_of_plasma_facing_component` | 0.000000 | 0 | 0 | NEITHER |
| `thermal_ion_heat_flux` | 0.000000 | 0 | 0 | NEITHER |
| `thermal_power_density_due_to_conductive_losses` | 0.000000 | 0 | 0 | NEITHER |
| `tilt_angle_of_langmuir_probe` | 0.000000 | 0 | 0 | NEITHER |
| `tilt_angle_of_neutral_beam_injector` | 0.000000 | 0 | 0 | NEITHER |
| `time_derivative_of_plasma_stored_energy` | 0.000000 | 0 | 0 | NEITHER |
| `total_gas_count_at_midplane_due_to_gas_injection` | 0.000000 | 0 | 0 | NEITHER |
| `total_momentum_flux_due_to_perturbed_parallel_vector_potential` | 0.000000 | 0 | 0 | NEITHER |
| `total_particle_count_due_to_gas_injection` | 0.000000 | 0 | 0 | NEITHER |
| `total_particle_perturbed_pressure_of_gyrokinetic_eigenmode` | 0.000000 | 0 | 0 | NEITHER |
| `total_prefill_gas_count` | 0.000000 | 0 | 0 | NEITHER |
| `transit_time_of_plant_component_port` | 0.000000 | 0 | 0 | NEITHER |
| `trapped_particle_density` | 0.000000 | 0 | 0 | NEITHER |
| `tritium_count_due_to_gas_injection` | 0.000000 | 0 | 0 | NEITHER |
| `turn_count_of_coil_conductor_element` | 0.000000 | 0 | 0 | NEITHER |
| `turn_count_of_passive_loop` | 0.000000 | 0 | 0 | NEITHER |
| `twist_phase_of_fiber_optic_current_sensor` | 0.000000 | 0 | 0 | NEITHER |
| `upper_bound_current` | 0.000000 | 0 | 0 | NEITHER |
| `upper_bound_width` | 0.000000 | 0 | 0 | NEITHER |
| `volume_averaged_ion_average_temperature` | 0.000000 | 0 | 0 | NEITHER |
| `volume_averaged_total_hydrogenic_density` | 0.000000 | 0 | 0 | NEITHER |
| `volumetric_source_rate` | 0.000000 | 0 | 0 | NEITHER |
| `wave_opacity_at_ece_channel_emission_position` | 0.000000 | 0 | 0 | NEITHER |
| `wavelength_of_optical_element` | 0.000000 | 0 | 0 | NEITHER |
## Method and reproducibility

Cohort, presence and ratification reads ran against the live graph via the repository's
`GraphClient` on 2026-09-09T08:22Z. The spend formula is the canonical ledger measure used by
the sibling censuses: for every `LLMCost` row whose `standard_name_ids` names the identity,
add `llm_cost / size(standard_name_ids)`. Rows that list an identity more than once must be
collapsed (`WITH DISTINCT c, sid`) before summing, or the share is double-counted; this census
collapses them. The formula reproduces the sibling census controls exactly
(`etendue_of_spectrometer_channel` $0.844849, `spectral_etendue_of_spectrometer_channel`
$4.025814).

Catalog-cut membership is read from the issued export report, not from review-stage manifests:
`git -C <catalog> show v0.4.0rc6+west-task-2e:.export_report.json`, `emitted_identities`
(208).

The cohort is a point-in-time measurement. Restores and pipeline runs are in flight on this
surface (the pedestal-family and spectral-child restores land concurrently), and each
re-creation moves an identity out of the absentee set; the counts above are the snapshot at
08:22Z. The spend and verdicts are properties of the identities and do not move with
presence.

## Evidence limit

`StandardNameChange` records *that* a deletion and a ratification happened, not the deletion-time
topology; as with the 2026-09-08 pass, the change row carries only identity, operation, origin,
reason and timestamp. The ratification records survive only because `DETACH DELETE` orphans the
change chain rather than deleting it; any cleanup that prunes orphaned `StandardNameChange` nodes
would erase the only durable evidence that these three identities were ratified.
