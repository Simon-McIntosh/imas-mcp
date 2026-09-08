# Reasonless failed-source triage

**Live count: 59 of 101 `StandardNameSource` rows at `status='failed'` have null, empty, or whitespace-only `last_error` (measured 2026-09-08T06:43:27+02:00).** The section-currency audit also measured 59 of 101, so the cohort has not moved.

These are the pre-existing rows deliberately left untouched when the failure-reason guard landed. No graph state was changed for this report: no reason was backfilled, no counter was reset, and no pipeline pool was run.

## How the causes were assigned

The partition uses durable graph state in precedence order, with code defining what each trace means:

1. A removed backing `IMASNode` would establish **upstream path removed**. All 59 sources still have a backing node and none has `lifecycle_status='removed'`, so this class has zero rows.
2. An attachment refusal would establish **attachment conflict**. The binding path in `_lock_claimed_name_bindings` releases such a source to `extracted` and writes a candidate-specific `last_error`; none of these terminal blank-error rows carries a collision receipt. This class therefore has zero supported rows; unsupported suspicions remain genuinely unknown.
3. A `HAS_STANDARD_NAME_VOCAB_GAP` edge or `vocab_gap_grammar_signature` establishes **vocabulary gap**. The latter is not a generic version stamp: `retry_vocab_gap_sources_on_grammar_change` writes it only while reviving a source parked at `vocab_gap`, and deliberately removes the old edge. The union is 28 rows: 4 retain a live gap edge, 25 carry the stamp, and one carries both.
4. With no vocabulary trace, backing categories `geometry` or `coordinate` establish **geometry dimension** for 8 rows.
5. Two untraced `/summary/pedestal_fits/.../rhostar_pedestal.../value` leaves are **fit-artifact bookkeeping**: fitted pedestal-output holders rather than independently named quantities. This follows the code’s fit-output exclusion boundary, not the generic `/value` spelling; three ordinary quantity `/value(s)` rows are retained as unknown below.
6. The remaining 21 rows have no surviving causal trace. They are **genuinely unknown** rather than being assigned from path resemblance.

“Name reachable” below is an exact graph traversal, not a prediction of whether recomposition could eventually succeed: it means the source has a `PRODUCED_NAME` edge and its backing DD node has the matching `HAS_STANDARD_NAME` projection. Exactly 2 of 59 do, both to terminal names; the other 57 have neither edge.

| Cause | Rows | Share of 59 | Directly reachable name |
|---|---:|---:|---:|
| Upstream path removed | 0 | 0.0% | 0 |
| Attachment conflict | 0 | 0.0% | 0 |
| Vocabulary gap | 28 | 47.5% | 1, superseded |
| Geometry dimension | 8 | 13.6% | 0 |
| Fit-artifact bookkeeping | 2 | 3.4% | 0 |
| Genuinely unknown | 21 | 35.6% | 1, exhausted |
| **Total** | **59** | **100%** | **2 terminal** |

## Upstream path removed — 0

All 59 sources retain exactly one `FROM_DD_PATH` edge to an `IMASNode`; none of those nodes is marked removed. The graph therefore supports no assignment to this cause.

## Attachment conflict — 0

No row retains an attachment-refusal reason or candidate receipt. Current code makes this class non-terminal and explicit: a lifecycle, unit, or attachment collision is released to `status='extracted'` with a candidate-specific error. Assigning any blank terminal row to this class would therefore be guesswork.

## Vocabulary gap — 28

“Signature” means the source was previously parked at `vocab_gap` and was revived by the grammar-change retry path, which stamped the vocabulary digest and removed the old edge. “Gap edge” is stronger current evidence and names the surviving `VocabGap` token. Neither is inferred from the English-looking path.

| Source path | Failed at (UTC) | Surrounding state that states the cause | Name reachable |
|---|---|---|---|
| `bolometer/camera/channel/subcollimators_separation` | 2026-07-28T07:40:20.876Z | vocabulary-retry signature | no |
| `camera_ir/fibre_bundle/geometry/x2_width` | 2026-07-28T07:47:46.581Z | vocabulary-retry signature | yes — `height_of_diagnostic_aperture` (`superseded`) |
| `camera_visible/channel/fibre_bundle/fibre_positions/x1` | 2026-07-28T07:30:28.030Z | vocabulary-retry signature | no |
| `ec_launchers/beam/polarization/ellipticity_angle` | 2026-07-28T07:35:52.948Z | vocabulary-retry signature | no |
| `ferritic/object/axisymmetric/oblique/length_beta` | 2026-07-28T07:19:57.672Z | live gap edge `qualifier:beta_inclined` (and signature) | no |
| `gyrokinetics_local/linear/wavevector/eigenmode/moments_norm_gyrocenter/pressure_parallel` | 2026-07-28T07:13:40.556Z | vocabulary-retry signature | no |
| `gyrokinetics_local/linear/wavevector/eigenmode/moments_norm_gyrocenter_bessel_0/density` | 2026-07-28T07:13:40.556Z | vocabulary-retry signature | no |
| `gyrokinetics_local/linear/wavevector/eigenmode/moments_norm_gyrocenter_bessel_0/j_parallel` | 2026-07-28T07:13:40.556Z | vocabulary-retry signature | no |
| `gyrokinetics_local/linear/wavevector/eigenmode/moments_norm_gyrocenter_bessel_0/pressure_parallel` | 2026-07-28T07:13:40.556Z | vocabulary-retry signature | no |
| `gyrokinetics_local/linear/wavevector/eigenmode/moments_norm_gyrocenter_bessel_0/pressure_perpendicular` | 2026-07-28T07:13:40.556Z | vocabulary-retry signature | no |
| `gyrokinetics_local/linear/wavevector/eigenmode/moments_norm_gyrocenter_bessel_0/v_perpendicular_square_energy` | 2026-07-28T07:13:40.556Z | vocabulary-retry signature | no |
| `gyrokinetics_local/linear/wavevector/eigenmode/moments_norm_gyrocenter_bessel_1/pressure_parallel` | 2026-07-28T07:13:40.556Z | vocabulary-retry signature | no |
| `gyrokinetics_local/linear/wavevector/eigenmode/moments_norm_gyrocenter_bessel_1/v_perpendicular_square_energy` | 2026-07-28T07:13:40.556Z | vocabulary-retry signature | no |
| `gyrokinetics_local/non_linear/fields_zonal_2d/a_field_parallel_perturbed_norm` | 2026-07-28T07:20:21.388Z | vocabulary-retry signature | no |
| `gyrokinetics_local/non_linear/fields_zonal_2d/b_field_parallel_perturbed_norm` | 2026-07-28T07:20:21.388Z | vocabulary-retry signature | no |
| `gyrokinetics_local/non_linear/fields_zonal_2d/phi_potential_perturbed_norm` | 2026-07-28T07:13:40.556Z | live gap edge `qualifier:zonal` | no |
| `gyrokinetics_local/species/temperature_log_gradient_norm` | 2026-07-28T07:20:21.388Z | vocabulary-retry signature | no |
| `mhd_linear/time_slice/toroidal_mode/plasma/stress_reynolds/imaginary` | 2026-07-28T05:39:07.206Z | vocabulary-retry signature | no |
| `mhd_linear/time_slice/toroidal_mode/plasma/stress_reynolds/real` | 2026-07-28T05:39:07.206Z | vocabulary-retry signature | no |
| `nbi/unit/beamlets_group/focus/width_min_vertical` | 2026-07-28T07:23:00.535Z | vocabulary-retry signature | no |
| `nbi/unit/beamlets_group/tilting/delta_position/phi` | 2026-07-28T07:15:15.633Z | vocabulary-retry signature | no |
| `spectrometer_uv/channel/grating/summit/r` | 2026-07-28T07:40:20.876Z | vocabulary-retry signature | no |
| `spectrometer_visible/channel/fibre_bundle/fibre_radius` | 2026-07-28T07:40:20.876Z | live gap edge `qualifier:fiber` | no |
| `spectrometer_visible/channel/fibre_image/outline/phi` | 2026-07-28T07:48:42.783Z | live gap edge `position:outline_point` | no |
| `summary/pedestal_fits/linear/pressure_electron/d_dpsi_norm_max_position/value` | 2026-07-28T07:13:30.132Z | vocabulary-retry signature | no |
| `summary/pedestal_fits/mtanh/n_e/d_dpsi_norm_max_position/value` | 2026-07-28T07:13:30.132Z | vocabulary-retry signature | no |
| `summary/pedestal_fits/mtanh/pressure_electron/d_dpsi_norm_max_position/value` | 2026-07-28T07:13:30.132Z | vocabulary-retry signature | no |
| `summary/pedestal_fits/mtanh/t_e/d_dpsi_norm_max_position/value` | 2026-07-28T07:13:30.132Z | vocabulary-retry signature | no |

## Geometry dimension — 8

These rows are backed by graph nodes classified as geometry or coordinate fields. That structural classification supplies the cause; no spelling inference is needed.

| Source path | Failed at (UTC) | Surrounding state that states the cause | Name reachable |
|---|---|---|---|
| `camera_visible/channel/fibre_bundle/geometry/radius` | 2026-07-28T07:36:41.427Z | geometry field | no |
| `ic_antennas/antenna/module/strap/geometry/outline/z` | 2026-07-28T07:39:39.358Z | geometry field | no |
| `ntms/time_slice/mode/detailed_evolution/rho_tor_norm` | 2026-07-28T07:36:39.723Z | coordinate field | no |
| `pf_passive/loop/element/geometry/arcs_of_circle/curvature_radii` | 2026-07-28T07:13:46.854Z | geometry field | no |
| `spectrometer_uv/channel/line_of_sight/second_point/phi` | 2026-07-28T07:47:25.012Z | geometry field | no |
| `spectrometer_visible/channel/fibre_bundle/geometry/centre/z` | 2026-07-28T07:40:20.876Z | geometry field | no |
| `spectrometer_visible/channel/fibre_bundle/geometry/radius` | 2026-07-28T07:40:20.876Z | geometry field | no |
| `thomson_scattering/channel/line_of_sight/first_point/z` | 2026-07-28T07:51:49.259Z | geometry field | no |

## Fit-artifact bookkeeping — 2

Both rows are scalar holders inside the graph's `summary/pedestal_fits` fit-output structure. The fit-output context, rather than the generic `/value` suffix, is the evidence for this bookkeeping cause.

| Source path | Failed at (UTC) | Surrounding state that states the cause | Name reachable |
|---|---|---|---|
| `summary/pedestal_fits/linear/rhostar_pedestal_top_electron_magnetic_axis/value` | 2026-07-28T07:13:30.132Z | fitted `rhostar` pedestal output holder | no |
| `summary/pedestal_fits/mtanh/rhostar_pedestal_top_electron_magnetic_axis/value` | 2026-07-28T07:13:30.132Z | fitted `rhostar` pedestal output holder | no |

## Genuinely unknown — 21

These sources have a live backing path but no surviving vocabulary-gap edge or retry signature, no removal state, no attachment-refusal receipt, and no geometry/coordinate or fit-output classification that states a cause. Their paths alone cannot establish why they failed, so the honest classification is unknown.

| Source path | Failed at (UTC) | Surrounding state that states the cause | Name reachable |
|---|---|---|---|
| `charge_exchange/channel/zeff` | 2026-07-28T05:39:40.020Z | live quantity backing; no durable causal trace | no |
| `charge_exchange/etendue` | 2026-07-28T07:45:34.132Z | live quantity backing; no durable causal trace | no |
| `ece/channel/delta_position_suprathermal/rho_tor_norm` | 2026-07-28T07:25:06.128Z | live quantity backing; no durable causal trace | no |
| `edge_transport/model/ggd/momentum/flux_limiter/radial` | 2026-07-28T07:15:02.529Z | stored quantity; no durable causal trace | no |
| `gyrokinetics_local/linear/wavevector/eigenmode/fields/phi_potential_perturbed_norm` | 2026-07-28T07:13:40.556Z | stored quantity; no durable causal trace | no |
| `gyrokinetics_local/linear/wavevector/eigenmode/linear_weights_rotating_frame/momentum_phi_perpendicular_phi_potential` | 2026-07-28T07:13:40.556Z | stored quantity; no durable causal trace | no |
| `gyrokinetics_local/non_linear/fields_4d/phi_potential_perturbed_norm` | 2026-07-28T07:20:21.388Z | stored quantity; no durable causal trace | no |
| `gyrokinetics_local/non_linear/fields_intensity_1d/phi_potential_perturbed_norm` | 2026-07-28T07:13:40.556Z | stored quantity; no durable causal trace | no |
| `gyrokinetics_local/non_linear/fields_intensity_2d_surface_average/phi_potential_perturbed_norm` | 2026-07-28T07:20:21.388Z | stored quantity; no durable causal trace | no |
| `gyrokinetics_local/non_linear/fluxes_2d_k_x_k_y_sum/energy_a_field_parallel` | 2026-07-28T07:13:40.556Z | stored quantity; no durable causal trace | no |
| `gyrokinetics_local/non_linear/fluxes_2d_k_x_k_y_sum_rotating_frame/momentum_phi_perpendicular_phi_potential` | 2026-07-28T07:13:40.556Z | stored quantity; no durable causal trace | no |
| `nbi/unit/beam_current_fraction` | 2026-07-28T07:34:13.971Z | live quantity backing; no durable causal trace | no |
| `nbi/unit/source/centre/phi` | 2026-07-28T07:19:13.540Z | live quantity backing; no durable causal trace | no |
| `neutron_diagnostic/neutron_flux_total` | 2026-07-27T18:02:15.081Z | no durable causal trace; terminal name edge survives | yes — `power_due_to_fusion_reactions` (`exhausted`) |
| `plasma_sources/source/ggd/ion/state/momentum/diamagnetic` | 2026-07-28T08:02:35.420Z | stored quantity; no durable causal trace | no |
| `plasma_transport/model/ggd/ion/momentum/flux_limiter/parallel` | 2026-07-28T07:22:04.659Z | stored quantity; no durable causal trace | no |
| `plasma_transport/model/profiles_1d/neutral/state/momentum/flux_limiter/parallel` | 2026-07-28T07:38:03.519Z | stored quantity; no durable causal trace | no |
| `pulse_schedule/nbi/unit/species/element/a` | 2026-07-28T05:40:44.262Z | live quantity backing; no durable causal trace | no |
| `spectrometer_x_ray_crystal/channel/bin/instrument_function/values` | 2026-07-28T07:17:58.715Z | ordinary quantity output; `/values` does not prove fit-artifact bookkeeping | no |
| `spectrometer_x_ray_crystal/channel/instrument_function/values` | 2026-07-28T07:17:58.715Z | ordinary quantity output; `/values` does not prove fit-artifact bookkeeping | no |
| `summary/local/pedestal/zeff/value` | 2026-07-28T07:22:47.794Z | ordinary quantity output; `/value` does not prove fit-artifact bookkeeping | no |

## Write-path closure

The current implementation rejects an empty failure reason before opening the graph on all three routes that can write terminal failure state:

- `persist_claimed_source_outcomes` guards ordinary claimed-source outcomes.
- `persist_claimed_vocab_gaps` guards vocabulary-gap outcomes.
- `mark_sources_failed` guards direct failure marking.

The guards prevent new reasonless failures; they deliberately do not invent causes for these 59 pre-existing rows. Establishing a cause for any of the 21 unknown rows requires a surviving historical execution record or an independently authorized retry. Neither is present in the graph, and this audit performed no graph mutation, counter reset, or pipeline run.
