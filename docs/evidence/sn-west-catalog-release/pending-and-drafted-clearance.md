# WEST pending and drafted clearance

## Scope and evidence rule

This run is limited to the 13 identities named below. Every row was re-read
from the live graph before mutation. Live graph reads ran on the login node
because the Neo4j endpoint is exposed through a login-local tunnel; both reads
were bounded by the exact identity list.

The pipeline may advance a row only when its own evidence supports the next
state. A name with a real below-threshold review remains reviewed. A validation
verdict without `validated_at` is re-observed through the validation worker,
not repaired by setting a timestamp. Documentation is generated or reviewed
only when its text or score is absent. The exact-name pipeline run uses the
default gap-only behaviour with `--skip-global-maintenance`, never `--reseed`
or `--force`, and has a paid-work ceiling of 25 USD.

## Before state

The bounded read found all 13 identities present. All were at
`name_stage=drafted` with a null `reviewer_score_name`. Five already carried an
observed valid verdict. The other eight carried `validation_status=pending`
with no observation time. Four rows already had accepted, scored documentation;
the other nine had no documentation text or docs score.

Review-edge counts below are direct `StandardName-[:HAS_REVIEW]->StandardNameReview`
relationships grouped by `review_axis`. A null scalar beside retained review
edges is reported as disagreement, not treated as authority to choose a winner.

| Identity | name_stage | docs_stage | validation_status | validated_at | reviewer_score_name | reviewer_score_docs | review edges name/docs | documentation chars | all-time spend before, USD |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| `accumulated_thermal_energy` | drafted | pending | valid | 2026-09-08T15:34:30.985Z | null | null | 0 / 0 | 0 | 0.000000 |
| `accumulated_total_gas_count` | drafted | pending | pending | null | null | null | 3 / 0 | 0 | 0.000000 |
| `distance_of_antenna_strap` | drafted | pending | pending | null | null | null | 3 / 0 | 0 | 0.000000 |
| `flux_surface_averaged_parallel_current_density` | drafted | pending | valid | 2026-09-08T15:36:12.784Z | null | null | 0 / 0 | 0 | 0.000000 |
| `gap_at_closest_wall_point` | drafted | pending | pending | null | null | null | 3 / 0 | 0 | 0.000000 |
| `hard_xray_emissivity` | drafted | accepted | pending | null | null | 0.93125 | 2 / 10 | 1472 | 1.030296 |
| `line_integrated_opacity` | drafted | pending | valid | 2026-09-08T15:37:17.492Z | null | null | 0 / 0 | 0 | 0.000000 |
| `neutral_pressure` | drafted | accepted | pending | null | null | 0.85000 | 8 / 4 | 1552 | 1.216970 |
| `outer_hard_xray_half_width` | drafted | pending | pending | null | null | null | 2 / 0 | 0 | 0.167773 |
| `surface_temperature` | drafted | accepted | pending | null | null | 0.86875 | 7 / 4 | 818 | 1.124002 |
| `toroidal_width_of_antenna_strap` | drafted | accepted | pending | null | null | 0.91250 | 3 / 4 | 1054 | 0.544532 |
| `total_energy_of_calorimetry_component` | drafted | pending | valid | 2026-09-08T15:41:38.689Z | null | null | 0 / 0 | 0 | 0.000000 |
| `total_neutron_rate` | drafted | pending | valid | 2026-09-08T15:41:38.689Z | null | null | 0 / 0 | 0 | 0.000000 |

The cohort's all-time spend before this run was **4.083573 USD**. That is
historical cost, not this run's spend; per-row run cost is the after-minus-before
ledger delta reported below.

## Source bindings and descriptions

| Identity | Bound WEST DD source | Description before rotation |
|---|---|---|
| `accumulated_thermal_energy` | `calorimetry/group/component/energy_cumulated` | Cumulative thermal energy extracted from a calorimetry component via its coolant since the start of the pulse. |
| `accumulated_total_gas_count` | `summary/gas_injection_accumulated/total/value` | Total accumulated count of injected gas particles, summed over species and normalized to equivalent electrons. |
| `distance_of_antenna_strap` | `ic_antennas/antenna/module/strap/distance_to_conductor` | Distance from the antenna strap rear surface to the conducting wall behind it. |
| `flux_surface_averaged_parallel_current_density` | `equilibrium/time_slice/profiles_1d/j_parallel` | Flux-surface-averaged parallel current density along magnetic field lines. |
| `gap_at_closest_wall_point` | `summary/boundary/gap_limiter_wall/value` | Distance between the plasma boundary and the nearest wall or limiter element. |
| `hard_xray_emissivity` | `hard_x_rays/emissivity_profile_1d/emissivity` | Local hard-X-ray bremsstrahlung photon production per plasma volume and emission solid angle, integrated over an energy band. |
| `line_integrated_opacity` | `ece/channel/optical_depth` | Optical depth along the ECE line of sight at the channel measurement position. |
| `neutral_pressure` | `barometry/gauge/pressure` | Scalar kinetic pressure of a neutral-particle population before a thermal, fast, or internal-state partition is specified. |
| `outer_hard_xray_half_width` | `hard_x_rays/emissivity_profile_1d/half_width_external` | Outward half-width of the hard-X-ray emissivity peak in normalized toroidal-flux coordinate. |
| `surface_temperature` | `camera_ir/channel/camera/frame/apparent_temperature` | Thermodynamic temperature of the exposed material boundary of a plasma-facing component at a specified surface point. |
| `toroidal_width_of_antenna_strap` | `ic_antennas/antenna/module/strap/width_phi` | Full toroidal width of the rectangular cross-section of an ICRH antenna strap conductor. |
| `total_energy_of_calorimetry_component` | `calorimetry/group/component/energy_total/data` | Total energy extracted from a calorimetry component over the whole discharge. |
| `total_neutron_rate` | `summary/fusion/neutron_rates/total/value` | Total neutron emission rate from all fusion reactions combined. |

## Rotation and after state

Pending. The next operations are the exact eight-row validation drain followed
by one exact-name, gap-only pipeline run over all 13 rows. The final table will
repeat every scalar above, give the review-edge count on each touched axis, and
report the per-row and total cost deltas.
