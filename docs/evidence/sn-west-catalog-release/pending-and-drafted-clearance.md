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

The validation worker re-observed the exact eight rows whose pending verdicts
lacked timestamps. It returned `validated=8`, `cleared=8`, and
`quarantined=0`; all 13 rows therefore finish `valid` with a non-null
`validated_at`. This was graph-only work and spent no model budget.

The first exact-cohort command was gap-only, named all and only the 13 identities,
passed `--skip-global-maintenance`, and used neither `--reseed` nor `--force`.
It exposed an orchestration hazard: the default pool set includes refinement,
so a below-threshold name verdict immediately made the same row eligible for a
refinement attempt. The run was interrupted when that happened. Six names had
already reached accepted, `neutral_pressure` had reached exhausted on a real
0.66250 verdict, and two below-threshold rows had been claimed for refinement.
No refined successor was persisted. Canonical worker-failure cleanup released
all six interrupted claims and restored the two claimed rows to reviewed. The
unwanted claims nevertheless incremented `refine_attempts` once on
`line_integrated_opacity` and `total_energy_of_calorimetry_component`; the
table retains that collateral rather than concealing it.

Completion then used exact single-pool rotations. `--only review_name` gave
each of the four still-drafted names one verdict and could not refine them.
`--only review_docs` gave each of the four accepted names with drafted
documentation one verdict and could not refine or regenerate them. Every
below-threshold result was left where its evidence placed it. In particular,
`flux_surface_averaged_parallel_current_density` remains docs-reviewed at
0.56250, and no retry was used to chase acceptance.

The final bounded read found all 13 identities, zero live claims, and these
evidence-backed states:

| Identity | name_stage after | docs_stage after | validation_status after | validated_at after | reviewer_score_name after | reviewer_score_docs after | review edges name/docs after | documentation chars | paid delta, USD |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| `accumulated_thermal_energy` | reviewed | pending | valid | 2026-09-08T15:34:30.985Z | 0.63125 | null | 2 / 0 | 0 | 0.144370 |
| `accumulated_total_gas_count` | accepted | accepted | valid | 2026-09-09T15:44:47.943Z | 0.91250 | 0.98750 | 5 / 2 | 1238 | 0.185935 |
| `distance_of_antenna_strap` | reviewed | pending | valid | 2026-09-09T15:44:47.943Z | 0.68750 | null | 5 / 0 | 0 | 0.119048 |
| `flux_surface_averaged_parallel_current_density` | accepted | reviewed | valid | 2026-09-08T15:36:12.784Z | 1.00000 | 0.56250 | 2 / 3 | 1795 | 0.506462 |
| `gap_at_closest_wall_point` | accepted | accepted | valid | 2026-09-09T15:44:47.943Z | 0.99375 | 1.00000 | 5 / 2 | 1072 | 0.188562 |
| `hard_xray_emissivity` | accepted | accepted | valid | 2026-09-09T15:44:47.943Z | 0.98750 | 0.93125 | 4 / 10 | 1472 | 0.062256 |
| `line_integrated_opacity` | reviewed | pending | valid | 2026-09-08T15:37:17.492Z | 0.84375 | null | 2 / 0 | 0 | 0.086049 |
| `neutral_pressure` | exhausted | accepted | valid | 2026-09-09T15:44:47.943Z | 0.66250 | 0.85000 | 11 / 4 | 1552 | 0.199252 |
| `outer_hard_xray_half_width` | reviewed | pending | valid | 2026-09-09T15:44:47.943Z | 0.53750 | null | 5 / 0 | 0 | 0.271350 |
| `surface_temperature` | exhausted | accepted | valid | 2026-09-09T15:44:47.943Z | 0.83750 | 0.86875 | 10 / 4 | 818 | 0.222701 |
| `toroidal_width_of_antenna_strap` | accepted | accepted | valid | 2026-09-09T15:44:47.943Z | 1.00000 | 0.91250 | 5 / 4 | 1054 | 0.034917 |
| `total_energy_of_calorimetry_component` | reviewed | pending | valid | 2026-09-08T15:41:38.689Z | 0.66875 | null | 2 / 0 | 0 | 0.083019 |
| `total_neutron_rate` | accepted | accepted | valid | 2026-09-08T15:41:38.689Z | 0.97500 | 0.95000 | 2 / 2 | 1027 | 0.166936 |

The cohort's all-time ledger total after the work is **6.354430 USD**. Against
the exact same rows' 4.083573 USD baseline, this run spent **2.270857 USD**,
leaving **22.729143 USD** of the authorised ceiling unused. The per-row deltas
sum to that total. Costs on rows that did not need a new score reflect work the
first default-pool run performed before interruption, including documentation
generation and quorum calls; no cost row was deleted or rewritten.

### Disposition

- Name accepted: 6 — `accumulated_total_gas_count`,
  `flux_surface_averaged_parallel_current_density`,
  `gap_at_closest_wall_point`, `hard_xray_emissivity`,
  `toroidal_width_of_antenna_strap`, and `total_neutron_rate`.
- Name reviewed below threshold: 5 — `accumulated_thermal_energy` (0.63125),
  `distance_of_antenna_strap` (0.68750), `line_integrated_opacity` (0.84375),
  `outer_hard_xray_half_width` (0.53750), and
  `total_energy_of_calorimetry_component` (0.66875). These require semantic
  refinement as separate work; they were not retried here.
- Name exhausted: 2 — `neutral_pressure` (0.66250) and
  `surface_temperature` (0.83750). Both retain already-accepted documentation.
- Documentation accepted: 7; reviewed below threshold: 1
  (`flux_surface_averaged_parallel_current_density`, 0.56250); pending: 5,
  exactly the five names that did not reach name acceptance.

Each non-null final name or documentation scalar is backed by at least two
`HAS_REVIEW` edges on that axis. The final edge counts include retained
historical reviews as well as the edges written here, so they prove the scalar
has graph evidence without pretending every edge was created by this run.

## Operational follow-on

An operator-grade `release-refine-claim` dry-run for the exact owned
`line_integrated_opacity` claim refused with “collateral projection drift.” It
was not retried or bypassed. The normal pool failure-release function then
released the same exact claim successfully, as it did the other interrupted
claims. The projection check therefore needs investigation outside this
documentation-only scope; it did not leave any identity claimed.
