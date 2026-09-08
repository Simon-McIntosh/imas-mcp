# WEST pipeline-drain receipt

Date: 2026-09-08. This receipt records the live graph state before and after
the bounded pipeline attempts. The graph reads ran on the login node because
the Neo4j tunnel is login-node-local; the standard-names test suite ran on
`all_debug`.

## Scope and budget

The compose scope was the 15 `west_production_dd_paths` members whose source
stage was `extracted`. It ran under run identifier
`c7444e69-8e7c-4675-9e42-f9b4a232a0cf`, `--only compose`, `-t 30`, and a
$100.00 sub-cap. Generate-name replicas were explicitly limited to four, the
measured two-H200 knee. The four requested documentation identities were
preflighted separately under a $50.00 sub-cap, leaving the combined ceiling at
$150.00.

Neither grammar item awaiting an upstream vocabulary decision was in the
15-source compose set: the disruption instant is explicitly declined at this
stage, and `signal_to_noise_ratio` has not been released. No identity was
composed against either unavailable term.

## Compose cohort

The table uses the initial live read as the before state and the persisted
source/name projection as the after state. A `vocab_gap` is a held result, not
a retry target.

| Source path | Before | After | Result or reason |
| --- | --- | --- | --- |
| `barometry/gauge/pressure` | extracted | composed | `neutral_pressure` |
| `calorimetry/group/component/energy_cumulated` | extracted | composed | `accumulated_thermal_energy` |
| `calorimetry/group/component/energy_total/data` | extracted | composed | `total_energy_of_calorimetry_component` |
| `calorimetry/group/component/power` | extracted | vocab_gap | held: missing `device:component` token |
| `camera_ir/channel/camera/frame/apparent_temperature` | extracted | composed | `surface_temperature` |
| `ece/channel/optical_depth` | extracted | composed | `line_integrated_opacity` |
| `equilibrium/time_slice/profiles_1d/darea_dpsi` | extracted | extracted | held: proposal recorded a missing physical base, `rate_of_change_of_area_with_respect_to_poloidal_magnetic_flux` |
| `equilibrium/time_slice/profiles_1d/j_parallel` | extracted | composed | `flux_surface_averaged_parallel_current_density` |
| `hard_x_rays/emissivity_profile_1d/emissivity` | extracted | composed | `hard_xray_emissivity` |
| `hard_x_rays/emissivity_profile_1d/half_width_external` | extracted | composed | `outer_hard_xray_half_width` |
| `ic_antennas/antenna/module/strap/distance_to_conductor` | extracted | composed | `distance_of_antenna_strap` |
| `ic_antennas/antenna/module/strap/width_phi` | extracted | composed | `toroidal_width_of_antenna_strap` |
| `summary/boundary/gap_limiter_wall/value` | extracted | composed | `gap_at_closest_wall_point` |
| `summary/fusion/neutron_rates/total/value` | extracted | composed | `total_neutron_rate` |
| `summary/gas_injection_accumulated/total/value` | extracted | composed | `accumulated_total_gas_count` |

The first persisted read found 13 composed sources, one terminal vocabulary
gap, and one source still extracted with its own vocabulary-gap evidence. The
active compose process is allowed to finish or reach its bounded timeout before
the final cost and completion fields are recorded.

## Documentation cohort

The live lifecycle preflight, rather than the earlier roster snapshot, controls
this rotation.

| Identity | Before | After | Reason |
| --- | --- | --- | --- |
| `hot_neutral_temperature` | name reviewed; docs accepted; valid | unchanged | Documentation is already accepted, so no docs work is claimable. |
| `normalized_toroidal_beta` | name reviewed; docs pending; valid; status superseded | unchanged | Exact-name scope refused a terminal lifecycle before any model call. |
| `inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width` | name drafted; docs pending; quarantined | unchanged | Held at validation refusal; docs must not be generated before a valid name-stage result. |
| `vertical_outline_of_plasma_boundary` | name drafted; docs pending; quarantined | unchanged | Held at validation refusal; docs must not be generated before a valid name-stage result. |

The mixed four-name invocation was refused atomically because the superseded
identity is terminal. It made no model call and spent $0.00. Splitting it would
not advance the other three: one already has accepted documentation and the two
drafts remain validation-quarantined.

## Spend and verification

The compose run uses the local `hosted_vllm` route, whose generate pool is
accounted as zero-cost. Its final ledger total remains pending while the scoped
process holds an active request; no paid documentation rotation was admitted.
The authorised ceiling is $150.00. Any remainder is unspent because two compose
rows require grammar vocabulary and the four documentation rows are either
terminal, already complete, or validation-blocked; spending through those
guards would not produce a valid catalog identity.

The baseline `tests/standard_names` run at `ea277fdbbaceea4580f22ac56dba28e70b58fdbc`
completed with 20 failures, 7,227 passes, 11 skips, and 323 deselections. Its
log is retained with the worker manifest. The completed-suite count differs
from the expected 18-failure base and is retained as observed rather than
normalised.
