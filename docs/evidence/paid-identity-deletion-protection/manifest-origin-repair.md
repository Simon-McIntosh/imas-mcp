# Manifest rows outside the emitted cut: origin repair

## Outcome

**Blocked without a signed apply receipt or graph write.** The frozen WEST
review roster has 227 identities, the issued export report emits 208, and the
outside-cut cohort therefore has 19 identities. At census time, ten of those
19 were present, had `origin=null`, and carried a `PRODUCED_NAME` producer.
Every producer is a composed `dd` source, so the truthful value proposed for
each is `origin=pipeline`.

The first exact signed preview admitted all ten rows with zero refusals, but its
apply re-read a different closure and raised `SignedManifestConflict` before
commit. A second fresh preview also admitted all ten rows with zero refusals;
its apply attempt produced no receipt, and a subsequent full census still found
all ten producer-backed null-origin rows. No value is claimed written below.

## Cohort boundary

| input | value |
| --- | ---: |
| frozen manifest | `v0.4.0rc6+west-task-2e.sn_names.yaml` |
| manifest SHA-256 | `18e7beb7b77882c5a72ba095aa49340ee8e69ff620269f659302ca05676db68c` |
| manifest identities | 227 |
| issued export identities | 208 |
| identities outside the emitted cut | 19 |
| null-origin identities outside the cut | 10 |
| producer-backed null origins | 10 |
| producer-less null origins | 0 |

## Per-identity evidence

All ten rows were live with `status=draft`; eight were `name_stage=drafted`
and two were `name_stage=accepted`. Each row has one direct `dd` producer. The
absence of a producer-less row means no verdict-only identity exists in this
cohort at the recorded census time.

| Identity | Name stage | `PRODUCED_NAME` producer | Proposed value | Signed receipt / verdict |
| --- | --- | --- | --- | --- |
| `accumulated_thermal_energy` | drafted | `dd:calorimetry/group/component/energy_cumulated` (dd; composed) | `pipeline` | no receipt: apply did not commit |
| `accumulated_total_gas_count` | drafted | `dd:summary/gas_injection_accumulated/total/value` (dd; composed) | `pipeline` | no receipt: apply did not commit |
| `flux_surface_averaged_parallel_current_density` | drafted | `dd:equilibrium/time_slice/profiles_1d/j_parallel` (dd; composed) | `pipeline` | no receipt: apply did not commit |
| `gap_at_closest_wall_point` | drafted | `dd:summary/boundary/gap_limiter_wall/value` (dd; composed) | `pipeline` | no receipt: apply did not commit |
| `line_integrated_opacity` | drafted | `dd:ece/channel/optical_depth` (dd; composed) | `pipeline` | no receipt: apply did not commit |
| `outer_hard_xray_half_width` | drafted | `dd:hard_x_rays/emissivity_profile_1d/half_width_external` (dd; composed) | `pipeline` | no receipt: apply did not commit |
| `radial_outline_of_plasma_boundary` | accepted | `dd:equilibrium/time_slice/boundary/outline/r` (dd; composed) | `pipeline` | no receipt: apply did not commit |
| `radial_outline_of_wall` | accepted | `dd:wall/description_2d/mobile/unit/outline/r` (dd; composed) | `pipeline` | no receipt: apply did not commit |
| `total_energy_of_calorimetry_component` | drafted | `dd:calorimetry/group/component/energy_total/data` (dd; composed) | `pipeline` | no receipt: apply did not commit |
| `total_neutron_rate` | drafted | `dd:summary/fusion/neutron_rates/total/value` (dd; composed) | `pipeline` | no receipt: apply did not commit |

## Signed preview evidence

The authority contains ten exact `StandardName` participants and ten
`set_properties` mutations, each assigning only `origin=pipeline`.

| preview | file SHA-256 | payload SHA-256 | manifest SHA-256 | admitted | refused | result |
| --- | --- | --- | --- | ---: | ---: | --- |
| first | `68312d693e66af7fdfb465ec7c6775fa86327918afd0d63d529ace40a750a0b9` | `1e07dcf7b2bdf080e785e3a37a7d0e467016fed5bb9b0c7d13629c14e30a6157` | `1644cc3242cb3ff0f38b78775ea5235fcc323d7caeac2c732cc272abe958da75` | 10 | 0 | apply refused at commit-time closure comparison |
| second | same authority | same authority | `ce30d9eb1325179a5ef6b733fe23e0e9572c6afdf8169aa1c408faab523bd94b` | 10 | 0 | no apply receipt; post-attempt census unchanged |

The distinct manifest hashes with an unchanged ten-row cohort establish that
the contention is in signed collateral closure rather than in these identities'
producer topology. A third apply was not attempted.

## Required next action

Resume from a fresh zero-refusal preview when the signed-manifest closure is
stable, authorize only its new manifest SHA-256, apply once atomically, and
record the ten per-identity receipt IDs. Then rerun the same full 227-name
manifest census and require zero producer-backed null-origin identities. A
producer-less identity, if one appears in a fresh census, must receive a stated
verdict rather than an origin value.

## Artifacts

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260909T103104976349-n-pidp-manifest-rows-outside-the-cut-carry-a-true-origin/before-census.json`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260909T103104976349-n-pidp-manifest-rows-outside-the-cut-carry-a-true-origin/origin-authority.json`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260909T103104976349-n-pidp-manifest-rows-outside-the-cut-carry-a-true-origin/origin-preview.json`
