# Manifest rows outside the emitted cut: origin repair

## Outcome

**Complete.** The frozen WEST review roster has 227 identities, the issued
export report emits 208, and the outside-cut cohort therefore has 19 identities.
At the final pre-apply census, ten of those 19 had `origin=null` and carried one
direct `dd:` `PRODUCED_NAME` producer. Zero were derived-only and zero were
producer-less, so `origin=pipeline` was independently justified for every row.

After graph writers were quiesced, an exact ten-row signed preview admitted all
ten with zero refusals. Its fresh digest was authorized once and applied
atomically. The persisted receipt cohort contains ten rows, and replaying that
same digest returned `already_applied`, ten receipt rows, and zero replay
writes. The full post-apply manifest census now contains zero producer-backed
null origins.

## Cohort boundary

| input | value |
| --- | ---: |
| frozen manifest | `v0.4.0rc6+west-task-2e.sn_names.yaml` |
| manifest SHA-256 | `18e7beb7b77882c5a72ba095aa49340ee8e69ff620269f659302ca05676db68c` |
| manifest identities | 227 |
| issued export identities | 208 |
| identities outside the emitted cut | 19 |
| null-origin identities outside the cut | 10 |
| producer-backed null origins before / after | 10 / 0 |
| producer-less null origins | 0 |

## Per-identity evidence

All ten rows were live with `status=draft`; eight were `name_stage=drafted`
and two were `name_stage=accepted`. Each row has one direct `dd` producer. The
absence of a producer-less row means no verdict-only identity exists in this
cohort at the recorded census time.

| Identity | Name stage | `PRODUCED_NAME` producer | Final origin | Signed receipt |
| --- | --- | --- | --- | --- |
| `accumulated_thermal_energy` | drafted | `dd:calorimetry/group/component/energy_cumulated` (dd; composed) | `pipeline` | `sn-change:signed-manifest:66aed4071228b294607a643554bde031f51be8c3e4c35bbfc48f306f31f5cf78:b1ba6c2312bb680ebe6fa092` |
| `accumulated_total_gas_count` | drafted | `dd:summary/gas_injection_accumulated/total/value` (dd; composed) | `pipeline` | `sn-change:signed-manifest:66aed4071228b294607a643554bde031f51be8c3e4c35bbfc48f306f31f5cf78:c8080a5f9d5cf3c819d5de38` |
| `flux_surface_averaged_parallel_current_density` | drafted | `dd:equilibrium/time_slice/profiles_1d/j_parallel` (dd; composed) | `pipeline` | `sn-change:signed-manifest:66aed4071228b294607a643554bde031f51be8c3e4c35bbfc48f306f31f5cf78:d3b56095eb92d716fd557425` |
| `gap_at_closest_wall_point` | drafted | `dd:summary/boundary/gap_limiter_wall/value` (dd; composed) | `pipeline` | `sn-change:signed-manifest:66aed4071228b294607a643554bde031f51be8c3e4c35bbfc48f306f31f5cf78:8c347b678c4c52d02b41e614` |
| `line_integrated_opacity` | drafted | `dd:ece/channel/optical_depth` (dd; composed) | `pipeline` | `sn-change:signed-manifest:66aed4071228b294607a643554bde031f51be8c3e4c35bbfc48f306f31f5cf78:e453750465670a777d44139a` |
| `outer_hard_xray_half_width` | drafted | `dd:hard_x_rays/emissivity_profile_1d/half_width_external` (dd; composed) | `pipeline` | `sn-change:signed-manifest:66aed4071228b294607a643554bde031f51be8c3e4c35bbfc48f306f31f5cf78:1bcb84cf2bdf65a2da283167` |
| `radial_outline_of_plasma_boundary` | accepted | `dd:equilibrium/time_slice/boundary/outline/r` (dd; composed) | `pipeline` | `sn-change:signed-manifest:66aed4071228b294607a643554bde031f51be8c3e4c35bbfc48f306f31f5cf78:3814c0e48bf698a7918f71a6` |
| `radial_outline_of_wall` | accepted | `dd:wall/description_2d/mobile/unit/outline/r` (dd; composed) | `pipeline` | `sn-change:signed-manifest:66aed4071228b294607a643554bde031f51be8c3e4c35bbfc48f306f31f5cf78:df8f1cf8692e099d23cbfb83` |
| `total_energy_of_calorimetry_component` | drafted | `dd:calorimetry/group/component/energy_total/data` (dd; composed) | `pipeline` | `sn-change:signed-manifest:66aed4071228b294607a643554bde031f51be8c3e4c35bbfc48f306f31f5cf78:ce71972d343b6dc367e6295a` |
| `total_neutron_rate` | drafted | `dd:summary/fusion/neutron_rates/total/value` (dd; composed) | `pipeline` | `sn-change:signed-manifest:66aed4071228b294607a643554bde031f51be8c3e4c35bbfc48f306f31f5cf78:d6986fc60adda9e5c3c74c49` |

## Signed preview evidence

The authority contains ten exact `StandardName` participants and ten
`set_properties` mutations, each assigning only `origin=pipeline`.

| preview | file SHA-256 | payload SHA-256 | manifest SHA-256 | admitted | refused | result |
| --- | --- | --- | --- | ---: | ---: | --- |
| first | `68312d693e66af7fdfb465ec7c6775fa86327918afd0d63d529ace40a750a0b9` | `1e07dcf7b2bdf080e785e3a37a7d0e467016fed5bb9b0c7d13629c14e30a6157` | `1644cc3242cb3ff0f38b78775ea5235fcc323d7caeac2c732cc272abe958da75` | 10 | 0 | apply refused at commit-time closure comparison |
| second | same authority | same authority | `ce30d9eb1325179a5ef6b733fe23e0e9572c6afdf8169aa1c408faab523bd94b` | 10 | 0 | no apply receipt; post-attempt census unchanged |
| quiet graph | same authority | same authority | `66aed4071228b294607a643554bde031f51be8c3e4c35bbfc48f306f31f5cf78` | 10 | 0 | applied atomically; ten persisted receipts |

The distinct earlier manifest hashes with an unchanged ten-row cohort establish
that the contention was in signed collateral closure rather than in these
identities' producer topology. Once graph writers were quiet, exactly one fresh
digest was authorized. Reading that persisted receipt cohort through the replay
path returned `outcome=already_applied`, `receipt_rows=10`, `changed=0`, and
`persistent_writes=0`; the replay verified the original transaction without
performing another mutation.

## Post-apply gate

The full 227-name census, not only the 19 outside-cut rows, reports 226 live
identities: 225 `pipeline`, zero `derived`, and two null rows with no producer.
It returns **zero producer-backed null-origin identities**. Inside the
outside-cut cohort all 19 identities now read `pipeline`; the ten repaired rows
were re-read individually beside the same `dd:` producer edges shown above.

No producer-less null row existed in the outside-cut repair cohort, so no
verdict-only row was needed there. The two producer-less nulls in the complete
manifest are outside this ten-row mutation and remain explicit rather than
receiving invented provenance.

## Artifacts

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260909T103104976349-n-pidp-manifest-rows-outside-the-cut-carry-a-true-origin/before-census.json`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260909T103104976349-n-pidp-manifest-rows-outside-the-cut-carry-a-true-origin/origin-authority.json`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260909T103104976349-n-pidp-manifest-rows-outside-the-cut-carry-a-true-origin/origin-preview.json`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260909T103104976349-n-pidp-manifest-rows-outside-the-cut-carry-a-true-origin/origin-apply.json`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260909T103104976349-n-pidp-manifest-rows-outside-the-cut-carry-a-true-origin/after-census.json`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260909T103104976349-n-pidp-manifest-rows-outside-the-cut-carry-a-true-origin/final-census.json`
