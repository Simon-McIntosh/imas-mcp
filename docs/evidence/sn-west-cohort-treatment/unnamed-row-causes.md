# Unnamed roster rows: every row ends with a name or a stated cause

**Plan section:** sn-west-cohort-treatment §9a (Tail A: 31 source paths that reached no name)
**Node:** n-swct-every-unnamed-row-states-its-cause
**Roster identity:** frozen review batch `v0.10.0rc1+west-task-2e` (`manifest_sources`),
minted 2026-09-07T09:28Z — the same roster §9a was counted against on 2026-09-07.
**Readout:** live graph, login node, 2026-09-08; every query bounded to the 31-row
cohort, each < 10 s.
**Rule being enforced:** a row must end with a name or a stated cause. An empty reason
field is not a neutral absence — it is indistinguishable from an unprocessed row.

## Summary

The frozen roster carries 355 `manifest_sources` rows: 324 with a name, 31 without.
Of the 31 unnamed rows, **13 record a cause and 18 record no cause at all**. This
document gives all 18 their actual cause from recorded pipeline state, reconciles the
four fit-constraint weights to one verdict, and corrects the time-axis and failed rows
the plan flagged. Measured figures override plan prose where they differ (two
discrepancies, flagged inline).

The controlling finding is at the bottom: **eleven of the eighteen causeless rows have a
full cause recorded in the graph that the roster failed to transcribe.** The roster's
`non_nameable_reason` is populated only for sources whose `source_status == "skipped"`,
copied from `skip_reason`/`skip_reason_detail`. It never reads `last_error` (the field
that carries every compose failure), and it blanks rows whose status is
`not_physical_quantity`, `extracted`, or `failed` even when a deterministic verdict or an
error string is present. The empty-reason defect is therefore partly a transcription bug
in the roster-assembly path, not only an absence in the pipeline.

---

## 1. The 18 causeless rows, each given its actual cause

"Before" is what the roster records (`source_status` / empty `non_nameable_reason`).
"After" is the cause read out of the source node and its backing DD node on 2026-09-08.
Rows are grouped by what the recorded state supports. None of these 18 rows has a
PRODUCED_NAME edge or a `produced_sn_id` scalar — the silence is about cause, not a
missing name.

### 1a. Eleven rows whose graph state DOES record a cause (the roster dropped it)

| source_path | before (status / reason) | recorded state | stated cause |
|---|---|---|---|
| `calorimetry/group/component/energy_cumulated` | extracted / *(empty)* | `last_error`: "missing device vocabulary token 'calorimetry_component': … the injected device registry has no token for this generic calorimetry-component carrier …" | compose attempt declined on a vocabulary gap — no device token exists for a generic calorimetry component; the composer refused to collapse the carrier onto `plasma_facing_component` against the source lineage |
| `calorimetry/group/component/power` | extracted / *(empty)* | `last_error`: "missing device vocabulary token 'calorimetry_component': … no registered device/object token represents a generic calorimetry component …" | compose attempt declined on a vocabulary gap — generic calorimetry-component device token missing; carrier identity required to keep the name unambiguous |
| `calorimetry/group/component/energy_total/data` | extracted / *(empty)* | `last_error`: "compose claim-attempt cap reached" (attempt_count 3); also `skip_reason: vocab_gap_nonactionable`, detail `:` | compose claim-attempt cap reached at 3 attempts; an earlier skip recorded a non-actionable vocabulary gap |
| `ece/channel/optical_depth` | extracted / *(empty)* | `last_error`: "compose claim-attempt cap reached"; `failed_at` 2026-09-04 | compose claim-attempt cap reached (recorded failure 2026-09-04); counter since reset to extracted |
| `equilibrium/time_slice/profiles_1d/darea_dpsi` | extracted / *(empty)* | `last_error`: "compose claim-attempt cap reached"; `failed_at` 2026-07-29 | compose claim-attempt cap reached (recorded failure 2026-07-29); counter since reset to extracted |
| `hard_x_rays/emissivity_profile_1d/emissivity` | extracted / *(empty)* | `last_error`: "compose claim-attempt cap reached"; `failed_at` 2026-09-04 | compose claim-attempt cap reached (recorded failure 2026-09-04); counter since reset to extracted |
| `camera_x_rays/camera/camera_dimensions` | failed / *(empty)* | `last_error`: "compose claim-attempt cap reached"; `attempt_count` 5; `failed_at` 2026-09-04; earlier `skip_reason: vocab_gap`, detail `position:camera_dimensions` (2026-06-19) | terminal compose failure: claim-attempt cap reached at 5 attempts; the same row was earlier declined on a vocabulary gap for the `position:camera_dimensions` token |
| `equilibrium/time_slice/profiles_1d/j_parallel` | extracted / *(empty)* | `skip_reason: dd_unit_unresolvable`, detail `None` | skipped at eligibility by an unresolvable DD unit (unit `A.m^-2` present on the DD node now — recorded at the time as unresolved) |
| `hard_x_rays/emissivity_profile_1d/half_width_external` | extracted / *(empty)* | `skip_reason: dd_unit_unresolvable`, detail `None` (attempt_count 3) | skipped at eligibility by an unresolvable DD unit (normalised-toroidal-flux half width, unit `1` on the DD node now) |
| `equilibrium/time_slice/constraints/b_field_pol_probe/weight` | not_physical_quantity / *(empty)* | `skip_reason: dd_node_category_ineligible`, detail "Backing DD node category `fit_artifact` cannot realize a StandardName" | deterministic verdict: excluded — a constraint fit weight, DD node_category `fit_artifact`, not a physical quantity |
| `equilibrium/time_slice/constraints/flux_loop/weight` | not_physical_quantity / *(empty)* | `skip_reason: dd_node_category_ineligible`, detail "Backing DD node category `fit_artifact` cannot realize a StandardName" | deterministic verdict: excluded — a constraint fit weight, DD node_category `fit_artifact`, not a physical quantity |

### 1b. Seven rows whose graph state is genuinely silent

For these the state does not say why no name was produced. Status is `extracted`,
attempt_count 0, with no `last_error` and no `skip_reason`. Recording a plausible cause
would be guessing; the honest statement is that the instrument did not record one. The
field that would have carried it is named for each.

| source_path | before (status / reason) | recorded state | stated cause |
|---|---|---|---|
| `barometry/gauge/pressure` | extracted / *(empty)* | status `extracted`, attempt_count 0; two `retry_events` recorded; no `last_error`, no `skip_reason` | **state is silent** — never claimed to a compose attempt with a recorded outcome; field that would carry the cause: `last_error` / `skip_reason` (both absent) |
| `camera_ir/channel/camera/frame/apparent_temperature` | extracted / *(empty)* | status `extracted`, attempt_count 0; one `retry_events`; `vocab_gap_grammar_signature: aeb602c3dc5196b7` recorded; no `last_error`, no `skip_reason` | **state is silent, with one exception** — a vocabulary-gap grammar signature (`aeb602c3dc5196b7`) was persisted, indicating a vocab-gap refusal, but no skip reason string was written; field that would carry the cause: `skip_reason` (present as a signature only) |
| `ic_antennas/antenna/module/strap/distance_to_conductor` | extracted / *(empty)* | status `extracted`, attempt_count 0; one `retry_events`; no `last_error`, no `skip_reason` | **state is silent** — never reached a recorded compose outcome; field: `last_error` / `skip_reason` (both absent) |
| `ic_antennas/antenna/module/strap/width_phi` | extracted / *(empty)* | status `extracted`, attempt_count 0; one `retry_events`; no `last_error`, no `skip_reason` | **state is silent** — never reached a recorded compose outcome; field: `last_error` / `skip_reason` (both absent) |
| `summary/boundary/gap_limiter_wall/value` | extracted / *(empty)* | status `extracted`, attempt_count 0; one `retry_events`; no `last_error`, no `skip_reason` | **state is silent** — never reached a recorded compose outcome; field: `last_error` / `skip_reason` (both absent) |
| `summary/fusion/neutron_rates/total/value` | extracted / *(empty)* | status `extracted`, attempt_count 0; `failed_at` 2026-07-28 **present** but no `last_error` string survives | **state is partial** — a failure timestamp was recorded (2026-07-28) but the error string was never persisted, so the cause is unreadable; field that would carry it: `last_error` (absent) |
| `summary/gas_injection_accumulated/total/value` | extracted / *(empty)* | status `extracted`, attempt_count 0, no `retry_events`, no `last_error`, no `skip_reason`; `batch_key: cc0abcaa0be044cc/1` (a different focus scope than the cohort's `focus`) | **state is silent** — seeded under a different focus scope, never claimed to a recorded compose outcome; field: `last_error` / `skip_reason` (both absent) |

Of the seven, the six with `retry_events` show the source was passed to a compose pool
and retried, but no attempt outcome was preserved. The one with a persisted
`vocab_gap_grammar_signature` and the one with a lone `failed_at` are the two where the
pipeline clearly *knew* a cause and still did not write it — those are instrument gaps,
not unknowns.

---

## 2. The four fit-constraint weights: one concept, one verdict

Plan §9a: "Four equilibrium fit-constraint weights, and the same concept is classified
two different ways: two report `not_physical_quantity` and two report a compose skip."

The four rows, all `equilibrium/time_slice/constraints/<probe>/weight`, all carrying
unit `1`, data_type `FLT_0D`, documentation "Weight given to the measurement":

| source_path | DD node_category | recorded status / reason |
|---|---|---|
| `b_field_pol_probe/weight` | **fit_artifact** | `not_physical_quantity` ← `dd_node_category_ineligible` ("Backing DD node category fit_artifact cannot realize a StandardName") |
| `flux_loop/weight` | **fit_artifact** | `not_physical_quantity` ← `dd_node_category_ineligible` (same detail) |
| `faraday_angle/weight` | **fit_artifact** | `skipped` / `compose_model_skipped` |
| `n_e_line/weight` | **fit_artifact** | `skipped` / `compose_model_skipped` |

They are **one concept, not two**: all four DD nodes carry node_category `fit_artifact`,
identical unit, identical documentation, identical locus (a per-probe weight in the
equilibrium constraint set). There is no semantic distinction between b-field/loop probe
weights and faraday/ne-line weights to defend a second verdict.

The verdict the graph supports on all four is the deterministic one — **excluded: a
constraint fit weight, DD node_category `fit_artifact`, not a physical quantity** — which
two of the rows already carry under `status = not_physical_quantity`. The two rows the
compose model declined (`compose_model_skipped`) reached the model instead of the
deterministic eligibility classifier and so were recorded with the weaker, free-form
reason. Reconciliation: route all four through the same verdict with the reason

> `not_physical_quantity: backing DD node category fit_artifact cannot realize a StandardName`

— one concept, one verdict, and the reason that both currently leave empty is now filled.
This matches the lead ruling of 2026-09-04 (fit weights and iteration counts are neither
physical nor geometric quantities and are excluded) and the existing `fit_artifact`
classification on the 417-node vocabulary the plan cites.

---

## 3. Time axes: the coordinate rule, not a compose skip

Plan §9a: "Three …/time rows are genuine axes but report the compose skip rather than
the coordinate rule that actually applies." Measured against the frozen roster, the
compose-model-skipped time-axis rows are **two**, not three — the third row the plan's
prose may have counted (`camera_x_rays/detector_temperature/time`) already carries the
correct `non_nameable_coordinate:time` reason, so the count discrepancy is some rows
being corrected between the §9a reading and this readout.

The two rows that must record the coordinate rule instead of the compose skip:

| source_path | before (reason) | DD node | stated cause (after) |
|---|---|---|---|
| `camera_x_rays/detector_humidity/time` | `compose_model_skipped` | node_category `coordinate`, unit `s`, doc "Time" | time axis: `non_nameable_coordinate:time` — bare non-nameable token `time` (dimension axis for time-varying data) |
| `camera_x_rays/frame/time` | `compose_model_skipped` | node_category `coordinate`, unit `s`, doc "Time" | time axis: `non_nameable_coordinate:time` — bare non-nameable token `time` (dimension axis for time-varying data) |

Both are excluded *because they are time coordinates*, not because the compose model
declined them. This is the same reason string the genuinely-correct rows already carry —
e.g. `core_profiles/profiles_1d/time` and `hard_x_rays/emissivity_profile_1d/time`
(`non_nameable_coordinate:time: bare non-nameable token: time`) and
`equilibrium/time_slice/time` (`temporal_coordinate: Nested time coordinate array …`).

### 3a. The disruption instant (separate but material)

`summary/disruption/time/value` is recorded with `non_nameable_coordinate:time: bare
non-nameable token: time` — but the plan's lead ruling (2026-09-04, §Decisions) says this
row is **not** an axis: it is the instant at which the disruption occurred (DD parent
doc "Time of the disruption"; DD node_category `quantity`, unit `s`), declined only
because the vocabulary admits no event-instant base. Its reason field must say that,
rather than the coordinate rule, which would suggest it was mistaken for an axis.

| source_path | before (reason) | stated cause (after) |
|---|---|---|
| `summary/disruption/time/value` | `non_nameable_coordinate:time: bare non-nameable token: time` | declined at this stage: no event-instant base in the vocabulary; the row is the disruption instant, not a dimension axis |

---

## 4. The failed row that "records nothing whatsoever"

Plan §9a: "`camera_x_rays/camera/camera_dimensions`. A terminal failure with nothing
recorded about it is the weakest row in the whole artifact."

The **roster** row records nothing — but the **graph** records a full cause. The
statement "nothing recorded" is true of the artifact, false of the pipeline:

| source_path | before (roster status / reason) | recorded graph state | stated cause (after) |
|---|---|---|---|
| `camera_x_rays/camera/camera_dimensions` | failed / *(empty)* | `status: failed`, `last_error: compose claim-attempt cap reached`, `attempt_count: 5`, `failed_at: 2026-09-04T16:09:10Z`; an earlier skip `vocab_gap` / `position:camera_dimensions` (skipped_at 2026-06-19) | terminal compose failure: claim-attempt cap reached at 5 attempts; earlier declined on a vocabulary gap for the `position:camera_dimensions` token |

The row ends with a stated cause; the roster need only transcribe the already-recorded
`last_error` (see §5).

---

## 5. The instrument finding: why the reason field was empty

The roster-assembly writes `non_nameable_reason` from `skip_reason`/`skip_reason_detail`
**only when `source_status == "skipped"`** (graph_ops.py, roster `resolved` builder); for
every other status it writes `""` unconditionally. Consequences, all measured on this
cohort:

- **`last_error` is never transcribed.** Seven of the eighteen causeless rows carry a
  recorded `last_error` (vocab-token gaps, claim-attempt caps, including the failed
  `camera_dimensions` row) that the roster dropped.
- **Deterministic verdicts are dropped when the status differs from `skipped`.** The two
  `not_physical_quantity` / `dd_node_category_ineligible` rows carry a complete,
  authoritative reason — and the roster still wrote `""` because the status is not
  literally `skipped`.
- **`skip_reason` is dropped when the status is `extracted`.** `j_parallel` and
  `half_width_external` carry `skip_reason: dd_unit_unresolvable` at status `extracted`;
  the roster omitted it.

So of the 18 empty reasons, **11 hide a recoverable recorded cause**, 7 are genuinely
silent, and 4 of those 7 have a partial instrument trace (a grammar signature, a failure
timestamp, retries) indicating the pipeline knew more than it kept. The empty-reason
defect is therefore mostly a lost-transcription defect: fixing the roster builder to
fall back to `last_error` and to transcribe `skip_reason`/deterministic verdicts
regardless of status would resolve **11 of the 18** rows from state that already exists.
The two time-axis reason-string repairs (§3) and the disruption-instant restatement
(§3a) fall on rows that already carry *a* reason — the wrong one — so they are a
separate, smaller repair to the 13 that do state a cause, not to the 18 that state none.

---

## 6. Measured totals and evidence

- Roster: 355 `manifest_sources`, 324 named / 31 unnamed (matches plan §9a).
- 18 empty `non_nameable_reason` rows: 11 with recorded cause dropped by the roster,
  7 genuinely silent (1 with a persisted vocab-gap signature, 1 with a lone `failed_at`).
- 4 fit-constraint weights = one concept (`fit_artifact`), reconciled to one verdict.
- 2 time-axis rows re-stated as the coordinate rule (plan prose said three; measured
  against the frozen roster is two), plus the disruption-instant reason re-stated per the
  lead ruling.
- 1 failed row given its already-recorded cause.
- No source among the 31 has a produced name on any axis (`PRODUCED_NAME` edge or
  `produced_sn_id` scalar) — all 31 rows are verified unnamed, not just unnamed in the
  roster.

Graph readout: live graph via the login-node tunnel (login-node-local), 2026-09-08.
Per-query row set: the 31-row cohort; every query returned in well under 10 s. Full
per-row property dump: `/tmp/west_unnamed_state.json` plus the `sn_*` probe scripts in
`/tmp/` at the time of writing.
