# Exhaustion before a name-refinement attempt

Measured on the live graph on 2026-09-08 through the login-node-local graph
endpoint. Every query was bounded to an aggregate, one exact identity, or five
exact identities and completed below the ten-second query ceiling. No graph
write was made.

## Population and the two counters

There are **274** identities at `name_stage='exhausted'`. Of those, **259**
have null or zero `refine_name_count`: **5** are `validation_status='valid'`
and **254** are not valid. This reproduces the coordinator's census exactly,
but `refine_name_count` does not prove that no refinement was attempted.

The lifecycle decision reads the separate durable budget counter
`refine_attempts`. Among the 259 rows with no `refine_name_count`, **224 have a
positive `refine_attempts` value** and **35 have null or zero
`refine_attempts`**. None of those 35 is validation-valid. Across all 274
exhausted rows, **44** have null or zero `refine_attempts`.

The disagreement is explained by the write paths. `refine_attempts` is charged
at a verified claim before the model call and follows the identity lineage.
`refine_name_count` is updated by `write_standard_names` only when it rewrites
an identity that already has a positive `generate_name_count`; it is not
inherited by `persist_refined_name`. Three of the five valid rows also have
explicit `REFINED_FROM` lineage despite null `refine_name_count`. The 259-row
figure is therefore a telemetry-gap census, not a no-attempt census.

Before this repair, `persist_reviewed_name` projected a missing attempt counter
as `coalesce(sn.refine_attempts, coalesce(sn.chain_length, 0))`. When both
properties are missing, `refine_attempts` evaluates to **0**. The runtime
`rotation_cap` evaluates to **3**, so the documented comparison is **`0 >= 3`,
false**. A zero cap would have made **`0 >= 0`, true**, which is why a positive
attempt is now independently required. The same invariant now guards the
strict-grammar override and `stop_refine_name_attempt`: no scoped write path
can produce `exhausted` from a null or zero durable attempt count.

The repair changes no stored row and does not alter
`REFINE_NAME_ELIGIBILITY_WHERE`. It makes **0 current identities** eligible or
accepted, and it changes **0 of the five valid exhausted identities** because
all five already carry 2 or 3 positive attempts. On a future first review, a
zero/null attempt count remains `reviewed` and can take its first ordinary
refinement rather than becoming terminal.

## Thresholds and the five validation-valid rows

The review threshold actually imported by the workers is
`DEFAULT_MIN_SCORE = 0.85`; the runtime read returned **0.85** and a rotation
cap of **3**. The `0.75` default on `pool_pending_counts` is not the acceptance
threshold used by `process_review_name_batch`. Consequently,
`gap_at_plasma_boundary` at 0.8125 and `width_of_antenna_strap` at 0.8 are
below the configured review threshold and are **not wrongly exhausted on score
ordering alone**.

`run_export` has a separate default score floor of **0.65**. Four of the five
rows clear that floor; the derivative name at 0.6125 does not. The asymmetry is
real: export is willing on score grounds to publish four names that ordinary
review will not accept, while `name_stage='exhausted'` withholds all five.
Which score floor governs publication is a contract decision; this change does
not resolve it by lowering either threshold.

| Identity | Stored state and evidence | Recommendation |
|---|---|---|
| `gap_at_plasma_boundary` | Score **0.8125**, draft, valid, 3 attempts, stop reason `attempts_exhausted`; 17 name reviews span 0.65–0.975. Description: “Minimum geometric clearance between the plasma separatrix and the nearest limiter or wall element.” Bound to `dd:equilibrium/time_slice/boundary_separatrix/closest_wall_point/distance`; a stored unit-dimension issue remains. | Do not auto-accept. Revalidate the description/unit diagnosis against the exact DD path, then run one exact rescore under fresh authority; it is close to the review floor and has materially conflicting reviewers. |
| `width_of_antenna_strap` | Score **0.8**, draft, valid, 2 attempts, stop reason `successor_collision`; 5 reviews span 0.7875–0.9125. Description: “Width of an ICRH antenna strap in the toroidal direction.” No current direct source binding was returned. | Resolve the named successor collision and restore source authority before review. A collision needs the sanctioned edit/identity-resolution path, not automatic acceptance or another blind refine. |
| `neutron_rate` | Score **0.7375**, draft, valid, 3 attempts, stop reason `attempts_exhausted`; 11 reviews span 0.675–0.8. Description says total fusion neutron production across channels and populations. Its lineage includes `total_neutron_rate_due_to_fusion_reactions`, but no current direct source binding was returned. | Keep withheld and adjudicate whether the shortened identity lost the total/fusion semantics before an exact rescore. It clears export's score floor but not review's. |
| `gap_of_antenna_strap` | Score **0.6625**, draft, valid, 3 attempts, stop reason `attempts_exhausted`; 11 reviews span 0.6625–0.95. The description is a rear-facing strap-to-wall clearance; a unit-dimension issue remains and no current direct source binding was returned. | Re-establish source binding and settle the description/unit issue before rescore. Its score barely clears export's floor and is far below review's. |
| `derivative_of_area_of_flux_surface_with_respect_to_normalized_poloidal_flux_coordinate` | Score **0.6125**, draft, valid, 3 attempts, stop reason `attempts_exhausted`; 11 reviews span 0.5625–0.7125. The description says rate of change; stored diagnostics flag verb drift and repeated `flux`; no current direct source binding was returned. | Keep withheld. Repair the name/description semantics and source provenance before any review; it clears neither score floor. |

These rows can withhold four score-exportable names from the upstream batch,
so the threshold contract and the per-name dispositions are release-path work.
This guard itself does not promote or restage any of them.

## The adjudicated identity and review evidence

The dispatch snapshot recorded
`net_power_due_to_ion_cyclotron_heating` as exhausted with null generation and
refinement call counters, valid validation, no issue, and no vocabulary gap.
The live row changed while this node was running under the separately owned
adjudication: it now reads `name_stage='accepted'`, `refine_attempts=3`,
`chain_length=1`, and `reviewer_score_name=0.6875`. This node made no write to
that identity.

Its scalar score is not selected by a weakest-reviewer/minimum rule. The latest
review group, written at 2026-09-08 11:53:51 UTC, contains scores **0.7125,
0.8, and 0.6875**; the 0.6875 row is the escalator with
`resolution_method='authoritative_escalation'`. `persist_reviewed_name` writes
the quorum's `winning_score`; `update_review_aggregates` updates only the mean,
count, and disagreement fields. The stored scalar happens to equal the minimum
of all nine rows because the latest authoritative escalator supplied that
value, not because the implementation minimizes reviews.

Comments are persisted on this path. All **9 of 9** current name-axis review
rows have non-empty `comments`, and `write_reviews` writes that field. The
dispatch report that all nine comments were empty no longer describes the live
rows; the concurrent rescore produced reasoning before this node's bounded
read.
