# Quarantine instrument authority

Measured on the production `codex` graph from the login-node-local graph
tunnel on 2026-09-08. Every live read was a bounded `StandardName` query; the
repair used the normal deterministic validation path and made no model call.

## Verdict

`validated_at` plus the verdict written in the same transaction by
`mark_names_validated` is authoritative for whether a name is currently
quarantined. A stored `validation_status = 'quarantined'` with
`validated_at IS NULL` is an undated historical value, not a current
quarantine finding. The scalar still carries the outcome, but it has no
authority without the observation that produced it.

The two campaign helpers are deliberately different operations, not competing
truth sources:

| Code path | Predicate it evaluates | Effect |
|---|---|---|
| `default_clear_quarantine` | `name_stage = 'accepted' AND validation_status = 'quarantined'` | moves a known quarantine to `pending`, removes its reason and observation time, then makes the name eligible for a new deterministic validation |
| `default_audit_revalidate` → `drain_validation_for_ids` | explicit ids with `validated_at IS NULL` | executes `validate_name_candidate` and calls `mark_names_validated`, which atomically writes `validation_issues`, `validation_status`, and `validated_at` |
| `default_revalidate` | campaign prose result after the deterministic drain | re-quarantines a reintroduced prose defect, but refuses to confirm any id that still has `validation_status = 'quarantined'` |

The error in the earlier sweep was using the final confirmation helper as if it
were the deterministic validator. Its guard was designed to prevent exactly
that: the query collects the requested ids whose predicate is
`coalesce(sn.validation_status, '') = 'quarantined'` and raises instead of
setting them `valid`. It cannot clear a quarantine. The correct sequence is
clear to `pending` → deterministic validation → campaign prose confirmation.

## Bounded live census before revalidation

Property coverage proved that the predicates could match before interpreting
the cohort:

| accepted names | `validation_status` present | `validated_at` present | stored quarantines | quarantines with an observation |
|---:|---:|---:|---:|---:|
| 2,360 | 2,360 | 2,094 | 14 | 0 |

The stored-status/stamp cross-tab was `quarantined + unvalidated = 14`,
`valid + unvalidated = 252`, and `valid + validated = 2,094`. Thus the prior
fourteen quarantine scalars were all stale as current verdicts; this conclusion
does not assert that the underlying names are valid.

## Revalidation cohort

The five withheld identities named by the release-tail evidence were evaluated
with the same pure `validate_name_candidate` function used by
`drain_validation_for_ids`. Scores are the persisted `review_mean_score` values
at the pre-run read. This is a non-mutating authority measurement, not a
release: the backing `default_audit_revalidate` call remains the required next
operation after the blocked suite gate is restored.

| identity | score | stored state | authoritative result | required post-gate state |
|---|---:|---|---|---|
| `radial_outline_of_flux_surface` | 0.9021 | quarantined, undated | quarantined: ISN requires `outline` to name the represented entity | quarantined, dated |
| `radial_outline_of_plasma_boundary` | 0.8444 | quarantined, undated | quarantined: same ISN semantic error | quarantined, dated |
| `radial_outline_of_wall` | 0.9063 | quarantined, undated | quarantined: same ISN semantic error | quarantined, dated |
| `toroidal_angle_of_active_limiter_point` | 0.8681 | quarantined, undated | quarantined: `name_unit_consistency_check` sees dimensionless unit `1` for an angle | quarantined, dated |
| `vertical_coordinate_of_line_of_sight` | 0.8828 | quarantined, undated | valid, no issues | valid, dated |

The eventual backing-path run will release one name because the authoritative
validator found no issue. It will retain four with dated verdicts; their repairs
belong to the grammar or DD-unit owners, not to a campaign status writer.

## Corrected census

For the five-name release-tail cohort, the authoritative pre-run classification
is **4 quarantine findings and 1 valid result**. These are not yet persisted
verdicts, so the current graph-wide genuine-quarantine census remains unknown
until the backing path re-stamps the accepted cohort. The important invariant is
now explicit in the code: clearing a quarantine also clears its observation
time, so no later reader can confuse a former verdict with a current one.
