# Blocked normalized-beta migration

## Result

The ordered migration cannot be completed through the sanctioned Standard Names
paths presently available. The identity selected by the live-plan ruling,
`normalized_toroidal_beta`, holds all eight producers but is catalog-terminal
(`status='superseded'`). The sanctioned rescore operation can change its
pipeline stage, but does not revive its catalog status; the exact scoped review
path then correctly refuses the same identity as terminal. No hand-written
Cypher workaround was used.

All live graph reads were bounded to these four identities on the login node,
where the Neo4j tunnel is available. They completed in under ten seconds.

## Live state before attempting the migration

| Identity | status | name_stage | docs_stage | producer count | docs length | docs score | docs review edges |
|---|---|---|---|---:|---:|---:|---:|
| `normalized_toroidal_beta` | `superseded` | `reviewed` | `pending` | 8 | 0 | null | 0 |
| `normalized_toroidal_plasma_beta` | `superseded` | `superseded` | `accepted` | 0 | 1,504 | 0.93125 | 7 |
| `normalized_toroidal_thermal_plasma_beta` | `superseded` | `superseded` | `pending` | 0 | 0 | null | 0 |
| `toroidal_beta` | `draft` | `accepted` | `accepted` | 5 | 1,449 | 0.925 | 9 |

The 1,504-character source document contains four erroneous links whose label
describes unnormalised toroidal beta but whose target is the document's own
identity. The intended replacements are all
`[toroidal_beta](name:toroidal_beta)`. The document was not copied because the
sanctioned docs edit correctly refused its target before mutation.

## Sanctioned-path results

1. `sn edit normalized_toroidal_beta --docs … --scope self --dry-run` refused:
   `target name_stage='reviewed' — docs edits require an accepted name`.
   This protects the docs review gate; it means the recovered documentation
   cannot be placed until the name has re-entered acceptance.

2. `sn supersede normalized_toroidal_plasma_beta --into
   normalized_toroidal_beta --dry-run` refused: its target was
   `name_stage='reviewed'`, not `accepted`. That operation cannot be used to
   revive the selected total-pressure identity before it has itself passed name
   review.

3. `sn rescore normalized_toroidal_beta --dry-run` was admitted, and the live
   sanctioned rescore transitioned the name axis from `reviewed` to `drafted`
   under `run_id=sn-rescore-20260909T172147Z`. It cleared the stale name score
   as designed. It did **not** change `status`, leaving it `superseded`.

4. The exact scoped continuation required to review that drafted identity,
   `sn run --name normalized_toroidal_beta --only review_name
   --skip-global-maintenance --dry-run`, then refused with
   `normalized_toroidal_beta: terminal StandardName lifecycle`.

The zero-cost read of `LLMCost` for the rescore run found no associated rows, so
the $15 cap has not been consumed. The live partial state after the sanctioned
rescore is therefore:

| Identity | status | name_stage | docs_stage | name score | docs score | producer count |
|---|---|---|---:|---:|---:|---:|
| `normalized_toroidal_beta` | `superseded` | `drafted` | `pending` | null | null | 8 |

No name-review edge was added in this attempt; the count remains 10 historical
name-axis edges. The four-identity lineage still contains both directions:
`normalized_toroidal_beta REFINED_FROM normalized_toroidal_plasma_beta` and
the inverse. No source path, documentation, link, or lineage edge was moved.

## Cause

The two sanctioned mechanisms disagree on the status of a rescore candidate:

* `stage_name_for_rescore` explicitly admits a `reviewed` predecessor, changes
  only the name-axis state to `drafted`, and leaves a superseded predecessor's
  catalog status intact so its lineage survives.
* `scope_exact_standard_names`, used by `sn run --name`, refuses either
  `name_stage='superseded'` **or** `status='superseded'` as a terminal
  lifecycle.

The rescore API therefore creates a drafted review candidate that its own
scoped pipeline cannot claim whenever the historical catalog status is
`superseded`. A docs edit is also unavailable because it requires both
`name_stage='accepted'` and an already settled docs axis.

## Required authority and next action

This needs a narrowly sanctioned identity-revival path that atomically changes
the selected, source-bound predecessor from catalog-superseded to reviewable
draft while preserving its eight `PRODUCED_NAME` bindings and lineage. The path
must be available through the Standard Names CLI, re-review the unchanged name,
and allow the documentation edit only after name acceptance. It must not be a
hand-written graph patch and must retain the current terminal-state guard for
ordinary exact scopes.

After that capability exists, resume in this order: revive and name-review
`normalized_toroidal_beta`; migrate and independently review the corrected
1,504-character document; move only
`dd:summary/global_quantities/beta_tor_thermal_norm/value` to the thermal-only
identity and review both axes; then remove the inverse `REFINED_FROM` edge.
