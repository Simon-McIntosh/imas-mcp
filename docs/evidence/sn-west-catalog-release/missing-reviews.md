# Missing review evidence: partial recovery and blocked name cohort

## Verdict

`net_power_due_to_ion_cyclotron_heating` now has genuine docs-axis review
evidence behind its accepted scalar. The four accepted-but-unscored name rows
remain unreviewed because the sanctioned exact-cohort restager fails its own
post-state proof before committing. This is a hard implementation blocker, not
a spent-budget or review-quality result.

Only the five identities named here were read or selected. No signed manifest
apply, no global maintenance, no `--reseed`, and no `--force` operation ran.

## Baseline: scalar claims versus review evidence

The first bounded login-node read was restricted to this exact five-row set and
completed in under one second. It reads the relevant lifecycle scalar beside
the authoritative `HAS_REVIEW` relationship count.

| Identity | Axis required | Baseline scalar state | Baseline review edges | Interpretation |
| --- | --- | --- | ---: | --- |
| `net_power_due_to_ion_cyclotron_heating` | docs | `docs_stage=accepted`; `reviewer_score_docs=null` | 0 | False acceptance: the scalar had no docs-axis evidence behind it. |
| `atomic_mass` | name | `name_stage=accepted`; `reviewer_score_name=null` | 0 | No name-axis evidence. |
| `gas_flow` | name | `name_stage=accepted`; `reviewer_score_name=null` | 6 | The scalar is missing, but six historical name reviews exist; this is a missing projection, not a no-evidence row. |
| `plasma_pressure` | name | `name_stage=accepted`; `reviewer_score_name=null` | 0 | No name-axis evidence. |
| `wave_phase_of_ion_cyclotron_heating_antenna` | name | `name_stage=accepted`; `reviewer_score_name=null` | 0 | No name-axis evidence. |

The `gas_flow` measurement matters: a null scalar is not enough to infer that
the axis has no evidence. Its six name-axis reviews have latest timestamp
`2026-07-02T05:18:50.243960Z` and include an
`authoritative_escalation` resolution method.

## Completed docs-axis rotation

The lifecycle owner was used to resolve the actual false-acceptance row rather
than setting a scalar by hand:

1. `mark_members_for_regen(['net_power_due_to_ion_cyclotron_heating'],
   dry_run=True)` returned `eligible=1`, `reset=0`.
2. Its live invocation returned `eligible=1`, `reset=1`, and scoped the row to
   run `b65c5568-a8d6-4989-a8eb-d415ba50e6fa`.
3. The ordinary gap-only docs pipeline ran only that scope:

   ```text
   imas-codex sn run --scope-run-id b65c5568-a8d6-4989-a8eb-d415ba50e6fa \
     --docs-only --flush --skip-global-maintenance --cost-limit 4 --time 10
   ```

The reset snapshots the pre-existing document as a `DocsRevision`, resets the
docs lifecycle to `pending`, and then lets ordinary generation and review
claims establish new evidence. It is neither a scalar-only acceptance nor a
`--reseed`/`--force` operation.

The final bounded read proves the result: `docs_stage=accepted`,
`reviewer_score_docs=1.0000`, and **3** docs-axis `HAS_REVIEW` relationships.
Their final resolution method is `authoritative_escalation`; the newest review
timestamp is `2026-09-09T12:53:37.204416Z`.

The four `LLMCost` rows associated with this identity at the rotation time total
**$0.280291**: one docs generation call ($0.008595) and three docs review calls
($0.056256, $0.076835, and $0.138605). This is below the allocated $4 docs
rotation ceiling and below the node's $8 total ceiling.

## Name-axis restage blocked before any name review

The four exact identities were dry-run through the sanctioned accepted-name
restager with `--include-accepted`. Its receipt was `would_apply`, named all
four rows, proposed `accepted -> drafted` for each, wrote zero scores, and
preserved all 300 relationships in the cohort snapshot.

The live invocation was:

```text
imas-codex sn restage-accepted atomic_mass gas_flow plasma_pressure \
  wave_phase_of_ion_cyclotron_heating_antenna --include-accepted --apply
```

It exited non-zero with:

```text
RuntimeError: accepted restage post-state proof failed
```

The transaction rolled back. A fresh bounded read confirmed every requested
row remains `name_stage=accepted`, `reviewer_score_name=null`, claim-free, and
without a new `run_id`; no name-axis review was generated and no unrelated
identity was touched. The ordinary `review_name` pool only claims
`name_stage=drafted`, so it cannot proceed without a successful restage.

| Identity | Required axis | Resulting stage scalar | Resulting reviewer score | Resulting review edges | Outcome |
| --- | --- | --- | ---: | ---: | --- |
| `net_power_due_to_ion_cyclotron_heating` | docs | `docs_stage=accepted` | 1.0000 | 3 | Complete: scalar is backed by genuine docs reviews. |
| `atomic_mass` | name | `name_stage=accepted` | null | 0 | Blocked before name restage. |
| `gas_flow` | name | `name_stage=accepted` | null | 6 | Blocked before projection repair; review evidence already exists. |
| `plasma_pressure` | name | `name_stage=accepted` | null | 0 | Blocked before name restage. |
| `wave_phase_of_ion_cyclotron_heating_antenna` | name | `name_stage=accepted` | null | 0 | Blocked before name restage. |

## Required follow-up

Repair `restage_accepted_names_for_review()` so its atomic post-state proof
accepts the intended, relationship-preserving `accepted -> drafted` transition.
Then rerun the exact four-name review pool, re-read the same five-row set, and
require a non-null name score plus a name-axis review relationship for each
name that genuinely has none. `gas_flow` needs separate projection
adjudication: its live review history must not be overwritten merely because
its stored scalar is null.
