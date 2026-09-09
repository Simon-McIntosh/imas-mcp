# Four restaged name reviews

## Verdict

All four named rows now carry a non-null name-axis score backed by one or
more `HAS_REVIEW` edges on that same axis. Three rows needed new review work;
`gas_flow` did not. Its six retained reviews all attach to `gas_flow` itself,
so the prior null score was a falsely restrictive projection, not missing
evidence. The recovery selected the most recent authoritative escalation
verdict. Its score is below the current `0.85` acceptance threshold, so the
truthful post-state is `name_stage='reviewed'`, not a fabricated acceptance.

The live graph was read and written on the login node because the Neo4j tunnel
is login-node-local. Every read was bounded to the four identities below (or
to `gas_flow`'s six edges) and completed in under ten seconds.

## Initial evidence and final state

The four rows were already restaged to `drafted`; no restage command was run
here. The initial read, taken before any review work, established the reported
split. The final read checks both the mirrored scalar and the authoritative
edge count.

| Identity | Initial `name_stage` | Initial score | Initial name-axis edges | Action | Final `name_stage` | Final score | Final name-axis edges |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: |
| `atomic_mass` | `drafted` | null | 0 | New quorum review | `accepted` | 1.00000 | 2 |
| `gas_flow` | `drafted` | null | 6 | Re-derived from the latest authoritative review | `reviewed` | 0.68750 | 6 |
| `plasma_pressure` | `drafted` | null | 0 | New quorum review | `reviewed` | 0.71875 | 2 |
| `wave_phase_of_ion_cyclotron_heating_antenna` | `drafted` | null | 0 | New quorum review | `accepted` | 0.86875 | 2 |

No row is `accepted` with zero name-axis evidence. In particular, `gas_flow`
is deliberately not accepted: its current authoritative result says the bare
name is under-specified, and its `0.68750` score does not meet the `0.85`
threshold.

## `gas_flow`: verify before spend

The six historical edges do apply to the same current identity, rather than a
superseded spelling or another projection: every edge has
`review_axis='name'` and `standard_name_id='gas_flow'`, and every edge is
attached directly from `StandardName(id='gas_flow')`. The table quotes the
identity and each required review field from the bounded live read.

| Review id | `review_axis` | `reviewed_at` | `score` | Resolves to |
| --- | --- | --- | ---: | --- |
| `gas_flow:names:64f612ca-eb55-49fc-b9ee-a68261a93c26:0` | `name` | `2026-07-02T05:18:50.243960Z` | 0.8250 | `gas_flow` |
| `gas_flow:names:64f612ca-eb55-49fc-b9ee-a68261a93c26:1` | `name` | `2026-07-02T05:18:50.243960Z` | 0.6500 | `gas_flow` |
| `gas_flow:names:64f612ca-eb55-49fc-b9ee-a68261a93c26:2` | `name` | `2026-07-02T05:18:50.243960Z` | 0.6875 | `gas_flow` |
| `gas_flow:names:5059d567-99e2-4e8e-9c52-af9e4a492609:0` | `name` | `2026-06-22T13:43:12.426041Z` | 0.9750 | `gas_flow` |
| `gas_flow:names:5059d567-99e2-4e8e-9c52-af9e4a492609:1` | `name` | `2026-06-22T13:43:12.426041Z` | 0.7750 | `gas_flow` |
| `gas_flow:names:5059d567-99e2-4e8e-9c52-af9e4a492609:2` | `name` | `2026-06-22T13:43:12.426041Z` | 0.9375 | `gas_flow` |

The selected edge is the latest review group’s escalation record,
`gas_flow:names:64f612ca-eb55-49fc-b9ee-a68261a93c26:2`. It is explicitly
marked `resolution_method='authoritative_escalation'`, has score `0.6875`, and
was written on `2026-07-02T05:18:50.243960Z`. A bounded GraphClient transaction
matched only that identity and only its authoritative name-axis review edges,
copied its score, rubric, comments, reviewer, tier, and review timestamp to
the mirrored name fields, and set `name_stage='reviewed'` by the current
threshold. It created no review node, no review edge, and no cost event.

## Paid gap-only review rotation

The exact command named only the three rows with zero evidence, never used
`--reseed` or `--force`, and bypassed global maintenance as required:

```text
imas-codex sn run --name atomic_mass --name plasma_pressure \
  --name wave_phase_of_ion_cyclotron_heating_antenna --only review_name \
  --skip-global-maintenance --cost-limit 8 --time 20
```

The resulting run id was `d8fb6e9a-a867-4d90-984c-e8a8616233bd`. Its durable
CLI log reports `pool=review_name processed=3 spent=$0.2229
mean_cost=$0.074309`; summing the six review records gives the precise total
of **$0.22292672**. This is below the $8.00 ceiling.

| Paid identity | New name-review edges | Review time | Resulting score | Cost (USD) |
| --- | ---: | --- | ---: | ---: |
| `atomic_mass` | 2 | `2026-09-09T13:22:30.946292Z` | 1.00000 | 0.07348805 |
| `plasma_pressure` | 2 | `2026-09-09T13:22:47.808188Z` | 0.71875 | 0.06710931 |
| `wave_phase_of_ion_cyclotron_heating_antenna` | 2 | `2026-09-09T13:23:17.297351Z` | 0.86875 | 0.08232936 |
| `gas_flow` | 0 | existing evidence retained | 0.68750 | 0.00000000 |

`atomic_mass` and `wave_phase_of_ion_cyclotron_heating_antenna` reached
`accepted` on quorate review results. `plasma_pressure` reached `reviewed`
with a quorate but below-threshold score, which preserves the evidence without
claiming that it is accepted. `gas_flow` cost nothing because its evidence was
already present and applicable.

## Scope and follow-up

No signed-manifest apply was attempted. No identity outside the four named
rows was reviewed or changed by the review rotation. The pipeline emitted
unrelated global ledger warnings during startup, but the explicit scoped mode
and `--skip-global-maintenance` prevented mutation of those unrelated rows.

The remaining release decision is semantic rather than evidentiary:
`plasma_pressure` and `gas_flow` have genuine, below-threshold name review
results and therefore remain `reviewed`; they are not false-acceptance cases.
