# A semantics-preserving rename keeps its document

## Outcome

`net_power_due_to_ion_cyclotron_heating` now carries the accepted document of
its predecessor, `power_due_to_ion_cyclotron_heating`, through a reasoned and
attributed graph transaction. The operation did not approve the name or advance
catalog lifecycle state: `name_stage` remains `accepted` and `status` remains
`draft`.

| Field | Before | After |
|---|---:|---:|
| `docs_stage` | `pending` | `accepted` |
| documentation length | 0 characters | 1,679 characters |
| `docs_model` | null | `openrouter/openai/gpt-5.6-luna` |
| `docs_generated_at` | null | `2026-09-01T21:49:51.601Z` |
| `name_stage` | `accepted` | `accepted` |
| `status` | `draft` | `draft` |

The carried documentation is byte-equal to the predecessor document. The
successor remains bound to the two data-dictionary sources that define this
quantity:

- `summary/heating_current_drive/ic/power/value`
- `ic_antennas/antenna/power_launched`

Its 168-character description was already unchanged across the rename:

> Net ion-cyclotron radio-frequency power launched through a heating antenna
> into the vacuum vessel, equal to forward power minus reflected power at the
> antenna boundary.

The added `net` qualifier therefore makes the description's existing
forward-minus-reflected definition explicit; it does not change the quantity
the document describes.

## Design ruling and rejected alternative

The grammar remains the automatic authority for structural re-renderings.
Exact grammar-IR equality and the existing locus-token substitution still
carry documentation automatically. Any other grammar difference still answers
`False` unless a caller supplies both a substantive reason and the asserting
actor.

The implementation deliberately does **not** classify every added qualifier as
meaning-preserving. A qualifier ordinarily narrows or otherwise modifies the
referent. For example,
`power_due_to_ion_cyclotron_heating` becoming
`perpendicular_power_due_to_ion_cyclotron_heating` adds an axis projection and
would inherit prose about total power under a blanket widening. The automatic
predicate returns `False` for both that pair and the measured `net` pair. The
measured pair crosses the gate only through the explicit assertion.

That explicit path writes a separate `StandardNameChange` receipt in the same
transaction as the documentation carry. The receipt records:

- declaration: `semantics_preserving_rename`
- predecessor document identity:
  `standard-name-document:power_due_to_ion_cyclotron_heating:sha256:ff36bff0b27b1392a345d61f7141c1e66d5c1cdb93dcfcc1d7617d81413c352a`
- asserting actor: `Simon McIntosh`
- receipt identity: `sn-change:736f5778-8056-5fef-b854-ed8849605bed`
- reason: the rename retained the same 168-character description and two
  authoritative DD source bindings; the added `net` qualifier makes the
  description's existing forward-minus-reflected definition explicit without
  changing the documented quantity

The repair route is idempotent. Its first invocation reported `changed: true`;
an immediate second invocation reported `changed: false`, returned the same
receipt identity, and left every measured field unchanged. It refuses a missing
reason, a missing actor, absent `REFINED_FROM` lineage, a predecessor without a
non-empty accepted document, or a successor whose docs axis already contains
independent content.

## Population census

Before the repair, 164 successor identities matched the same structural class:
a superseded predecessor with non-empty accepted documentation linked through
`REFINED_FROM` to a successor at `docs_stage=pending` with empty documentation.
Those successors came from 163 distinct predecessors. After repairing the one
authorized pair, 163 successor identities from 162 predecessors remain. Thus
there are **163 other successor identities** to adjudicate; this node did not
infer that their meanings are preserved and did not mutate them.

## Verification

The full `tests/standard_names` gate ran on the `all_debug` SLURM partition at
both revisions, using the repository's shared environment and default markers.

| Revision | Result |
|---|---|
| base `53b5d885171b19348b9dd4f74720abfa06726afb` | `7275 passed, 11 skipped, 323 deselected, 34 warnings in 284.56s (0:04:44)` |
| implementation `f4c1d604c795fb385fa28b61abec1af7fe89da8a` | `7287 passed, 11 skipped, 323 deselected, 34 warnings in 277.90s (0:04:37)` |

The change adds 12 regression cases and zero failures. A focused run covering
the documentation-carry tests and the existing locus-rename tests passed all 41
cases. The live mutation statement was also compiled with Neo4j `EXPLAIN`
before execution and produced no planner notifications.

Live graph reads and the one bounded mutation ran on the login node because the
configured Neo4j endpoint is reachable there but not from the compute nodes.
Each query was restricted to the named pair or the 4,852-node Standard Name
population and completed within the ten-second ceiling.
