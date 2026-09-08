# Non-canonical successor route census

## Outcome

The exact per-successor route census is **blocked before route classification**.
The live population itself was re-measured successfully and has moved from the
prior report: the graph still carries **105** drafted-and-open successors, with
the same validation partition of **94 valid, 8 quarantined, and 3 pending**, but
the strict-parser partition is now **56 strict-valid, 48 strict-non-canonical,
and 1 generic parse failure**. The prior `49 of 105` strict-non-canonical figure
therefore no longer holds exactly. There are still 49 parser-invalid rows in
total, but one of them exposes no canonical spelling and cannot enter the route
table required by this node.

The generic parse failure is
`inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width`
(`validation_status='quarantined'`). The strict parser reports:

```text
residue 'inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width'
does not match any physical_base or geometry_carrier; nearest candidates:
['normalized_toroidal_flux_coordinate']
```

It has no parser-supplied canonical form. Its migration route is therefore
blocked by the **strict ISN parse guard**; inventing a spelling here would
violate the plan's parse, re-render, and reparse contract.

## Intended route instrument

The read-only instrument is preserved beside the worker manifest as
`route_census.py`. It performs these checks for every parser-supplied
non-canonical pair:

1. Read the current drafted-and-open cohort and obtain the canonical spelling
   only from the strict parser's `canonical_form`.
2. Read whether the canonical `StandardName.id` exists and whether its
   `name_stage` is live.
3. For an occupied spelling, call `supersede_into(..., dry_run=True)` so the
   complete `_fold_guard_reason` decision is evaluated in a rolled-back
   transaction.
4. For an unoccupied spelling, call `apply_edit(..., dry_run=True)`, then
   reproduce `retarget_standard_name_sources`'s exact live-binding, source
   status, active-claim, and scalar compare-and-set predicates. The attachment
   pairing and DD unit-authority checks are also evaluated without writing.

This produces exactly the requested route set: `staged rename`, `fold`, or
`blocked` with the refusing guard and its detail. It stages no edit, executes no
fold, changes no source, and never invokes `sn run`.

## Execution blocker

The route instrument is a multi-minute live-graph census and therefore belongs
on a debug compute node under the workstation execution policy. The first
`all_debug` launch failed because the compute-shell `uv` function already adds
`--no-sync`, making the explicit flag a duplicate. After removing that
duplicate, the second launch reached the instrument but could not reach Neo4j:
the profile resolves through the login-node-local tunnel
`bolt://localhost:17687`, while the allocated compute node has no listener on
that loopback port and cannot start its own tunnel because SSH to `iter:22` is
refused.

The two-failure node fence now requires a stop. The exact route groups and the
dominant-blocker count are **not measured**, so no row is assigned a guessed
route and no migration node should be cut from this partial record.

## Recovery options

1. Make the graph endpoint reachable from `all_debug` through a non-loopback
   URI or a compute-node-valid tunnel, then rerun the preserved instrument.
   This retains the required compute placement and is the preferred route.
2. Explicitly authorize this read-only, I/O-bound census on the login node.
   The preliminary strict-parser pass completed there, but full fold/source
   preflights exceeded the ten-second heavy-work threshold.
3. Supply a current graph snapshot or an already-produced route-preflight
   artifact containing target occupancy, source binding state, and fold-guard
   verdicts for the 48 parser-supplied pairs; the report can then be completed
   without a live scan.

Choosing the wrong recovery can require re-running the entire 48-row census.
In particular, classifying from the prior 2026-09-05 ten-row refusal set would
be unsound because the relevant cohort has already moved and the current
drafted-and-open population is a different lifecycle slice.
