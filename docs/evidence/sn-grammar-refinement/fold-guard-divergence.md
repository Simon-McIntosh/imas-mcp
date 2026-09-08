# The fold ambiguity guard reads the change ledger, not the edge alone

The folder's ambiguity guard asks whether some identity other than the fold
target currently claims the source's meaning. It was answering that question
from the existence of a `REFINED_FROM` edge; it must answer it from the
change ledger's ordering, because an edge records every succession a rename
ever made — including renames a later event revoked. The divergence set was
enumerated and recorded against the live graph **before** the guard changed,
the guard now decides from `StandardNameChange` ordering, and no fold outside
the enumerated set changes verdict.

## Divergence set, enumerated before the change

The rule under test: for a name `old`, the edge-existence reading treats
every `X -[:REFINED_FROM]-> old` edge as a live successor claim; the ledger
reading treats `X` as a current successor only when the most recent rename or
supersede event on the `(old, X)` pair moved the identity **onto** `X`, and
as a reverted (dead) claim when that latest event moved it back onto `old`.
A name is divergent when the two readings disagree.

Measured on the live graph (2026-09-07, worktree
`n-sgr-the-fold-guard-reads-the-present`, base `69f24046b`):

| Names with ≥ 1 successor edge | Verdict differs | Verdict unchanged |
|---|---:|---:|---:|
| 1710 | **7** | 1703 |

The divergence set is small and consists entirely of reversed lineage, exactly
the reverted-rename class the design expects. The seven names:

| Fold source (`old`) | Edge successors | Ledger-current successors | Why the excluded successor is dead |
|---|---|---|---|
| `flux_surface_averaged_deuterium_tritium_density_at_plasma_boundary` | `deuterium_tritium_density_flux_surface_averaged_at_plasma_boundary`, `flux_surface_averaged_deuterium_tritium_density` | `deuterium_tritium_density_flux_surface_averaged_at_plasma_boundary` | `flux_surface_averaged_deuterium_tritium_density`: latest lineage event `backfill_refine` moved the identity back onto `old` (2026-07-17T05:11) |
| `normalized_toroidal_beta` | `normalized_toroidal_plasma_beta` | — | `fold_identity` (2026-09-06T11:04) moved `normalized_toroidal_plasma_beta` into `old`; its old edge onto `old` is the folded-away residue |
| `ratio_of_mean_runaway_electron_velocity_to_runaway_electron_speed` | `ratio_of_runaway_electron_mean_velocity_to_runaway_electron_speed` | — | the `refine` (2026-07-28T05:42) renamed that identity back onto `old` |
| `reflected_wave_voltage_of_reflectometer_antenna_amplitude` | `reflected_voltage_of_reflectometer_antenna` | — | the `refine` (2026-07-28T05:40) renamed that identity back onto `old` |
| `toroidal_contravariant_flux_surface_averaged_metric` | `derivative_with_respect_to_toroidal_flux_coordinate_of_volume`, `flux_surface_averaged_metric` | `derivative_with_respect_to_toroidal_flux_coordinate_of_volume` | `flux_surface_averaged_metric`: refined back onto `old` (2026-07-22T11:25) |
| `x_direction_unit_vector_of_camera` | `x_image_up_unit_vector_of_camera` | — | `supersede_into_ancestor` (2026-08-18T23:55) moved that identity into `old` |
| `z_image_up_unit_vector_of_camera` | `z_image_up_unit_vector` | — | `human_edit` (2026-07-28T11:34) renamed it back onto `old`, three events after the `regenerate` that created the claim — the same shape as beta |

Every excluded successor shares one property: the latest lineage event
between it and the source moved the identity **back onto the source**, so the
remaining edge is history, not a live claim. A divergence set of this size
and shape is evidence the new rule is right, not evidence the old guard was.

## The canonical case: the normalized beta family

The live graph records the pair that originally motivated the change:

- `regenerate` moved the normalized identity's spelling to `beta` at
  2026-07-22T13:07:04 (change `sn-change:3d2943a9…`, owned by `beta`).
- `human_edit` moved it back onto the normalized identity at 13:10:05
  (change `sn-change:2cfc3637…`, owned by `normalized_toroidal_plasma_beta`).

Both edges survive (`beta -> normalized_toroidal_plasma_beta` and the
reverse), so the edge-existence reading treated `beta` as "another successor
lineage" and refused `sn supersede normalized_toroidal_plasma_beta --into
normalized_toroidal_beta` with "fold is ambiguous" forever. The ledger's
latest word on the pair is the `human_edit` back — `beta` is a reverted
rename, not a live claim — so under the ledger rule the fold is unambiguous.

Note on ownership: the create event lives on the third-party successor
(`beta`), the revert on the source, so a fold's own snapshot — which carries
only the source/targets' rows — cannot see both. The guard therefore re-reads
the pair ledger by snapping the successor/source pair.

### Current live state of the beta pair

The fold itself has already landed (2026-09-06T11:04:55, `fold_identity`
`normalized_toroidal_plasma_beta -> normalized_toroidal_beta`), after a
manual lineage adjudication on 2026-09-06T10:40 removed the `beta -> source`
edge (`remove_refined_from_relationship`, change
`sn-change:f5633cf1…`). The target `normalized_toroidal_beta` has since been
promoted to `reviewed`. Consequently the live CLI dry run of the same command
now stops at a *different* gate — the target-stage gate
(`target 'normalized_toroidal_beta' is name_stage='reviewed', not 'accepted'`)
— not the ambiguity gate. The ambiguity decision this node changes is no
longer the live obstruction for this particular pair because the fold already
landed; what the guard change removes is the *necessity of the manual edge
removal* for any future same-shape fold: a reverted rename no longer blocks
it, and no edge needs to be deleted to make it go through.

The demonstration that the beta fold clears is therefore run on the exact
historical pre-fold state — source accepted, canonical target a free
tombstone, mutual `beta` lineage present, both ledger rows present — through
`supersede_into(..., dry_run=True)` (see tests below). That dry run returns
`ok=True` under the new guard and `ok=False` "another successor lineage"
under the old one.

## The change

`_fold_guard_reason` (`imas_codex/standard_names/edit.py`) still enumerates
the successor edges onto the fold source and still refuses a duplicate
target lineage, but the "another successor lineage" refusal is now issued only
for a successor whose claim is *current*. A helper
`_fold_successor_is_current` reads every rename/supersede event touching the
`(source, successor)` pair — by re-snapshotting the pair, so the fold's own
snapshot-shape and persisted receipts are untouched — and lets the most recent
one decide: an event from the source onto the successor means the successor is
current; an event back onto the source means the rename was reverted and the
claim is dead. An edge with **no** recorded lineage event (one that predates
the ledger) keeps the edge-existence reading — the conservative direction,
since a fold the ledger cannot vouch for stays refused exactly as before.

Only operations that move an identity's own spelling count as lineage events
(`human_edit`, `regenerate`, `refine`, `backfill_refine`, `fold`,
`fold_identity`, `supersede_legacy_spelling`, `supersede_into_ancestor`,
`source_migration_manifest`). The allowlist is load-bearing, not cosmetic:
the same pair carries a `remove_refined_from_relationship` row dated
2026-09-06 whose `from_name`/`to_name` read `source -> beta`; were it counted
as a rename it would look like a fresh claim and flip the verdict. Source
moves, attachment repairs and reconciles touch `from_name`/`to_name` without
changing lineage and are excluded for the same reason.

`REFINED_FROM` is left as the complete history it is — no edge is deleted and
the two-cycle remains, as it must.

## After: no other fold's verdict moves

The new guard's ambiguity verdict differs from the old guard's exactly on the
enumerated set: the comparison term is `edge_successors vs
ledger-current-successors`, which disagree on precisely the 7 names above and
nowhere else (1703 of 1710 names with successor edges are byte-identical
between the two readings). All 7 disagreements are the reverted-lineage class;
there is no case where a live, non-reverted successor stops blocking a fold.

## Tests

`tests/standard_names/test_fold_guard_current_successor.py` (new) covers the
reverted-rename case by name:

- `test_reverted_beta_rename_clears_the_tombstone_fold_dry_run` — the beta
  pre-fold state folds onto the free tombstone; **fails** under the old
  edge-existence rule, passes now.
- `test_reverted_rename_clears_an_accepted_target_fold` — same rule with an
  accepted target; also fails under the old rule.
- `test_live_rename_is_still_a_current_successor` — a succession the ledger
  has not reverted still blocks the fold (no-regression).
- `test_edge_with_no_ledger_keeps_the_edge_existence_reading` — an edge with
  no ledger rows still blocks the fold (conservative direction).
- `test_a_reclaim_after_revert_is_current_again` — a rename onto `beta`, a
  revert, and a second rename onto `beta` leave `beta` current: the most
  recent event decides, not "any revert kills the claim".

Verified against the old guard by temporarily restoring the edge-existence
block: exactly `test_reverted_beta_rename_clears_the_tombstone_fold_dry_run`
and `test_reverted_rename_clears_an_accepted_target_fold` fail by name; the
other three hold under both readings.

Run results (debug partition `all_debug`, worktree base `69f24046b`):

- New file alone / full fold gate: 5 new tests pass; the broad fold gate
  (`test_fold_guard_current_successor`, `test_tombstone_supersede`,
  `test_tombstone_fold_lineage`, `test_supersede_into_ancestor`,
  `test_lineage_edge_removal`, `test_source_backing_removal`,
  `tests/cli/test_sn_identity_fold`) passes with **0 added failures**.
- Full `tests/standard_names` suite: see manifest `after_suite` for the
  revision, command, exit status and failure list.

## Live-graph sanity check of the new decision

The helper was exercised directly against the live graph through a
read-only transaction:

```
pair events between beta and normalized_toroidal_plasma_beta:
   ('regenerate', 'normalized_toroidal_plasma_beta', 'beta', '2026-07-22T13:07:04Z')
   ('human_edit', 'beta', 'normalized_toroidal_plasma_beta', '2026-07-22T13:10:05Z')
   ('refine',     'beta', 'normalized_toroidal_plasma_beta', '2026-07-23T15:16:02Z')
   ('remove_refined_from_relationship', 'normalized_toroidal_plasma_beta', 'beta', '2026-09-06T10:40:30Z')
_fold_successor_is_current(old=normalized_toroidal_plasma_beta, successor=beta) -> False  (reverted)
_fold_successor_is_current(old=invalid_duplicate, successor=electron_density)   -> True   (no ledger -> edge-existence stands)
```

The latest lineage event on the beta pair is a move back onto the source
(now `human_edit`, here reinforced by the later `refine` in the same
direction), so `beta` is correctly not a current successor; the
`remove_refined_from_relationship` row — which would read as a *create* if
counted — is excluded by the operation allowlist.
