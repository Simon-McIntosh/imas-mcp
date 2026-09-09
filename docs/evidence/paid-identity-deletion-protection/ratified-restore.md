# Ratified identity archive reconstruction

## Outcome

**Blocked on one structural-retention code gap after two successful atomic
applies.** `time` and
`flux_surface_averaged_current_density_due_to_wave_driven_current_drive` each
passed an independent signed preview with zero refusals and each received an
`applied` receipt for its complete 18-edge archive authority. Both live nodes
have the exact signed node properties, `status=draft`, `origin=pipeline`, one
matching direct producer, and three unchanged-ratification records. The later
independent reproduction query found that both retained every archived
relationship-type floor except the second `HAS_PARENT` edge.

The quantitative result is therefore:

- signed previews admitted: **2/2**;
- preview refusals: **0/2**;
- atomic applies reported `applied`: **2/2**;
- restored nodes at `status=draft` and `origin=pipeline`: **2/2**;
- restored nodes carrying exactly three unchanged-ratification records: **2/2**;
- restored nodes with every archived relationship count at or above its archive
  floor afterward: **0/2**, solely because `HAS_PARENT` is 1 live against 2
  archived on each;
- live relationship surpluses treated as retained evidence rather than
  failures: `FOR_STANDARD_NAME +22` on `time`, `FOR_STANDARD_NAME +11` and
  `HAS_INTERNAL_CHANGE +1` on the flux-surface identity;
- `plasma_beta` live nodes created: **0**.

No further graph mutation was attempted after the independent floor failure.
The live-graph commands ran on the login node because the configured Neo4j
endpoint is a login-local tunnel; every query was restricted to the named
identities or their exact producer cohort and completed within the ten-second
query ceiling. No model calls were made and the measured model spend was
**$0.00**.

## Signed authorities and receipts

The archive snapshots were read on an `all_debug` compute node. Each apply was
authorized only after a fresh, separate preview returned `would_apply` with a
refusal count of zero.

| Identity | Archive snapshot | Authority file SHA-256 | Signed payload SHA-256 | Authorized manifest SHA-256 | Preview | Apply receipt |
| --- | --- | --- | --- | --- | --- | --- |
| `time` | `imas-codex-graph-dev-198ec82-20260902T085827Z.tar.gz` | `13349c1c85234d15388d5e6c94db4a6544c7a60c14dbca538c19756bfad9ac6c` | `4c7bfb387899353a6888306aa8b2c584cc7a1dcbcb3bd6fd0bda5c77381deaa7` | `f0fc2e603ce7548955facaf1895055d7a2c52bfa76e4d2ffec9fde7847efe19d` | `would_apply`; 0 refusals | `applied`; changed 1; 30 mutations; 0 refusals; 2026-09-09 12:42:58 CEST |
| `flux_surface_averaged_current_density_due_to_wave_driven_current_drive` | `imas-codex-graph-dev-5b5faf1-20260905T121249Z.tar.gz` | `b319eb12b16787a3f1d070afb157985eae858a13183b7a761302a4e9dd2b9913` | `ba83ec96947803933f45960107c942fc5f40e822ce4fc371e8d7c859a2b4a784` | `207cfb8b21615faeb3759c0711da298a6d2a1047ea31f125bc07e2f1aa949d8b` | `would_apply`; 0 refusals | `applied`; changed 1; 29 mutations; 0 refusals; 2026-09-09 12:44:59 CEST |

The applied adapter revision was `6f5f526f83d6ee4e5901d4435dccc18264009bb3`.
Its own copy of `signed_manifest.py` contains
`_ARCHIVE_RECONSTRUCTABLE_COUNTERPART_LABELS` and typed counterpart identity
handling; its archive-reconstruction test file contains ten tests.

## `time`

The signed node state matches live property for property. Its origin is
truthful because the direct producer is `derived:time`, whose status is
`composed`; its scalar mirror is `produced_sn_id=time` and its
`PRODUCED_NAME` edge also targets `time`.

The archive authority declared these 18 incident edges:

| Relationship type | Archive floor | Live | Meets floor |
| --- | ---: | ---: | :---: |
| `DOCS_REVISION_OF` | 2 | 2 | yes |
| `FOR_STANDARD_NAME` | 0, absent from archive | 22 | yes; surplus 22 |
| `HAS_INTERNAL_CHANGE` | 3 | 3 | yes |
| `HAS_PARENT` | 2 | 1 | **no** |
| `HAS_REVIEW` | 8 | 8 | yes |
| `HAS_STRUCTURAL_AUTHORITY` | 1 | 1 | yes |
| `HAS_UNIT` | 1 | 1 | yes |
| `PRODUCED_NAME` | 1 | 1 | yes |

All 22 live-only `FOR_STANDARD_NAME` relationships come from `LLMCost` nodes.
The pipeline re-minted this correct cost evidence after deletion; removing it
to reproduce archive equality would destroy newer authority and is forbidden.
The archive's two `HAS_PARENT` incidents were incoming edges from
`alfven_time` and `ratio_of_coolant_mass_to_time`. The live reproduction query
found only the incoming `alfven_time` edge. The missing counterpart is a
`StandardName` node and the archived relationship direction is incoming:
`(ratio_of_coolant_mass_to_time)-[:HAS_PARENT]->(time)`. That counterpart is
currently `name_stage=superseded`, `status=superseded`.

The three retained unchanged-ratification records are:

| Change id | Operation | From | To |
| --- | --- | --- | --- |
| `sn-change:1be1f4a6-acc8-43e1-bd25-c1adc0e2ad65` | `unchanged_ratification` | `time` | `time` |
| `sn-change:6d81891b-9fa2-43f3-9a8a-b7e11d4c1d88` | `unchanged_ratification` | `time` | `time` |
| `sn-change:7b42fa0e-ae41-487c-8a8f-d2e65e0c2ed5` | `unchanged_ratification` | `time` | `time` |

## `flux_surface_averaged_current_density_due_to_wave_driven_current_drive`

The signed node state matches live property for property. Its origin is
truthful because the direct producer is
`derived:flux_surface_averaged_current_density_due_to_wave_driven_current_drive`,
whose status is `composed`; its scalar mirror and `PRODUCED_NAME` edge both
target the restored identity.

The archive authority declared these 18 incident edges:

| Relationship type | Archive floor | Live | Meets floor |
| --- | ---: | ---: | :---: |
| `DOCS_REVISION_OF` | 2 | 2 | yes |
| `ENTAILED_FROM_CHILD` | 1 | 1 | yes |
| `FOR_STANDARD_NAME` | 0, absent from archive | 11 | yes; surplus 11 |
| `HAS_INTERNAL_CHANGE` | 3 | 4 | yes; surplus 1 |
| `HAS_PARENT` | 2 | 1 | **no** |
| `HAS_REVIEW` | 7 | 7 | yes |
| `HAS_STRUCTURAL_AUTHORITY` | 1 | 1 | yes |
| `HAS_UNIT` | 1 | 1 | yes |
| `PRODUCED_NAME` | 1 | 1 | yes |

All 11 live-only `FOR_STANDARD_NAME` relationships come from `LLMCost` nodes
re-minted by the pipeline after deletion and are correct retained evidence. The
fourth live `HAS_INTERNAL_CHANGE` is
`sn-change:f3c39d04-1418-4771-9872-10bb8519a910`, a
`realign_grammar_segments` reconciliation record created because stored grammar
segments disagreed with the canonical parse. Both surpluses are explained
observations and satisfy the corrected archive-floor measure.

The archive's two `HAS_PARENT` incidents were an incoming edge from
`parallel_flux_surface_averaged_current_density_due_to_wave_driven_current_drive`
and an outgoing edge to `current_density_due_to_wave_driven_current_drive`.
Only the outgoing edge is live. The missing counterpart is a `StandardName`
node and the archived relationship direction is incoming:
`(parallel_flux_surface_averaged_current_density_due_to_wave_driven_current_drive)-[:HAS_PARENT]->(flux_surface_averaged_current_density_due_to_wave_driven_current_drive)`.
That counterpart is currently `name_stage=superseded`, `status=superseded`.

The three retained unchanged-ratification records are:

| Change id | Operation | From | To |
| --- | --- | --- | --- |
| `sn-change:0400a3f1-60b5-420c-b1ab-dbae5ce0f4fa` | `unchanged_ratification` | `flux_surface_averaged_current_density_due_to_wave_driven_current_drive` | same identity |
| `sn-change:b62b5384-476d-46f2-9183-4a6d88c6e0ad` | `unchanged_ratification` | `flux_surface_averaged_current_density_due_to_wave_driven_current_drive` | same identity |
| `sn-change:caf7b3ee-9dba-471b-a4d7-608a20a75cc3` | `unchanged_ratification` | `flux_surface_averaged_current_density_due_to_wave_driven_current_drive` | same identity |

## `plasma_beta`: held semantic collision

`plasma_beta` remains absent. Its three archived unchanged-ratification records
remain live as editorial records but have no owning `StandardName` edge:

- `sn-change:0371b379-0808-4741-84ee-cb48bfc1c268`;
- `sn-change:0b3387cd-ae52-4140-9380-f7a4cb1e4fdc`;
- `sn-change:f4d5fb4a-4c4b-44ad-8a13-419a26d747fa`.

The archived source
`dd:summary/global_quantities/beta_tor/value` now has `status=attached`, a
`produced_sn_id=toroidal_beta` scalar, and one `PRODUCED_NAME` edge to
`toroidal_beta`. Restoring its archived edge to `plasma_beta` would create dual
source authority and would contradict that scalar. A partial closure for
`plasma_beta` was therefore refused.

The hold is accepted as correct and is now a rename question under the lead's
beta ruling. The three unchanged-ratification records remain evidence about the
former catalog row; they do not authorize this restore to adjudicate the
quantity's current home. No further beta mutation or disposition decision was
undertaken.

No choice was made between the two possible dispositions:

1. leave `plasma_beta` absent because its quantity survives as
   `toroidal_beta`; or
2. reconstruct `plasma_beta` without the producer edge and immediately
   supersede it to `toroidal_beta`, accepting a knowing exception to archive
   parity rather than calling the operation a restore.

This is a general restore collision, not a name-specific adapter accident. If
a source has been retargeted since the archive was taken, its current scalar
and sole live producer edge are authority. Reinstating an archived producer
edge would create dual authority. The restore must refuse the whole identity
unless a separate ruling explicitly chooses a non-parity disposition.

## Missing-edge diagnosis

`_ARCHIVE_RECONSTRUCTABLE_COUNTERPART_LABELS` is **not** the reason either edge
is missing. That registry controls which missing counterpart nodes may be
created from a signed payload. Both counterparts already exist as
`StandardName` nodes, both authority files carry the incoming edges, and the
archive adapter's edge writer directly matches existing `StandardName`
endpoints. Adding `StandardName` to the reconstructable counterpart registry
would therefore address the wrong mechanism and could authorize unwanted
identity creation.

The gap is in the later structural reconciliation path in
`imas_codex/standard_names/graph_ops.py`. `rederive_structural_edges()` deletes
every operator-bearing `HAS_PARENT` relationship originating at a
`superseded` or `exhausted` child. It does not consult the target identity's
durable ratification protection. Both missing archive relationships meet that
deletion predicate: their child counterpart is superseded and their archived
edge carries `operator_kind` (`binary` for the ratio child and `projection` for
the parallel child).

The smallest change that carries the second edge is to bind the target in that
dead-edge cleanup and exclude a `HAS_PARENT` relationship when its target has
an `unchanged_ratification` protection record. A focused test must cover a
superseded child pointing to a ratified target and prove that ordinary dead
edges remain deletable. That change belongs to `graph_ops.py` and its structural
protection tests, which are outside this node's write scope and already owned
by the concurrent protected-identity work. It is recorded as a follow-on; no
change was made to `signed_manifest.py`, no signed apply was repeated, and no
edge was hand-edited.

## Evidence artifacts

- `archive-ratified-identities.json` — exact raw incident closures from the two
  pre-deletion snapshots.
- `admitted-ratified-restore-previews.json` — independent signed previews and
  their zero-refusal receipt digests.
- `time-reconstruction-authority.json` and
  `flux_surface_averaged_current_density_due_to_wave_driven_current_drive-reconstruction-authority.json`
  — the two signed one-identity authorities.
- `time-reconstruction-receipt.json` and
  `flux_surface_averaged_current_density_due_to_wave_driven_current_drive-reconstruction-receipt.json`
  — the two atomic apply receipts.
- `ratified-restore-postflight.json` — node properties, producer mirrors,
  ratifications, full incident counts, and the held beta evidence.
- `live-restore-drift.json` — exact current incident counterparts and
  counterpart lifecycle states.

These artifacts are under
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260909T095558811143-n-pidp-restore-the-three-ratified-identities/`.
The repository-wide suite was not run here; merged-head verification belongs to
the separately dispatched test node.
