# Ratified identity archive reconstruction

## Outcome

**Blocked after two successful atomic applies.** `time` and
`flux_surface_averaged_current_density_due_to_wave_driven_current_drive` each
passed an independent signed preview with zero refusals and each received an
`applied` receipt for its complete 18-edge archive authority. Both live nodes
have the exact signed node properties, `status=draft`, `origin=pipeline`, one
matching direct producer, and three unchanged-ratification records. The later
independent reproduction query nevertheless found that neither node still has
full per-relationship-type incident parity with its archive snapshot.

The quantitative result is therefore:

- signed previews admitted: **2/2**;
- preview refusals: **0/2**;
- atomic applies reported `applied`: **2/2**;
- restored nodes at `status=draft` and `origin=pipeline`: **2/2**;
- restored nodes carrying exactly three unchanged-ratification records: **2/2**;
- restored nodes with complete incident-type archive parity afterward: **0/2**;
- `plasma_beta` live nodes created: **0**.

No further graph mutation was attempted after the independent parity failure.
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

| Relationship type | Archive | Live at 12:47:53 CEST | Equal |
| --- | ---: | ---: | :---: |
| `DOCS_REVISION_OF` | 2 | 2 | yes |
| `FOR_STANDARD_NAME` | 0, absent from archive | 22 | **no** |
| `HAS_INTERNAL_CHANGE` | 3 | 3 | yes |
| `HAS_PARENT` | 2 | 1 | **no** |
| `HAS_REVIEW` | 8 | 8 | yes |
| `HAS_STRUCTURAL_AUTHORITY` | 1 | 1 | yes |
| `HAS_UNIT` | 1 | 1 | yes |
| `PRODUCED_NAME` | 1 | 1 | yes |

All 22 live-only `FOR_STANDARD_NAME` relationships come from `LLMCost` nodes.
The archive's two `HAS_PARENT` incidents were incoming edges from
`alfven_time` and `ratio_of_coolant_mass_to_time`. The live reproduction query
found only the incoming `alfven_time` edge; the latter child is live but has no
`HAS_PARENT` target.

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

| Relationship type | Archive | Live at 12:47:53 CEST | Equal |
| --- | ---: | ---: | :---: |
| `DOCS_REVISION_OF` | 2 | 2 | yes |
| `ENTAILED_FROM_CHILD` | 1 | 1 | yes |
| `FOR_STANDARD_NAME` | 0, absent from archive | 11 | **no** |
| `HAS_INTERNAL_CHANGE` | 3 | 3 | yes |
| `HAS_PARENT` | 2 | 1 | **no** |
| `HAS_REVIEW` | 7 | 7 | yes |
| `HAS_STRUCTURAL_AUTHORITY` | 1 | 1 | yes |
| `HAS_UNIT` | 1 | 1 | yes |
| `PRODUCED_NAME` | 1 | 1 | yes |

All 11 live-only `FOR_STANDARD_NAME` relationships come from `LLMCost` nodes.
The archive's two `HAS_PARENT` incidents were an incoming edge from
`parallel_flux_surface_averaged_current_density_due_to_wave_driven_current_drive`
and an outgoing edge to `current_density_due_to_wave_driven_current_drive`.
Only the outgoing edge is live; the parallel child is live but has no
`HAS_PARENT` target.

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

The lead's independent observation reported seven `beta_tor` sources, all
attached or composed, with scalar and edge agreement on `toroidal_beta`. The
bounded live reproduction at 12:47:53 and 12:54:03 CEST found only the five
rows below. It found no additional `StandardNameSource` with
`produced_sn_id=toroidal_beta`, so two exact source identities and statuses
cannot be recorded from the current graph without inventing evidence.

| Live source | Status | Scalar | Edge target |
| --- | --- | --- | --- |
| `dd:core_profiles/global_quantities/beta_tor` | `attached` | `toroidal_beta` | `toroidal_beta` |
| `dd:equilibrium/time_slice/global_quantities/beta_tor` | `attached` | `toroidal_beta` | `toroidal_beta` |
| `dd:plasma_profiles/global_quantities/beta_tor` | `attached` | `toroidal_beta` | `toroidal_beta` |
| `dd:summary/global_quantities/beta_tor/value` | `attached` | `toroidal_beta` | `toroidal_beta` |
| `dd:summary/global_quantities/beta_tor_mhd/value` | `composed` | `toroidal_beta` | `toroidal_beta` |

The target itself has `name_stage=accepted`, `status=draft`, and
`origin=pipeline`; it has `HAS_PARENT` to `beta` and `REFINED_FROM` to
`mhd_beta`. This is a coherent current home for the observed five producers and
is stronger authority than the deleted row's old producer edge. The three
unchanged-ratification records are evidence about the former catalog row; they
do not authorize this restore to adjudicate the quantity's current home.

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

## Additional adapter capability required

The two successful receipts demonstrate that the current closed adapter can
reconstruct its registered archive edge types atomically. The independent
query demonstrates that this is not enough to prove complete incident closure.
The missing capability is **signed full-incident postflight parity**, including
relationship types absent from the archive and the persistence of both
directions of current `HAS_PARENT` topology. It must either keep the committed
node at the complete signed closure or refuse and roll back when live-only
`FOR_STANDARD_NAME` edges or non-persistent archived parent edges prevent that
closure. Re-running the current adapter or manually deleting or recreating
edges would bypass the signed authority and was not attempted.

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
- `live-restore-drift.json` — exact current incident counterparts and the
  bounded beta-source census.

These artifacts are under
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260909T095558811143-n-pidp-restore-the-three-ratified-identities/`.
The repository-wide suite was not run here; merged-head verification belongs to
the separately dispatched test node.
