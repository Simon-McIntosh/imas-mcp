# WEST recut preflight receipt

Date: 2026-09-08

## Result

No request was closed or opened. The required recut cannot carry
`net_power_due_to_ion_cyclotron_heating` because the live graph and the
release export agree that the successor is not publishable.

A bounded login-node graph read returned:

| Identity | `name_stage` | `validation_status` | catalog `status` | score |
|---|---|---|---|---:|
| `power_due_to_ion_cyclotron_heating` | `superseded` | `valid` | `superseded` | 0.9750 |
| `net_power_due_to_ion_cyclotron_heating` | `exhausted` | `valid` | `draft` | 0.6875 |
| `total_power_due_to_ion_cyclotron_heating` | `accepted` | `valid` | `draft` | 1.0000 |

The dispatch premise that the new identity was drafted is stale: its inline
name review finished before this recut and exhausted the successor. The old
spelling is therefore correctly absent from the export population, but the
replacement cannot take its place.

## Release preflight

The repository release recipe was followed through its non-mutating batch
preflight:

```text
imas-codex sn release --batch west_production_dd_paths --target fork --no-notes --dry-run -m "WEST review recut"
```

It exited 0, resolved the catalog fork and RC series correctly, and produced
an export report. The report's accounting closes:

```text
225 candidates - 206 exported = 19 accounted exclusions; residue = 0
```

The previous review cut exported 208 names. This preflight would export 206,
two fewer. It is not a valid replacement for request 18 because the new
identity is one of the seven `name_not_accepted` exclusions:

```text
net_power_due_to_ion_cyclotron_heating:
  reason: name_not_accepted
  detail: name_stage='exhausted'
```

The other exclusion bucket is `invalid_validation_status=12`; together with
`name_not_accepted=7`, these make the 19-accounted total. The exported
`auxiliary_heating.yml` contains neither the new nor the superseded spelling.
The report itself is present in staging, but it cannot ride a release commit
without violating the required inclusion predicate.

## Preserved boundaries

The default subtree dry-run of the rename had already shown why the total
variant stays untouched: it is a protected catalog-edited identity. This recut
did not pass `--override-edits`, did not rename either descendant, and did
not alter `total_power_due_to_ion_cyclotron_heating`.

No `--skip-gate` option was used. No fork request was closed, no branch was
force-pushed, no tag was created, and no catalog commit was made. Closing
request 18 before the successor becomes exportable would leave reviewers with
no request carrying the intended identity.

## Required next action

The lead adjudicated the exhausted proposal after this first preflight. The
backing path was `promote._contest` followed by
`promote.revert_contested`: the first represents the human-versus-rubric
disagreement, and the second records the human resolution while returning the
identity to the exportable name-axis stage. No threshold or review row was
changed.

Before adjudication the identity was `exhausted`, valid, catalog-draft, with
`reviewer_score_name=0.6875`. Exactly nine name-axis review rows were present,
spanning scores `0.6875` through `1.0`. After adjudication it read
`name_stage=accepted`, `validation_status=valid`, catalog `status=draft`,
the same reviewer score, the same nine review rows, and a
`contested_resolution` carrying the physics reason: net is the
forward-minus-reflected antenna-boundary power and distinguishes this identity
from the forward, reflected, and total variants.

## Second release preflight

A fresh execution of the same release preflight still could not produce the
required catalog. Its accounting also closes:

```text
224 candidates - 206 exported = 18 accounted exclusions; residue = 0
```

The successor is no longer an exclusion, but it is also not a candidate.
A bounded state read explains the population-boundary result:

```text
name_stage=accepted
docs_stage=pending
status=draft
source_paths=[]
PRODUCED_NAME source count=0
```

The exported entry files again contain neither the accepted successor nor the
superseded predecessor. Compared with request 18's 208 published entries, the
second preflight remains two entries smaller. Because the successor has no
source provenance and no accepted docs axis, opening the 206-name request would
still omit the exact identity this recut exists to carry.

The next action is a governed provenance/docs repair, not a release override:
restore the successor's source bindings through their backing lifecycle path
and complete its documentation axis, then repeat the preflight. Request 18
remains open until a zero-residue export positively contains the renamed
identity.

## Source-retarget refusal

The two source rows were then inspected directly before repair:

| Source | DD version | status | `produced_sn_id` | `PRODUCED_NAME` targets |
|---|---|---|---|---|
| `dd:ic_antennas/antenna/power_launched` | `4.1.0` | `extracted` | null | none |
| `dd:summary/heating_current_drive/ic/power/value` | `4.1.0` | `extracted` | null | none |

`retarget_standard_name_sources` was read before use. Its compare-and-set
contract requires every explicit source to have exactly the expected current
`PRODUCED_NAME` target and matching scalar, or to carry the already-completed
migration manifest. It maintains the edge, scalar mirror, upstream
`HAS_STANDARD_NAME` projection, and both names' `source_paths` together.

The guarded call named both sources, expected the superseded identity, and
planned the accepted net identity as its target. It refused before mutation:

```text
source migration compare-and-set failed:
dd:ic_antennas/antenna/power_launched(
  exists=True, status='extracted', claimed=False, bindings=[], scalar=None
),
dd:summary/heating_current_drive/ic/power/value(
  exists=True, status='extracted', claimed=False, bindings=[], scalar=None
)
```

This proves the rename dropped both the authoritative edge and its scalar
mirror. The ordinary retarget path cannot move a source that no longer points
at the predecessor, and forcing past that refusal would erase the very
concurrency protection the function provides. No edge, scalar, or
`source_paths` cache was written.

The recut remains stopped. A separate rename-path repair must prevent this data
loss, and a governed orphan-recovery path must establish authority for rebinding
these two null/null sources before this release node can resume.

## Source attachment recovery

The compare-and-set refusal identified the sources as wholly unbound, so the
correct backing path was `sn attach`, not retargeting. Each source was dry-run
separately before it was applied. The launcher path planned only this change:

```text
would attach ic_antennas/antenna/power_launched to
net_power_due_to_ion_cyclotron_heating
  source dd:ic_antennas/antenna/power_launched -> 'attached'
  target at name_stage 'accepted'
```

It was attached because it is the total ICRF power over one antenna's straps
launched into the vacuum vessel, which is the net antenna-boundary power after
reflection rather than a system-level total. The word "Total" in its source
documentation aggregates straps within that antenna.

The summary path independently produced the same one-source dry run:

```text
would attach summary/heating_current_drive/ic/power/value to
net_power_due_to_ion_cyclotron_heating
  source dd:summary/heating_current_drive/ic/power/value -> 'attached'
  target at name_stage 'accepted'
```

The Data Dictionary defines that value as IC resonance-heating power coupled
to the plasma from a specific launcher. In this context, "coupled" is not a
second quantity: it is the net forward-minus-reflected power accepted across
the launcher/plasma load boundary. The name's documentation draws the same
boundary by excluding the power ultimately absorbed by plasma particles.
Both sources are therefore launcher-resolved realizations of the same net
boundary power; neither realizes the system-level total.

The mechanical dimensionality, locus, and vector-family guard accepted both
pairings. After applying the two guarded attachments, a bounded read returned:

| Source | source status | scalar `produced_sn_id` | producer-edge targets |
|---|---|---|---|
| `dd:ic_antennas/antenna/power_launched` | `attached` | `net_power_due_to_ion_cyclotron_heating` | exactly the net identity |
| `dd:summary/heating_current_drive/ic/power/value` | `attached` | `net_power_due_to_ion_cyclotron_heating` | exactly the net identity |

Both DD nodes also carry the matching `HAS_STANDARD_NAME` projection. The net
identity now has exactly two producers and these two `source_paths`. The
superseded `power_due_to_ion_cyclotron_heating` has zero producers and an empty
`source_paths` list, so the recovery introduced no co-binding. Its successor
reads `name_stage=accepted`, `validation_status=valid`, catalog `status=draft`,
and `docs_stage=pending` immediately after attachment.

The rename operation dropping both bindings remains a defect in the rename
path. `sn attach` repaired the affected live state through the ordinary
composition-equivalent writer; it does not fix that underlying path.
