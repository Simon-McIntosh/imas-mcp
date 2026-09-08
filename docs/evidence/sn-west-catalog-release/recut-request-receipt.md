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

The reviewer or lead must decide how to resolve the exhausted rename proposal.
A later node may re-steer or otherwise adjudicate
`net_power_due_to_ion_cyclotron_heating` through the governed name-review
path. Only after it reaches an exportable state can request 18 be closed and a
new branch and fork review request be cut.
