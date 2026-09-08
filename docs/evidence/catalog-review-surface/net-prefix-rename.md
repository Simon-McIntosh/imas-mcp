# Net boundary-power rename dry-run receipt

Date: 2026-09-08

## Outcome

The required `sn edit --rename` dry run refused the proposal before any graph
write. The default edit scope is `subtree` because
`power_due_to_ion_cyclotron_heating` is the parent of two live
`HAS_PARENT` children. The cascade reaches identities beyond the requested
identity and its own bindings, so the node stopped as required and did not pass
`--override-edits`, force `--scope self`, apply the edit, approve a name, or
write catalog status `active`.

The physics-grounded reason passed to the command was:

> The quantity is net boundary power, equal to forward power minus reflected
> power at the antenna boundary; the unqualified name does not distinguish it
> from the forward, reflected, and total variants beside it.

The command exited 2 with this refusal:

```text
cascade plan conflict: 'total_power_due_to_ion_cyclotron_heating' has
origin='catalog_edit'; pass override_edits=True to rename anyway
```

## Bounded dry-run census

The underlying ordinary cascade planner was then read directly in dry-run mode
to expose the complete plan hidden behind the command's refusal panel. It
discovered exactly two descendants:

| Role | Current identity | Planned identity | Result |
|---|---|---|---|
| root | `power_due_to_ion_cyclotron_heating` | `net_power_due_to_ion_cyclotron_heating` | planned |
| descendant | `launched_power_due_to_ion_cyclotron_heating` | `launched_net_power_due_to_ion_cyclotron_heating` | planned |
| descendant | `total_power_due_to_ion_cyclotron_heating` | not materialized | conflict: protected `catalog_edit` origin |

The planner reported `total_descendants=2`, two rename rows including the
root, zero independent skips, and one conflict. Both proposed destination ids
were absent in the graph. Because cascade conflicts are all-or-nothing, the
plan wrote zero rows.

The direct topology and source-binding set that makes this a wider-than-root
cascade is:

- `launched_power_due_to_ion_cyclotron_heating`
  `-[:HAS_PARENT]-> power_due_to_ion_cyclotron_heating`, with
  `dd:summary/heating_current_drive/power_launched_ic/value
  -[:PRODUCED_NAME]-> launched_power_due_to_ion_cyclotron_heating`.
- `total_power_due_to_ion_cyclotron_heating
  -[:HAS_PARENT]-> power_due_to_ion_cyclotron_heating`, with
  `dd:ic_antennas/power_launched
  -[:PRODUCED_NAME]-> total_power_due_to_ion_cyclotron_heating` and
  `dd:summary/heating_current_drive/power_ic/value
  -[:PRODUCED_NAME]-> total_power_due_to_ion_cyclotron_heating`.
- The requested root itself has two producing bindings:
  `dd:ic_antennas/antenna/power_launched
  -[:PRODUCED_NAME]-> power_due_to_ion_cyclotron_heating` and
  `dd:summary/heating_current_drive/ic/power/value
  -[:PRODUCED_NAME]-> power_due_to_ion_cyclotron_heating`.

Had the plan been admissible, the ordinary path would mint the root successor
with `REFINED_FROM` lineage, retarget the two root `PRODUCED_NAME` bindings
and their scalar and `HAS_STANDARD_NAME` mirrors, and move both incoming
`HAS_PARENT` edges to the successor. On root acceptance, the same cascade
would rename the launched-power descendant and refresh its one producing
source mirror; it refuses before doing so because the total-power descendant is
protected. No such edge or identity change was applied.

## Read-back

A bounded five-id read after the refused dry run confirmed:

| Identity | Exists | `name_stage` | `validation_status` | catalog `status` |
|---|---:|---|---|---|
| `power_due_to_ion_cyclotron_heating` | yes | `accepted` | `valid` | `draft` |
| `net_power_due_to_ion_cyclotron_heating` | no | — | — | — |
| `launched_power_due_to_ion_cyclotron_heating` | yes | `drafted` | `valid` | `draft` |
| `launched_net_power_due_to_ion_cyclotron_heating` | no | — | — | — |
| `total_power_due_to_ion_cyclotron_heating` | yes | `accepted` | `valid` | `draft` |

Thus the graph remains unchanged and no involved identity is catalog-active.
The requested new identity cannot yet be read back because it was correctly not
minted after the wider cascade was found.

All live reads ran on the login node because the graph tunnel is
login-node-local. They were bounded to the five identities above; the three
query durations were 0.674 seconds or less, below the ten-second ceiling.

## Withdrawn premise

The earlier claim that
`launched_power_of_ion_cyclotron_heating_antenna` was an unresolved
documentation link was withdrawn. In-document name links resolve against the
full graph, where that identity exists, rather than against one deployment
batch. No defect was investigated or recorded for that link.

## Verification disposition

The full Standard Names suite was not started. This receipt changes no product
or test code, and the exact execution contract requires stopping before apply
when the cascade reaches another identity. A suite result could not turn this
blocked graph mutation into an admissible one.

