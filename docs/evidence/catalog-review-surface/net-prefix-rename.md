# Net boundary-power rename receipt

Date: 2026-09-08

## Outcome

The first `sn edit --rename` dry run refused the proposal before any graph
write. The default edit scope is `subtree` because
`power_due_to_ion_cyclotron_heating` is the parent of two live
`HAS_PARENT` children. The cascade reaches identities beyond the requested
identity and its own bindings, so the node stopped as required. The lead then
authorized `--scope self` and explicitly prohibited `--override-edits` or
either descendant rename. The self-scoped dry run planned exactly one root
rename and exited 0.

The physics-grounded reason passed to the command was:

> The quantity is net boundary power, equal to forward power minus reflected
> power at the antenna boundary; the unqualified name does not distinguish it
> from the forward, reflected, and total variants beside it.

The command exited 2 with this refusal:

```text
cascade plan conflict: 'total_power_due_to_ion_cyclotron_heating' has
origin='catalog_edit'; pass override_edits=True to rename anyway
```

The authorized command then created
`net_power_due_to_ion_cyclotron_heating` as a successor and entered
`review_name` under run `sn-edit-20260908T113856Z`. The ordinary reviewer
scored it `0.6875`, below the acceptance threshold, and exhausted the edit
rather than accepting it. The actual review spend was `$0.7850` against the
`$1.00` ceiling; the `$0.2150` remainder went unspent because the single
scoped review had reached a terminal refusal. No approval command ran and no
catalog status became `active`.

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

Had the default subtree plan been admissible, the ordinary path would mint the root successor
with `REFINED_FROM` lineage, retarget the two root `PRODUCED_NAME` bindings
and their scalar and `HAS_STANDARD_NAME` mirrors, and move both incoming
`HAS_PARENT` edges to the successor. On root acceptance, the same cascade
would rename the launched-power descendant and refresh its one producing
source mirror; it refuses before doing so because the total-power descendant is
protected. The authorized self-scoped command did not rename either descendant
and did not override the protected catalog wording.

## Applied state and read-back

A bounded five-id read after the self-scoped review confirmed:

| Identity | Exists | `name_stage` | `validation_status` | catalog `status` |
|---|---:|---|---|---|
| `power_due_to_ion_cyclotron_heating` | yes | `superseded` | `valid` | `superseded` |
| `net_power_due_to_ion_cyclotron_heating` | yes | `exhausted` | `valid` | `draft` |
| `launched_power_due_to_ion_cyclotron_heating` | yes | `drafted` | `valid` | `draft` |
| `launched_net_power_due_to_ion_cyclotron_heating` | no | — | — | — |
| `total_power_due_to_ion_cyclotron_heating` | yes | `accepted` | `valid` | `draft` |

The successor has `edit_scope=only_self`, `edit_status=exhausted`, and one
`REFINED_FROM` edge to the predecessor. Both original root sources now have
`PRODUCED_NAME` edges and `produced_sn_id` mirrors naming the successor:
`dd:ic_antennas/antenna/power_launched` remains `attached`, and
`dd:summary/heating_current_drive/ic/power/value` remains `composed`.

Because the review refused acceptance, the two descendant `HAS_PARENT` edges
remain on the superseded predecessor. Their identities and producing sources
are unchanged. In particular, the protected total name remains accepted with
catalog `status=draft`. The new root is valid but exhausted and catalog
`status=draft`, so it is explicitly not approved and not active.

The reviewer found the requested name grammatical but scored down its missing
antenna or launcher locus and its proximity to the launched and total
identities. That refusal was preserved as the result; no threshold was lowered
and no second review was issued.

All live reads ran on the login node because the graph tunnel is
login-node-local. They were bounded to the five identities above; the three
query durations were 0.674 seconds or less, below the ten-second ceiling.

## Reviewer-owned open question

After this rename, `total_power_due_to_ion_cyclotron_heating` describes a sum
of net powers while its spelling does not visibly name the summand as net.
Whether that total variant should also carry `net` is a physics-naming
judgement for the reviewer. This node records the question and takes no action:
the human-curated total identity was neither renamed nor overridden.

## Withdrawn premise

The earlier claim that
`launched_power_of_ion_cyclotron_heating_antenna` was an unresolved
documentation link was withdrawn. In-document name links resolve against the
full graph, where that identity exists, rather than against one deployment
batch. No defect was investigated or recorded for that link.

## Verification

The full Standard Names test gate is recorded below once complete.
