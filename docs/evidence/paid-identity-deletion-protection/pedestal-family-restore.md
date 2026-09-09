# Pedestal-top density family restore

## Outcome

**Blocked with zero identities restored.** The exact 21-source pipeline scope
was valid, but its first persistence step repeatedly refused its own
skeleton-placeholder cleanup because the candidate identities carry recorded
LLM spend. The run was stopped after the same deterministic refusal recurred;
continuing would only repeat free composition calls without committing a name.

The graph postcondition remains:

- **0 of 19** intended DD-backed children are live;
- **0 of 21** intended DD producer edges are live;
- `density_at_pedestal_top` is absent and has **0 of 19** required incoming
  `HAS_PARENT` edges;
- no `origin` value was written on the parent; and
- **$0.000000** was spent in 9 recorded `generate_name` calls against the
  authorised **$100.00** restore ceiling.

This is a hard implementation blocker, not a provider, credential, source-data,
or budget failure.

## Authority and exact cohort

The live `paid-identity-deletion-protection` plan was read at version 22 before
execution. Its archive census identifies exactly nineteen direct children of
`density_at_pedestal_top`. The apparent twenty-name discrepancy in prose is
resolved by the archive topology: `total_ion_density_at_pedestal_top` is a
direct-DD identity in the broader incident set, but it is not one of these
nineteen archived incoming `HAS_PARENT` children.

The nineteen children map to twenty-one surviving DD sources because
`electron_density_at_pedestal_top` has three direct DD bindings:

| Intended child | Exact DD producer path | Archive name stage | Pre-run live source |
|---|---|---|---|
| `argon_density_at_pedestal_top` | `summary/local/pedestal/n_i/argon/value` | accepted | extracted, unbound |
| `beryllium_density_at_pedestal_top` | `summary/local/pedestal/n_i/beryllium/value` | accepted | extracted, unbound |
| `boron_density_at_pedestal_top` | `summary/local/pedestal/n_i/boron/value` | accepted | extracted, unbound |
| `carbon_density_at_pedestal_top` | `summary/local/pedestal/n_i/carbon/value` | accepted | extracted, unbound |
| `deuterium_density_at_pedestal_top` | `summary/local/pedestal/n_i/deuterium/value` | accepted | extracted, unbound |
| `deuterium_tritium_density_at_pedestal_top` | `summary/local/pedestal/n_i/deuterium_tritium/value` | accepted | extracted, unbound |
| `electron_density_at_pedestal_top` | `summary/local/pedestal/n_e/value` | reviewed | extracted, unbound |
| `electron_density_at_pedestal_top` | `summary/pedestal_fits/linear/n_e/pedestal_height/value` | reviewed | extracted, unbound |
| `electron_density_at_pedestal_top` | `summary/pedestal_fits/mtanh/n_e/pedestal_height/value` | reviewed | extracted, unbound |
| `helium_3_density_at_pedestal_top` | `summary/local/pedestal/n_i/helium_3/value` | accepted | extracted, unbound |
| `helium_4_density_at_pedestal_top` | `summary/local/pedestal/n_i/helium_4/value` | accepted | extracted, unbound |
| `hydrogen_density_at_pedestal_top` | `summary/local/pedestal/n_i/hydrogen/value` | accepted | extracted, unbound |
| `iron_density_at_pedestal_top` | `summary/local/pedestal/n_i/iron/value` | accepted | extracted, unbound |
| `krypton_density_at_pedestal_top` | `summary/local/pedestal/n_i/krypton/value` | accepted | extracted, unbound |
| `lithium_density_at_pedestal_top` | `summary/local/pedestal/n_i/lithium/value` | accepted | extracted, unbound |
| `neon_density_at_pedestal_top` | `summary/local/pedestal/n_i/neon/value` | accepted | extracted, unbound |
| `nitrogen_density_at_pedestal_top` | `summary/local/pedestal/n_i/nitrogen/value` | accepted | extracted, unbound |
| `oxygen_density_at_pedestal_top` | `summary/local/pedestal/n_i/oxygen/value` | accepted | extracted, unbound |
| `tritium_density_at_pedestal_top` | `summary/local/pedestal/n_i/tritium/value` | accepted | extracted, unbound |
| `tungsten_density_at_pedestal_top` | `summary/local/pedestal/n_i/tungsten/value` | accepted | extracted, unbound |
| `xenon_density_at_pedestal_top` | `summary/local/pedestal/n_i/xenon/value` | accepted | extracted, unbound |

All 21 source nodes were found under their schema-owned `id='dd:' + path` key.
Each was `status='extracted'`, had no active claim, had no live
`PRODUCED_NAME` target, and retained its `FROM_DD_PATH` edge to the identically
keyed `IMASNode`. All nineteen child identities and the parent were absent.

## Preflight and bounded live attempt

The gap-only dry run used the exact 21 paths with
`--skip-global-maintenance`; it exited 0 and reported 21 focused paths with no
graph writes. Neither `--reseed` nor `--force` was used.

The live command used the same exact focus set with a 30-minute wall-clock
limit and a $99.69 command budget, preserving headroom under the authorised
$100 restore ceiling after the earlier etendue work. It created scope id
`72a13df1-c940-4f4b-90fe-09a5c85c3c37` and accounting run
`0c979912-b685-4057-b254-057ecbd1a142`.

The first persistence attempts failed in
`write_standard_names` before any name was committed. Representative refusals
were:

- `helium_4_density_at_pedestal_top`, whose recorded prior spend is
  $0.255943;
- `tungsten_density_at_pedestal_top`, whose recorded prior spend is
  $0.383553; and
- the remaining 17 children plus the absent parent in the final batch, each
  refused for the same positive-spend protection.

The final batch refusal names every intended child except the two already
refused separately and also names `density_at_pedestal_top`. The pool recorded
9 errors, processed 0 names, and stabilized at 19 pending source rows. After a
bounded positive check still returned zero live target identities, the run was
interrupted rather than allowed to retry until its 30-minute deadline.

## Root cause and required repair

`write_standard_names` obtains `skeleton_candidate_ids` from relationship-side
materialisation. It passes that whole set to
`refuse_protected_automatic_deletion` before its deletion query applies the
positive skeleton predicates. Those predicates require, among other things,
that a deletable node have no creation or generation timestamp, no validated
identity fields, no DD binding, no `PRODUCED_NAME` edge, and no incoming
structural edge.

The guard therefore evaluates **relationship candidates**, not the narrower
set the following query can actually delete. A paid, fully composed identity is
correctly protected by the guard but is not a deletable skeleton. Passing it to
the guard anyway turns the protection into a refusal of ordinary composition.
That is exactly what happened here: the protection fired, but it was aimed one
selection boundary too early.

The required repair is outside this node's exclusive write scope. A corrective
node must first select the exact rows satisfying the existing skeleton deletion
predicate, then pass only those positively proven placeholder ids to the paid
deletion refusal and deletion statement in the same transaction. Focused tests
must prove both halves: a paid real identity may be composed without entering
the delete candidate set, while a paid id-only placeholder remains undeletable.

## Post-run graph state and relationship comparison

The interrupted run left all 21 sources `extracted`, unclaimed and unbound. It
stamped their exact ephemeral scope id, but created no target node and no
producer edge. The authoritative accounting row reports `stop_reason=interrupted`,
`names_composed=0`, `names_reviewed=0`, `cost_limit=99.69`, and
`cost_spent=0.0`. The nine `LLMCost` rows for this scope are all
`phase='generate_name'`, model `hosted_vllm/deepseek-v4-flash`, and sum to
**$0.000000**.

Because none of the twenty target identities exists live, every live incident
relationship count is zero and the requested successful live-versus-archive
comparison cannot be produced. The lineage artifact already establishes these
archive-side positive controls:

| Identity scope | Archive relationship type | Archive count | Live count after attempt | Result |
|---|---|---:|---:|---|
| parent | incoming `HAS_PARENT` from the intended children | 19 | 0 | restore absent |
| parent | outgoing `HAS_PARENT` | 1 | 0 | restore absent |
| parent | `PRODUCED_NAME` from `derived:density_at_pedestal_top` | 1 | 0 | restore absent |
| parent | `HAS_REVIEW` | 30 | 0 | restore absent |
| each non-electron child | `PRODUCED_NAME` from its direct DD source | 1 | 0 | restore absent |
| electron child | `PRODUCED_NAME` from its direct DD sources | 3 | 0 | restore absent |
| each child | `PRODUCED_NAME` from its archived structural `derived:` source | 1 | 0 | restore absent |
| each child | outgoing `HAS_PARENT` to `density_at_pedestal_top` | 1 | 0 | restore absent |

The prior lineage artifact is not a full incident-edge export: it does not
enumerate every archived grammar, unit, COCOS, domain, cluster, docs-revision,
change-history or structural-authority relationship. Therefore naming
archive-absent relationship types or claiming per-type equality would be false.
That comparison remains gated behind a successful restore and the isolated
archive inventory required by the live plan.

## Recovery point

No name or edge needs undoing. After the skeleton-candidate boundary is
repaired and tested, re-run the same gap-only exact-source command: the 21
sources are still `extracted` and unbound, so they remain the correct substrate.
The restore node must then verify all nineteen children at `origin='pipeline'`,
`status='draft'`, each with its own DD producer, before accepting the structural
tail's parent; it must state the parent origin actually written together with
the producer topology that supports that value. Only then can the exact
archive/live per-relationship-type comparison be completed.
