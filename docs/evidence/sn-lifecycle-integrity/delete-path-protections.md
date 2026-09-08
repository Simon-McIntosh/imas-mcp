# Automatic deletion protection receipt

Date: 2026-09-08

## Result

Automatic standard-name deletion now refuses a complete candidate batch when
any identity carries publication authority or any non-zero recorded LLM spend.
Structural cleanup additionally refuses more than 80 candidates and can delete
only a node that positively carries `needs_composition=true`, the marker that
it is still a placeholder. Failure of the parent-admission test is no longer
sufficient authority to delete a materialized physics identity.

The ceiling is derived from the observed daily reaper totals: 74 was the
smallest normal daily total, while the destructive pass removed 93 identities
in eight seconds. A ceiling of 80 admits the measured low-volume day and
refuses that pass in full. It does not delete an initial subset.

## Protection evidence

`automatic_deletion_protections` reads seven publication signals independently
of `origin`: `name_stage=approved`, `catalog_pr_number`,
`catalog_merge_commit_sha`, `catalog_commit_sha`, `exported_at`,
`unchanged_ratification`, and `content_edit`. It returns every reason per
identity. `refuse_protected_automatic_deletion` passes that explicit identity
set through `filter_protected`, then raises one refusal naming every protected
identity and its evidence.

Positive LLM spend is an eighth signal with no monetary floor. One scan per
graph-client pass materializes
`(LLMCost)-[:FOR_STANDARD_NAME]->(StandardName)` from the retained
`standard_name_ids` lists for rows with `llm_cost > 0`. Every subsequent
protection check uses the edge and reports the summed spend as
`recorded_llm_spend_usd=<amount>`. Live materialization created or matched
32,203 edges in 1.29 seconds. An absent identity cannot receive an edge;
therefore a later restore recreating `etendue_of_spectrometer_channel` will be
followed by the same pass materializing its retained 0.84 USD history before
any automatic deletion decision.

The measured destroyed cohort shows why this signal is independent: spend
protects 75 of the 93 identities removed in the destructive pass, including
nine the provenance classification misses, while 18 have no spend. Across the
491 still-absent identities, 1,894 cost rows record 100.47 USD; the single pass
destroyed identities carrying 785 rows and 43.12 USD across all seven pipeline
phases.

## Ledgered deletion census

There are nine relevant `deletion_change_cypher` surfaces when the shared
builder is counted with its eight uses:

| Surface | Invocation | Automatic protection |
|---|---|---|
| `deletion_change_cypher` | shared receipt builder | every use gets recovery material |
| `cancel_staged_rename` | operator requests one named cancellation | operator authority |
| `retire_unrecoverable_provenance_orphans` | provenance rebuild retires an exact reviewed list | protected batch refusal added |
| `compact_unapproved_superseded` | operator invokes explicit compaction | operator authority |
| `_delete_derived_parent_nodes` | lifecycle maintenance cleanup | protected batch refusal, 80 ceiling, placeholder proof |
| `write_standard_names` skeleton sweep | automatic post-write cleanup | protected batch refusal added |
| `clear_standard_names`, source-scoped branch | operator invokes selected clear | operator authority |
| `clear_standard_names`, name-scoped branch | operator invokes selected clear | operator authority |
| `clear_sn_subsystem` | operator invokes destructive full clear | operator authority |

The three automatic routes use the common refusal. The five operator routes
retain their explicit human scope rather than being recast as maintenance.

## Recoverability

Every ledgered deletion now creates an adjacent
`StandardNameDeletionSnapshot` before deleting the name. `SET snapshot =
properties(name)` preserves the original typed property values. One
`StandardNameDeletedEdge` is created for every relationship, carrying its type,
direction, neighbor identity and labels, and relationship properties. The
change links to the node snapshot with `HAS_DELETION_SNAPSHOT`; the node
snapshot links to its edge inventory with `HAS_EDGE_SNAPSHOT`. This is graph
recovery material, not merely evidence that deletion occurred.

The cost-edge materialization, protection query, and shared deletion receipt
all passed live Neo4j `EXPLAIN`. No deleted identity was restored.

## Exposure measurement and gate state

The new focused contract passes:

```text
5 passed, 1 warning in 5.80s
```

Closing the guard exposed four existing fixture failures across a 47-test
cohort:

```text
4 failed, 43 passed, 1 warning in 10.83s
```

The failures are complete and attributable:

- `test_pending_single_child_shadow_is_retired_with_deletion_ledger`
- `test_derived_parent_delete_records_change_atomically`
- `test_skeleton_delete_records_change_atomically`
- `test_retirement_is_list_scoped_and_ledgered_atomically`

Each fixture rejects or misinterprets the new one-time LLMCost edge
materialization query before reaching the deletion it intended to simulate.
They are legitimate exposure of a test graph that modelled the old fail-open
contract, not evidence that the production guard should be removed. Updating
those three existing test files is outside this node's exclusive write paths,
so the full suite was not run and the node stops with that exact follow-on.
