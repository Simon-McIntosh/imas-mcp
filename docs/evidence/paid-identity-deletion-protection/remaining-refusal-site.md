# Remaining global-maintenance refusal site

## Diagnosis before implementation

The remaining routine global-maintenance failure reaches
`refuse_protected_automatic_deletion` at
`imas_codex/standard_names/graph_ops.py:3675`, inside
`_delete_derived_parent_nodes`. The guard is still the correct deletion-time
backstop. The defect is the childless-derived-parent selector feeding it at
`graph_ops.py:4627-4639`: that selector sends every childless derived identity
to the deleter, including identities carrying durable authority.

The earlier selector correction did reach the same deleter, but through a
different feed. `_query_derived_parents_for_admission_cleanup` filters its
candidate ids through `filter_automatic_deletion_candidates` at
`graph_ops.py:3646`. `normalize_derived_parent_lifecycle` then assembles a
second candidate set for childless derived parents inline and passes that set
to `_delete_derived_parent_nodes` without the equivalent selection-time
exclusion. One protected row is therefore enough to turn an otherwise routine
"nothing to reap" result into a whole-maintenance failure.

```text
run_sn_pools global maintenance
  -> normalize_derived_parent_lifecycle
     -> childless derived-parent query              unfiltered
        -> _delete_derived_parent_nodes
           -> refuse_protected_automatic_deletion   correct backstop, fires
```

## Three-call-site census

| Guard call | Candidate origin | Global-maintenance disposition |
|---|---|---|
| `graph_ops.py:3675`, `_delete_derived_parent_nodes` | Admission-recheck candidates and childless derived parents | **Offending site.** Admission-recheck candidates are already filtered, but the childless query at `graph_ops.py:4627-4639` bypasses that exclusion and reaches this shared deleter. |
| `graph_ops.py:5774`, `write_standard_names` skeleton cleanup | Relationship endpoints narrowed by `_query_skeleton_placeholders_for_cleanup` and the positive id-only placeholder predicate | Exempt from this routine maintenance failure. This site is reached by composition, not the maintenance-only path, and the narrowed set contains actual placeholders rather than every relationship endpoint. Its immediate refusal remains the required backstop for a protected placeholder. |
| `provenance_lifecycle.py:1784`, `retire_unrecoverable_provenance_orphans` | Explicit, reviewed list of source-less identities | Exempt from routine global maintenance. Production reaches it only from `rebuild_provenance` when the operator opts into `retire_unresolved=True`; that flag defaults to false, and `run_sn_pools` does not call `rebuild_provenance`. The list-scoped refusal must remain. |

The third site was the initial expectation, but the reproduction does not
reach it. The observed traceback names the first site and, more specifically,
the unfiltered childless candidate feed into that site.

## Pre-change reproduction

Base revision: `d26e6f2f2afb923792125a31d1ba2140999d2915`

Command, run on the login node because the live Neo4j endpoint is available
only through its login-local tunnel:

```bash
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv \
PYTHONPATH=<assigned-worktree> \
/home/ITER/mcintos/.local/bin/uv run --no-sync \
  imas-codex sn run --only reconcile --quiet
```

Result: **exit 1**. The command entered the real global-maintenance sequence,
performed no LLM work, and failed at the structural-parent normalization
stage. The durable application log is
`/home/ITER/mcintos/.local/share/imas-codex/logs/sn_sn-compose.log`, with the
reproduction beginning near line 73215 at 2026-09-09 13:09:57 local time.

The traceback route was:

```text
loop.py:2019   run_sn_pools
loop.py:1451   _global_maintenance_call
graph_ops.py:4639 normalize_derived_parent_lifecycle
graph_ops.py:3675 _delete_derived_parent_nodes
protection.py:164 refuse_protected_automatic_deletion
```

The refused candidate set contained six protected childless identities:

| Identity | Durable authority reported by the guard |
|---|---:|
| `flux_at_first_wall` | recorded spend USD 0.001137 |
| `flux_at_wall_due_to_eddy_current` | recorded spend USD 0.450517 |
| `flux_at_wall_due_to_pumping` | recorded spend USD 0.008069 |
| `ion_diffusivity` | catalog commit plus recorded spend USD 1.672799 |
| `permeability_of_ferritic_element` | recorded spend USD 0.078877 |
| `width_of_spectrometer_channel` | recorded spend USD 0.088525 |

The refusal is correct for a batch handed directly to a deleter. The routine
selector is wrong for proposing these identities for deletion at all.

## Post-change reproduction

Pending implementation. Acceptance requires the same maintenance-only command
to exit zero while the deletion-time refusal remains demonstrably active when
a protected identity is handed to `_delete_derived_parent_nodes` directly.
