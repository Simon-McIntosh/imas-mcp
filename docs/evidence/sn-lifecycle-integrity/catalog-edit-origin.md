# Catalog-edit origin is earned

Measured 2026-09-08 against the live Standard Names graph. The graph carried
`origin='catalog_edit'` on 2,096 identities, but the population had zero
`catalog_pr_number`, zero `catalog_approved_at`, and zero `name_stage='approved'`.
There has never been a catalog review or merged catalog request in this graph,
so none of those 2,096 identity-level markers represented a legitimate catalog
edit. The marker was protecting identities on an authority that did not exist.

## What wrote the marker

The current Standard Names package does not write `StandardName.origin` as
`catalog_edit`. The three catalog-edit writers in `edit.py` write that value on
the `StandardNameChange` ledger event instead:

- `edit.py:1191-1202` records `reclassify_kind` with
  `change.origin='catalog_edit'`;
- `edit.py:1457-1468` records `fold_identity` with the same event origin;
- `edit.py:2478-2492` builds the same fold receipt in Python before it is
  persisted.

`graph_ops.py` has the same event-level convention for internal repairs, such
as domain and grammar-segment realignment. None of these statements assigns
`sn.origin='catalog_edit'`.

The historical cause is documented by
`standard_names/provenance_rebuild.py:3-5`: an export/import round trip strips
pipeline provenance and re-enters names with `origin='catalog_edit'` and no
`StandardNameSource`. That is the explanation for the residue, not evidence
of a catalog request. The catalog checker itself is explicitly read-only in
`standard_names/catalog_import.py:1-18`; it does not recreate names or rebuild
provenance. The 2,096 identity markers are therefore historical import residue
from a path that is no longer an active writer.

The live ledger corroborates the distinction. Only 29 of the 2,096 affected
identities had an attached `StandardNameChange` whose event origin was
`catalog_edit`; there were 29 such attached event rows, and they were internal
operations rather than catalog approval records.

The operation census over all `catalog_edit`-origin ledger events is broader
than the affected-identity census because it includes events attached to
other identities; it returned 47 `reclassify_domain`, 41 `fold_identity`, 1
`compact`, and 1 `reclassify_kind`. Neither census contains a catalog PR or
approval event.

## Repair

Commit `c201f4449` adds
`reconcile_catalog_edit_origins()` to `imas_codex/standard_names/graph_ops.py`.
It is a graph-backed, idempotent repair route, not an ad-hoc Cypher mutation.
It selects only `catalog_edit` identities with no catalog request number, no
approval timestamp, and no approved lifecycle stage. It then recovers the
strongest surviving provenance:

| surviving evidence | replacement | rows moved |
| --- | --- | ---: |
| DD or signal source, with no derived source | `pipeline` | 1,179 |
| derived source present, including mixed DD/derived bindings | `derived` | 246 |
| no DD, signal, or derived source evidence | origin property removed (`null`) | 671 |
| **total** |  | **2,096** |

The dry run returned `eligible=2096, pipeline=1179, derived=246, unset=671`.
The apply returned `changed=2096` and recorded one
`lineage_reconciliation` `StandardNameChange` event per identity. The
cascade protection guard was not widened or weakened.

The post-apply live census is:

| identity origin | count |
| --- | ---: |
| `pipeline` | 2,657 |
| `derived` | 516 |
| origin absent | 1,955 |
| `catalog_edit` | **0** |

The post-apply query also returned zero rows with catalog-edit origin carrying
any of the three authority markers, and a second invocation returned
`eligible=0, pipeline=0, derived=0, unset=0, changed=0`. The repair is therefore
idempotent and leaves future, genuinely approved catalog edits protected.

Representative repaired identities and their remaining evidence:

| name | description | replacement | source binding |
| --- | --- | --- | --- |
| `absorbed_radiated_power_at_divertor_target` | Net photon power deposited on one divertor target surface after subtracting radiative power reflected from that surface. | `pipeline` | `dd:divertors/divertor/target/power_radiated` |
| `absorbed_radiated_power_of_breeder_blanket_module` | Net photon power retained by a breeding blanket module after reflection of incident plasma radiation at its plasma-facing surface. | `pipeline` | `dd:breeding_blanket/module/time_slice/power_thermal_radiated` |
| `absorbed_power_of_beam_tracing_beam` | Absorbed power of a beam-tracing beam is the total electromagnetic wave power transferred from one coherent beam to the plasma along its trajectory. | `derived` | `derived:absorbed_power_of_beam_tracing_beam` |
| `absorbed_power_at_inside_flux_surface` | Total absorbed wave power inside a flux surface, cumulative volume integral of power density. | origin absent | no surviving source binding |

## Validation

The focused gate covering the changed provenance and graph operation surfaces
passed `59 passed, 2 deselected` in `6.65s`:

```text
uv run --no-sync pytest -p no:cacheprovider \
  tests/standard_names/test_provenance_ledger.py \
  tests/standard_names/test_graph_ops.py
```

The full `tests/standard_names` gate then passed `7267 passed, 11 skipped,
323 deselected, 34 warnings` in `292.86s` on the `all_debug` partition, with
exit status 0. No failure ids were emitted and no failure was attributable to
either changed path.
