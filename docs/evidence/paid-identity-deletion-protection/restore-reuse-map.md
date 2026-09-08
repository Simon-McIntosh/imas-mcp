# Restore reuse map

## Authority and scope

This map was derived from the live plan at version 9 and the source tree at
`271536947e417ddb36a3091c1559fc606f62c475`. It identifies the existing
entry points that recovery work must use and the places where no existing
entry point satisfies the locked restore contract. It performs no graph write
and does not claim that any identity has been restored.

The governing recovery census is
`docs/evidence/sn-lifecycle-integrity/unshielded-identity-deletions.md`: 67
`RESTORE`, 11 `CORRECTLY REMOVED`, and 0 `UNDETERMINED`. Its ordered recovery
routes are the 28 directly DD-backed identities, the pedestal-top density
family with children before parent, selective archive reconstruction, and the
`etendue_of_spectrometer_channel` parent followed by its ratified spectral
child. Every restored `StandardName` must finish with `origin='pipeline'`.

## Primitive entry-point verdicts

| Required operation | Existing entry point | Verdict | Reason and boundary |
|---|---|---|---|
| Return a terminal or attempt-capped `StandardNameSource` to `extracted` | CLI `sn_retry` at `imas_codex/cli/sn.py:6956`, selecting `retry_failed_sources` at `imas_codex/cli/sn.py:6979`; implementation `imas_codex/standard_names/graph_ops.py:11187` | **REUSE** | This is the sanctioned path for both `status='failed'` and `status='extracted'` at the attempt cap (`imas_codex/standard_names/graph_ops.py:11194`). It writes a `StandardNameSourceRetry` event before clearing the counter and uses status plus attempt count as a compare-and-set fence (`imas_codex/standard_names/graph_ops.py:11226`, `imas_codex/standard_names/graph_ops.py:11276`). Use `imas-codex sn retry --failed <exact-path> --reason <reason>` first as a dry run and then live. Do not use `reset_standard_name_sources` for an already-unbound deleted target: that operation requires a non-empty expected binding set and scalar, while these recovery sources have no surviving `PRODUCED_NAME` edge. The 30 ordinary direct-DD sources are already eligible at `extracted` and need no reset; the reset entry point is required for the attempt-capped `soft_x_rays/channel/etendue` source. |
| Compose a missing name from an exact source or attach the source to a stable existing name | CLI `sn_run` at `imas_codex/cli/sn.py:1753`, exact-source focus routing at `imas_codex/cli/sn.py:2331`; pool processor `imas_codex/standard_names/workers.py:6554`; transactional compose writer `imas_codex/standard_names/graph_ops.py:7165`; claimed attachment writer `imas_codex/standard_names/graph_ops.py:10899` | **REUSE** | The public route is `imas-codex sn run --focus <dd-path> --skip-global-maintenance`, first with `--dry-run`, then with an explicit cost and time bound. Focus seeds/scopes exact DD paths and hands them to the normal pool orchestrator (`imas_codex/cli/sn.py:2408`, `imas_codex/cli/sn.py:2463`). `process_generate_name_batch` routes generated candidates through the exact token-and-sequence persistence transaction and attachment candidates through `persist_claimed_attachments`; neither direct `write_standard_names` calls nor hand-written Cypher carry the claim fences. The compose transaction writes the name, `StandardNameSource` state, `PRODUCED_NAME`, and the backing `HAS_STANDARD_NAME` projection together (`imas_codex/standard_names/graph_ops.py:7184`). |
| Derive and materialise a missing structural parent from its children | Normal compose persistence reaches `write_standard_names` at `imas_codex/standard_names/graph_ops.py:5165`; its tail calls `_write_standard_name_edges` at `imas_codex/standard_names/graph_ops.py:2949`, which calls `_bootstrap_missing_derived_parent_targets` at `imas_codex/standard_names/graph_ops.py:2836` | **CANNOT REUSE UNCHANGED** | The topology and admission machinery must be reused: it derives current `HAS_PARENT` edges from the ISN grammar, expands the closure, requires the exact live child set, inherits child authority, and validates the parent before creating an absent endpoint (`imas_codex/standard_names/graph_ops.py:2997`, `imas_codex/standard_names/graph_ops.py:3071`, `imas_codex/standard_names/graph_ops.py:3100`). However, its canonical materializer `_materialize_derived_parent_rows` (`imas_codex/standard_names/graph_ops.py:3839`) unconditionally writes `parent.origin='derived'` (`imas_codex/standard_names/graph_ops.py:4041`), contradicting the locked `origin='pipeline'` decision for every restored identity. `seed_parent_sources` at `imas_codex/standard_names/graph_ops.py:4467` is not an alternative for the deleted density parent because it selects an already-existing parent whose `name_stage IS NULL`; it cannot bootstrap an absent node. Recovery must extend the existing bootstrap/materializer boundary so the restore-authorised parent uses the same derivation and admission checks while atomically receiving `origin='pipeline'`. It must not duplicate the grammar or parent-admission logic. |
| Write `origin` on a `StandardName` with signed, exact authority | Generic `apply_signed_manifest` at `imas_codex/standard_names/signed_manifest.py:6821`; the source-backed maintenance writer `reconcile_reviewable_name_stage` writes `pipeline` at `imas_codex/standard_names/graph_ops.py:13755` | **CANNOT REUSE AS THE COMPLETE RESTORE WRITER** | `apply_signed_manifest` can safely apply a `set_properties` mutation to an existing, signed participant, so it is the reusable property-update boundary for a name that already exists. It cannot create an absent name: preflight refuses a missing participant at `imas_codex/standard_names/signed_manifest.py:3445`. `reconcile_reviewable_name_stage` is narrower still: it requires a live non-derived `PRODUCED_NAME` source and only matches names below `drafted` (`imas_codex/standard_names/graph_ops.py:13757`), so it cannot stamp source-less archive reconstructions and is not guaranteed to touch a newly composed name already at `drafted`. `reconcile_catalog_edit_origins` at `imas_codex/standard_names/graph_ops.py:13249` is explicitly wrong for this recovery: it only matches `origin='catalog_edit'` and assigns null to a source-less name (`imas_codex/standard_names/graph_ops.py:13276`, `imas_codex/standard_names/graph_ops.py:13282`). Because restoration and reclassification are one action, archive-created nodes and the restored structural parent must receive `origin='pipeline'` inside their creation transaction; a later bulk reconcile is forbidden. |
| Load the archived store into an isolated database for exact reads | `start_temp_neo4j` at `imas_codex/graph/temp_neo4j.py:207`, paired with `stop_temp_neo4j` at `imas_codex/graph/temp_neo4j.py:323` | **REUSE** | Stage the archive's single `graph.dump` as `<temp-dir>/dumps/neo4j.dump` on a debug-partition allocation, then call this loader with dedicated loopback Bolt and HTTP ports. It binds every database write path to the caller's temporary directory, loads with `neo4j-admin`, writes an auth-disabled temporary configuration, waits for a real Bolt exchange, and returns the process handle for guaranteed cleanup (`imas_codex/graph/temp_neo4j.py:217`, `imas_codex/graph/temp_neo4j.py:227`, `imas_codex/graph/temp_neo4j.py:251`, `imas_codex/graph/temp_neo4j.py:284`). Do **not** use CLI `graph_load` at `imas_codex/cli/graph/data.py:644`: it requires the target to be the currently active profile (`imas_codex/cli/graph/data.py:653`, `imas_codex/cli/graph/data.py:663`), stops/backs up that active database through `Neo4jOperation` (`imas_codex/cli/graph/data.py:726`), and therefore is the live-store replacement path, not an isolated evidence reader. |
| Recreate an absent archived `StandardName` together with every archived incident edge | No conforming entry point exists. The closest generic boundary is `apply_signed_manifest` at `imas_codex/standard_names/signed_manifest.py:6821` | **CANNOT REUSE** | The generic signed-manifest registry only admits node labels `StandardName` and `StandardNameSource` and relationship types `PRODUCED_NAME` and `HAS_PARENT` (`imas_codex/standard_names/signed_manifest.py:58`). It snapshots every participant and refuses if one does not exist (`imas_codex/standard_names/signed_manifest.py:3443`), so it cannot create the missing identity. It also cannot express the archive's reviews, docs revisions, internal changes, unit/COCOS/domain/cluster edges, grammar-role edges, structural authority, or other relationship roles. The archive route therefore needs a new closed reconstruction adapter inside the signed-manifest transaction boundary. That adapter must consume a signed exact-identity manifest produced from the isolated database, create each absent node with `origin='pipeline'`, recreate only allowlisted archived edges whose counterpart exists or is included, and compare per-type edge counts with the archive before commit. It must not add general arbitrary-Cypher or arbitrary-label authority. |

## Restore-route wiring

| Restore route | Required reuse sequence | Cannot-as-is seam and stop condition |
|---|---|---|
| Direct DD-backed identities | Confirm the exact 30 source rows are still unbound, unclaimed, `status='extracted'`, and below the attempt cap. Run the `sn run --focus ... --skip-global-maintenance` dry run, then the same bounded live focus cohort. Let `process_generate_name_batch` choose compose versus attachment and let `persist_generated_name_winners` or `persist_claimed_attachments` own the write. | The existing compose route does not provide a reliable atomic `origin='pipeline'` postcondition for every newly restored node. The live operation must stop unless its implementation adds that postcondition at the existing persistence transaction and verifies it for all 28 identities. |
| Pedestal-top density family | Feed the 19 DD-backed child paths through the same exact focus route. Parent derivation must occur only through the normal `write_standard_names` structural tail, so current grammar, admission, child-set, unit, domain, and relationship logic are reused. Verify all 19 intended children exist before accepting `density_at_pedestal_top`. | The current parent materializer stamps `derived`. Do not invoke `seed_parent_sources` or replay an archived parent first. Stop unless the restore-aware call into the canonical materializer atomically writes `pipeline` and the archive-to-live per-type count comparison passes for the parent and every child. |
| Source-less ratified and spend-protected identities | Load the archive only through `start_temp_neo4j` on a debug partition. Query the signed exact identity set and record node properties plus every incident edge with direction, type, properties, and counterpart identity. Feed that result to the new closed reconstruction adapter built inside `apply_signed_manifest`'s preview/hash/apply/receipt pattern. | There is no current reconstruction mutation. Do not substitute `graph_load`, catalog reconcile, `write_standard_names`, or free-form Cypher. Stop if an archived edge's counterpart cannot be proven live or included, if a relationship type is outside the explicit reconstruction registry, if a node already exists with non-identical state, or if any per-type archive/live count differs. |
| `etendue_of_spectrometer_channel` and its ratified spectral child | Dry-run and apply `sn retry --failed soft_x_rays/channel/etendue --reason <reason>` to clear the extracted-at-cap source, then run the exact focus route to recompose the DD-backed parent. Restore `spectral_etendue_of_spectrometer_channel` through the same signed archive reconstruction path, using the restored parent's structural relationship as part of the signed edge inventory. | `persist_claimed_attachments` cannot restore the spectral child because it attaches an exact claimed DD source to a stable target; the child has no direct DD source. Stop unless both names have `origin='pipeline'`, the parent retains the direct DD provenance, the child retains its ratification evidence, and both identities' per-type edge counts equal the isolated archive counts. |

## Invariants for the implementation nodes

- Exact cohorts only: every reset, focus run, archive query, signed manifest,
  receipt, and verification query names the identities or source IDs it owns.
- The isolated archive is read-only evidence. No load, switch, or replay targets
  the active graph profile or database name.
- Pipeline source claims remain authoritative. Composition and attachment use
  the existing token-and-sequence fenced writers; recovery code does not write
  their edges or scalar mirrors independently.
- Structural derivation remains grammar-owned. Recovery supplies authority and
  the required provenance value; it does not hand-author `HAS_PARENT` topology.
- Every restored node is created or completed with `origin='pipeline'` in the
  same restore action. No archived `catalog_edit` or materializer-produced
  `derived` value is allowed to survive.
- Archive reconstruction is all-or-nothing for its signed cohort. A name-only
  success is a failure. The quantitative gate is equality of archived and live
  incident-edge counts per relationship type for every identity, including
  explicit zeroes for roles absent from the archive.
- The ordinary source-backed and special etendue runs report actual spend next
  to the authorised 100 USD ceiling. The archive route spends no LLM budget.

## Headline result

The reuse map identifies six existing boundaries that later work must build
on: `sn retry --failed`, exact `sn run --focus`, the transactional compose and
attachment writers, the canonical structural derivation/bootstrap chain, the
signed-manifest preview/hash/apply transaction, and the temporary Neo4j loader.
Three operations are directly reusable: attempt-cap reset, governed
compose/attachment, and isolated archive loading. Structural materialisation is
reusable only below its origin assignment, and generic signed repair is reusable
only as the transaction envelope: neither currently satisfies the locked
creation-plus-`pipeline` contract. No existing entry point can reconstruct the
full archived subgraph, so that narrowly typed adapter is the only new machinery
the restore implementation is justified in adding.
