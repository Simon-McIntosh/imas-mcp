# Pedestal-top density family restore

## Outcome

**Complete.** The ordinary gap-only pipeline restored all nineteen intended
pedestal-top density children from the exact twenty-one DD source paths. The
structural materializer derived `density_at_pedestal_top` from that family.
The final bounded live-graph gate measured:

- **19 of 19** intended child identities live;
- **21 of 21** exact DD sources joined by `PRODUCED_NAME` to their intended
  child;
- every child at `origin=pipeline`, `status=draft`, and
  `name_stage=accepted`;
- `density_at_pedestal_top` live at `origin=derived`, `status=draft`, and
  `name_stage=accepted`;
- **19 of 19** incoming `HAS_PARENT` edges on the parent; and
- the parent's only producer is
  `derived:density_at_pedestal_top` (`source_type=derived`), with no direct DD
  producer.

The successful restore spent **$3.284246** against the authorised **$100.00**
ceiling: $2.976525 in the first clean drain and $0.307721 in the final drain.
The two pre-fix attempts and two intervening no-work/failure runs spent
$0.000000. Total attributable spend is therefore **3.284246%** of the ceiling,
leaving **$96.715754** unused. More of the ceiling was unnecessary because all
twenty-one sources converged without regeneration.

Eighteen children are also at `docs_stage=accepted`. A concurrent run stamped
`electron_density_at_pedestal_top` with its separate run id and left that one
row at `docs_stage=pending` after this node's successful drain. Documentation
stage is outside this restore's acceptance measure; the concurrent residue is
reported under follow-ons rather than attributed to this run.

## Exact child and producer evidence

The archive topology fixes the family at nineteen direct children. The broader
incident cohort also contains `total_ion_density_at_pedestal_top`, but that
identity is not an archived incoming child of `density_at_pedestal_top` and is
not part of this restore.

The final gate read each relationship from the live graph. Each row below is a
distinct live child with the stated direct DD producer evidence; the electron
identity deliberately has three producers. The `origin=pipeline` value was
written only after this positive producer query returned at least one
`StandardNameSource {source_type: 'dd'}-[:PRODUCED_NAME]->(child)` row. All
nineteen rows also returned `status=draft` and one outgoing `HAS_PARENT` edge
to the intended parent.

| Child | Direct live DD producer path(s) | DD producers | Origin | Status |
|---|---|---:|---|---|
| `argon_density_at_pedestal_top` | `summary/local/pedestal/n_i/argon/value` | 1 | pipeline | draft |
| `beryllium_density_at_pedestal_top` | `summary/local/pedestal/n_i/beryllium/value` | 1 | pipeline | draft |
| `boron_density_at_pedestal_top` | `summary/local/pedestal/n_i/boron/value` | 1 | pipeline | draft |
| `carbon_density_at_pedestal_top` | `summary/local/pedestal/n_i/carbon/value` | 1 | pipeline | draft |
| `deuterium_density_at_pedestal_top` | `summary/local/pedestal/n_i/deuterium/value` | 1 | pipeline | draft |
| `deuterium_tritium_density_at_pedestal_top` | `summary/local/pedestal/n_i/deuterium_tritium/value` | 1 | pipeline | draft |
| `electron_density_at_pedestal_top` | `summary/local/pedestal/n_e/value`; `summary/pedestal_fits/linear/n_e/pedestal_height/value`; `summary/pedestal_fits/mtanh/n_e/pedestal_height/value` | 3 | pipeline | draft |
| `helium_3_density_at_pedestal_top` | `summary/local/pedestal/n_i/helium_3/value` | 1 | pipeline | draft |
| `helium_4_density_at_pedestal_top` | `summary/local/pedestal/n_i/helium_4/value` | 1 | pipeline | draft |
| `hydrogen_density_at_pedestal_top` | `summary/local/pedestal/n_i/hydrogen/value` | 1 | pipeline | draft |
| `iron_density_at_pedestal_top` | `summary/local/pedestal/n_i/iron/value` | 1 | pipeline | draft |
| `krypton_density_at_pedestal_top` | `summary/local/pedestal/n_i/krypton/value` | 1 | pipeline | draft |
| `lithium_density_at_pedestal_top` | `summary/local/pedestal/n_i/lithium/value` | 1 | pipeline | draft |
| `neon_density_at_pedestal_top` | `summary/local/pedestal/n_i/neon/value` | 1 | pipeline | draft |
| `nitrogen_density_at_pedestal_top` | `summary/local/pedestal/n_i/nitrogen/value` | 1 | pipeline | draft |
| `oxygen_density_at_pedestal_top` | `summary/local/pedestal/n_i/oxygen/value` | 1 | pipeline | draft |
| `tritium_density_at_pedestal_top` | `summary/local/pedestal/n_i/tritium/value` | 1 | pipeline | draft |
| `tungsten_density_at_pedestal_top` | `summary/local/pedestal/n_i/tungsten/value` | 1 | pipeline | draft |
| `xenon_density_at_pedestal_top` | `summary/local/pedestal/n_i/xenon/value` | 1 | pipeline | draft |

Representative restored definitions and review evidence are:

- `argon_density_at_pedestal_top` — “Charge-state-summed local number density
  of argon ions at the H-mode edge pedestal top, excluding neutral argon.”
  Name and documentation review scores are both 1.000.
- `electron_density_at_pedestal_top` — “Electron number density at the top of
  the H-mode edge pedestal is the local concentration of free electrons at the
  transport-barrier shoulder.” Its name review score is 1.000 and the three
  source bindings above converge on the same identity.
- `helium_4_density_at_pedestal_top` — “Helium-4 ion number density at the top
  of the H-mode edge pedestal is the local concentration of helium-4 ions
  summed over all ionization states.” Name and documentation review scores are
  both 1.000.
- `tungsten_density_at_pedestal_top` — “Tungsten ion number density at the top
  of the H-mode edge pedestal is the local concentration of ionized tungsten,
  summed over all tungsten charge states.” Name and documentation review scores
  are both 1.000.

## Parent origin judgement

`density_at_pedestal_top` was not relabelled as pipeline. Its producer topology
is qualitatively different from every child:

| Identity | Direct DD producers | Structural producers | Incoming family edges | Truthful origin |
|---|---:|---:|---:|---|
| nineteen children | 21 in total | 0 required for the restored live rows | 0 | pipeline |
| `density_at_pedestal_top` | 0 | 1: `derived:density_at_pedestal_top` | 19 | derived |

The parent materializer had already written `origin=derived`. The final
origin-setting write was bounded to the nineteen named children and required a
positive DD producer edge, so it could not touch the producer-less parent. The
result records what the live producer graph makes true rather than copying the
archive's historical `catalog_edit` label or minting a false pipeline label.

## Execution and recovery record

The worktree first merged the corrective history at
`55e8420c7cff29c0fd9a728baf91a4856a1d8dad`. Content inspection at that exact
`HEAD` established both halves of the protection boundary:

- `graph_ops.py` defines `_SKELETON_PLACEHOLDER_PREDICATE`, uses it to query the
  positively proven placeholder subset, and passes only that subset to
  `refuse_protected_automatic_deletion`; and
- `protection.py` still defines `refuse_protected_automatic_deletion` and raises
  `ProtectedDeletionError` for a protected candidate immediately before a real
  deletion.

The preflight found all twenty-one sources extracted, unclaimed, DD-linked,
and unbound, with all twenty target identities absent. The exact twenty-one
focus paths then passed a dry run with zero writes. No execution used
`--reseed` or `--force`.

The first clean live drain used scope
`4a1bd026-05cf-4241-b1c6-f16bc2276435` and accounting run
`7536348c-8390-4980-919e-040631419f50`. It exited 0 with
`stop_reason=no_eligible_work`, 18 source compositions, 99 LLM events, and
$2.976525 spend. That drain created seventeen distinct children; two children
and the electron linear-fit binding remained. This was not a provider or
deletion failure: the three residual source rows were still `extracted` but
carried `attempt_count=5` consumed by the two pre-fix refusals, so the normal
gap-only selector could seed them but the workers could not claim them.

The exact recovery command `sn retry --failed` previewed and released 3 of 3
named sources. It wrote the repository's durable retry events before resetting
only those counters; this was not a reseed or force operation. The same complete
twenty-one-path gap-only focus command then ran with scope
`719c7a7c-e008-4826-84ea-5963b4983563` and accounting run
`d2636d4e-a2bb-42ba-ba59-0119b249ca8e`. It exited 0 with
`stop_reason=no_eligible_work`, three source compositions, thirteen LLM events,
and $0.307721 spend. The three compositions were the two missing children plus
the third source binding on the existing electron identity.

An intervening run without `--skip-global-maintenance` set the restored rows'
catalog status to draft and re-derived all seventeen then-live child edges. It
subsequently failed outside this family while global childless-parent cleanup
refused six other protected identities. That out-of-scope failure spent zero,
did not claim the three residual sources, and is recorded under follow-ons; the
scoped final drain completed the assigned cohort.

The two earlier blocked attempts are retained as negative evidence. Scope
`72a13df1-c940-4f4b-90fe-09a5c85c3c37` / accounting run
`0c979912-b685-4057-b254-057ecbd1a142` recorded nine errors, while scope
`ab862cd7-2e3c-4b06-9bec-6903f6337fa3` / accounting run
`da581695-fe36-4326-ae05-5f7bd8bae52f` recorded eleven. Both stopped with zero
names and zero spend because relationship-side skeleton candidates reached the
paid-deletion refusal before positive placeholder selection. Their exact
diagnosis is the boundary corrected in the merged content above.

## Live versus archive relationship inventory

The named archive
`imas-codex-graph-dev-002bf65-20260906T220012Z.tar.gz` was loaded into a
temporary Neo4j instance on `all_debug` job 1267661. A single exact-name query
found all 20 of 20 family identities and counted every incident relationship by
direction and type. A separate bounded live query counted the same 20
identities. In the tables below, each number is **archive/live**, and direction
is relative to the `StandardName` row.

The parent comparison is:

| Identity | Complete directional relationship counts, archive/live | Archive types absent live | Live-only types |
|---|---|---|---|
| `density_at_pedestal_top` | `in:HAS_PARENT` 19/19; `in:PRODUCED_NAME` 1/1; `out:DOCS_REVISION_OF` 7/0; `out:HAS_COCOS` 1/1; `out:HAS_INTERNAL_CHANGE` 3/1; `out:HAS_LOCUS` 2/1; `out:HAS_PHYSICS_DOMAIN` 1/0; `out:HAS_REVIEW` 30/0; `out:HAS_STRUCTURAL_AUTHORITY` 1/0; `out:HAS_UNIT` 1/1; `in:FOR_STANDARD_NAME` 0/23 | `DOCS_REVISION_OF`, `HAS_PHYSICS_DOMAIN`, `HAS_REVIEW`, `HAS_STRUCTURAL_AUTHORITY` | incoming `FOR_STANDARD_NAME` |

The child comparisons are complete per identity and type:

| Child | Complete directional relationship counts, archive/live | Archive types absent live | Live-only types |
|---|---|---|---|
| `argon_density_at_pedestal_top` | `in:ENTAILED_FROM_CHILD` 1/0; `in:FOR_STANDARD_NAME` 0/27; `in:HAS_STANDARD_NAME` 1/1; `in:PRODUCED_NAME` 2/1; `out:DOCS_REVISION_OF` 2/0; `out:HAS_COCOS` 1/1; `out:HAS_INTERNAL_CHANGE` 3/0; `out:HAS_LOCUS` 2/1; `out:HAS_PARENT` 1/1; `out:HAS_PHYSICAL_BASE` 0/1; `out:HAS_PHYSICS_DOMAIN` 2/0; `out:HAS_POSITION` 1/1; `out:HAS_REVIEW` 14/4; `out:HAS_SEGMENT` 2/3; `out:HAS_SUBJECT` 1/1; `out:HAS_UNIT` 1/1; `out:IN_CLUSTER` 1/1 | `ENTAILED_FROM_CHILD`, `DOCS_REVISION_OF`, `HAS_INTERNAL_CHANGE`, `HAS_PHYSICS_DOMAIN` | incoming `FOR_STANDARD_NAME`, `HAS_PHYSICAL_BASE` |
| `beryllium_density_at_pedestal_top` | `in:ENTAILED_FROM_CHILD` 1/0; `in:FOR_STANDARD_NAME` 0/15; `in:HAS_STANDARD_NAME` 1/1; `in:PRODUCED_NAME` 2/1; `out:DOCS_REVISION_OF` 1/0; `out:HAS_COCOS` 1/1; `out:HAS_INTERNAL_CHANGE` 1/0; `out:HAS_LOCUS` 2/1; `out:HAS_PARENT` 1/1; `out:HAS_PHYSICAL_BASE` 0/1; `out:HAS_PHYSICS_DOMAIN` 2/0; `out:HAS_POSITION` 1/1; `out:HAS_REVIEW` 6/5; `out:HAS_SEGMENT` 2/3; `out:HAS_SUBJECT` 1/1; `out:HAS_UNIT` 1/1; `out:IN_CLUSTER` 1/1 | `ENTAILED_FROM_CHILD`, `DOCS_REVISION_OF`, `HAS_INTERNAL_CHANGE`, `HAS_PHYSICS_DOMAIN` | incoming `FOR_STANDARD_NAME`, `HAS_PHYSICAL_BASE` |
| `boron_density_at_pedestal_top` | `in:ENTAILED_FROM_CHILD` 1/0; `in:FOR_STANDARD_NAME` 0/18; `in:HAS_STANDARD_NAME` 1/1; `in:PRODUCED_NAME` 2/1; `out:DOCS_REVISION_OF` 1/0; `out:HAS_COCOS` 1/1; `out:HAS_INTERNAL_CHANGE` 1/0; `out:HAS_LOCUS` 2/1; `out:HAS_PARENT` 1/1; `out:HAS_PHYSICAL_BASE` 0/1; `out:HAS_PHYSICS_DOMAIN` 1/0; `out:HAS_POSITION` 1/1; `out:HAS_REVIEW` 9/4; `out:HAS_SEGMENT` 2/3; `out:HAS_SUBJECT` 1/1; `out:HAS_UNIT` 1/1; `out:IN_CLUSTER` 0/1 | `ENTAILED_FROM_CHILD`, `DOCS_REVISION_OF`, `HAS_INTERNAL_CHANGE`, `HAS_PHYSICS_DOMAIN` | incoming `FOR_STANDARD_NAME`, `HAS_PHYSICAL_BASE`, `IN_CLUSTER` |
| `carbon_density_at_pedestal_top` | `in:ENTAILED_FROM_CHILD` 1/0; `in:FOR_STANDARD_NAME` 0/17; `in:HAS_STANDARD_NAME` 1/1; `in:PRODUCED_NAME` 2/1; `out:DOCS_REVISION_OF` 1/0; `out:HAS_COCOS` 1/1; `out:HAS_INTERNAL_CHANGE` 2/0; `out:HAS_LOCUS` 2/1; `out:HAS_PARENT` 1/1; `out:HAS_PHYSICAL_BASE` 0/1; `out:HAS_PHYSICS_DOMAIN` 2/0; `out:HAS_POSITION` 1/1; `out:HAS_REVIEW` 8/4; `out:HAS_SEGMENT` 2/3; `out:HAS_SUBJECT` 1/1; `out:HAS_UNIT` 1/1; `out:IN_CLUSTER` 1/1 | `ENTAILED_FROM_CHILD`, `DOCS_REVISION_OF`, `HAS_INTERNAL_CHANGE`, `HAS_PHYSICS_DOMAIN` | incoming `FOR_STANDARD_NAME`, `HAS_PHYSICAL_BASE` |
| `deuterium_density_at_pedestal_top` | `in:ENTAILED_FROM_CHILD` 1/0; `in:FOR_STANDARD_NAME` 0/17; `in:HAS_STANDARD_NAME` 1/1; `in:PRODUCED_NAME` 2/1; `out:DOCS_REVISION_OF` 1/0; `out:HAS_COCOS` 1/1; `out:HAS_INTERNAL_CHANGE` 1/0; `out:HAS_LOCUS` 2/1; `out:HAS_PARENT` 1/1; `out:HAS_PHYSICAL_BASE` 0/1; `out:HAS_PHYSICS_DOMAIN` 2/0; `out:HAS_POSITION` 1/1; `out:HAS_REVIEW` 8/4; `out:HAS_SEGMENT` 2/3; `out:HAS_SUBJECT` 1/1; `out:HAS_UNIT` 1/1; `out:IN_CLUSTER` 1/1 | `ENTAILED_FROM_CHILD`, `DOCS_REVISION_OF`, `HAS_INTERNAL_CHANGE`, `HAS_PHYSICS_DOMAIN` | incoming `FOR_STANDARD_NAME`, `HAS_PHYSICAL_BASE` |
| `deuterium_tritium_density_at_pedestal_top` | `in:ENTAILED_FROM_CHILD` 1/0; `in:FOR_STANDARD_NAME` 0/36; `in:HAS_STANDARD_NAME` 1/1; `in:PRODUCED_NAME` 2/1; `out:DOCS_REVISION_OF` 3/0; `out:HAS_COCOS` 1/1; `out:HAS_INTERNAL_CHANGE` 3/0; `out:HAS_LOCUS` 2/1; `out:HAS_PARENT` 1/1; `out:HAS_PHYSICAL_BASE` 0/1; `out:HAS_PHYSICS_DOMAIN` 2/0; `out:HAS_POSITION` 1/1; `out:HAS_REVIEW` 20/5; `out:HAS_SEGMENT` 2/3; `out:HAS_SUBJECT` 1/1; `out:HAS_UNIT` 1/1; `out:IN_CLUSTER` 1/1 | `ENTAILED_FROM_CHILD`, `DOCS_REVISION_OF`, `HAS_INTERNAL_CHANGE`, `HAS_PHYSICS_DOMAIN` | incoming `FOR_STANDARD_NAME`, `HAS_PHYSICAL_BASE` |
| `electron_density_at_pedestal_top` | `in:ENTAILED_FROM_CHILD` 1/0; `in:FOR_STANDARD_NAME` 0/18; `in:HAS_STANDARD_NAME` 3/3; `in:PRODUCED_NAME` 4/3; `out:DOCS_REVISION_OF` 1/1; `out:HAS_COCOS` 1/1; `out:HAS_INTERNAL_CHANGE` 1/0; `out:HAS_LOCUS` 2/1; `out:HAS_PARENT` 1/1; `out:HAS_PHYSICAL_BASE` 0/1; `out:HAS_PHYSICS_DOMAIN` 2/0; `out:HAS_POSITION` 1/1; `out:HAS_REVIEW` 7/4; `out:HAS_SEGMENT` 2/3; `out:HAS_SUBJECT` 1/1; `out:HAS_UNIT` 1/1; `out:IN_CLUSTER` 1/0 | `ENTAILED_FROM_CHILD`, `HAS_INTERNAL_CHANGE`, `HAS_PHYSICS_DOMAIN`, `IN_CLUSTER` | incoming `FOR_STANDARD_NAME`, `HAS_PHYSICAL_BASE` |
| `helium_3_density_at_pedestal_top` | `in:ENTAILED_FROM_CHILD` 1/0; `in:FOR_STANDARD_NAME` 0/14; `in:HAS_STANDARD_NAME` 1/1; `in:PRODUCED_NAME` 2/1; `out:DOCS_REVISION_OF` 1/0; `out:HAS_COCOS` 1/1; `out:HAS_INTERNAL_CHANGE` 1/0; `out:HAS_LOCUS` 2/1; `out:HAS_PARENT` 1/1; `out:HAS_PHYSICAL_BASE` 0/1; `out:HAS_PHYSICS_DOMAIN` 1/0; `out:HAS_POSITION` 1/1; `out:HAS_REVIEW` 6/4; `out:HAS_SEGMENT` 2/3; `out:HAS_SUBJECT` 1/1; `out:HAS_UNIT` 1/1; `out:IN_CLUSTER` 0/1 | `ENTAILED_FROM_CHILD`, `DOCS_REVISION_OF`, `HAS_INTERNAL_CHANGE`, `HAS_PHYSICS_DOMAIN` | incoming `FOR_STANDARD_NAME`, `HAS_PHYSICAL_BASE`, `IN_CLUSTER` |
| `helium_4_density_at_pedestal_top` | `in:ENTAILED_FROM_CHILD` 1/0; `in:FOR_STANDARD_NAME` 0/15; `in:HAS_STANDARD_NAME` 1/1; `in:PRODUCED_NAME` 2/1; `out:DOCS_REVISION_OF` 1/0; `out:HAS_COCOS` 1/1; `out:HAS_INTERNAL_CHANGE` 1/0; `out:HAS_LOCUS` 2/1; `out:HAS_PARENT` 1/1; `out:HAS_PHYSICAL_BASE` 0/1; `out:HAS_PHYSICS_DOMAIN` 2/0; `out:HAS_POSITION` 1/1; `out:HAS_REVIEW` 6/4; `out:HAS_SEGMENT` 2/3; `out:HAS_SUBJECT` 1/1; `out:HAS_UNIT` 1/1; `out:IN_CLUSTER` 1/1 | `ENTAILED_FROM_CHILD`, `DOCS_REVISION_OF`, `HAS_INTERNAL_CHANGE`, `HAS_PHYSICS_DOMAIN` | incoming `FOR_STANDARD_NAME`, `HAS_PHYSICAL_BASE` |
| `hydrogen_density_at_pedestal_top` | `in:ENTAILED_FROM_CHILD` 1/0; `in:FOR_STANDARD_NAME` 0/14; `in:HAS_STANDARD_NAME` 1/1; `in:PRODUCED_NAME` 2/1; `out:DOCS_REVISION_OF` 1/0; `out:HAS_COCOS` 1/1; `out:HAS_INTERNAL_CHANGE` 1/0; `out:HAS_LOCUS` 2/1; `out:HAS_PARENT` 1/1; `out:HAS_PHYSICAL_BASE` 0/1; `out:HAS_PHYSICS_DOMAIN` 1/0; `out:HAS_POSITION` 1/1; `out:HAS_REVIEW` 6/4; `out:HAS_SEGMENT` 2/3; `out:HAS_SUBJECT` 1/1; `out:HAS_UNIT` 1/1; `out:IN_CLUSTER` 0/1 | `ENTAILED_FROM_CHILD`, `DOCS_REVISION_OF`, `HAS_INTERNAL_CHANGE`, `HAS_PHYSICS_DOMAIN` | incoming `FOR_STANDARD_NAME`, `HAS_PHYSICAL_BASE`, `IN_CLUSTER` |
| `iron_density_at_pedestal_top` | `in:ENTAILED_FROM_CHILD` 1/0; `in:FOR_STANDARD_NAME` 0/19; `in:HAS_STANDARD_NAME` 1/1; `in:PRODUCED_NAME` 2/1; `out:DOCS_REVISION_OF` 1/0; `out:HAS_COCOS` 1/1; `out:HAS_INTERNAL_CHANGE` 2/0; `out:HAS_LOCUS` 2/1; `out:HAS_PARENT` 1/1; `out:HAS_PHYSICAL_BASE` 0/1; `out:HAS_PHYSICS_DOMAIN` 2/0; `out:HAS_POSITION` 1/1; `out:HAS_REVIEW` 8/5; `out:HAS_SEGMENT` 2/3; `out:HAS_SUBJECT` 1/1; `out:HAS_UNIT` 1/1; `out:IN_CLUSTER` 1/1 | `ENTAILED_FROM_CHILD`, `DOCS_REVISION_OF`, `HAS_INTERNAL_CHANGE`, `HAS_PHYSICS_DOMAIN` | incoming `FOR_STANDARD_NAME`, `HAS_PHYSICAL_BASE` |
| `krypton_density_at_pedestal_top` | `in:ENTAILED_FROM_CHILD` 1/0; `in:FOR_STANDARD_NAME` 0/23; `in:HAS_STANDARD_NAME` 1/1; `in:PRODUCED_NAME` 2/1; `out:DOCS_REVISION_OF` 1/0; `out:HAS_COCOS` 1/1; `out:HAS_INTERNAL_CHANGE` 1/0; `out:HAS_LOCUS` 2/1; `out:HAS_PARENT` 1/1; `out:HAS_PHYSICAL_BASE` 0/1; `out:HAS_PHYSICS_DOMAIN` 2/0; `out:HAS_POSITION` 1/1; `out:HAS_REVIEW` 10/4; `out:HAS_SEGMENT` 2/3; `out:HAS_SUBJECT` 1/1; `out:HAS_UNIT` 1/1; `out:IN_CLUSTER` 1/1 | `ENTAILED_FROM_CHILD`, `DOCS_REVISION_OF`, `HAS_INTERNAL_CHANGE`, `HAS_PHYSICS_DOMAIN` | incoming `FOR_STANDARD_NAME`, `HAS_PHYSICAL_BASE` |
| `lithium_density_at_pedestal_top` | `in:ENTAILED_FROM_CHILD` 1/0; `in:FOR_STANDARD_NAME` 0/14; `in:HAS_STANDARD_NAME` 1/1; `in:PRODUCED_NAME` 2/1; `out:DOCS_REVISION_OF` 1/0; `out:HAS_COCOS` 1/1; `out:HAS_INTERNAL_CHANGE` 1/0; `out:HAS_LOCUS` 2/1; `out:HAS_PARENT` 1/1; `out:HAS_PHYSICAL_BASE` 0/1; `out:HAS_PHYSICS_DOMAIN` 1/0; `out:HAS_POSITION` 1/1; `out:HAS_REVIEW` 6/4; `out:HAS_SEGMENT` 2/3; `out:HAS_SUBJECT` 1/1; `out:HAS_UNIT` 1/1; `out:IN_CLUSTER` 0/1 | `ENTAILED_FROM_CHILD`, `DOCS_REVISION_OF`, `HAS_INTERNAL_CHANGE`, `HAS_PHYSICS_DOMAIN` | incoming `FOR_STANDARD_NAME`, `HAS_PHYSICAL_BASE`, `IN_CLUSTER` |
| `neon_density_at_pedestal_top` | `in:ENTAILED_FROM_CHILD` 1/0; `in:FOR_STANDARD_NAME` 0/15; `in:HAS_STANDARD_NAME` 1/1; `in:PRODUCED_NAME` 2/1; `out:DOCS_REVISION_OF` 1/0; `out:HAS_COCOS` 1/1; `out:HAS_INTERNAL_CHANGE` 1/0; `out:HAS_LOCUS` 2/1; `out:HAS_PARENT` 1/1; `out:HAS_PHYSICAL_BASE` 0/1; `out:HAS_PHYSICS_DOMAIN` 1/0; `out:HAS_POSITION` 1/1; `out:HAS_REVIEW` 7/4; `out:HAS_SEGMENT` 2/3; `out:HAS_SUBJECT` 1/1; `out:HAS_UNIT` 1/1; `out:IN_CLUSTER` 0/1 | `ENTAILED_FROM_CHILD`, `DOCS_REVISION_OF`, `HAS_INTERNAL_CHANGE`, `HAS_PHYSICS_DOMAIN` | incoming `FOR_STANDARD_NAME`, `HAS_PHYSICAL_BASE`, `IN_CLUSTER` |
| `nitrogen_density_at_pedestal_top` | `in:ENTAILED_FROM_CHILD` 1/0; `in:FOR_STANDARD_NAME` 0/30; `in:HAS_STANDARD_NAME` 1/1; `in:PRODUCED_NAME` 2/1; `out:DOCS_REVISION_OF` 3/0; `out:HAS_COCOS` 1/1; `out:HAS_INTERNAL_CHANGE` 3/0; `out:HAS_LOCUS` 2/1; `out:HAS_PARENT` 1/1; `out:HAS_PHYSICAL_BASE` 0/1; `out:HAS_PHYSICS_DOMAIN` 2/0; `out:HAS_POSITION` 1/1; `out:HAS_REVIEW` 17/4; `out:HAS_SEGMENT` 2/3; `out:HAS_SUBJECT` 1/1; `out:HAS_UNIT` 1/1; `out:IN_CLUSTER` 1/1 | `ENTAILED_FROM_CHILD`, `DOCS_REVISION_OF`, `HAS_INTERNAL_CHANGE`, `HAS_PHYSICS_DOMAIN` | incoming `FOR_STANDARD_NAME`, `HAS_PHYSICAL_BASE` |
| `oxygen_density_at_pedestal_top` | `in:ENTAILED_FROM_CHILD` 1/0; `in:FOR_STANDARD_NAME` 0/24; `in:HAS_STANDARD_NAME` 1/1; `in:PRODUCED_NAME` 2/1; `out:DOCS_REVISION_OF` 2/0; `out:HAS_COCOS` 1/1; `out:HAS_INTERNAL_CHANGE` 1/0; `out:HAS_LOCUS` 2/1; `out:HAS_PARENT` 1/1; `out:HAS_PHYSICAL_BASE` 0/1; `out:HAS_PHYSICS_DOMAIN` 2/0; `out:HAS_POSITION` 1/1; `out:HAS_REVIEW` 13/4; `out:HAS_SEGMENT` 2/3; `out:HAS_SUBJECT` 1/1; `out:HAS_UNIT` 1/1; `out:IN_CLUSTER` 1/1 | `ENTAILED_FROM_CHILD`, `DOCS_REVISION_OF`, `HAS_INTERNAL_CHANGE`, `HAS_PHYSICS_DOMAIN` | incoming `FOR_STANDARD_NAME`, `HAS_PHYSICAL_BASE` |
| `tritium_density_at_pedestal_top` | `in:ENTAILED_FROM_CHILD` 1/0; `in:FOR_STANDARD_NAME` 0/27; `in:HAS_STANDARD_NAME` 1/1; `in:PRODUCED_NAME` 2/1; `out:DOCS_REVISION_OF` 2/0; `out:HAS_COCOS` 1/1; `out:HAS_INTERNAL_CHANGE` 2/0; `out:HAS_LOCUS` 2/1; `out:HAS_PARENT` 1/1; `out:HAS_PHYSICAL_BASE` 0/1; `out:HAS_PHYSICS_DOMAIN` 2/0; `out:HAS_POSITION` 1/1; `out:HAS_REVIEW` 14/5; `out:HAS_SEGMENT` 2/3; `out:HAS_SUBJECT` 1/1; `out:HAS_UNIT` 1/1; `out:IN_CLUSTER` 1/1 | `ENTAILED_FROM_CHILD`, `DOCS_REVISION_OF`, `HAS_INTERNAL_CHANGE`, `HAS_PHYSICS_DOMAIN` | incoming `FOR_STANDARD_NAME`, `HAS_PHYSICAL_BASE` |
| `tungsten_density_at_pedestal_top` | `in:ENTAILED_FROM_CHILD` 1/0; `in:FOR_STANDARD_NAME` 0/17; `in:HAS_STANDARD_NAME` 1/1; `in:PRODUCED_NAME` 2/1; `out:DOCS_REVISION_OF` 1/0; `out:HAS_COCOS` 1/1; `out:HAS_INTERNAL_CHANGE` 2/0; `out:HAS_LOCUS` 2/1; `out:HAS_PARENT` 1/1; `out:HAS_PHYSICAL_BASE` 0/1; `out:HAS_PHYSICS_DOMAIN` 2/0; `out:HAS_POSITION` 1/1; `out:HAS_REVIEW` 8/4; `out:HAS_SEGMENT` 2/3; `out:HAS_SUBJECT` 1/1; `out:HAS_UNIT` 1/1; `out:IN_CLUSTER` 1/1 | `ENTAILED_FROM_CHILD`, `DOCS_REVISION_OF`, `HAS_INTERNAL_CHANGE`, `HAS_PHYSICS_DOMAIN` | incoming `FOR_STANDARD_NAME`, `HAS_PHYSICAL_BASE` |
| `xenon_density_at_pedestal_top` | `in:ENTAILED_FROM_CHILD` 1/0; `in:FOR_STANDARD_NAME` 0/21; `in:HAS_STANDARD_NAME` 1/1; `in:PRODUCED_NAME` 2/1; `out:DOCS_REVISION_OF` 2/0; `out:HAS_COCOS` 1/1; `out:HAS_INTERNAL_CHANGE` 2/0; `out:HAS_LOCUS` 2/1; `out:HAS_PARENT` 1/1; `out:HAS_PHYSICAL_BASE` 0/1; `out:HAS_PHYSICS_DOMAIN` 2/0; `out:HAS_POSITION` 1/1; `out:HAS_REVIEW` 11/4; `out:HAS_SEGMENT` 2/3; `out:HAS_SUBJECT` 1/1; `out:HAS_UNIT` 1/1; `out:IN_CLUSTER` 1/1 | `ENTAILED_FROM_CHILD`, `DOCS_REVISION_OF`, `HAS_INTERNAL_CHANGE`, `HAS_PHYSICS_DOMAIN` | incoming `FOR_STANDARD_NAME`, `HAS_PHYSICAL_BASE` |

Across the whole family, the archive relationship types that are present in at
least one archive row but absent from the corresponding live row are explicitly:
incoming `ENTAILED_FROM_CHILD`; outgoing `DOCS_REVISION_OF`,
`HAS_INTERNAL_CHANGE`, `HAS_PHYSICS_DOMAIN`, `HAS_REVIEW`,
`HAS_STRUCTURAL_AUTHORITY`, and `IN_CLUSTER`. The live-only types are incoming
`FOR_STANDARD_NAME`, outgoing `HAS_PHYSICAL_BASE`, and, for the five children
whose archive row lacked it, outgoing `IN_CLUSTER`.

The differences are expected for a pipeline reconstruction rather than an
archive replay. The live graph preserves the required DD bindings, catalog
status, COCOS, unit, grammar, locus, subject, and family-parent topology while
starting fresh review and history populations. Archive `PRODUCED_NAME` exceeds
live by one per child because every archived child also carried a historical
`derived:` producer; the restored live children truthfully use their DD
producers, while only the producer-less parent carries a structural producer.

## Evidence artifacts

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260909T072310390297-n-pidp-restore-pedestal-density-family/archive-pedestal-relationship-counts.json`
  — all 20 archive identities and every directional relationship count.
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260909T072310390297-n-pidp-restore-pedestal-density-family/pedestal-live-archive-relationship-comparison.json`
  — exact per-identity union of archive and live relationship types, counts,
  deltas, archive-absent types, and live-only types.
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260909T072310390297-n-pidp-restore-pedestal-density-family/archive-relationship-extract.log`
  — the bounded `all_debug` archive load log for job 1267661.
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260908T153422670447-n-sli-the-origin-repair-unshielded-seventy-eight-identities/archive-identity-lineage.json`
  — the prior isolated-archive lineage census that fixed the nineteen-child and
  twenty-one-source cohort.

Live graph work ran on the login node because `NEO4J_URI` resolves through the
login-local tunnel. Every query named only these twenty identities or their
twenty-one exact sources and completed under ten seconds. Archive loading and
its relationship census ran on `all_debug`; no archive scan ran on the login
node.

## Follow-on outside this node

The unskipped maintenance attempt exposed a separate structural derived-parent
cleanup refusal for `flux_at_first_wall`, `flux_at_wall_due_to_eddy_current`,
`flux_at_wall_due_to_pumping`, `ion_diffusivity`,
`permeability_of_ferritic_element`, and `width_of_spectrometer_channel`. That
failure is outside this node's declared family and write scope. It did not
alter the successful 19-child acceptance gate and is returned for its owning
cleanup node rather than triaged here.

A separate concurrent accounting run
`1c51710c-e97c-49af-84f6-17393025e0e8` subsequently left
`electron_density_at_pedestal_top` at `docs_stage=pending`. Its identity,
origin, status, three DD producers, and parent edge remain correct; the owning
documentation run must drain or release that residual docs state.
