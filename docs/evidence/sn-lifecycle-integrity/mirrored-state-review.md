# Mirrored state review

Reviewed 2026-09-08 against code revision
`53b5d885171b19348b9dd4f74720abfa06726afb` and live-plan version 103. This is
a read-only design review: it changes no graph row, lifecycle scalar, predicate,
or product code and runs no Standard Names pipeline operation.

The lifecycle counts below come from
[`stage-scalar-agreement.md`](stage-scalar-agreement.md), measured against code
revision `1d7104d9c734c7a03e18f019b5340e31820c1d10`. I did not re-run those
predicates. I added count-only reads for relationship mirrors the earlier census
did not cover. Each supplemental query began at exactly one
`StandardName` or `StandardNameSource` label, used only per-row pattern
comprehensions or `EXISTS`, returned aggregates plus at most ten diagnostic
samples, and avoided Cartesian products. The slowest took 0.333 seconds through
the login-node-local graph tunnel, below the ten-second ceiling.

## Decision

The observation is confirmed, but the repair is not “delete every scalar.” This
review found **24 mirrored-state surfaces** and five apparent duplicates that
are semantically distinct:

- **DERIVE — 12 surfaces.** The scalar is only a second representation of an
  edge or immutable receipt. Read it from that authority and stop storing it.
- **KEEP AND RECONCILE — 12 surfaces.** The scalar is a useful indexed queue or
  aggregate projection. It needs one writer and an edge-to-scalar or
  receipt-to-scalar reconcile owned by that writer.
- **KEEP AS DISTINCT — five boundaries.** These fields resemble a mirror but
  encode a different fact. Conflating them would discard information.

The general failure mechanism is also confirmed: for the lifecycle pairs, the
queue/export predicate normally reads the scalar while the corroborating edge
or receipt is either ignored or consulted by a different code path. The stale
copy therefore gates the operation. The exceptions are full documentation
export, which now reads the winning `HAS_REVIEW` traversal, source liveness,
which reads the upstream edge/entity, and several search traversals that read
relationships directly.

| # | Scalar projection | Relationship, event, or stamp | What gates the work today | Verdict |
|---:|---|---|---|---|
| 1 | `source_paths` | DD `HAS_STANDARD_NAME` plus source `PRODUCED_NAME` | Scalar gates IDS filters; edges gate source migration | DERIVE |
| 2 | source `produced_sn_id` | `PRODUCED_NAME` | Edge gates composition/hints; release and recovery also trust scalar | DERIVE |
| 3 | `source_types` | producer node/edge kinds | Scalar gates reset/delete filters | DERIVE |
| 4 | `source_domains` | `HAS_PHYSICS_DOMAIN` set | Scalar/list gates search; edge gates neighborhood traversal | KEEP AND RECONCILE |
| 5 | source `dd_path` / `signal` | `FROM_DD_PATH` / `FROM_SIGNAL` | Edge plus `source_id` gates stale/revive | DERIVE |
| 6 | `unit` | `HAS_UNIT` | Both are checked by fold; scalar gates dimensionality, export prefers edge | KEEP AND RECONCILE |
| 7 | `cocos` | `HAS_COCOS` | Scalar gates audits; export reads edge | KEEP AND RECONCILE |
| 8 | `primary_cluster_id` | `IN_CLUSTER` | Search traverses edge; scalar mostly feeds writers | DERIVE |
| 9 | `links` | `REFERENCES` | Resolver/export read scalar; relationship readers see none | DERIVE |
| 10 | `deprecates` / `superseded_by` | `HAS_PREDECESSOR` / `HAS_SUCCESSOR` | Catalog/fold read scalar; terminal traversal reads edge | DERIVE |
| 11 | axis reviewer field bundles | axis `HAS_REVIEW` group | Pools/refine read scalar; full docs export reads edge | DERIVE |
| 12 | review count/mean/disagreement | aggregate over `HAS_REVIEW` | Vocabulary promotion reads mean scalar | KEEP AND RECONCILE |
| 13 | `reviewed_name_at` / `reviewed_docs_at` | axis Review `reviewed_at` | Name/docs admission reads timestamps | DERIVE |
| 14 | `name_stage` terminal/review states | Review, attempt, and lineage receipts | Scalar gates every name pool and export | KEEP AND RECONCILE |
| 15 | `docs_stage` terminal/review states | docs Review/generation/chain receipts | Scalar gates docs pools; full export reads edge | KEEP AND RECONCILE |
| 16 | name/docs quorum-shortfall markers | non-winning Review plus `*_at` | Scalar gates export, promotion, and refine | KEEP AND RECONCILE |
| 17 | `chain_length` | `REFINED_FROM` depth | Scalar is a legacy budget fallback | DERIVE |
| 18 | `docs_chain_length` | `DOCS_REVISION_OF` count | Scalar gates docs exhaustion/refine | DERIVE |
| 19 | `validation_status` | validator timestamp/detail receipt | Scalar gates review/export; timestamp gates validation | KEEP AND RECONCILE |
| 20 | `link_status` | target validity plus resolver receipt | Scalar gates resolver | KEEP AND RECONCILE |
| 21 | catalog `status` | approval/change/successor receipts | Scalar gates export and approval | KEEP AND RECONCILE |
| 22 | `origin` | producer/change/approval provenance | Scalar still gates derived-name paths | DERIVE |
| 23 | source `status` | state-specific edge/stamp/reason | Scalar gates composition | KEEP AND RECONCILE |
| 24 | `compose_hint_status` | request/consume stamps plus produced target | Scalar and live edge jointly gate exact hint admission | KEEP AND RECONCILE |

## Priority by exposed live population

“Otherwise publishable” uses the earlier census definition: accepted or
approved name, valid validation scalar, no name/docs shortfall, no catalog
tombstone, and a reachable winning docs review. For the supplemental rows it is
an intersection count, not a claim that the mirror itself currently excludes
the row from export. Source rows are ranked separately because a source is not
a publishable identity.

| Rank | Divergent mirror | Divergent rows | Otherwise publishable | Present consequence |
|---:|---|---:|---:|---|
| 1 | `links` versus declared `REFERENCES` | 2,948 names | 1,985 | Every relationship traversal sees no references: 6,682 stored `name:` references exist, but the live graph has zero `REFERENCES` edges. Export still reads the scalar. |
| 2 | `source_types` versus producer kinds | 3,549 names | 1,750 | Scalar filters and graph-derived provenance select different source populations. |
| 3 | `docs_chain_length` versus `DOCS_REVISION_OF` count | 1,579 names | 1,181 | The scalar can grant or deny a docs refinement rotation contrary to stored history. |
| 4 | `source_domains` versus `HAS_PHYSICS_DOMAIN` set | 1,654 names | 582 | Scalar/list filters and domain traversals disagree. |
| 5 | `chain_length` versus `REFINED_FROM` depth | 735 names | 180 | The scalar is a fallback budget input even though lineage records a different depth. |
| 6 | `primary_cluster_id` versus `IN_CLUSTER` | 900 names | 170 | 805 are scalar-only and 95 edge-only; cluster searches and scalar readers see different populations. |
| 7 | `validation_status='valid'` without `validated_at` | 538 names | 169 | The provisional scalar can admit export while the full validator remains claimable by timestamp. |
| 8 | `link_status='resolved'` with an absent or terminal target | 328 names | 39 | The resolver skips the row; export does not gate on link status. |
| 9 | `status='superseded'` with a live pipeline stage | 34 names | 31 | Export removes the identity at its population boundary. |
| 10 | `docs_stage='accepted'` without a winning docs review | 27 names | 27 | Scalar-driven docs queues skip the row; relationship-driven full export refuses it. |
| 11 | accepted name state without name-review authority | 6 names | 6 | Review is skipped because the stage says it already happened. |
| 12 | `reviewed_name_at` without a name review or structural authority | 5 names | 5 | Name review is skipped and docs review persistence is admitted. |
| 13 | `source_paths` versus live source edges | 279 names | 0 | All current disagreements are on terminal identities because the reconcile intentionally skips them. |
| 14 | `review_count`/mean/disagreement versus `HAS_REVIEW` | 59 names | 0 | Vocabulary promotion can read a stale mean; no otherwise-publishable row is currently exposed. |
| 15 | `superseded_by` versus `HAS_SUCCESSOR` | 6 names | 0 | All six are tombstones; lineage readers disagree, but no live batch is reduced. |
| 16 | `unit` versus `HAS_UNIT` | 4 names | 0 | All four are tombstones with a scalar and no edge. |
| 17 | historical `origin='catalog_edit'` without catalog evidence | 0 names | 0 | The earlier 2,096-row defect has been reconciled; this is the cheapest retirement because no repair population remains. |

The relationship supplement also found **117 `StandardNameSource` target
mirror disagreements**: 100 sources have multiple `PRODUCED_NAME` edges, 17
have a single edge disagreeing with `produced_sn_id`, 85 have a live upstream
entity, 98 have at least one live target, and 84 have both. Three DD sources
have `dd_path`/`source_id` disagreement or multiple `FROM_DD_PATH` edges. The
earlier lifecycle census independently ranks 398 live-upstream skipped sources
without a reason, 59 failed sources without complete failure evidence, three
extracted sources with a live produced target, and one open compose hint with a
live binding.

The first implementation order should therefore be: references and producer
types, docs/name lineage counters, domain/cluster membership, validation
receipts, link resolution, catalog tombstones, review authority/stages, then
terminal-only unit/lineage/cache cleanup. Source binding
cardinality is a separate high-priority repair before any migration trusts
`produced_sn_id`: 100 sources presently have more than one target edge, so
choosing either representation without adjudication would silently select an
identity.

## Source provenance mirrors

### `StandardName.source_paths` ↔ source-binding edges — DERIVE

**Pair and authority.** The non-`derived:` entries of `source_paths` duplicate
the union of `(IMASNode)-[:HAS_STANDARD_NAME]->(sn)` and non-derived
`(StandardNameSource)-[:PRODUCED_NAME]->(sn)`. The edge union is already named
as the source of truth by `reconcile_standard_name_source_paths`
(`graph_ops.py:13521-13544`). The `derived:` pseudo-URIs are the exception: no
node or edge represents them, so they must be materialized as derived
`StandardNameSource` identities before the scalar can disappear.

**Which copy gates work.** Name-review and validation fetches return
`sn.source_paths`, and IDS-scoped review uses
`ANY(p IN sn.source_paths WHERE p STARTS WITH $ids_prefix)`
(`graph_ops.py:8180-8193`); CLI inspection filters the same list
(`cli/sn.py:5935-6000`). Source migration, source liveness, and release lineage
use `PRODUCED_NAME` edges. The two sides can therefore select different
cohorts. Current supplement: 279 names differ (207 have at least one scalar-only
path and 72 at least one edge-only path), all outside the otherwise-publishable
set because the existing reconcile deliberately excludes terminal stages.

**Recommendation and cost.** Derive the list through one bounded adjacency
projection whenever it is returned or filtered; the cost is proportional to
the name's source degree and the largest lists already have to be transferred
to callers. `reconcile_standard_name_source_paths` is the existing single
obvious edge-to-scalar repair and is the migration bridge, not the permanent
second writer. Blast radius: migrate the 279 divergent names, then replace
scalar reads; preserve `derived:` entries until corresponding source nodes
exist.

### `StandardNameSource.produced_sn_id` ↔ `PRODUCED_NAME` — DERIVE

**Pair and authority.** The schema explicitly calls `produced_sn_id` a scalar
mirror. `_finalize_generated_name_stage` writes the scalar and edge in one
transaction (`graph_ops.py:7614-7687`), while `reconcile_provenance` says the
live edge is authoritative after a cascade retarget
(`graph_ops.py:13129-13154`).

**Which copy gates work.** Composition and hint admission test the edge
(`graph_ops.py:9701-9723`); manifest release reads both and prefers the scalar
when it is among the direct targets (`graph_ops.py:12243-12312`); provenance
repair uses the scalar to recreate a missing edge
(`graph_ops.py:13141-13169`). Current supplement: 5,441 sources have either
representation and 117 disagree. Of those, 100 have multiple target edges, 17
have a single nonmatching edge, and 84 combine a live upstream with a live
target. The scalar equals one of the edge targets on 100 rows, which shows why
“prefer the scalar” hides rather than resolves the cardinality defect.

**Recommendation and cost.** First adjudicate the 100 multi-edge sources with
the existing single-live-target guard; then derive the target id from the
cardinality-one edge. A one-hop lookup is cheaper than maintaining two values
and cannot select a stale id after a retarget. Remove the scalar-to-edge
recovery direction after migration: an atomic edge write cannot be repaired
safely from an independently stale scalar. Blast radius: 117 inconsistent
sources now, including 84 able to affect live work.

### `StandardName.source_types` ↔ producer kinds — DERIVE

**Pair and authority.** For DD, signal, and derived sources, the set duplicates
the labels/types of entities attached through `HAS_STANDARD_NAME` and
`PRODUCED_NAME`. The value `catalog` has no corresponding source/event node,
which is a missing provenance representation rather than a reason to keep an
unverifiable list.

**Which copy gates work.** Review context reads the scalar
(`review/pipeline.py:280-302`), and reset/delete source filters test membership
in it (`graph_ops.py:8268-8308`, `8475-8532`). Origin reconciliation instead
derives source kinds from `PRODUCED_NAME` (`graph_ops.py:13234-13245`). A live
union comparison including direct DD/signal projections found 3,549 divergent
names, 1,750 otherwise publishable. Representative disagreement:
`vacuum_magnetic_vector_potential` stores `['catalog']` while its only producer
kind is `derived`; many terminal names store `['dd']` after all producer edges
have moved away.

**Recommendation and cost.** Materialize catalog import/approval as a typed
provenance event, then derive the source-kind set from producer/event adjacency.
Cost is one deduplicated adjacency projection. Blast radius is 3,549 names;
1,750 can enter a catalog batch and currently present provenance the graph
cannot corroborate.

### `StandardName.source_domains` ↔ `HAS_PHYSICS_DOMAIN` set — KEEP AND
RECONCILE

**Pair and authority.** `source_domains` is the complete append-only domain set;
`_write_standard_name_edges` materializes one `HAS_PHYSICS_DOMAIN` edge per
value (`graph_ops.py:3251-3274`). The separate `physics_domain` scalar is the
ranked primary and is addressed below as a non-mirror.

**Which copy gates work.** Search/filter accepts the primary scalar or list
membership (`search.py:90-91`, `476-477`), while graph neighborhood queries can
traverse the edges. The supplemental exact-set comparison found 1,654 divergent
names, 582 otherwise publishable; the primary domain is present among the edge
set on 3,145 names, so this is not merely a primary-versus-all interpretation.

**Recommendation and cost.** Keep the list as an indexed filter projection, but
make domain promotion the sole writer of both it and the edge set, and reconcile
the full set in both directions from source-domain authority. Current repair
blast radius is 1,654 names, including 582 otherwise publishable.

### `StandardNameSource.dd_path` / `signal` ↔ `FROM_DD_PATH` /
`FROM_SIGNAL` — DERIVE

**Pair and authority.** The source schema declares both as relationship slots;
`source_id` is the stable source identity. Source materialization writes the
typed edge from the incoming path (`graph_ops.py:10130-10141`). The liveness
reconcile checks both the edge and the entity keyed by `source_id`, then repairs
the edge (`graph_ops.py:12101-12216`). That makes `source_id` plus the typed edge
the authority; separate `dd_path`/`signal` properties add no state.

**Which copy gates work.** `reconcile_standard_name_sources` treats a missing
typed edge or missing/removed entity as stale, while
`_standard_name_source_upstream_present_cypher` keys existence from
`source_id` (`graph_ops.py:12083-12098`). No queue gate reads `dd_path` or
`signal`. The supplement found three DD source rows with inconsistent/multiple
edges and no populated signal mirror population.

**Recommendation and cost.** Derive the convenience field from the single
typed adjacency on read; cost is one degree-one expansion. Repair the three DD
rows before enforcing cardinality one. The existing
`reconcile_standard_name_sources` is the obvious temporary repair owner.

## Catalog and structural mirrors

### `StandardName.unit` ↔ `HAS_UNIT` — KEEP AND RECONCILE

**Pair and authority.** Unit is deliberately dual-represented. The scalar is
the catalog/prompt value and the edge links to the controlled `Unit` node. The
current contract makes the scalar authoritative: the unit reconcile replaces
the edge set from `sn.unit` (`graph_ops.py:13391-13464`).

**Which copy gates work.** The rename/fold guard refuses multiple unit edges or
a scalar/edge disagreement (`edit.py:1739-1758`). Attachment checks and several
prompt/audit paths read `sn.unit`; export reads `coalesce(u.id, sn.unit)`
(`export.py:565-571`). Thus neither side silently wins at the destructive fold
gate, but ordinary readers can still disagree. Four current rows diverge; all
are superseded tombstones, so the otherwise-publishable exposure is zero.

**Recommendation and cost.** Keep the scalar for indexed dimensionality and
catalog serialization. Make every unit mutation go through the existing
cardinality-one writer and retain `reconcile_standard_name_unit_edges` as the
single repair owner. The current repair blast radius is four terminal names.

### `StandardName.cocos` ↔ `HAS_COCOS` — KEEP AND RECONCILE

**Pair and authority.** The integer scalar is the convention value used by
audits; the edge links to the convention node. The writer emits both
(`graph_ops.py:5612-5626`) and
`reconcile_standard_name_cocos_links` fills the scalar from current DD
authority before creating a missing edge (`graph_ops.py:13318-13388`).

**Which copy gates work.** COCOS audit predicates read `sn.cocos`
(`audits.py:3999-4000`, `4454-4462`); export reads the edge's `c.id`
(`export.py:565-571`). The supplemental census found 1,033 populated names and
zero disagreement.

**Recommendation and cost.** Keep both because the scalar is a compact audit
projection and the edge supplies convention metadata. The existing reconcile
is the single obvious owner and currently has a zero-row repair blast radius.

### `StandardName.primary_cluster_id` ↔ `IN_CLUSTER` — DERIVE

**Pair and authority.** The schema calls `primary_cluster_id` the scalar source
for `IN_CLUSTER`; `_backfill_cluster_from_sources` writes both
(`graph_ops.py:7561-7600`). Unlike `physics_domain`, a StandardName has only one
primary cluster edge in the live data.

**Which copy gates work.** Similarity/search paths traverse `IN_CLUSTER`
(`search.py:848-849`), while the scalar is consumed almost entirely as an edge
write input (`graph_ops.py:3234-3248`, `7567-7597`). The live split is severe:
805 scalar-only names, 95 edge-only names, no row with two cluster edges, and
170 divergent names otherwise publishable.

**Recommendation and cost.** Repair all 900 rows from their source-cluster
authority, enforce one primary edge, then derive `primary_cluster_id` by a
degree-one expansion. `_backfill_cluster_from_sources` is the existing obvious
writer to own the migration. Removing the scalar eliminates an otherwise idle
copy; query cost is one adjacency lookup only when a caller asks for the id.

### `StandardName.links` ↔ `REFERENCES` — DERIVE

**Pair and authority.** `links` stores `name:<id>` strings for catalog output;
the schema separately declares `(StandardName)-[:REFERENCES]->(StandardName)`
(`standard_name.yaml:1925-1935`). Documentation link normalization rebuilds the
scalar from accepted documentation (`graph_ops.py:9438-9579`), but no current
Standard Names writer materializes `REFERENCES`.

**Which copy gates work.** Link resolution claims on `link_status` and rewrites
the scalar list (`graph_ops.py:9103-9246`); graph reference traversals would
read `REFERENCES`. Today 2,948 names carry 6,682 `name:` scalar references and
the graph carries **zero** `REFERENCES` edges, so all 2,948 pair rows diverge;
1,985 are otherwise publishable. This is missing infrastructure, not evidence
that the 6,682 links are semantically wrong.

**Recommendation and cost.** Materialize the edges once from validated
`name:` links, make the link-normalization/resolution transaction the sole edge
writer, then derive the export list from those edges. The read cost is the
out-degree already represented by the list; sorting by target id makes output
deterministic. If documentation ordering is considered meaningful, preserve it
as an edge property rather than as an independent list. This is the largest
name blast radius in the review: 2,948 to migrate, 1,985 publishable.

### `deprecates` / `superseded_by` ↔ `HAS_PREDECESSOR` /
`HAS_SUCCESSOR` — DERIVE

**Pair and authority.** `_write_standard_name_edges` creates the lineage edges
from either scalar vocabulary (`graph_ops.py:2936-2965`, `3046-3057`,
`3209-3230`). Multi-hop consumers already use `HAS_SUCCESSOR`, for example the
manifest terminal-identity traversal (`graph_ops.py:12287-12301`).

**Which copy gates work.** Catalog serialization and fold revival inspect the
scalar summary (`edit.py:2085-2104`); terminal traversal reads the edge. Six
current `superseded_by` rows have no matching successor edge, all terminal, and
there are no populated predecessor pairs. The earlier census separately found
519 tombstones with neither scalar nor `REFINED_FROM` successor evidence; that
is an absent-authority problem rather than a scalar/edge disagreement.

**Recommendation and cost.** Once the six missing edges are repaired, derive
the one-step catalog fields from the degree-one edges. The same edge supports
arbitrary-depth traversal, so keeping a second one-step scalar buys no query
the graph cannot express cheaply. Current publishable blast radius is zero.

## Review-authority mirrors

### Axis review projections ↔ `HAS_REVIEW` records — DERIVE

**Pair and authority.** The name/docs bundles
`reviewer_score_*`, `reviewer_scores_*`, `reviewer_comments_*`,
`reviewer_comments_per_dim_*`, `reviewer_model_*`, and the name-axis
`review_resolution_method` duplicate the selected
`StandardNameReview` record or group. `write_reviews` persists the complete
record and edge (`graph_ops.py:5750-5886`). The relationship record carries the
axis, group, cycle, canonical flag, resolution role/method, score, rubric,
comments, model, time, and cost, so the node projection adds no review fact.

**Which copy gates work.** Name review/refine and stranded promotion read
`reviewer_score_name` (`graph_ops.py:16102-16113`, `11970-11976`,
`17410-17420`); docs generation/refinement and stranded promotion read
`reviewer_score_docs` (`graph_ops.py:24020-24055`, `24352-24359`,
`11979-11986`). Only full documentation export reads the winning
`HAS_REVIEW` traversal (`export.py:504-516`, `552-558`). The cause of drift is
visible in the writer: `persist_reviewed_name` commits the scalar/stage bundle
first (`graph_ops.py:16635-16677`), then writes the Review in a later call and
explicitly swallows failure (`graph_ops.py:16797-16861`); the docs path has the
same split (`graph_ops.py:17171-17194`, `17231-17287`).

The live supplement found six name-score projections without a name-axis edge
and two docs-score projections without a docs-axis edge. The earlier census
places the six name-stage cases in the otherwise-publishable set; the two docs
score cases are terminal. This is the direct, measured instance of “the scalar
gates while its authority failed to persist.”

**Recommendation and cost.** Select the current canonical/winning review group
once in the shared claim/fetch query and project its fields on read. The cost is
bounded by the review out-degree; the live corpus has 27,136 review records over
5,048 names, so this replaces several unverified scalar reads with one already
required adjacency/group selection. During migration, make review record + edge
the first atomic write and stop updating the node bundle. Blast radius: six
live name authorities and two terminal docs authorities are provably absent;
all scalar readers listed above then change together.

### `review_count`, `review_mean_score`, `review_disagreement` ↔ review
aggregate — KEEP AND RECONCILE

**Pair and authority.** These three are exact aggregates of attached
`StandardNameReview` nodes. `update_review_aggregates` is already the single
obvious writer and computes count, mean, and score spread from `HAS_REVIEW`
(`graph_ops.py:5996-6029`).

**Which copy gates work.** Vocabulary promotion gates on the stored mean
(`vocab_promotion.py:93-123`); no corresponding path computes the aggregate
inline. The supplement recomputed the writer formula at its current 0.2 spread
threshold and found 59 divergent names, none otherwise publishable.

**Recommendation and cost.** Keep the indexed mean because the promotion query
must filter every supporting name before grouping by grammar token; computing a
nested review aggregate for every candidate would increase that scan. Make
`update_review_aggregates` run in the same transaction as review-group terminal
persistence and retain an edge-to-projection reconcile. Current repair blast
radius: 59 names.

### `reviewed_name_at` / `reviewed_docs_at` ↔ review timestamps — DERIVE

**Pair and authority.** Each timestamp duplicates `reviewed_at` on the selected
axis review. Structural derived parents are a deliberate alternative name
authority and use `HAS_STRUCTURAL_AUTHORITY`, not a Review.

**Which copy gates work.** `reviewed_name_at IS NULL` admits name review and a
non-null value admits docs review persistence (`graph_ops.py:16102-16113`,
`6596-6651`). Review edges do not participate in either gate. The earlier
census found 179 name timestamps without a name Review, but 174 are supported
by structural authority; the true unsupported count is five, all otherwise
publishable. It also found 72 name edges without a timestamp, two docs
timestamps without a docs edge, and 36 docs edges without a timestamp.

**Recommendation and cost.** Replace the timestamp gate with existence of a
name-axis review or signed structural authority, and project `max(reviewed_at)`
when a timestamp is needed. For docs, project the selected docs group's time.
The cost is one bounded adjacency existence/maximum. Blast radius is five false
name timestamps, 72 missing name projections, two false docs timestamps, and
36 missing docs projections.

### `name_stage` review states ↔ name review/attempt/lineage evidence —
KEEP AND RECONCILE

**Pair and authority.** `name_stage` is not wholly derivable: `pending`,
`drafted`, `refining`, and `contested` are queue/claim states. Its terminal
assertions are evidence projections: reviewed/accepted require a name review,
exhausted requires a below-threshold review plus spent `refine_attempts`, and
superseded requires successor lineage. The review persistence function is the
obvious owner of reviewed/accepted/exhausted (`graph_ops.py:16341-16359`);
refine/fold owns supersession.

**Which copy gates work.** Name review requires `name_stage='drafted'`
(`graph_ops.py:16130-16139`), refinement requires `reviewed`
(`graph_ops.py:17410-17420`), and export admits accepted/approved
(`export.py:502-507`, `537-550`). The scalar wins in all three. Census:
six accepted non-derived rows have no name review; 44 exhausted rows have no
durable attempt count (259 lack the obsolete success counter); two exhausted
rows have no review; and 519 superseded rows have no successor evidence. The
five valid exhausted names have nonzero attempt authority, so the legacy
259-row observation must not size a current repair.

**Recommendation and cost.** Keep the indexed queue scalar, but make each
terminal transition atomic with its review, attempt, or lineage receipt and
run a receipt-to-stage reconcile before claims. Blast radii by transition are
six accepted, 44 unsupported exhausted by current counter, two exhausted
without review (overlapping the 44 is possible), and 519 unsupported
superseded. Only the six accepted cases are otherwise publishable today.

### `docs_stage` review states ↔ docs generation/review evidence — KEEP AND
RECONCILE

**Pair and authority.** Pending/drafted/refining remain queue states; reviewed,
accepted, and exhausted are projections of docs-axis review and chain evidence.
`persist_reviewed_docs` is the obvious transition owner
(`graph_ops.py:16971-17014`, `17171-17194`).

**Which copy gates work.** Generation reads pending, review reads drafted, and
refinement reads reviewed (`graph_ops.py:24002-24009`, `16918-16924`,
`24352-24359`). Full export deliberately ignores accepted and reads the winning
review traversal (`export.py:552-558`). Census: 27 otherwise-publishable names
say accepted without a winning review (22 have no docs review at all); one
reviewed row and three exhausted rows lack a docs review; and two rows carry the
out-of-vocabulary value `superseded`.

**Recommendation and cost.** Keep the queue scalar, but persist a review edge
and stage transition atomically and reconcile terminal stage from the winning
review/chain. Current repair blast radius: 27 publishable-risk accepted rows,
one reviewed row, three exhausted rows, and two invalid terminal values.

### Quorum-shortfall markers ↔ non-winning review + timestamp — KEEP AND
RECONCILE

**Pair and authority.** `review_quorum_shortfall` and
`docs_review_quorum_shortfall` contain a reason that a relationship alone does
not necessarily preserve; their `*_at` stamps and axis-matching non-winning
Review corroborate the state.

**Which copy gates work.** The name marker gates export, stranded promotion,
and name refinement (`export.py:540-550`; `graph_ops.py:11970-11976`,
`17429-17432`); the docs marker gates full export and docs refinement
(`export.py:552-558`; `graph_ops.py:11979-11986`, `24358-24359`). The census
found 126 name and ten docs markers and zero contradictions.

**Recommendation and cost.** Keep the reason projection, write it atomically
with its timestamp and non-winning Review, and reconcile/clear from the review
group outcome. The repair blast radius is zero today; this is a preventive
single-writer closure.

### `chain_length` ↔ `REFINED_FROM` depth — DERIVE

**Pair and authority.** The schema defines `chain_length` as the depth of the
persisted `REFINED_FROM` lineage (`standard_name.yaml:1283-1289`). The edge path
therefore already contains the entire fact.

**Which copy gates work.** `refine_attempts` is the primary budget authority,
but `REFINE_NAME_ATTEMPTS_SPENT` falls back to `chain_length` when the attempt
counter is absent (`graph_ops.py:17399-17420`), and review exhaustion reads the
same fallback (`graph_ops.py:16416-16424`). The live maximum-depth comparison
found 735 disagreements, 180 otherwise publishable.

**Recommendation and cost.** Complete `refine_attempts` coverage, remove the
budget fallback, and derive display/history depth from the bounded
`REFINED_FROM` path. The current rotation cap bounds newly written chains, so
the normal cost is a handful of edge hops; defensive readers should still cap
the traversal and report a cycle. Blast radius is 735 names.

### `docs_chain_length` ↔ `DOCS_REVISION_OF` count — DERIVE

**Pair and authority.** Each docs refinement snapshots the prior content in a
`DocsRevision` and increments the scalar. The harmonization path already takes
the maximum of stored revision ids and the scalar before choosing a new id
(`graph_ops.py:24636-24674`), acknowledging that the history is stronger than
the counter.

**Which copy gates work.** Docs review chooses exhausted from
`docs_chain_length` (`graph_ops.py:17055-17094`) and docs refinement admits only
rows below the scalar cap (`graph_ops.py:24320-24359`). Neither gate counts
`DOCS_REVISION_OF`. The live exact-count comparison found 1,579 disagreements,
1,181 otherwise publishable.

**Recommendation and cost.** Derive the spent depth with a degree count in the
claim/stage query and use `max(revision_number)+1` for ids. There are fewer
DocsRevision edges than names in the current graph, so this aggregation is
bounded and avoids a second counter. Blast radius is 1,579 names and is urgent:
unlike a display-only drift, it changes whether another paid refinement is
allowed.

## Lifecycle and provenance mirrors

### `validation_status` ↔ validation receipt — KEEP AND RECONCILE

**Pair and authority.** A completed validation writes `validated_at`, issues,
layer summary, and status atomically in `mark_names_validated`
(`graph_ops.py:8035-8074`). The status is the indexed outcome; the other fields
are its receipt. Compose also writes a provisional audit status without the
receipt (`graph_ops.py:5296-5330`), so the current writers do not implement one
meaning.

**Which copy gates work.** Validation claims on `validated_at IS NULL`
(`graph_ops.py:7994-8008`), while review and export gate on
`validation_status='valid'` (`graph_ops.py:16130-16139`; `export.py:537-550`).
The scalar and timestamp therefore admit opposing operations on the same row.
The census found 538 valid assertions with no timestamp, 169 otherwise
publishable; 594 quarantined assertions lack a timestamp and 242 of those also
lack issue/reason evidence.

**Recommendation and cost.** Keep the indexed status, but reserve `valid` and
`quarantined` for the receipt-bearing validator writer; composition should
leave it pending. Make `mark_names_validated` the sole terminal writer and add
a receipt-to-status reconcile. Blast radius: 538 valid plus 594 quarantined
rows, with a 169-name direct publishable intersection.

### `link_status` ↔ link targets and resolver receipt — KEEP AND RECONCILE

**Pair and authority.** Resolved means every `name:` link has a present,
non-terminal target and the resolver has recorded `link_checked_at`; failed or
unresolved also depend on retry state. `resolve_links_batch` is capable of
writing the list, status, timestamp, and retry count together
(`graph_ops.py:9147-9246`).

**Which copy gates work.** Claims read only `link_status='unresolved'` and the
retry count (`graph_ops.py:9103-9128`). `_compute_link_status` and docs cleanup
can stamp resolved without checking target existence
(`graph_ops.py:9449-9579`, `9636-9659`). The census found all 2,821 resolved
rows lack `link_checked_at`; 328 contain an absent or terminal target and 39 of
those are otherwise publishable.

**Recommendation and cost.** Keep the status as a retry/work-queue projection,
but make `resolve_links_batch` the only writer of `resolved` or `failed` and
have compose/enrichment write `unresolved` when links exist. Reconcile resolved
against target existence and terminal state. Current semantic repair blast
radius is 328 names; receipt backfill/forced recheck covers all 2,821.

### catalog `status` ↔ approval and successor receipts — KEEP AND RECONCILE

**Pair and authority.** This is distinct from `name_stage`: it is the catalog
lifecycle chosen by the live plan. `active` is earned by complete merged-PR
metadata and its `StandardNameChange`; `superseded` is earned by successor
lineage; draft is the pre-publication default. `_approve_one` atomically writes
active, the PR fields, and the change edge (`promote.py:1637-1673`).

**Which copy gates work.** Export excludes `status='superseded'` at the
population boundary (`export.py:470-484`) and approval accepts only draft
(`promote.py:1648-1661`). The scalar is therefore the gate; the receipt is not
re-read. The census found 34 superseded statuses against a non-superseded
pipeline stage, 31 otherwise publishable, and 519 superseded statuses without
successor evidence. There are no active/deprecated rows and two nulls.

**Recommendation and cost.** Keep the catalog scalar as the indexed export
gate, preserve `_approve_one` as the sole active writer, and make
`reconcile_catalog_status` the single repair owner for draft/superseded
(`graph_ops.py:13598-13666`). Its current semantic blast radius is 34
stage-disagreements plus 519 unsupported tombstones (the sets may overlap),
with 31 directly suppressed publishable names.

### `origin` ↔ source/change provenance — DERIVE

**Pair and authority.** The scalar's schema meaning is “most recent editorial
edit,” but the current reconcile describes it as how the identity entered the
graph. `StandardNameSource`/`PRODUCED_NAME` can answer source origin, and
`StandardNameChange.origin` plus catalog receipts can answer the provenance of
a particular edit. One scalar cannot truthfully answer both.

**Which copy gates work.** Name review excludes `origin='derived'`
(`graph_ops.py:16130-16139`), and derived-parent admission still uses derived or
legacy catalog-origin tests; automatic catalog protection now correctly gates
on `name_stage='approved'`, not origin (`protection.py:203-216`). The prior
2,096 false catalog-edit assertions have been reconciled to zero. The census
also found 16 derived scalars without a derived-source edge, all structurally
supported by an incoming child, and 951 pipeline scalars without a DD/signal
producer; every pipeline assertion lacks an immutable last-editor receipt.

**Recommendation and cost.** Retire the stored origin, in agreement with the
live plan decision. Ask the exact provenance question on read: source type from
`PRODUCED_NAME`, structural derivation from `HAS_PARENT`/structural authority,
and editorial provenance from the applicable change/approval receipt. Each is
a one-hop existence or latest-event lookup. Historical repair blast radius is
already zero; reader migration spans the remaining 3,093 populated/non-null
origin rows (2,657 pipeline plus 436 derived in the census), without inventing
one replacement value for two meanings.

### `StandardNameSource.status` ↔ state-specific evidence — KEEP AND
RECONCILE

**Pair and authority.** This indexed state machine is not reducible to one edge:
composed/attached require `produced_sn_id`, `PRODUCED_NAME`, and composition
time; vocabulary-gap requires `HAS_STANDARD_NAME_VOCAB_GAP` and an error;
failed requires failure time/error/attempt; skipped requires a reason; stale
requires missing upstream authority.

**Which copy gates work.** Composition claims only `status='extracted'`
(`graph_ops.py:10151-10204`, `10207-10250`), so a false terminal value suppresses
the only producer. `reconcile_source_status_liveness` is the shared
edge-to-state repair for produced targets (`graph_ops.py:12342-12482`), while
`reconcile_standard_name_sources` owns upstream stale/revive transitions
(`graph_ops.py:12101-12222`). Census: 398 skipped live-upstream sources lack a
reason, 59 failed live-upstream sources lack complete failure evidence, and
three extracted sources already have a live target. Separate target-cardinality
measurement found 117 produced-id/edge disagreements.

**Recommendation and cost.** Keep the indexed state for the 9,900-row work
queue, but route all transitions through one state-transition writer that
requires the state-specific receipt and invokes the existing liveness
reconciles. Immediate exclusion blast radius is 457 live-upstream terminal
rows; duplicate-work radius is three extracted rows, plus 117 target-mirror
repairs before the produced states can be trusted.

### `compose_hint_status` ↔ hint timestamps and produced target — KEEP AND
RECONCILE

**Pair and authority.** Open requires a hint, reason, request time, and no live
binding; consumed requires consumption time and a produced target. Rejected has
no timestamp or immutable receipt and is therefore not presently auditable.

**Which copy gates work.** Exact-source steering refuses an existing open
scalar after separately checking the live edge (`graph_ops.py:9692-9731`), and
finalizers consume the status with the produced binding
(`graph_ops.py:7356-7374`, `7418-7429`). The census found one open hint already
bound to live accepted/valid
`vertical_coordinate_of_pellet_path_point`; all other 71 non-null states carry
their required evidence.

**Recommendation and cost.** Keep the explicit status so rejection can remain
a future state and exact-id admission stays cheap. Make `set_source_compose_hint`
the sole opener, the atomic source finalizer the sole consumer, add a rejection
receipt before that enum value is used, and reconcile open+bound to consumed.
Current repair blast radius is one source.

## Apparent duplicates that are distinct

These five groups are deliberately **KEEP AS DISTINCT**. They are included so
the migration does not “fix” them by deleting one side.

### Primary `physics_domain` versus domain-set edge

`physics_domain` is the single highest-ranked home domain; `source_domains` and
the `HAS_PHYSICS_DOMAIN` edges are the complete contributing set
(`graph_ops.py:3251-3274`, `5416-5497`). Search intentionally tests the primary
or set membership. Live shape: 4,818 names have a primary scalar, 3,189 have at
least one domain edge, 201 have multiple edges, and the primary is among the
edges on 3,145. Keep the primary distinct, but require it to be a member of the
reconciled full set; the 1,654 exact set disagreements are owned by the
`source_domains` mirror above.

### Grammar segment columns versus grammar-token edges

The flat columns (`physical_base`, `subject`, `transformation`, `component`,
`coordinate`, `process`, `position`, `region`, `device`, `geometric_base`,
`aggregation`, `orbit`, `population`, plus non-edge `state`, `object`, and
`geometry`) are the strict parser projection. Typed `HAS_*`/`HAS_SEGMENT` edges
exist only when the value is in a synced closed-vocabulary `GrammarToken`.
`_write_grammar_decomposition` explicitly writes columns unconditionally and
edges conditionally (`graph_ops.py:6838-6857`, `6930-7007`). A missing edge for
an open-vocabulary value is therefore not drift. Vocabulary promotion reads
the scalar segment and its review-mean gate (`vocab_promotion.py:104-123`);
grammar-neighborhood queries may traverse edges. Keep both and retain this one
parser-owned writer; do not reconcile an intentionally absent token edge by
inventing a `GrammarToken`.

### `refine_attempts`, `refine_name_count`, and persisted lineage

These are three different counters: claimed attempts (including refused or
unpersisted results), actual refinement LLM calls, and persisted successors.
The budget correctly gates on `refine_attempts` with `chain_length` only as a
legacy fallback (`graph_ops.py:17399-17420`). The earlier 259 exhausted rows
with zero `refine_name_count` do not prove 259 unspent budgets; only 44 lack the
durable attempt evidence. Keep attempt and call counters distinct, remove the
lineage-depth fallback after coverage is complete, and derive `chain_length`
as recommended above.

### `edit_status` / `edit_origin` versus the edit request envelope

The request envelope proves a steer was attached; it does not prove whether a
later review applied, exhausted, or rejected it, and actor category is not an
identity/signature receipt. The scalar `edit_status='open'` gates edit-only
claims and prevents bare stranded promotion (`graph_ops.py:11970-11976`,
`15371-15443` in the census revision). Census: all 1,065 non-null origins lack
an actor receipt, two human origins lack an edit status, and all 883 terminal
statuses lack an immutable outcome event. Keep the workflow and claimed actor
category distinct from the request timestamp; add immutable outcome/actor
receipts before calling either corroborated. No proven false-terminal count is
available today, so this is an auditability blast radius of 883 outcomes and
1,065 origins rather than a contradiction count.

### `superseded_from_stage` versus successor lineage

This scalar records a historical stage for possible fold revival; successor
edges record identity lineage. They are not the same fact. Fold revival reads
the scalar and falls back to drafted (`edit.py:2008-2029` in the census
revision). Of 1,222 non-null values, 39 have no successor evidence and 568 say
the prior state was refining, which cannot safely be restored without a claim.
Keep it distinct only if an immutable change event records the prior stage;
until then treat the entire 1,222-row population as uncorroboratable restoration
input, not as a relationship mirror that can be recomputed.

`StandardName.docs_review_admission` is not an additional scalar mirror. It is
the relationship slot itself, and the target id is stored on the
`DocsReviewAdmission` authority node. Likewise timestamps such as
`updated_at`, embedding freshness hashes, harmonization signatures, and LLM
cost/call counts record events or payload freshness rather than duplicating a
relationship fact; they are outside this mirror inventory.

## Supplemental predicates and reproducibility

All list comparisons were set comparisons in both directions:
`any(x IN scalar WHERE NOT x IN edges) OR any(x IN edges WHERE NOT x IN
scalar)`. Cardinality-one pairs used
`NOT (size(edges) = 1 AND edges[0] = scalar)`. Null-only coverage was not called
a contradiction unless the relationship side carried the asserted fact.

The exact pair projections were:

| Pair | Scalar projection | Relationship projection / divergence qualification |
|---|---|---|
| unit | `sn.unit` | `[(sn)-[:HAS_UNIT]->(u:Unit) \| u.id]`; compare whenever either side is populated |
| COCOS | `toString(sn.cocos)` | `[(sn)-[:HAS_COCOS]->(c:COCOS) \| toString(c.id)]`; compare whenever either side is populated |
| primary cluster | `sn.primary_cluster_id` | `[(sn)-[:IN_CLUSTER]->(c:IMASSemanticCluster) \| c.id]`; scalar must occur in the cardinality-one edge list |
| successor | `sn.superseded_by` | `[(sn)-[:HAS_SUCCESSOR]->(s:StandardName) \| s.id]`; exact cardinality-one comparison |
| predecessor | `sn.deprecates` | `[(sn)-[:HAS_PREDECESSOR]->(p:StandardName) \| p.id]`; exact cardinality-one comparison |
| references | `[x IN coalesce(sn.links,[]) WHERE x STARTS WITH 'name:' \| substring(x,5)]` | `[(sn)-[:REFERENCES]->(t:StandardName) \| t.id]`; bidirectional set difference |
| produced target | `sns.produced_sn_id` | `[(sns)-[:PRODUCED_NAME]->(sn:StandardName) \| sn.id]`; exact cardinality-one comparison |
| DD upstream | `coalesce(sns.dd_path,sns.source_id)` for `source_type='dd'` | `[(sns)-[:FROM_DD_PATH]->(n:IMASNode) \| n.id]`; exact cardinality-one comparison |
| signal upstream | `coalesce(sns.signal,sns.source_id)` for `source_type='signals'` | `[(sns)-[:FROM_SIGNAL]->(n:FacilitySignal) \| n.id]`; exact cardinality-one comparison |
| source paths | non-`derived:` values from `sn.source_paths` | `'dd:' + imas.id` over inbound `HAS_STANDARD_NAME`, union non-derived source ids over inbound `PRODUCED_NAME`; bidirectional set difference |
| source types | `coalesce(sn.source_types,[])` | producer `source_type`, plus `dd`/`signals` for direct inbound projections; deduplicate then compare sets |
| source domains | `coalesce(sn.source_domains,[])` | `[(sn)-[:HAS_PHYSICS_DOMAIN]->(d:PhysicsDomain) \| d.id]`; bidirectional set difference |
| review aggregates | stored count, mean, and disagreement | `count(r)`, `avg(r.score)`, and `count(r)>1 AND max(r.score)-min(r.score)>=0.2` over outbound `HAS_REVIEW` |
| axis review support | any non-null score/time for an axis | absence of `HAS_REVIEW` to a Review with the same `review_axis`; structural authority was then applied as the documented name-axis alternative |
| name lineage depth | `coalesce(sn.chain_length,0)` | `coalesce(max(length((sn)-[:REFINED_FROM*1..]->(:StandardName))),0)` |
| docs lineage depth | `coalesce(sn.docs_chain_length,0)` | `size([(sn)-[:DOCS_REVISION_OF]->(r:DocsRevision) \| r])` |

The supplemental otherwise-publishable intersection was exactly:

```cypher
sn.name_stage IN ['accepted', 'approved']
AND coalesce(sn.status, '') <> 'superseded'
AND sn.validation_status = 'valid'
AND sn.review_quorum_shortfall IS NULL
AND sn.docs_review_quorum_shortfall IS NULL
AND docs_review_eligibility_where()
```

Here `docs_review_eligibility_where()` was expanded from the code at
`graph_ops.py:6371-6412` and used its production winning/non-winning method
parameters. Every query returned only aggregate counts; diagnostic probes used
`collect({...})[0..10]`. No live query exceeded 1.882 seconds; the slower domain
set comparison remained well inside the login-node exception.

## Conclusion

The most important distinction for implementation is **authority before
reconciliation**. For exact mirrors, select the edge/event first and derive or
repair the scalar from it. For indexed workflow projections, make the receipt
and scalar one atomic transition and reconcile only in that direction. Never
let a stale scalar recreate an edge when more than one live edge exists: the
100 multi-target sources prove that recovery direction can turn ambiguity into
a silent identity choice.

This sequencing retires the mechanism, not just today's rows. The first four
name-facing migrations cover 2,948 reference mirrors, 3,549 source-type
mirrors, 1,579 docs-depth mirrors, and 1,654 domain-set mirrors. The most urgent
state-gate repairs then cover 538 unstamped valid assertions, 328 false resolved
links, 34 contradictory catalog tombstones, and 27 unsupported docs
acceptances. Counts overlap and must not be summed into a unique-identity total.
