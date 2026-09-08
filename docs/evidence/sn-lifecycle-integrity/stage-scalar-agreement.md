# Lifecycle scalar agreement census

Measured 2026-09-08 against code revision
`1d7104d9c734c7a03e18f019b5340e31820c1d10` and the live Standard Names graph.
This is a read-only census: it applies no repair, changes no graph row, and runs
no pipeline operation.

## Method

A contradiction is counted only when a stored lifecycle or provenance scalar
asserts a state for which the code-defined corroborating relationship, receipt,
counter, or timestamp is absent. Each query is bounded to one label and one
scalar, returns grouped counts rather than graph entities, and avoids Cartesian
products. Queries ran through the login-node-local Neo4j tunnel because compute
nodes cannot reach that endpoint; any query taking ten seconds would have been
stopped and reported.

For each scalar, the census records:

1. the operation gated by the scalar and the code predicate that performs the
   gate;
2. the evidence that earns the asserted state, or an explicit statement that
   no independent corroboration exists;
3. the exact bounded Cypher predicate and live contradiction count, split by
   whether affected rows are otherwise publishable.

"Otherwise publishable" means that the row clears the other currently
independent export gates after ignoring the scalar under examination:
`name_stage IN ['accepted', 'approved']`, `validation_status = 'valid'`, no
name/docs quorum-shortfall marker, no catalog tombstone, and (for a full export)
a reachable winning docs-axis review. For a source row, the corresponding risk
classification is instead whether a live upstream entity is being withheld from
composition. Counts are identity counts unless explicitly labelled as source
counts.

## `StandardName.name_stage`

**Gated work.** The name review pool requires
`sn.name_stage = 'drafted'` (`graph_ops.py:16131-16135`), name refinement uses
the shared `REFINE_NAME_ELIGIBILITY_WHERE` predicate beginning with
`sn.name_stage = 'reviewed'` and excluding `superseded`, `exhausted`, and
`contested` (`graph_ops.py:17396-17421`), and export admits only `accepted` or
`approved` (`export.py:547-550`). A false terminal value therefore suppresses
review/refinement and publication; a false live value can admit work that its
evidence has not earned.

**Corroborating evidence.** `reviewed` and `accepted` require a name-axis
`HAS_REVIEW` record and its score; `approved` requires the PR number, URL,
merge SHA, approval timestamp, and `status='active'` written atomically by
`_approve_one` (`promote.py:1649-1661`). `refining` requires an active claim.
`exhausted` requires a below-threshold review after the refinement budget has
actually been spent; the current authority is `refine_attempts`, while
`refine_name_count` is an older persisted-success counter. `superseded`
requires either an incoming `REFINED_FROM` edge or the catalog
`superseded_by` receipt. `drafted` is written with `generated_at` by the
transactional compose finalizer. `pending` is only a queue marker and has no
independent corroborating relationship or receipt, so its 12 assertions cannot
be audited independently.

**Live census (5,048 StandardName rows).** Distribution: 2,328 accepted,
2,134 superseded, 274 exhausted, 181 drafted, 119 reviewed, and 12 pending;
there are no approved or refining rows.

- 519 superseded identities have neither `superseded_by` nor an incoming
  `REFINED_FROM` edge; 414 are otherwise `validation_status='valid'`. Predicate:
  `sn.name_stage='superseded' AND sn.superseded_by IS NULL AND NOT EXISTS {
  MATCH (:StandardName)-[:REFINED_FROM]->(sn) }`. Examples include
  `time_derivative_of_poloidal_magnetic_flux_at_constant_toroidal_flux`,
  `poloidal_diffusion_coefficient`, and `parallel_total_current`.
- 259 of the 274 exhausted identities have
  `coalesce(sn.refine_name_count,0)=0`; five of those 259 are valid. The newer
  attempt authority changes the interpretation: only 44 have
  `coalesce(sn.refine_attempts,0)=0`, and two exhausted rows have no name-axis
  review at all. All five valid exhausted rows have non-zero
  `refine_attempts`. Four score at least 0.65 (the release-export default), but
  none reaches 0.85 (the current review default). Exact predicates:
  `sn.name_stage='exhausted' AND coalesce(sn.refine_name_count,0)=0`, and the
  same predicate with `refine_attempts`; the score check adds
  `sn.validation_status='valid' AND
  coalesce(sn.reviewer_score_name,-1.0) >= <threshold>`. The five valid rows are
  `derivative_of_area_of_flux_surface_with_respect_to_normalized_poloidal_flux_coordinate`
  (0.6125), `gap_at_plasma_boundary` (0.8125), `gap_of_antenna_strap`
  (0.6625), `neutron_rate` (0.7375), and `width_of_antenna_strap` (0.8).
- Six accepted, non-derived draft identities have no name-axis `HAS_REVIEW`
  edge; all six are valid. Predicate:
  `sn.name_stage='accepted' AND coalesce(sn.origin,'') <> 'derived' AND
  coalesce(sn.status,'') <> 'active' AND NOT EXISTS { MATCH
  (sn)-[:HAS_REVIEW]->(r:StandardNameReview) WHERE r.review_axis='name' }`.
  They are the x/y/z first/second measurement-direction unit vectors of a
  strain gauge.
- Eight drafted identities have no `generated_at`; none is valid. Predicate:
  `sn.name_stage='drafted' AND sn.generated_at IS NULL`. Examples are
  `toroidal_width_of_antenna_strap`, `neutral_pressure`, and
  `hard_xray_emissivity`.
- Zero reviewed rows lack a score or name-axis review; zero approved rows lack
  their complete catalog receipt; zero refining rows lack a live claim.

The earlier 259/274 `refine_name_count` observation therefore reproduces, but
it is not the whole current contract: `refine_attempts` is now the gate's
authority and reduces the unsupported-exhaustion count to 44 (two with no
review). The report retains both counts because a migration sized only from the
legacy counter would overstate current refusal by 215 rows.

## `StandardName.docs_stage`

**Gated work.** Documentation generation requires
`sn.name_stage='accepted' AND sn.docs_stage='pending'`
(`graph_ops.py:24037`); docs review requires `docs_stage='drafted'`
(`graph_ops.py:16905-16909`); docs refinement requires
`docs_stage='reviewed'`, a below-threshold score, remaining rotation budget,
and a winning review (`graph_ops.py:24337-24348`). The graph fetch for full
export now uses `docs_review_eligibility_where()` rather than trusting the
scalar (`export.py:552-558`), but the export classifier still reports
`documentation_not_accepted` when its projected `docs_stage` is not accepted
(`export.py:795-797`). The relationship gate prevents a false accepted scalar
from publishing; the scalar can still suppress the review/refine pools and
misclassify the exclusion reason.

**Corroborating evidence.** `drafted` requires documentation plus
`docs_generated_at`; `reviewed` requires a docs-axis review and score;
`accepted` requires a reachable review group whose method is one of
`quorum_consensus`, `authoritative_escalation`, or `single_review` (the shared
predicate at `graph_ops.py:6410-6412`). `exhausted` requires a docs review,
below-threshold score, and non-zero `docs_chain_length`. `pending` is described
by the schema as earned when the name becomes accepted, although the compose
finalizer currently initializes it earlier; that conflict means the scalar is
not independently reliable as an acceptance receipt.

**Live census.** Distribution: 3,092 accepted, 1,820 pending, 89 drafted, 30
reviewed, 12 exhausted, three null, and two values literally equal to
`superseded`, which is outside the documented docs-stage vocabulary.

- 27 otherwise-publishable accepted rows have no reachable winning docs
  review. Of these, 22 have no docs-axis `HAS_REVIEW` edge and every docs-review
  projection is null; the other five have only non-winning review evidence.
  Predicate: `sn.docs_stage='accepted'` plus the other export gates and
  `NOT EXISTS { MATCH (sn)-[:HAS_REVIEW]->(r:StandardNameReview) WHERE
  r.review_axis='docs' AND r.resolution_method IN
  ['quorum_consensus','authoritative_escalation','single_review'] }`. Examples
  without any docs review include
  `radiated_energy_accumulated_due_to_impurity_radiation`,
  `magnetic_field_magnitude_at_pedestal_top_high_field_side`, and
  `poloidal_ion_velocity_at_charge_exchange_channel`.
- Across all lifecycle states, 464 accepted-docs scalars have no docs-axis
  review and 472 have no winning review. Most are already-retired name
  identities; the 27-row publishability intersection above is the batch-risk
  count. The earlier nine-row finding has therefore moved to 22 strict
  no-review rows in the current publishability intersection.
- One reviewed row lacks a docs score or docs review; three exhausted rows
  lack a docs review. No drafted row lacks generation evidence and no exhausted
  row has zero `docs_chain_length` or a valid score at/above 0.85.
- 1,787 pending docs rows do not yet have an accepted/approved name. Predicate:
  `sn.docs_stage='pending' AND NOT (sn.name_stage IN
  ['accepted','approved'])`. This is a schema/writer disagreement rather than a
  present work-admission defect because the generation predicate also checks
  the name stage.
- `flux_due_to_thermal_fusion` and `lower_energy` carry the invalid
  `docs_stage='superseded'`; both names are themselves superseded, so neither
  currently shrinks a publishable batch.

## `StandardName.validation_status`

**Gated work.** Full export requires `sn.validation_status='valid'`
(`export.py:540,549`); name review uses the same value
(`graph_ops.py:16131-16135`), and enrichment/consolidation also select it.
`claim_names_for_validation` instead selects the absence of the corroborating
timestamp, `sn.validated_at IS NULL` (`graph_ops.py:8001-8007`). A false
`valid` value can publish an unvalidated row, while a false `quarantined` value
silently suppresses every downstream pool.

**Corroborating evidence.** The validator's atomic mark writes
`validated_at`, `validation_issues`, `validation_layer_summary`, and
`validation_status` together (`graph_ops.py:8061-8068`). That is the strongest
independent receipt available. Compose also writes a provisional status from
inline audits without `validated_at`; therefore a missing timestamp proves the
full validation receipt is absent, not necessarily that the provisional audit
verdict is semantically wrong.

**Live census.** Distribution: 4,428 valid, 599 quarantined, 18 pending, and
three null.

- 538 valid assertions have `validated_at IS NULL`; 169 clear every other full
  export gate, so the scalar is the only validation authority on rows that can
  reach a catalog batch. Predicate: `sn.validation_status='valid' AND
  sn.validated_at IS NULL`, with the publishability predicate from Method.
  Examples include `perturbed_pressure_bessel_1`, `torque_density`, and
  `neutron_flux_due_to_fusion`.
- 594 quarantined assertions have no `validated_at`; 242 additionally have
  neither a non-empty `validation_issues` list nor `merge_quarantine_reason`.
  Predicates: `sn.validation_status='quarantined' AND sn.validated_at IS NULL`
  and `... AND size(coalesce(sn.validation_issues,[]))=0 AND
  sn.merge_quarantine_reason IS NULL`. These rows are already non-publishable,
  so their immediate batch cost is zero, but the unsupported terminal scalar
  prevents deterministic validation from being the sole gate.
- Zero pending assertions carry a completed validation timestamp; there are no
  invalid non-null enum values. Three nulls are schema-coverage gaps, not false
  assertions, and none is otherwise name-publishable.

## `StandardName.link_status`

**Gated work.** Link resolution claims only
`sn.link_status='unresolved'` with fewer than five retries
(`graph_ops.py:9117-9128`). A false `resolved` or `failed` value therefore
prevents the resolver from revisiting the row. The current export predicate
does not read `link_status`, so a false resolved value can pass through unless
the catalog validator independently rejects the links.

**Corroborating evidence.** A real resolver attempt writes `link_checked_at`,
increments `link_retry_count`, rewrites DD links, and sets the scalar in one
operation (`graph_ops.py:9220-9245`). Independently, every `name:` target must
exist and must not be in a terminal name stage. The lightweight
`_compute_link_status` writer classifies any non-`dd:` spelling as resolved
without checking the target (`graph_ops.py:469-480`), so it cannot serve as
independent corroboration.

**Live census.** 2,821 rows assert resolved and 2,227 are null; there are no
unresolved or failed values.

- All 2,821 resolved rows have `link_checked_at IS NULL`: none carries the
  resolver receipt. Predicate: `sn.link_status='resolved' AND
  sn.link_checked_at IS NULL`.
- 328 resolved rows have at least one objectively invalid `name:` target: 81
  point to an absent identity, 275 point to a superseded/exhausted/contested
  identity, and 28 rows contain both kinds. Thirty-nine of the 328 are
  otherwise name-publishable. Exact predicate:
  `sn.link_status='resolved' AND ANY(link IN coalesce(sn.links,[]) WHERE link
  STARTS WITH 'name:' AND (NOT EXISTS { MATCH (:StandardName
  {id:substring(link,5)}) } OR EXISTS { MATCH (target:StandardName
  {id:substring(link,5)}) WHERE target.name_stage IN
  ['superseded','exhausted','contested'] }))`. Examples include
  `source_rate_due_to_thermal_fusion` pointing to absent `neutron_power`, and
  `area_of_flux_surface` pointing at a terminal identity.
- Zero resolved rows contain a `dd:` link or an unsupported prefix; zero
  unresolved/failed assertions lack their state-specific evidence. Another
  313 rows have non-empty links and a null scalar; these are coverage gaps, not
  contradictory assertions.

## `StandardName.status`

**Gated work.** This is the catalog-vocabulary axis. The shared export
tombstone predicate excludes `status='superseded'` before the candidate
population is constructed (`export.py:470-484`), and catalog approval accepts
only draft rows before atomically setting `status='active'` with the merged-PR
receipt (`promote.py:1649-1661`). A false superseded status therefore makes a
row invisible even when its name, docs, and validation axes are publishable.

**Corroborating evidence.** `active` requires the catalog PR number, URL, merge
SHA, approval timestamp, and import observation; `superseded` requires the
same successor evidence as the name-stage tombstone. `draft` is a default
queue state and has no independent receipt. No code path presently writes
`deprecated` (`graph_ops.py:13610-13613`), so if that value appears its
retirement evidence must come from catalog lineage rather than this pipeline.

**Live census.** 2,878 draft, 2,168 superseded, and two null; no active or
deprecated rows exist.

- 34 `status='superseded'` rows have a non-superseded `name_stage`; 31 would be
  name-publishable if this status tombstone were ignored. Predicate:
  `sn.status='superseded' AND coalesce(sn.name_stage,'') <> 'superseded'`.
  Representative accepted/valid rows include
  `flux_surface_averaged_nitrogen_density_at_plasma_boundary`,
  `flux_surface_averaged_electron_density_at_plasma_boundary`, and
  `accumulated_carbon_count_due_to_gas_injection`.
- 519 superseded-status rows have neither `superseded_by` nor an incoming
  `REFINED_FROM` edge. Predicate: `sn.status='superseded' AND
  sn.superseded_by IS NULL AND NOT EXISTS { MATCH
  (:StandardName)-[:REFINED_FROM]->(sn) }`. This is the same unsupported
  tombstone population found through `name_stage`, not an additional 519.
- Zero draft rows carry approval evidence, and there are no active/deprecated
  assertions to contradict. The two null rows,
  `toroidal_ion_velocity_at_plasma_boundary` and `pressure_at_pedestal_top`,
  are accepted and valid; null is a coverage failure rather than an asserted
  false state, but it leaves their exported catalog status undefined.

## `StandardName.origin`

**Gated work.** At this revision, automatic editorial protection is keyed to
`name_stage='approved'`, not origin (`protection.py:203-216`). Origin still
affects work: `origin='derived'` is excluded from name review/refinement, and
`origin='catalog_edit'` or an accepted stage grants legacy protection in the
derived-parent admission gate (`graph_ops.py:2638-2640`). A false origin can
therefore preserve an inadmissible parent or suppress the ordinary name pools.

**Corroborating evidence.** `catalog_edit` requires a catalog request number,
approval timestamp, or approved stage; this is also the exact reconciliation
contract (`graph_ops.py:13234-13247`). `derived` requires structural child
evidence and should also have a derived `StandardNameSource`. `pipeline` has no
immutable event identifying the most recent editorial writer: a DD/signal
producer can corroborate how the identity entered the pipeline, but cannot
prove who last edited it. That value is therefore not fully auditable from the
present graph.

**Live census.** 2,657 pipeline, 436 derived, 1,955 null, and zero catalog_edit.

- The previously measured 2,096 false `catalog_edit` origins are now exactly
  zero under `sn.origin='catalog_edit' AND sn.catalog_pr_number IS NULL AND
  sn.catalog_approved_at IS NULL AND coalesce(sn.name_stage,'') <> 'approved'`.
  The current graph therefore confirms that repair.
- Sixteen derived assertions lack a derived-source `PRODUCED_NAME` edge, but
  all sixteen do have an incoming `HAS_PARENT` child and therefore retain
  structural corroboration; 12 are otherwise name-publishable. Examples are
  `beta`, `gas_flow`, and `neutron_flux`.
- 951 pipeline assertions lack a DD/signal `StandardNameSource` producer; 896
  of those also lack any direct IMASNode/FacilitySignal binding and 53 are
  otherwise name-publishable. More importantly, all 2,657 pipeline assertions
  lack an immutable "most recent editorial edit" receipt, so the scalar cannot
  be verified to the semantics its schema claims. Predicate for the weaker
  source-coverage check: `sn.origin='pipeline' AND NOT EXISTS { MATCH
  (src:StandardNameSource)-[:PRODUCED_NAME]->(sn) WHERE src.source_type IN
  ['dd','signals'] }`.

## `StandardNameSource.status`

**Gated work.** Composition claims only `status='extracted'`
(`graph_ops.py:10163-10176` and `10227-10231`); composed/attached sources are
treated as having produced a name, while failed, stale, skipped,
not-physical-quantity, and vocabulary-gap states are excluded. A false
terminal status therefore removes a source from the only operation that can
produce its catalog candidate. Conversely, a false extracted status can spend
composition work against an already-bound source.

**Corroborating evidence.** `composed` requires `composed_at`,
`produced_sn_id`, and a `PRODUCED_NAME` edge; `attached` requires the id mirror
and edge. `vocab_gap` requires `last_error` plus
`HAS_STANDARD_NAME_VOCAB_GAP`; `failed` requires `failed_at`, `last_error`, and
a spent attempt; skipped/not-physical states require `skip_reason`; and
`stale` requires the absence of the source-type-specific upstream entity. The
same upstream-presence predicate is centralized in
`_standard_name_source_upstream_present_cypher` (`graph_ops.py:12083-12100`).

**Live census (9,900 source rows at the final source query).** Distribution:
3,053 composed, 2,249 attached, 1,745 stale, 1,413 skipped, 1,327 extracted,
103 failed, ten not-physical-quantity, and zero vocabulary-gap.

- 398 skipped sources have no `skip_reason`; all 398 still have a live
  upstream entity. Predicate: `sns.status='skipped' AND sns.skip_reason IS
  NULL`, with the source-type upstream existence predicate. Examples include
  `dd:bolometer/grid/volume_element`, `dd:camera_visible/latency`, and
  `dd:camera_x_rays/exposure_time`. This is direct candidate-loss risk: the
  terminal scalar suppresses composition and carries no reason for doing so.
- 59 failed sources lack `last_error`; all 59 have `failed_at`, non-zero
  attempts, and a live upstream entity. Predicate: `sns.status='failed' AND
  (sns.failed_at IS NULL OR sns.last_error IS NULL OR
  coalesce(sns.attempt_count,0)=0)`. Examples include
  `dd:bolometer/camera/channel/subcollimators_separation` and
  `dd:camera_visible/channel/fibre_bundle/geometry/radius`.
- Three extracted sources already have a live `PRODUCED_NAME` target, all to
  the quarantined drafted identity
  `total_plasma_momentum_field_aligned_convection_velocity`. Predicate:
  `sns.status='extracted' AND EXISTS { MATCH
  (sns)-[:PRODUCED_NAME]->(sn:StandardName) WHERE NOT
  (coalesce(sn.name_stage,'') IN
  ['superseded','exhausted','contested']) }`. These do not currently shrink a
  valid batch; they expose duplicate-work risk instead.
- Zero composed or attached assertions lack their complete produced-name
  evidence; zero stale assertions have a live upstream entity; zero
  not-physical assertions lack a reason; there are no null or invalid values.

The source population changed from 9,894 to 9,900 while the read-only census
was in progress (attached +2, extracted +4), then stabilized for the final
source query. Counts above are the exact final per-scalar result, not a claim
that the mutable live graph was transactionally frozen across the entire
report.

## `StandardNameSource.compose_hint_status`

**Gated work.** Opening source steering requires an extracted, unclaimed DD
source below the attempt cap, with no live name binding and (unless replacing)
no existing open hint (`graph_ops.py:9692-9731`). An open hint is consumed in
the same transaction that finalizes a source-to-name binding
(`graph_ops.py:7357-7360`, `7419-7422`). A false open state can reject a new
hint and continue injecting stale steering; a false consumed state hides
steering that never reached a target.

**Corroborating evidence.** `open` requires non-empty hint and reason plus
`compose_hint_requested_at` and no live binding. `consumed` requires
`compose_hint_consumed_at` plus a `PRODUCED_NAME` edge. The schema defines
`rejected`, but there is no rejection timestamp or immutable rejection receipt;
if that value appears it cannot be independently corroborated.

**Live census.** Seven open, 65 consumed, 9,828 null, and zero rejected.
Every open/consumed value has its required request/consumption fields except
one open hint that already has a live binding:
`dd:pellets/time_slice/pellet/path_geometry/second_point/z` is attached to the
accepted, valid `vertical_coordinate_of_pellet_path_point`. Exact predicate:
`sns.compose_hint_status='open' AND EXISTS { MATCH
(sns)-[:PRODUCED_NAME]->(sn:StandardName) WHERE NOT
(coalesce(sn.name_stage,'') IN ['superseded','exhausted','contested']) }`.
This one contradiction blocks replacement steering unless an operator uses the
explicit replacement path.

## Additional state-bearing StandardName scalars

### Review quorum-shortfall markers

The name marker gates export, stranded promotion, and name refinement
(`export.py:541,550`; `graph_ops.py:11971-11976,17420`); the docs marker gates
full export and docs refinement (`export.py:557`; `graph_ops.py:11980-11986,
24343`). Evidence is a paired `*_shortfall_at` timestamp plus an axis-matching
non-winning review record. There are 126 name-shortfall and ten docs-shortfall
assertions. Exact predicate for each axis:
`sn.<axis>_review_quorum_shortfall IS NOT NULL AND
(sn.<axis>_review_quorum_shortfall_at IS NULL OR NOT EXISTS { MATCH
(sn)-[:HAS_REVIEW]->(r:StandardNameReview) WHERE r.review_axis=<axis> AND
r.resolution_method IN <non-winning methods> })`. The contradiction count is
zero on both axes. One name-marker row and nine docs-marker rows otherwise
clear the name/validation gates, but their withholding evidence is present, so
they are correctly blocked rather than false exclusions.

### `edit_status` and `edit_origin`

`edit_status='open'` gates `--edits` claim scope and prevents the bare stranded
promotion path (`graph_ops.py:15371-15443,11971-11976`). Every non-null status
must be backed by `edit_mode`, `edit_reason`, `edit_origin`, and
`edit_requested_at`; an open status additionally requires the mode-appropriate
name/docs payload. There are 180 open, 837 applied, 38 exhausted, eight
rejected, and 3,985 null status values. Zero non-null states lack their request
envelope, zero open states lack their payload, and zero name-axis open edits
are stranded at accepted/exhausted.

The 883 terminal edit outcomes have no immutable applied/exhausted/rejected
receipt. Current stage is not a safe substitute because later rescore,
supersession, or docs regeneration legitimately changes it. These assertions
are therefore uncheckable, not counted as proven contradictions.

`edit_origin` has 571 human, 494 agent, and 3,983 null values. Two human-origin
assertions (`pulse_duration` and
`net_power_due_to_ion_cyclotron_heating`) have no `edit_status`, although both
retain mode, reason, and request timestamp. More fundamentally, no actor
identity or signed request receipt corroborates human versus agent for any of
the 1,065 assertions; `edit_origin` does not independently gate an operation,
but it is provenance that cannot be audited to the semantics it claims. This
census only reports those rows and did not touch either identity.

### `superseded_from_stage`

Fold revival restores a tombstoned target from this scalar (capping accepted to
reviewed) and falls back to drafted only when it is null
(`edit.py:2008-2029`). Of 1,222 non-null assertions, 92 remain on a currently
non-superseded name and 39 have neither catalog successor evidence nor an
incoming `REFINED_FROM` edge. Predicate: `sn.superseded_from_stage IS NOT NULL
AND sn.superseded_by IS NULL AND NOT EXISTS { MATCH
(:StandardName)-[:REFINED_FROM]->(sn) }`. There is no immutable prior-stage
field on the change ledger against which any of the 1,222 values can be checked.
In addition, 568 store `refining`; that may truthfully describe the historical
instant of supersession, but reviving it literally would create a refining
current state without a claim. This is an uncorroboratable restoration input,
not included in the primary contradiction totals below.

### Review-completion and enrichment timestamps

`reviewed_name_at IS NULL` admits name-review work
(`graph_ops.py:16102-16113`), while a non-null value is also the hard gate that
allows docs-review results to be written (`graph_ops.py:6596-6651`). Its normal
evidence is a name-axis `HAS_REVIEW`; the explicit alternative is a signed
`HAS_STRUCTURAL_AUTHORITY` for a derived parent. There are 179 non-null
timestamps without a name review, but 174 have structural authority and are
therefore supported. The remaining five are accepted, valid strain-gauge
measurement-direction unit vectors with neither form of evidence. Predicate:
`sn.reviewed_name_at IS NOT NULL AND NOT EXISTS { MATCH
(sn)-[:HAS_REVIEW]->(r:StandardNameReview) WHERE r.review_axis='name' } AND NOT
EXISTS { MATCH (sn)-[:HAS_STRUCTURAL_AUTHORITY]->(:StructuralNameAuthority) }`.
All five are otherwise name-publishable. Conversely, 72 names have a review
edge but no timestamp; that is missing-state coverage that can cause duplicate
review work, not a false asserted state.

Two `reviewed_docs_at` timestamps have no docs review edge, on the already
superseded `x_direction_unit_vector_of_shatter_cone` and
`z_direction_unit_vector_of_shatter_cone`; 36 docs-review relationships have
no timestamp. Neither false timestamp currently shrinks a publishable batch.

`enriched_at` directly suppresses the enrichment pool when non-null
(`graph_ops.py:16064-16087`), but there are currently zero non-null values, so
there is no asserted state to test. Other timestamps and counters named above
are treated as evidence for their owning lifecycle scalar. Embedding,
consolidation, and harmonization timestamps are output freshness markers whose
worklists also inspect their payload/signature; they are not independent
lifecycle/provenance states and are outside the state-scalar count.

## Ranked findings

The first table ranks assertions with absent or contradictory evidence by
direct exposure to otherwise publishable names. Source rows cannot yet be
called publishable, so their live-upstream exclusion exposure is listed
separately rather than mixed into the name count.

| Rank | Scalar assertion | Affected | Otherwise publishable | What is silently skipped |
|---:|---|---:|---:|---|
| 1 | `validation_status='valid'`, no `validated_at` | 538 | 169 | Full validation is skipped while export trusts the provisional scalar. |
| 2 | `link_status='resolved'`, absent/terminal target | 328 | 39 | Link resolver skips the row; full export does not read this scalar. |
| 3 | `status='superseded'`, live `name_stage` | 34 | 31 | Export removes the identity before population accounting. |
| 4 | `docs_stage='accepted'`, no winning docs review | 27 | 27 | Docs review/refine pools skip it; relationship-aware export correctly refuses it. |
| 5 | accepted non-derived name, no name review | 6 | 6 | Name review is skipped and export currently has no name-review relationship gate. |
| 6 | `reviewed_name_at` without review/structural authority | 5 | 5 | Name review is skipped and docs-review persistence is admitted. |
| 7 | open source hint despite live binding | 1 | 1 bound valid target | New steering is refused or stale steering remains open. |
| 8 | invalid `docs_stage='superseded'` | 2 | 0 | No live batch impact because both names are tombstones. |

Additional large contradictions are terminal rather than immediately
publishable: 519 name/status tombstones lack successor evidence (414 are
valid), 44 exhausted names lack the durable `refine_attempts` evidence, 594
quarantined rows lack a validation timestamp (242 also lack failure detail),
and two docs-review timestamps lack review edges.

| Rank | Source assertion | Affected | Live upstream | Work impact |
|---:|---|---:|---:|---|
| 1 | `status='skipped'`, no reason | 398 | 398 | Composition is permanently excluded without a recorded justification. |
| 2 | `status='failed'`, incomplete failure evidence | 59 | 59 | Composition is excluded; every row lacks `last_error`. |
| 3 | `status='extracted'`, live produced target already exists | 3 | 3 | Duplicate composition is enabled; all three targets are quarantined. |

The following are larger *auditability* gaps, ranked by population, but not
proven semantic contradictions: 2,821 resolved-link assertions have no resolver
timestamp; all 2,657 pipeline-origin assertions lack an immutable last-editor
receipt; 1,222 `superseded_from_stage` values lack a prior-stage receipt; and
883 terminal edit-status values lack an immutable outcome receipt. These are
listed separately so absence of an instrument is not misreported as proof that
every value is wrong.

## Reproducibility and conclusion

Every live query had exactly one leading label scan (`StandardName` or
`StandardNameSource`), used `EXISTS` subqueries keyed from that row, returned
only counts and bounded samples, and avoided Cartesian products. The slowest
query took 0.281 seconds, well below the ten-second login-node ceiling. Queries
ran between the first 5,048-name snapshot and the final 9,900-source snapshot
on 2026-09-08; the live graph briefly exposed six additional in-flight rows,
so each scalar section reports its own exact observation rather than pretending
the independent reads formed one transaction.

The highest-priority migration sizing numbers are therefore **169**
otherwise-publishable names relying on an unstamped valid scalar, **39** with a
false resolved-link assertion, **31** suppressed by a contradictory catalog
tombstone, **27** with accepted docs but no winning review, and **398 + 59**
live source rows terminally excluded without their required reason evidence.
The historical catalog-origin defect is closed at **0**, while the historical
exhaustion count refines from **259 legacy-counter absences** to **44 missing
durable attempt counters**.
