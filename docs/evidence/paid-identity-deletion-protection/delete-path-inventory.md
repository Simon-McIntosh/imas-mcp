# Delete-path inventory: does every recorded delete path consult spend, publication state, and a volume ceiling?

**Node:** static inventory over the two deletion-bearing modules, read at
`da96aa1d4` — all line numbers below are from this checkout.
**Files reviewed:** `imas_codex/standard_names/graph_ops.py` and
`imas_codex/standard_names/provenance_lifecycle.py` (the two files the plan
names), every literal `DETACH DELETE` and every call site of the
`deletion_change_cypher` helper.
**Method:** text-level identification of every statement, then per-path (and
per statement) classification of the three protections by reading the
surrounding function. No graph was touched — this is a read of the tree, not a
live census.

## Totals and reconciliation

| Artifact | Plan claim | Found in tree | Reconciliation |
|---|---|---|---|
| `DETACH DELETE` literals ("across graph_ops.py and provenance_lifecycle.py") | 37 | **37 grep matches**, of which **3 are comments** (`graph_ops.py:8839`, `9112`, `9114` — prose, not executable) → **34 executable statements** (25 in `graph_ops.py`, 9 in `provenance_lifecycle.py`) | claim matches the raw grep count; the executable count is 34 |
| `deletion_change_cypher` call sites | nine | **8** in production source (3 in `provenance_lifecycle.py` + 5 in `graph_ops.py`); the 9th is a test import (`tests/standard_names/test_derived_parent_cleanup_protection.py:78`) | plan's "nine" includes the test; 8 production sites |
| recorded delete paths | nine | **10 distinct functions** hold executable deletions (7 in `graph_ops.py` + 3 in `provenance_lifecycle.py`; `_count` on `graph_ops.py:9100` is a nested count helper, not a delete owner) | tree has one more path than the plan recorded |
| inventory rows | — | **42** = 34 executable `DETACH DELETE` statements + 8 helper call sites | row count equals total statements + call sites found |

Three of the 37 raw matches are comments; they carry no row. Every executable
statement and every helper call site below has a row.

## What the three protections mean here

Checked per site, on the *path that removes the node*, before/at the delete:

- **Spend check** — the path consults recorded `LLMCost` spend attribution
  (currently the materialized `(cost:LLMCost)-[:FOR_STANDARD_NAME]->(sn)`
  edges read by `automatic_deletion_protections`, `protection.py`) and refuses
  to remove an identity that carries non-zero recorded spend. `ABSENT (N/A)`
  marks a path that deletes a non-identity label (`DocsRevision`, `VocabGap`,
  a `StandardNameSource`), where the spend instrument does not attach.
- **Publication-state check** — the path refuses (or the operator must opt in
  to) removing an identity that is `approved`, accepted-with-catalog-binding,
  ratified (`unchanged_ratification`), or a catalog-cut member. The strong form
  is `refuse_protected_automatic_deletion` / `filter_automatic_deletion_candidates`;
  the weak form is a scope predicate that structurally excludes published
  states (e.g. `include_accepted` gating, `catalog_approved_at IS NULL`).
- **Volume ceiling** — the path refuses the whole pass when the candidate set
  exceeds a stated numeric ceiling, or is otherwise structurally bounded to a
  fixed maximum. `ABSENT` means the pass can take unbounded rows in one go.

## Inventory — one row per helper call site and per `DETACH DELETE` statement

`C` = AUTOMATIC (runs inside a pipeline, global-maintenance, or startup pass
without a human in the loop) · `O` = OPERATOR-INVOKED (CLI command / governed
repair with explicit confirmation or single explicit target).

### graph_ops.py — `_delete_derived_parent_nodes` (def 3649) — the reaper

Automatic: called from derived-parent admission cleanup at `graph_ops.py:4578`,
`4639`, `4710` and from the manifest-signed repair path `signed_manifest.py:6826`.
Admission candidates are pre-filtered by `filter_automatic_deletion_candidates`
(`graph_ops.py:3646`), then the path guards and caps.

| file:line | statement | C/O | spend | publication | volume |
|---|---|---|---|---|---|
| graph_ops.py:3682 | `deletion_change_cypher("sn")` (receipt emitted per identity) | C | PRESENT | PRESENT | PRESENT |
| graph_ops.py:3730 | `FOREACH (source IN derived_sources \| DETACH DELETE source)` | C | PRESENT | PRESENT | PRESENT |
| graph_ops.py:3731 | `FOREACH (r IN reviews \| DETACH DELETE r)` | C | PRESENT | PRESENT | PRESENT |
| graph_ops.py:3732 | `FOREACH (d IN revisions \| DETACH DELETE d)` | C | PRESENT | PRESENT | PRESENT |
| graph_ops.py:3733 | `DETACH DELETE sn` | C | PRESENT | PRESENT | PRESENT |

Protections: `refuse_protected_automatic_deletion` (`3675`) refuses the whole
batch when any candidate is protected (spend via `recorded_llm_spend_usd`,
publication via `name_stage=approved` / `catalog_pr_number` /
`catalog_merge_commit_sha` / `catalog_commit_sha` / `exported_at` /
`unchanged_ratification` / `content_edit`); volume ceiling
`DERIVED_PARENT_CLEANUP_DELETION_LIMIT = 80` (`72`, checked at `3667` — refuses
the whole pass above it); the delete itself requires the
`needs_composition=true` placeholder marker. This is the fully protected path.

### graph_ops.py — `sweep_orphaned_docs_revisions` (def 3747)

Automatic: runs every startup / global maintenance (`graph_ops.py:4676`).
Deletes `DocsRevision` snapshots whose owning `StandardName` is gone.

| file:line | statement | C/O | spend | publication | volume |
|---|---|---|---|---|---|
| graph_ops.py:3764 | `WITH dr LIMIT 50000` + `DETACH DELETE dr` | C | ABSENT (N/A) | ABSENT (N/A) | PRESENT (`LIMIT 50000`) |

### graph_ops.py — `write_standard_names` skeleton sweep (def 5168)

Automatic: the pipeline writer (called from the review pipeline
`review/pipeline.py:1502` and `graph_ops.py:7340`). Deletes relationship-side
skeleton placeholders created by this write that never became composed names.

| file:line | statement | C/O | spend | publication | volume |
|---|---|---|---|---|---|
| graph_ops.py:5733 | `deletion_change_cypher("sn")` | C | PRESENT | PRESENT | ABSENT |
| graph_ops.py:5756 | `DETACH DELETE sn` (after bare-skeleton predicates + receipt) | C | PRESENT | PRESENT | ABSENT |

Protections: `refuse_protected_automatic_deletion` (`5728`) refuses the batch
on any protected candidate; the deletion query then applies strict
bare-skeleton predicates (`created_at`/`generated_at`/`validation_status`/
`unit`/`kind`/`needs_composition` null, non-empty stage, no parent/error/source
edges). **Volume ceiling ABSENT** — the candidate set is whatever
relationship skeletons this write produced, with no numeric cap. Note for the
reader: the guard is evaluated *before* the bare-skeleton predicates are
applied (plan §3a records the placement concern this raises; the placement fix
is owned by the restore work, not this inventory).

### graph_ops.py — `clear_standard_names` (def 8486)

Operator-invoked: `sn clear` (`cli/sn.py:2222`, `2520`, `5598`, `5632`), with
`dry_run` preview and an `include_accepted` opt-in. Statement branchs differ
by selection style; protection posture is uniform across them.

| file:line | statement | C/O | spend | publication | volume |
|---|---|---|---|---|---|
| graph_ops.py:8853 | `deletion_change_cypher("sn")` (path-allowlist branch) | O | ABSENT | PRESENT | ABSENT |
| graph_ops.py:8895 | `DETACH DELETE r` (review, path-allowlist branch) | O | ABSENT | PRESENT | ABSENT |
| graph_ops.py:8897 | `DETACH DELETE sn` (name, path-allowlist branch) | O | ABSENT | PRESENT | ABSENT |
| graph_ops.py:8929 | `DETACH DELETE source` (derived source of retired parent) | O | ABSENT | PRESENT | ABSENT |
| graph_ops.py:8930 | `DETACH DELETE parent` (retired bare scaffold parent) | O | ABSENT | PRESENT | ABSENT |
| graph_ops.py:8959 | `DETACH DELETE r` (review, scope-join branch) | O | ABSENT | PRESENT | ABSENT |
| graph_ops.py:8961 | `DETACH DELETE sn` (name, scope-join branch) | O | ABSENT | PRESENT | ABSENT |
| graph_ops.py:8967 | `deletion_change_cypher("sn")` (no-src-join branch) | O | ABSENT | PRESENT | ABSENT |
| graph_ops.py:8981 | `DETACH DELETE r` (review, no-src-join branch) | O | ABSENT | PRESENT | ABSENT |
| graph_ops.py:8983 | `DETACH DELETE sn` (name, no-src-join branch) | O | ABSENT | PRESENT | ABSENT |
| graph_ops.py:8995 | orphan-review sweep `DETACH DELETE r` `LIMIT 10000` | O | ABSENT (N/A) | ABSENT (N/A) | PRESENT (`LIMIT 10000`) |
| graph_ops.py:9021 | unscoped full clear: `MATCH (c:LLMCost) DETACH DELETE c` | O | ABSENT (wipes the spend ledger) | ABSENT | ABSENT |

Publication checks present: accepted names require `include_accepted` (`8510`
+"required" gate); relationship-first delete keeps names attached to any
out-of-scope/accepted path alive; terminal `superseded`/`exhausted` stages are
deleted only when explicitly listed. Spend: an unscoped clear **deletes the
LLMCost ledger itself** (`9021`) — the reverse of consulting it. Scoped clears
leave the ledger intact but do not check it against the names being removed.

### graph_ops.py — `clear_sn_subsystem` (def 9054; nested `_count` at 9100 is a count helper, not a delete owner)

Operator-invoked: `sn clear --subsystem` (`cli/sn.py:5457–5499`), `dry_run`
preview available. Wholesale wipe of every label.

| file:line | statement | C/O | spend | publication | volume |
|---|---|---|---|---|---|
| graph_ops.py:9116 | `deletion_change_cypher("sn")` | O | ABSENT | ABSENT | ABSENT |
| graph_ops.py:9115 | `MATCH (r:StandardNameReview) DETACH DELETE r` | O | ABSENT | ABSENT | ABSENT |
| graph_ops.py:9121 | `DETACH DELETE sn` (whole label) | O | ABSENT | ABSENT | ABSENT |
| graph_ops.py:9129 | `MATCH (s:StandardNameSource) DETACH DELETE s` | O | ABSENT | ABSENT | ABSENT |
| graph_ops.py:9130 | `MATCH (d:DocsRevision) DETACH DELETE d` | O | ABSENT | ABSENT | ABSENT |
| graph_ops.py:9131 | `MATCH (v:VocabGap) DETACH DELETE v` | O | ABSENT | ABSENT | ABSENT |
| graph_ops.py:9132 | `MATCH (rr:SNRun) DETACH DELETE rr` | O | ABSENT | ABSENT | ABSENT |
| graph_ops.py:9133 | `MATCH (c:LLMCost) DETACH DELETE c` | O | ABSENT (wipes the spend ledger) | ABSENT | ABSENT |

### graph_ops.py — `reconcile_vocab_gaps` (def 12679)

Automatic: global maintenance (`loop.py:1523`). Deletes `VocabGap` rows
reclassified as `false_positive` / `invalid_segment` / `open_segment`.

| file:line | statement | C/O | spend | publication | volume |
|---|---|---|---|---|---|
| graph_ops.py:12761 | `UNWIND $ids ... MATCH (vg:VocabGap {id}) DETACH DELETE vg` | C | ABSENT (N/A) | ABSENT (N/A) | ABSENT |

### graph_ops.py — `reconcile_provenance` (def 13174)

Automatic: global maintenance (`loop.py:1595`). Deletes orphaned derived-parent
`StandardNameSource` scaffolding whose intended name no longer exists.

| file:line | statement | C/O | spend | publication | volume |
|---|---|---|---|---|---|
| graph_ops.py:13229 | `MATCH (sns:StandardNameSource {batch_key:'derived_parent'}) ... DETACH DELETE sns` | C | ABSENT (N/A) | ABSENT (N/A) | ABSENT |

### provenance_lifecycle.py — `cancel_staged_rename` (def 153)

Operator-invoked: a single-target governed cancel API — no automatic caller in
the tree today (only `tests/standard_names/test_cancel_staged_rename.py`
exercises it); it requires an explicit `successor_id`, a non-empty `reason` and
a `dry_run` stage. Restores the superseded predecessor and deletes the
unaccepted rename successor.

| file:line | statement | C/O | spend | publication | volume |
|---|---|---|---|---|---|
| provenance_lifecycle.py:208 | `deletion_change_cypher("successor")` | O | ABSENT | PRESENT | PRESENT |
| provenance_lifecycle.py:256 | `FOREACH (item IN reviews \| DETACH DELETE item)` | O | ABSENT | PRESENT | PRESENT |
| provenance_lifecycle.py:257 | `FOREACH (item IN revisions \| DETACH DELETE item)` | O | ABSENT | PRESENT | PRESENT |
| provenance_lifecycle.py:258 | `DETACH DELETE successor` | O | ABSENT | PRESENT | PRESENT |

Publication scope present by construction: the successor must be an open
`edit_mode='rename'` edit at `drafted`/`reviewed`/`exhausted` with a
`superseded` predecessor — an accepted or approved identity cannot match.
Volume present by construction: exactly one identity.

### provenance_lifecycle.py — `retire_unrecoverable_provenance_orphans` (def 1741)

Automatic: called from the provenance-rebuild pipeline
(`provenance_rebuild.py:709`). Deletes a reviewed, list-scoped set of
source-less names with atomic ledger records.

| file:line | statement | C/O | spend | publication | volume |
|---|---|---|---|---|---|
| provenance_lifecycle.py:1788 | `deletion_change_cypher("sn")` | C | PRESENT | PRESENT | PRESENT |
| provenance_lifecycle.py:1808 | `FOREACH (item IN reviews \| DETACH DELETE item)` | C | PRESENT | PRESENT | PRESENT |
| provenance_lifecycle.py:1809 | `FOREACH (item IN revisions \| DETACH DELETE item)` | C | PRESENT | PRESENT | PRESENT |
| provenance_lifecycle.py:1810 | `DETACH DELETE sn` | C | PRESENT | PRESENT | PRESENT |

Protections: retired targets must still be source-less at mutation time;
accepted orphans require `include_accepted=True` (`1780`, publication gate);
`refuse_protected_automatic_deletion` (`1784`) refuses any protected identity
(spend + publication authority); volume is bounded by design — the function is
"deliberately list-scoped … cannot widen into an unbounded graph cleanup".

### provenance_lifecycle.py — `compact_unapproved_superseded` (def 1866)

Operator-invoked: `sn provenance-cleanup` (`cli/sn.py:5699` manifest,
`5725` apply), with an explicit `--apply`/`--force` confirmation over a
read-only manifest. Compacts superseded names that survived to exactly one
live tip after retargeting sources to it.

| file:line | statement | C/O | spend | publication | volume |
|---|---|---|---|---|---|
| provenance_lifecycle.py:1924 | `deletion_change_cypher("old")` | O | ABSENT | PRESENT | ABSENT |
| provenance_lifecycle.py:1934 | `FOREACH (item IN reviews \| DETACH DELETE item)` | O | ABSENT | PRESENT | ABSENT |
| provenance_lifecycle.py:1935 | `FOREACH (item IN revisions \| DETACH DELETE item)` | O | ABSENT | PRESENT | ABSENT |
| provenance_lifecycle.py:1936 | `DETACH DELETE old` | O | ABSENT | PRESENT | ABSENT |

Publication scope present: rows must be `name_stage='superseded'` with
`catalog_approved_at IS NULL` (`1885`, `1928`), retargeted to `target`, and
only `safe_to_compact` singletons are applied (`1887`); ambiguous/dead-end rows
are never touched. One publication state (`approved`) and the spend instrument
are not consulted — an approved name cannot match the predicates, but a
superseded name that once carried spend can be compacted (its
`LLMCost` records are removed with it). Volume absent: the whole manifest can
be applied in one `--apply`; the operator confirm is the gate, not a number.

## Closing table — every AUTOMATIC path lacking any of the three protections

| Path (def) | Automatic caller | Spend | Publication | Volume | Lacking |
|---|---|---|---|---|---|
| `write_standard_names` skeleton sweep (`graph_ops.py:5168`) | pipeline writer / `review/pipeline.py:1502` | PRESENT | PRESENT | **ABSENT** | volume ceiling |
| `sweep_orphaned_docs_revisions` (`graph_ops.py:3747`) | startup / global maintenance (`4676`) | ABSENT (N/A — `DocsRevision`) | ABSENT (N/A) | PRESENT (`LIMIT 50000`) | spend, publication (non-identity target) |
| `reconcile_vocab_gaps` (`graph_ops.py:12679`) | global maintenance (`loop.py:1523`) | ABSENT (N/A — `VocabGap`) | ABSENT (N/A) | ABSENT | all three (non-identity target) |
| `reconcile_provenance` (`graph_ops.py:13174`) | global maintenance (`loop.py:1595`) | ABSENT (N/A — derived `StandardNameSource`) | ABSENT (N/A) | ABSENT | all three (non-identity target) |

The two remaining automatic paths are fully protected: `_delete_derived_parent_nodes`
(all three: spend + publication guard, `=80` ceiling) and
`retire_unrecoverable_provenance_orphans` (all three: refusal guard,
`include_accepted` gate, list-scoped).

**Reading:** the only automatic path that deletes `StandardName` *identities*
and lacks one of the three protections is the `write_standard_names` skeleton
sweep, which is missing a volume ceiling (spend and publication are both
consulted via the refusal guard). The three other automatic paths lacking
protections all delete non-identity scaffolding (`DocsRevision`, `VocabGap`,
derived `StandardNameSource`), where the spend and publication instruments do
not attach, and the volume question for them is a "does it cap its pass" matter
rather than a "can it remove a paid identity" matter. Every operator-invoked
path (`sn clear`, `sn clear --subsystem`, `sn provenance-cleanup`, and the
`cancel_staged_rename` API) carries at least a dry-run/manifest, a
single-target bound, or an `include_accepted` publication gate before it
removes anything; none consults recorded spend, and two wipe the LLMCost
ledger themselves.

## Per-path notes justifying the verdicts

- **Protection function locations:** the spend and publication authority is
  `automatic_deletion_protections` / `filter_automatic_deletion_candidates` /
  `refuse_protected_automatic_deletion` in `protection.py`; spend comes from
  materialised `(LLMCost)-[:FOR_STANDARD_NAME]->(sn)` edge sums
  (`recorded_spend > 0.0`), publication from approved stage, catalog PR /
  merge / commit fields, `exported_at`, and `unchanged_ratification` /
  `content_edit` change operations.
- **Counting note:** a `deletion_change_cypher` call and the neighbouring
  `DETACH DELETE` statements inside the same function are separate rows per the
  measure; each call site is counted once.
- **Census deltas versus the plan text are flagged inline above** (37 raw
  matches → 34 executable; helper call sites 9 → 8 in production; delete paths
  "nine" → ten functions holding deletes). None changes the shape of the
  finding: the reaper already carries all three protections, and the one
  partially-open automatic identity path is `write_standard_names`
  (missing volume).

## Method and limits

Read-only inspection at `da96aa1d4`; no graph writes, no model calls, no tests
run (a documentation census does not change behaviour). Each verdict was taken
by reading the enclosing function; where a statement sits in a branch the
verdict follows that branch's path. The live graph was not consulted, so
"consult spend" here means *the code path consults the spend instrument before
removing*, not that any particular identity was checked.
