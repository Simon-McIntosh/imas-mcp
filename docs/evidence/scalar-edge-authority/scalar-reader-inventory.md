# Reader inventory for relationship-mirrored scalars

This is a static inventory of the twelve surfaces marked **DERIVE** in the mirrored-state review. It does not query the graph, run a pipeline, or change application behaviour.

![Read-site counts split by consequence class](/imas-codex/figures/scalar-edge-authority/scalar-reader-counts.svg)

## Method and boundary

The package was searched rather than inferred from its owning modules. The raw package-wide lexical result is retained at `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260909T112247593387-n-sea-who-reads-the-twelve-mirrored-scalars/scalar-reader-raw-rg.txt`; the twelve per-surface `reads-*.txt` results sit beside it. Searches covered `imas_codex/**/*.py` and both names of each paired surface.

A *read site* is an executable operation that obtains a stored graph scalar, compares or filters on it, or puts that obtained value into an output/context payload. Adjacent accesses in one function or Cypher statement are one site. Schema declarations, docstrings, `SET`-only writes, pre-persistence payload construction, and unrelated local variables with the same spelling are not reads. That filtering explains why the raw lexical files have more lines than the table. Every retained site below is named by file and first reading line; the cited line grounds its classification and consequence.

| Class | Meaning |
|---|---|
| publication-gating | An export, release, catalog, or admission path where the value can decide whether a name is emitted. |
| pipeline-internal | A compose, review, docs, reconcile, edit, repair, or worker path. |
| informational | A report, display, audit, benchmark, ledger, or census that observes rather than emits a catalog name. |

`dd_path`/`signal`, `deprecates`/`superseded_by`, the axis-review projection bundle, and `reviewed_name_at`/`reviewed_docs_at` are paired/bundled fields but one mirror each. Accordingly there are twelve surfaces, not sixteen-plus field names.

## Measured consequence summary

| # | DERIVE surface | Publication | Internal | Informational | Retention recommendation, driven by |
|---:|---|---:|---:|---:|---|
| 1 | `source_paths` | 1 | 16 | 9 | **remove** once all non-`derived:` attachments are edges; `graph_ops.py:8265`. |
| 2 | `StandardNameSource.produced_sn_id` | 2 | 18 | 4 | **remove after adjudication**; `graph_ops.py:12309`. |
| 3 | `source_types` | 1 | 7 | 2 | **remove** after producer-kind projection; `graph_ops.py:5585`. |
| 4 | `StandardNameSource.dd_path` / `signal` | 0 | 14 | 3 | **remove**; `graph_ops.py:12216`. |
| 5 | `primary_cluster_id` | 0 | 1 | 0 | **remove**; `graph_ops.py:3250`. |
| 6 | `links` | 2 | 6 | 2 | **remove only after full REFERENCES parity**; `export.py:984`. |
| 7 | `deprecates` / `superseded_by` | 1 | 5 | 1 | **remove**; `edit.py:2104`. |
| 8 | axis reviewer projections | 2 | 20 | 7 | **remove after atomic selected-review write**; `export.py:504`. |
| 9 | `reviewed_name_at` / `reviewed_docs_at` | 0 | 7 | 1 | **remove** after one shared authority predicate; `review/pipeline.py:351`. |
| 10 | `chain_length` | 1 | 10 | 3 | **remove** after replacing the legacy fallback; `graph_ops.py:16477`. |
| 11 | `docs_chain_length` | 1 | 8 | 2 | **remove** after revision counting replaces gates; `graph_ops.py:17122`. |
| 12 | `origin` | 3 | 20 | 4 | **remove, not cache**; `export.py:1072`. |

Cheap removals are `primary_cluster_id`, `dd_path`/`signal`, `deprecates`/`superseded_by`, and, after a shared predicate, the review timestamps: their replacements are cardinality-one or one-hop reads. `source_types`, the two counters, and `source_paths` require complete-set projection but no retained cache. The non-cheap set is `links` (materialisation exclusions remain), `produced_sn_id` (multi-target adjudication), axis projections (atomic selection/persistence), and `origin` (one value has incompatible meanings). No repository source reports query timing or cost for any surface, so this audit has no evidence to recommend **keep-as-cache** for any one of them.

## Complete static read-site inventory

### 1. `StandardName.source_paths`

| Class | Reader sites (file:line) |
|---|---|
| publication-gating | `imas_codex/cli/sn.py:5935`; `imas_codex/standard_names/promote.py:654` |
| pipeline-internal | `imas_codex/standard_names/graph_ops.py:5460,8098,8265,8282,12505`; `imas_codex/standard_names/context.py:629,711`; `imas_codex/standard_names/workers.py:4873,5004,5412,8793,8959,9880,9942`; `imas_codex/standard_names/review/pipeline.py:282,667,778,856`; `imas_codex/standard_names/edit.py:673`; `imas_codex/standard_names/consolidation.py:104`; `imas_codex/standard_names/provenance_rebuild.py:413` |
| informational | `imas_codex/llm/sn_tools.py:270`; `imas_codex/standard_names/audits.py:3451,3582`; `imas_codex/standard_names/review/audits.py:121`; `imas_codex/standard_names/benchmark.py:560,800,879`; `imas_codex/standard_names/benchmark_roles.py:352,421,479,921,1007,1140`; `imas_codex/standard_names/release_notes.py:201` |

Without the scalar, the hard reader `graph_ops.py:8265` cannot restrict its IDS source filter and can admit or omit the wrong cohort; once all non-derived attachments are relationships, a path-set query returns the same value.

### 2. `StandardNameSource.produced_sn_id`

| Class | Reader sites (file:line) |
|---|---|
| publication-gating | `imas_codex/standard_names/graph_ops.py:12309,12318,12335`; `imas_codex/standard_names/ledger.py:76,80` |
| pipeline-internal | `imas_codex/standard_names/graph_ops.py:1014,1177,1458,3702,3711,4792,4803,5086,5098,5184,8716,8729,11384,11446,13979,14011,14022,19713,20629,21162,21925,22309,22469,22772`; `imas_codex/standard_names/provenance_lifecycle.py:496,820,1106,1135,1144,1185,1394`; `imas_codex/standard_names/provenance_rebuild.py:284,357,370,381`; `imas_codex/standard_names/edit.py:1259,1267,1308,1726,2559`; `imas_codex/standard_names/signed_manifest.py:1813,2799,2918,3114,3833,3884,3938,4160,5466,6011,6169,6403,6591`; `imas_codex/standard_names/attachment_audit.py:1640,1796` |
| informational | `imas_codex/standard_names/ledger.py:112`; `imas_codex/standard_names/attachment_audit.py:1640,1796`; `imas_codex/standard_names/provenance_rebuild.py:370,381`; `imas_codex/standard_names/signed_manifest.py:3833,3884` |

Removing it before adjudication leaves `graph_ops.py:12309` unable to select the intended target when one source has several `PRODUCED_NAME` edges; only a single adjudicated edge makes removal valid.

### 3. `StandardName.source_types`

| Class | Reader sites (file:line) |
|---|---|
| publication-gating | `imas_codex/cli/sn.py:5950` |
| pipeline-internal | `imas_codex/standard_names/graph_ops.py:5455,5585,5586,5730,7296,7601,8383,8607`; `imas_codex/standard_names/review/pipeline.py:298`; `imas_codex/standard_names/consolidation.py:113` |
| informational | `imas_codex/cli/sn.py:3795,3796,3797` |

The hard reader `graph_ops.py:5585` needs the DD/signal choice for reset routing; producer-edge kinds can supply that deduplicated set once materialised.

### 4. `StandardNameSource.dd_path` / `signal`

| Class | Reader sites (file:line) |
|---|---|
| pipeline-internal | `imas_codex/standard_names/graph_ops.py:7296,7601,12083,12101,12216`; `imas_codex/standard_names/provenance_lifecycle.py:496,820,1106,1135,1144,1185`; `imas_codex/standard_names/provenance_rebuild.py:370,381`; `imas_codex/standard_names/edit.py:1259,1267,1308`; `imas_codex/standard_names/signed_manifest.py:3833,3884,3938,4160`; `imas_codex/standard_names/attachment_audit.py:1640,1773,1796`; `imas_codex/standard_names/workers.py:4298,6247` |
| informational | `imas_codex/standard_names/attachment_audit.py:1773`; `imas_codex/standard_names/provenance_rebuild.py:370,381`; `imas_codex/standard_names/signed_manifest.py:3833,3884` |

The hard reader `graph_ops.py:12216` cannot decide stale versus live attachment without the typed scalar; with exactly one `FROM_DD_PATH` or `FROM_SIGNAL` edge it is a direct identity lookup.

### 5. `StandardName.primary_cluster_id`

| Class | Reader sites (file:line) |
|---|---|
| pipeline-internal | `imas_codex/standard_names/graph_ops.py:3250` |

`graph_ops.py:3250` is the only executable stored-scalar reader and feeds `IN_CLUSTER`; after that edge exists, a one-hop lookup replaces it, making this the cheapest removal.

### 6. `StandardName.links`

| Class | Reader sites (file:line) |
|---|---|
| publication-gating | `imas_codex/standard_names/export.py:984,1403,1835`; `imas_codex/cli/sn.py:5934` |
| pipeline-internal | `imas_codex/standard_names/graph_ops.py:2043,2060,5459,5478,8097,8281,9194,9243,9261,9537,9711,9714`; `imas_codex/standard_names/review/pipeline.py:281`; `imas_codex/standard_names/workers.py:5003`; `imas_codex/standard_names/review/audits.py:376,384` |
| informational | `imas_codex/llm/sn_tools.py:269`; `imas_codex/standard_names/review/audits.py:376,384` |

The hard reader `export.py:984` emits `name:<id>` entries. Removing the scalar before every target has a `REFERENCES` counterpart exports an incomplete link list; only full parity, including closure of the current exclusion ledger, permits edge-based output.

### 7. `StandardName.deprecates` / `superseded_by`

| Class | Reader sites (file:line) |
|---|---|
| publication-gating | `imas_codex/standard_names/catalog_import.py:306,307`; `imas_codex/standard_names/canonical.py:34,35,68,69` |
| pipeline-internal | `imas_codex/standard_names/graph_ops.py:3063,3067`; `imas_codex/standard_names/edit.py:2085,2104`; `imas_codex/standard_names/signed_manifest.py:4088,4095`; `imas_codex/standard_names/protection.py:33,34` |
| informational | `imas_codex/standard_names/signed_manifest.py:4088,4095` |

The hard reader `edit.py:2104` needs the successor for its compatibility summary, but a cardinality-one `HAS_SUCCESSOR`/`HAS_PREDECESSOR` lookup gives the same answer after reader migration.

### 8. Axis reviewer projections

This one surface bundles `reviewer_score_*`, `reviewer_scores_*`, `reviewer_comments_*`, `reviewer_comments_per_dim_*`, `reviewer_model_*`, and `review_resolution_method`, all mirrored by selected `HAS_REVIEW` records.

| Class | Reader sites (file:line) |
|---|---|
| publication-gating | `imas_codex/standard_names/export.py:504,516,552,558`; `imas_codex/cli/sn.py:5953,5956,5959` |
| pipeline-internal | `imas_codex/standard_names/graph_ops.py:16102,16113,16251,16477,16704,16995,17122,17244,17468,17593,17969,24436,24743,25741,26098,26245`; `imas_codex/standard_names/review/pipeline.py:292,299,301,351,385,389,406,409`; `imas_codex/standard_names/workers.py:5005,6801,6848,9874,10454`; `imas_codex/standard_names/edit.py:807,3260`; `imas_codex/standard_names/harmonize.py:413,426`; `imas_codex/standard_names/context.py:615,622`; `imas_codex/standard_names/catalog_reconcile.py:142,151` |
| informational | `imas_codex/standard_names/example_loader.py:62,65,68,71`; `imas_codex/standard_names/review/themes.py:161,167,176`; `imas_codex/standard_names/benchmark.py:556,587`; `imas_codex/standard_names/benchmark_roles.py:403,436`; `imas_codex/standard_names/campaign_pricing.py:176`; `imas_codex/standard_names/chain_history.py:37`; `imas_codex/standard_names/docs_holdout_eval.py:184` |

The hard reader `export.py:504` needs selected review score, model, and resolution for a catalog row. A selected `HAS_REVIEW` query replaces the bundle only once selection and persistence are atomic.

### 9. `reviewed_name_at` / `reviewed_docs_at`

| Class | Reader sites (file:line) |
|---|---|
| pipeline-internal | `imas_codex/standard_names/graph_ops.py:6711,6717,16167,16178,25741,26098,26245`; `imas_codex/standard_names/review/pipeline.py:301,351,389,391,406,409`; `imas_codex/standard_names/workers.py:9835`; `imas_codex/standard_names/progress.py:137,138` |
| informational | `imas_codex/standard_names/progress.py:137,138` |

The hard reader `review/pipeline.py:351` would skip docs review if a missing scalar meant no name review. Its replacement must return true when a selected name review or structural authority exists.

### 10. `chain_length`

| Class | Reader sites (file:line) |
|---|---|
| publication-gating | `imas_codex/cli/sn.py:518`; `imas_codex/standard_names/graph_ops.py:16251,17593,25433` |
| pipeline-internal | `imas_codex/standard_names/graph_ops.py:640,16477,16502,17456,17531,18238,24436,25745,26247`; `imas_codex/standard_names/workers.py:6801,6848,6964,6983,6999,7210,7226,7240`; `imas_codex/standard_names/fanout/trigger.py:108,131`; `imas_codex/standard_names/edit.py:809,3260`; `imas_codex/standard_names/chain_history.py:37` |
| informational | `imas_codex/standard_names/display.py:179,296`; `imas_codex/standard_names/campaign_pricing.py:176,441`; `imas_codex/standard_names/benchmark_roles.py:323`; `imas_codex/cli/sn.py:518` |

The hard reader `graph_ops.py:16477` uses `chain_length` if `refine_attempts` is absent, so removal without a bounded `REFINED_FROM` depth replacement can re-admit or prematurely exhaust refinement.

### 11. `docs_chain_length`

| Class | Reader sites (file:line) |
|---|---|
| publication-gating | `imas_codex/cli/sn.py:474,539,542`; `imas_codex/standard_names/graph_ops.py:24721,25453` |
| pipeline-internal | `imas_codex/standard_names/graph_ops.py:677,16995,17122,17139,17147,17174,18246,18757,18764,18850,18865,18917,24743,24849,24890,24918,24997,25038`; `imas_codex/standard_names/workers.py:9871,10454,10458,10474,10682,10692`; `imas_codex/standard_names/edit.py:3459` |
| informational | `imas_codex/cli/sn.py:539,542`; `imas_codex/standard_names/workers.py:10419` |

The hard reader `graph_ops.py:17122` decides docs exhaustion. If its scalar disappears before replacement, it returns zero/no depth; counting `DOCS_REVISION_OF` snapshots becomes correct only after the complete revision set exists.

### 12. `StandardName.origin`

| Class | Reader sites (file:line) |
|---|---|
| publication-gating | `imas_codex/standard_names/export.py:1072,1083,1294`; `imas_codex/cli/sn.py:398,430,441,485,496`; `imas_codex/standard_names/graph_ops.py:16194,16968,17484,24411,24711,25413,25428,25434` |
| pipeline-internal | `imas_codex/standard_names/graph_ops.py:1117,2632,2636,2652,2700,3532,3562,3582,3612,3639,3989,4558,4734,4750,5090,7538,13773,13842,13940,17969,18156,18281,18283,18970,19307,20646,21925,21998,22143,25413,25589,25638,25739,25924,26072,26207,26215,26231,26416,26453`; `imas_codex/standard_names/edit.py:745,807,1115,1128,3400`; `imas_codex/standard_names/parents.py:239,400,425`; `imas_codex/standard_names/workers.py:4835,5007,8368,8795,9055,9878,9898,10548`; `imas_codex/standard_names/review/pipeline.py:294,621,631`; `imas_codex/standard_names/cascade.py:554,562,793,934`; `imas_codex/standard_names/attachment_audit.py:182,2446`; `imas_codex/standard_names/campaign_pricing.py:178,254`; `imas_codex/standard_names/ledger.py:39,61,112` |
| informational | `imas_codex/standard_names/audits.py:4603,4665`; `imas_codex/standard_names/ledger.py:61,112`; `imas_codex/standard_names/campaign_pricing.py:178`; `imas_codex/standard_names/cascade.py:554,562`; `imas_codex/standard_names/provenance_lifecycle.py:1962` |

The hard reader `export.py:1072` changes catalog output for `derived` and `catalog_edit`. It cannot be saved by a cache because this scalar conflates source entrance with last editorial action; the reader must ask its precise producer, structural-authority, or change-receipt question.

## Derive timing decision

**Decision: never derive before the relationship set is materialised and count-verified.** No current listed reader resolves a DERIVE relationship in place of its scalar: every publication and pipeline gate above still reads the stored value. Therefore no present reader can observe a *partially materialised relationship set*; it does not read that set at all. This is a static finding, not a claim about live transaction isolation.

The replacement readers could observe a partial set if edge creation commits in batches and a read falls between them. The repository text does not prove an all-pair transaction boundary, so the exact interleaving is **unverified** from source alone. The potential return is specific: an edge replacement for `export.py:984` returns only committed `REFERENCES` targets; `graph_ops.py:12309` can see several `PRODUCED_NAME` targets; `graph_ops.py:17122` undercounts docs revisions; `graph_ops.py:16477` undercounts refinement depth; and `review/pipeline.py:351` sees no selected review until its edge exists. Each is an admission or publication decision, so a verified complete set is required.

## Scalar-retention decision

No measured relationship-query cost appears in repository source for any of the twelve surfaces. This audit therefore supplies no evidence for **keep-as-cache**. The non-cheap rows are sequence blockers, not cache candidates: `links` waits for parity, `produced_sn_id` for adjudication, axis projections for atomic selected-review persistence, and `origin` for semantic separation. If profiling later proves a material cost, a retained value must have one named writer and an edge-to-scalar reconcile; it becomes a cache decision rather than a DERIVE scalar.
