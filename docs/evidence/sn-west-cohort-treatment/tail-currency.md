# WEST tail-step currency: which of the five ordered steps are already satisfied

**Node:** census over the live `codex` graph and the current tree, 2026-09-08.
**Frozen cohort:** review roster `v0.10.0rc1+west-task-2e` (214 names) and the
`west_production_dd_paths` manifest (355 source paths), the same artifacts the
§9 tail was measured against.
**Method:** every figure below is a bounded live query (indexed on the batch-ids
or the manifest paths), or the real WEST review export itself, run on the login
node because the Neo4j bolt endpoint is a login-node-local tunnel. Each query
completed well under the ten-second ceiling. The live graph is being written to
by concurrent peers, so exact snapshot counts shift between minutes; each row
records its own run and caveat.

The five rows follow the ordering of the driving followup's five steps. Four are
**SATISFIED**; the pipeline-spend step is **OPEN** because the *budget was
spent* but the *drain it bought has residual rows* that are vocabulary-gated or
held at validation, and the live graph has since gained a sixth extracted WEST
source that was not in the drained cohort.

| # | Ordered step | Verdict | Measure (returned figure) |
|---|---|---|---|
| 1 | Reconcile the quarantine instruments | **SATISFIED** | accepted-and-quarantined names: **22** (15 dated + 7 undated); in the WEST batch: **4, all dated** |
| 2 | Export accounts for every exit | **SATISFIED** | names leaving with no stated cause: **0** (211 − 202 − 9 = 0 residue) |
| 3 | Recover what the accounting exposes | **SATISFIED** | null `physics_domain` in batch: **0**; tombstoned rows: **2 in batch, 0 in candidate selection** |
| 4 | Spend the pipeline budget | **OPEN** | WEST sources still at `extracted`: **2**; batch names still at `reviewed`/`drafted`: **4** |
| 5 | Repair the roster's reason field | **SATISFIED** | unnamed roster rows recording no cause: **0** (fresh assembly: 19 unnamed, all with a stated cause) |

Cohort constants: review batch 214 unique ids; manifest source paths 355.
Graph totals at measurement time: 5,048 `StandardName` nodes; 2,328 acceptedor-approved.

---

## Step 1 — Reconcile the quarantine instruments — SATISFIED

The two instruments that were disagreeing now answer through one authority, and
the five withheld names were revalidated through it.

Bounded live query:

```cypher
MATCH (sn:StandardName)
WHERE sn.name_stage IN ['accepted','approved'] AND sn.validation_status='quarantined'
RETURN count(sn) AS total
```

Returned **22** accepted-and-quarantined names; the same cohort split by
observation stamp returned **15 dated / 7 undated**. Within the 214-name WEST
batch exactly **4** names carry `validation_status='quarantined`, and **all four
now carry a `validated_at` stamp of `2026-09-08T07:48:57Z`** — i.e. current,
observed quarantine findings from the revalidation, not stale scalars:

`inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width`,
`radial_outline_of_plasma_boundary`, `radial_outline_of_wall`,
`vertical_outline_of_plasma_boundary`.

The fifth batch quarantined name that §9b counted
(`vertical_coordinate_of_line_of_sight`) now reads `validation_status='valid'`
and publishes.

Code support — the authority and the refusal are both in the tree:

- `mark_names_validated` (`graph_ops.py:8077`) writes `validation_status` and
  `validated_at` in one transaction — the verdict and its observation cannot
  diverge.
- `default_audit_revalidate` (`campaign.py:769`) → `drain_validation_for_ids`
  (`workers.py:5016`) is the deterministic validator that owns quarantines.
- `default_revalidate` (`campaign.py:697`) matches
  `coalesce(validation_status,'')='quarantined'` and refuses — it cannot clear
  a quarantine; `run_audits` (`audits.py:3601`) produces evidence only.

Residual, not a defect: **7** dated-absent quarantine scalars persist on
accepted names outside this batch. Per the instrument authority they are
undated historical values, not current findings, and the export excludes them
at selection (`validation_status='valid' AND validated_at IS NOT NULL`) rather
than publishing them. A state-repair sweep over them is follow-on work.

## Step 2 — Export accounts for every exit — SATISFIED

The real WEST review export was run with the current code, the 214-name review
batch, and the freshly-assembled 355-row source roster:
`uv run --no-sync python /tmp/tail_export_run.py` (driver reads the roster and
manifest from the repo and calls `run_export`; retained log
`export-accounting-run2.log`). Post-run closure is exact:

```text
candidate_count      211
published_count      202
accounted_exclusions   9
candidate − published 9
accounting_residue     0
```

`exclusion_by_reason`: `invalid_validation_status` 4, `name_not_accepted` 1,
`unreviewed_name` 4 — every one of the nine exits has a stated cause. All four
gates passed in that run: `catalog_status`, `identity_token_collision`,
`exclusion_accounting`, `manifest_source_accounting`. The manifest-source level
closes too: all 355 rows accounted as **309 emitted, 18 documented
non-nameable, 28 excluded**, each `excluded` row carrying a reason.

Names leaving the export with no stated cause: **0**.

Live-graph caveat recorded for truthfulness: one later re-run of the same
export is now refused by the newer `catalog_status` gate, because
`etendue_of_spectrometer_channel` carries a null `status` in the graph — a
state introduced mid-flight by the concurrent etendue-restore peer whose repair
is in its own evidence file. That refusal is the new null-status guard refusing
fail-closed, not a second accounting hole; the accounting gates and every exit
with a stated cause hold whenever the selection state is stable.

## Step 3 — Recover what the accounting exposes — SATISFIED

Bounded live queries over the 214-name batch:

```cypher
MATCH (sn:StandardName) WHERE sn.id IN $batch AND sn.physics_domain IS NULL
RETURN sn.id
-- returned 0 rows
MATCH (sn:StandardName) WHERE sn.id IN $batch
  AND (coalesce(sn.status,'')='superseded' OR coalesce(sn.name_stage,'')='superseded')
RETURN sn.id, sn.status, sn.name_stage
-- returned 2 rows
```

- Null `physics_domain`: **0** in the batch. `total_electron_count` and
  `vertical_coordinate_of_ece_channel` both carry assigned domains
  (`electromagnetic_wave_diagnostics`, via the sanctioned
  `reclassify_standard_name_domain` path) and now publish.
- Tombstoned rows: **2** identities in the review batch
  (`normalized_toroidal_beta` — `status=superseded`, name stage `reviewed`;
  `power_due_to_ion_cyclotron_heating` — `superseded` on both axes). Both are
  suppressed at candidate selection by the shared predicate
  `_tombstone_exclusion_clause` (`export.py:482`, applied in both
  `_fetch_export_population` and `_fetch_candidates`), so **0 tombstoned rows
  enter the candidate population**. The batch population is consequently
  214 − 2 = 212 (measured directly), or 211 at the earlier export snapshot
  while a third identity was mid-transition by a peer.

## Step 4 — Spend the pipeline budget — OPEN

Bounded live queries:

```cypher
UNWIND $paths AS p OPTIONAL MATCH (sns:StandardNameSource {id:'dd:'+p})
WITH p, sns WHERE sns IS NOT NULL
RETURN sns.source_id, sns.status  -- over the 355 manifest paths
MATCH (sn:StandardName) WHERE sn.id IN $batch
  AND sn.name_stage IN ['reviewed','drafted'] RETURN sn.id, sn.name_stage, ...
```

- **2** WEST manifest sources are still parked at `extracted`:
  - `calorimetry/group/component/power` — the vocabulary hold (§9a), now at
    `extracted` with `last_error` "missing device vocabulary token 'component'";
    a compose attempt cannot proceed until the grammar admits a carrier token.
  - `soft_x_rays/channel/etendue` — **newly** extracted (attempt_count 2, a
    persisted `vocab_gap_grammar_signature`, no `last_error` string): this path
    was not one of the 15 the drain cohort defined, and its restore is the
    concurrent etendue-repair peer's scope.
  - `equilibrium/time_slice/profiles_1d/darea_dpsi` sits at `failed`
    ("compose claim-attempt cap reached") with the physical-base gap recorded —
    a third source without a name, held on vocabulary.
  The other 13 of the original 15 drained sources are `composed`.
- **4** batch names are still at `reviewed`/`drafted`
  (`hot_neutral_temperature` reviewed/docs-accepted; `normalized_toroidal_beta`
  reviewed/superseded; `inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width`
  drafted/quarantined; `vertical_outline_of_plasma_boundary`
  drafted/quarantined). The four-step docs rotation was discharged against all
  of them — one already had accepted documentation, one is terminal, two are
  withheld at validation — so none is recoverable by a further docs rotation,
  but `hot_neutral_temperature` still needs a name-stage review/rescore, and the
  two drafted names await a fix for their named validation errors.

**Why OPEN rather than satisfied:** the budget half of the step is genuinely
spent — the scoped compose ran (13 of 15 sources named, compose spend $0.00 on
the zero-cost local route per the pipeline receipts), and the docs rotation cost
$1.00 for the nine unreviewed names. But the step's outcome — a batch with no
recoverable WEST source parked — is not yet reached: **2** sources at
`extracted` plus `darea_dpsi` at `failed`, and **4** names at
`reviewed`/`drafted`. Two of the source residuals are deferred by explicit
vocabulary decisions (upstream grammar repository), one is a newly created
parked source in a peer's repair scope, and the name residuals are validation-
or review-gated. The figure makes the residual legible and it is non-zero.

## Step 5 — Repair the roster's reason field — SATISFIED

Measured by re-running the roster assembly the release consumes,
`fetch_manifest_source_release_rows` over the 355 manifest paths
(`graph_ops.py:12267`), and counting unnamed rows whose
`non_nameable_reason` is empty:

```text
fresh assembly: 355 rows → 336 named / 19 unnamed; unnamed rows with an empty
non_nameable_reason: 0
```

The stored frozen roster recorded **18** of its 31 unnamed rows with an empty
reason; a fresh assembly under the current code returns **0**. Every unnamed
row now ends with a stated cause — either a transcribed `last_error` /
`skip_reason` (e.g. `calorimetry/group/component/power` → "missing device
vocabulary token 'component'", `camera_x_rays/camera/camera_dimensions` →
"compose claim-attempt cap reached", the fit-constraint weights →
`dd_node_category_ineligible … fit_artifact`), or the explicit honest statement
"cause not recorded" when the graph state is genuinely silent. The transcription
fix (`non_nameable_reason = last_error or skip_cause or "cause not recorded"`,
blanked only when the row has a name) is present in this tree, so an empty
reason can no longer be written for an unnamed row.

Residual sub-items, not a "recording no cause" failure, left as follow-on work:
two `…/time` rows and two constraint weights still carry a weaker
`compose_model_skipped` reason where the deterministic coordinate /
`fit_artifact` verdict is the correct mechanism (the classification reason-string
repair §9a already names), and the genuinely-silent rows now say "cause not
recorded" rather than guessing.

---

## Method note

All live-graph reads ran on the login node under the login-local-tunnel
exception; each was bounded to the 214-name batch or the 355 paths and returned
in under a second except the export itself (~12 s, consisting of many indexed
queries). No graph write was issued by this census; the only files written are
this evidence document, the staged catalog files under `/tmp/tail-export-now/`
and `/tmp/tail-export-now-r4/` (the export's own staging), and the retained
logs. Census JSON: `census-steps1345.json`; export run-2 log:
`export-accounting-run2.log`; refused re-run log:
`export-blocked-etendue-run4.log` — all under the run directory.
