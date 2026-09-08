# Recovered exclusions: the three WEST rows a repair can return to the batch

## Result

Three exclusions were recovered so they stop leaving the WEST batch as
accounted exclusions, and the export accounting was re-run over the 214-name
batch at the newly merged roster. The arithmetic moved exactly as the repair
predicted, with one concurrent effect stated rather than claimed:

| Metric | Before | After |
| --- | ---: | ---: |
| Candidate count | 214 | **213** |
| Published count | 196 | **199** |
| `missing_physics_domain` | 2 | **0** |
| `accounting_residue` | 0 | **0** |
| Exclusions (total) | 18 | 14 |

The candidate count falls by exactly one because a superseded identity is no
longer *selected* as a candidate at all. Two null-domain identities now carry a
physics domain and publish. Of the two additional published names beyond that
pair, one is attributed to the concurrent quarantine revalidation (below), not
to this node.

## The three identities, before and after

### 1. `total_electron_count` — assigned `electromagnetic_wave_diagnostics`

- **Source path:** `interferometer/electrons_n`
- **Domain before:** null → **after:** `electromagnetic_wave_diagnostics`
- **Evidence:** the quantity is the total free-electron inventory inferred from
  a plasma interferometer, which measures electron density through the phase
  shift of a probing microwave ("electromagnetic wave") beam; the repository's
  IDS-to-physics-domain mapping
  (`definitions/physics/ids_domains.json`) files the `interferometer` IDS under
  `electromagnetic_wave_diagnostics`, matching every other interferometer-bound
  standard name in the graph.
- **Now:** exported. In the before run it was excluded only for the null
  domain; after the assignment it passes every subsequently-reached predicate
  (valid, accepted, documented, scored 0.96875) and emits.

### 2. `vertical_coordinate_of_ece_channel` — assigned `electromagnetic_wave_diagnostics`

- **Source path:** `ece/channel/position/z`
- **Domain before:** null → **after:** `electromagnetic_wave_diagnostics`
- **Evidence:** the quantity is the measurement position of an
  electron-cyclotron-emission channel, and ECE is the microwave radiometry of
  electron cyclotron radiation — an electromagnetic-wave diagnostic; the
  repository's IDS-to-physics-domain mapping files the `ece` IDS under
  `electromagnetic_wave_diagnostics`.
- **Now:** exported. Scored 0.99375; excluded before only on the null domain.

Both assignments went through the sanctioned repair path, the `sn reclassify`
command (``reclassify_standard_name_domain``), which reassigns
`physics_domain`/`source_domains` atomically with the `HAS_PHYSICS_DOMAIN`
edge and records a `StandardNameChange` event whose reason states the physics.
No ad-hoc Cypher was used, and the domain is grounded in what each source
measures rather than chosen to let the name through.

### 3. `normalized_toroidal_beta` — the tombstone, no longer selected

- **Identity:** `normalized_toroidal_beta` — `status: superseded`,
  `name_stage: reviewed`.
- **Before:** admitted into the candidate population through the review-batch
  membership clause `(sn.name_stage = 'approved' OR sn.id IN $batch)`, then
  dropped by eligibility as `name_not_accepted` and counted as an exclusion on
  every cut.
- **After:** not selected. Candidate selection now suppresses superseded
  identities before the population is formed, so the identity is never a
  candidate and the export never counts it.

**Predicate changed:** `_fetch_export_population` (the population selection
that feeds `candidate_count`) and, for consistency, `_fetch_candidates` now
both apply the shared tombstone clause
`coalesce(sn.status, '') <> 'superseded' AND coalesce(sn.name_stage, '') <> 'superseded'`
added to the `WHERE` of every selection branch. Both tombstone encodings are
suppressed because ``status`` carries the catalog tombstone and ``name_stage``
can carry the same state for lifecycle-retired identities. The two queries
share the one predicate so the candidate count and the eligibility decision
cannot drift apart.

**Measured:** the batch population fetched before the change was 214; after
the change it is **213** — down by exactly one — and `normalized_toroidal_beta`
is absent while both recovered names remain present.

## The export accounting after the repair

Re-run of the real WEST review export over the same 214-name batch
(`skip_gate=True`, login-node graph tunnel, report persisted to
`west-export-after-recovered/.export_report.json`):

```text
candidate_count             213
published_count             199
accounted exclusions         14
candidate - published        14
accounting residue             0
  invalid_validation_status   4
  missing_physics_domain      0
  name_not_accepted           1
  never_reviewed              9
```

- **candidate 214 → 213** — the tombstone left candidate selection.
- **`missing_physics_domain` 2 → 0** — both recovered names publish; each is
  in the emitted 199-identity set.
- **`name_not_accepted` 2 → 1** — only `hot_neutral_temperature` remains;
  `normalized_toroidal_beta` is no longer an exclusion because it is no longer
  a candidate.
- **`invalid_validation_status` 5 → 4** — `vertical_coordinate_of_line_of_sight`
  now reads `validation_status: valid` in the graph and publishes. This fifth
  quarantined row is **not** this node's recovery; the quarantine instrument
  node revalidated it concurrently, and the +1 in the published count that is
  not one of the two domain assignments belongs to that concurrent work. This
  node handled only its three rows and touched nothing in the quarantined set.
- **`never_reviewed` 9 → 9** — the docs-review lifecycle question, owned by a
  follow-on node, is untouched.

Both recovered names publish; neither is withheld by any later predicate after
the domain assignment, so no per-name failure needs stating for them.

## The repair paths used

- Domain assignment: `imas-codex sn reclassify <name> --domain ... --reason ...`
  (the backing `reclassify_standard_name_domain` code path) — reproducible,
  recorded, no ad-hoc Cypher.
- Tombstone selection: a source change to
  `imas_codex/standard_names/export.py` adding a shared tombstone predicate to
  the two selection queries, with a regression test
  (`tests/standard_names/test_export_tombstone_selection.py`) asserting the
  predicate is present in every selection branch and suppresses both retirement
  fields.
- No export predicate was weakened to admit a name: the two recovered names
  satisfy the unchanged predicates once their domain field is populated, and
  the tombstone satisfies no predicate because it is never asked.

## Run and gate records

- Live-graph work ran on the login node (the Neo4j tunnel is login-local),
  bounded to the 214-id batch and the three identities, every query under the
  ten-second ceiling, exception declared in the worker manifest.
- Export driver: `/tmp/run_west_export.py`; run log:
  `west_export_run.log`; report: `/tmp/west-export-after-recovered/.export_report.json`.
- Suite gate: `tests/standard_names` (default markers, on the `all_debug`
  partition):
  - baseline at `origin/main` `6a5525dc6`: **43 failed, 7209 passed**.
  - after at the merged head `14b8a9693` (`origin/main` + the tombstone-selection
    change): **43 failed, 7213 passed**.
  - added failures: **0**; added passes: **4** (the new tombstone-selection
    regression tests). The failure sets are identical between the two runs;
    the 43 are the current-main shared base, including a supersede/tombstone
    test cluster that lives in the successor-migration write scope rather than
    this node's. Logs: `baseline-suite-main.log`, `after-suite.log`.
