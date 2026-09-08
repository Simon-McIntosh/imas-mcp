# Section currency — which lifecycle-integrity rows are still open

**Recovered by the coordinator from the worker manifest of
`n-sli-which-section-rows-are-still-open`.** That run reported `status: complete`
with substantive findings but never committed the report it declared, and its
process was gone before it could be asked to. Its manifest is the durable
delivery and the file was the convenience that did not land — so the findings are
transcribed here, each marked with whether I re-measured it.

Base `69f24046b`; graph re-queried 2026-09-08.

## Verified against the live graph by the coordinator

| Claim | Verdict |
|---|---|
| status census: `draft` 2977, `superseded` 2126, null 5 | **exact** |
| `active` 0 and `deprecated` 0 — no catalog request has ever merged | **exact** (both absent from the census) |
| the null residual is 5 names, all `name_stage=drafted` | **exact** |
| those 5 were created 2026-09-07T07:10:40–07:14:20 | **exact**, to the second |
| sources at the compose attempt cap (≥5) = 221 | **exact** |
| 0 rows carry `superseded_by` while `accepted` + `draft` | **exact** — that footgun is closed |

## Not reconciled

`fit_artifact` — the manifest reports 53 live. Counting `IMASNode` with
`node_category='fit_artifact'` returns **473**. The manifest's population
definition is narrower than that query and is not stated, so the two are not
comparable rather than in conflict. Anyone acting on the 53 should re-derive it.

## Row verdicts

**Closed, and re-verified by the worker against code at HEAD:**

- §2, the exported status — the `active` literal is gone (grep 0); projections
  select `sn.status` at `export.py:533` and `:635`; a null-status cut is refused
  at `export.py:2402-2431`.
- §4, the import editorial-axis guard — the positive allow-list is
  `record_catalog_import_provenance` at `catalog_import.py:51-106`.
- §5a, `updated_at` — declared at `standard_name.yaml:988`, 83 stamps in
  `graph_ops.py` and 165 package-wide, behind the static gate
  `tests/graph/test_updated_at_coverage.py`.
- §5 footgun 6, the retarget pointer — guard at `signed_manifest.py:3110-3143`,
  and the live population is 0 as measured above.

**Write path closed, repair population still outstanding:**

- reasonless failures — the write path is closed at `graph_ops.py:10278`, but
  **59 of 101 rows remain** unexplained. This is the cohort the separately
  abandoned triage node was meant to classify, and it is still unclassified.
- synthetic review verdict — the gate now writes a typed
  `semantic_similarity_gate` method and makes no routing claim, but **38 rows
  remain scoreless**.

**Still open:**

- **the compose attempt cap** — 221 sources at the cap, 60 of them with no name.
  A repair population, not 221 physics problems.
- **the `superseded_by` scalar retirement** — the decision is locked and has
  **not landed**; the field is still declared and written in six modules. The
  retarget subclass is closed; the field-and-successorless cohort is not.

## Two findings the audit produced that were not asked for

1. **`persist_reviewed_name`'s authority-overwrite defect has no identified
   closing commit.** A non-reviewer verdict overwriting a quorate scalar was
   named on 2026-09-06; gate rows no longer write a scalar, so the live artifact
   population is the repaired 3 — but **the write path itself is unverified**. A
   focused check is owed before this is called closed.
2. **A strict "zero null status" invariant is not yet true.** The batch mint path
   at `graph_ops.py:5289` sets no status on CREATE, so freshly minted names carry
   null until the next `reconcile_catalog_status` pass heals them. The 5 residual
   rows above are exactly that window, and all 5 sit outside the accepted/approved
   export population, so no cut is exposed.
