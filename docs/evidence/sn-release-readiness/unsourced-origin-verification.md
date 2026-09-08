# Coordinator verification — n-srr-why-72-identities-have-no-source

Base `69f24046b`, checked 2026-09-08 against the live graph and the code.

## Confirmed exact

| Claim | Verdict |
|---|---|
| total `StandardName` = 5,108 | **exact** |
| bulk catalog-fold fingerprint `created_at 2026-07-04T21:20:38.632Z` | **exact, and larger than stated** — the timestamp is real and shared by **1,925** identities, a single bulk import; the report's 36 are the unsourced chain-cap subset of it |
| `write_standard_names` at `graph_ops.py:5125` | **exact** |
| `persist_generated_name_winners` at `graph_ops.py:7119` | **exact** |
| `RefinedNamePersistenceRefusalReason.AUTHORITATIVE_SOURCE_COHORT_EMPTY` | **exact** — declared `graph_ops.py:17640`, raised `:18170`, guarded by `tests/standard_names/test_refine_source_cohort_gate.py:89` |
| carve-out `_allow_empty_noop=(not authoritative_cohort_observed or bool(edit_mode))` | **verbatim** at `graph_ops.py:18408-18410` |
| `9c4403dba` 09-01 19:13 require refine source provenance | **exact** |
| `5c6c015fb` 09-01 19:17 merge refusing an ungrounded predecessor | **exact** |
| `8ee947b04` 09-01 21:35 preserve source-less human edits | **exact** |
| `4b949931` 07-09 remove the bulk catalog import | **exact** |

## Corrected — the runtime-lag exposure is more than double what was reported

The report's conclusion is right and is the finding worth keeping: automated refine
minted unsourced successors *after* the guard merged, so the running environment was
executing pre-fix code. The magnitude is understated.

Partitioning every unsourced identity created since `5c6c015fb` (09-01T19:17:55Z) by route:

| Route | Count |
|---|---|
| `edit_mode=rename` (governed carve-out, by design) | 36 |
| no `refine_reason` — compose or structural, outside the refine route | 25 |
| **automated refine** — `refine_reason` set, no `edit_mode`, not `sn-edit` | **20** |
| governed `sn-edit` run ids | 2 |
| `edit_mode=hint` | 1 |
| **total unsourced creations since the merge** | **84** |

- The report says **nine** automated-refine mints; the measured figure is **20**.
- The report says the window ends **2026-09-04T13:54Z**; the last automated-refine
  mint is **2026-09-04T15:27:31**, about ninety minutes later.
- The report says "since 09-04 13:54Z the only unsourced mints are `sn-edit`
  renames". Directionally right about the 09-05 cohort, which is the governed
  `edit_mode=rename` carve-out — but **false as stated**, because an automated
  refine sits after that cutoff.

**Consequence for the next node: a remediation sized on nine would under-scope by
more than half.** Use 20, and treat 09-04T15:27 as the boundary.

## Not verified by the coordinator

The chain-cap census itself — 168 at `chain_length>=3`, 93 unsourced, 84 with no
source-bearing ancestor, split 48 legacy / 36 catalog-fold. Reproducing it needs the
report's own chain-walk definition. The direct count of identities with no
`PRODUCED_NAME` producer at all is **2,280**, which is a different and much wider
measure and neither confirms nor refutes 93.

## Process note

`test_logs` cites `/tmp/census.py`, `/tmp/roots.py` and seven more. Those are
ephemeral and outside the repository, so the reproduction scripts behind a 407-line
report are already unreadable to anyone but that worker. The repository's own rule
puts session scratch in the harness scratchpad and evidence under `docs/evidence/`.
