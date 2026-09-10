# Drain of the live non-exhausted tail at the authorised ceiling

## Purpose

Drive the live non-exhausted StandardName tail toward accepted on both lifecycle
axes through a broad, cost-capped `sn run` at the authorised ceiling of
**USD 150**, and report the live tail shape before and after the run with
bounded reads. The exhausted cohort (name-axis rotation cap already reached) is
excluded from the run and reported separately — another rotation there spends
money and changes nothing; it waits for a separate rescore pass.

## Method

All counts come from bounded, read-only GraphClient queries against the live
graph on 2026-09-10, immediately before the run (and again after it). Definitions:

- **Live** = `name_stage <> 'superseded'`. Supersession is marked by
  `name_stage='superseded'` (`superseded_by` scalar is set on only 18 rows;
  no live row carries a successor marker), so this is the load-bearing
  discriminator.
- **Fully accepted on both axes** = `name_stage='accepted'` **and**
  `docs_stage='accepted'`.
- **Exhausted cohort** = `name_stage='exhausted'` (name-axis rotation cap
  reached); `contested` is currently empty (0 rows).
- **Run scope** = live, name-terminal states excluded (`exhausted`,
  `contested`), and not fully accepted on both axes. This is the population the
  pipeline run targets.
- **Missing a name score** = `reviewer_score_name IS NULL`.
- **Missing a docs score** = `reviewer_score_docs IS NULL`.
- **No documentation text** = `coalesce(trim(documentation), '') = ''`.

## Before-state (measured pre-run, 2026-09-10)

### Whole live partition

| Population | Count |
|---|---|
| Total StandardName nodes | 5,085 |
| Superseded (`name_stage='superseded'`) | 2,133 |
| **Live (non-superseded)** | **2,952** |
| Fully accepted on both axes | 2,182 |
| Live tail (live minus fully accepted) | 770 |
| — of which exhausted cohort (`name_stage='exhausted'`) | 259 |
| — of which contested | 0 |
| — of which terminal on the catalog-status axis (`status` superseded/deprecated) | 2 |
| **Run scope (tail minus exhausted minus status-terminal)** | **509** |

Two of the 770 tail identities are terminal on the catalog-status axis
(`status='superseded'` while pipeline name_stage is `reviewed`), so the
`--name` preflight refuses them as terminal lifecycle:
`accumulated_methane_carbon_13_count_due_to_gas_injection` and
`accumulated_total_gas_count_at_midplane_due_to_gas_injection`. They are
excluded from the run and reported here so the scope arithmetic closes.

### Run scope by name_stage

| name_stage | Count |
|---|---|
| accepted | 206 |
| drafted | 169 |
| reviewed | 123 |
| pending | 11 |
| **Total** | **509** |

(The T0 511 slice was accepted 206, drafted 169, reviewed 125, pending 11; the
two removed status-terminal rows were both `reviewed`, so reviewed drops 125 →
123 on the exact run scope.)

### Run scope by docs_stage

| docs_stage | Count |
|---|---|
| pending | 335 |
| drafted | 75 |
| accepted | 70 |
| reviewed | 24 |
| exhausted | 3 |
| (null) | 2 |
| **Total** | **509** |

(The two removed rows carry `docs_stage='accepted'`, so the T0 511-slice docs
stages are unchanged except accepted 72 → 70.)

### Missing-score / missing-docs figures

Within the **run scope (509)**, measured at T0 before the run:

| Measure | Count |
|---|---|
| Missing `reviewer_score_name` | 220 |
| Missing `reviewer_score_docs` | 466 |
| No documentation text | 252 |

Over the **full live population (2,952)**:

| Measure | Count |
|---|---|
| Missing `reviewer_score_name` | 493 |
| Missing `reviewer_score_docs` | 660 |
| No documentation text | 403 |

### Reconciliation with the dispatch-time figures

The dispatch carried: *2919 live identities, 2151 fully accepted on both axes,
leaving 768, of which 493 lack a name score, 660 lack a docs score and 403 have
no documentation text; ~258 name_stage=exhausted*.

- The **missing-count trio (493 / 660 / 403) is confirmed exactly** when those
  three measures are taken over the full live population — which is what they
  were measured against, not a subset of the tail.
- The **totals differ at ~1%**: this node reads live = 2,952 (vs 2,919),
  fully accepted = 2,182 (vs 2,151), tail = 770 (vs 768), exhausted = 259
  (vs ~258). The exhausted and tail deltas are single rows; the live/fully-
  accepted deltas (33 and 31 rows) are consistent with graph movement between
  the dispatch-time snapshot and this pre-run read, and with a slightly
  different population cut in the dispatch query. This node's numbers, with the
  definitions above, are the authoritative before-state.

## Pipeline run

Invocation (single scoped command, no piping):

```
uv run --no-sync imas-codex sn run \
  --name <the 511 run-scope identities> \
  --skip-global-maintenance \
  --cost-limit 150 \
  --time <fits the node wall-clock>
```

- `--skip-global-maintenance` on every invocation: global structural
  maintenance runs a superseded-child cleanup that removes protected parent
  edges without consulting protection — a broad run would exercise the defect.
- `--name` scope preflights the set atomically and never seeds DD sources, so
  the run drains existing identities rather than minting new ones.
- `--cost-limit 150` is the authorised ceiling; measured hard to within one
  call on recent capped runs. `--time` bounds wall-clock inside the node fence.
- No `--reseed`, no `--force`. Mid-pipeline stop on time or cost is a completed
  wave and is reported as such.

### Pool behaviour expected

The default all-pool loop claims whatever each pool finds eligible within the
scope: generate_name composes the 11 `pending` rows, review_name scores the
drafted/reviewed/pending rows, refine_name recovers reviewed rows below the
0.85 bar, and the docs pools (generate/review/refine) drive documentation for
name-accepted rows. The exhausted docs rows (3) and the exhausted name cohort
(259) are not claimable and are reported as remaining.

## After-state (measured post-run, 2026-09-10)

### Slice 1 (run `4e87869b-b8c8-49ae-ae3d-709e363b0219`, 04:21:15Z–04:26:58Z)

The first pipeline invocation ran broad-pool over the 509-identity scope with
`--skip-global-maintenance --cost-limit 150 --time 28`. The worker process was
interrupted after 414 s (the node's process ended mid-run), so `stop_reason`
is `interrupted` rather than a clean cap stop. That is a completed wave: the
run's own reported figure is the authoritative spend.

| Run record field | Value |
|---|---|
| `stop_reason` | interrupted |
| `cost_spent` | USD 30.504402 |
| `cost_limit` | USD 150.0 |
| `names_reviewed` | 174 |
| `names_enriched` | 172 |
| `names_regenerated` | 43 |
| `elapsed_s` | 413.8 |

Ledger (LLMCost, `llm_at >= 2026-09-10T04:00Z`, all rows in this one run):
**694 rows, USD 30.504402** — matches the run record exactly.

| Pool | Rows | USD |
|---|---|---|
| review (docs-axis review) | 169 | 12.014247 |
| review_name | 279 | 9.991374 |
| refine_name | 35 | 4.307425 |
| refine_docs | 28 | 3.267625 |
| generate_docs | 172 | 0.745194 |
| refine_name+fanout | 1 | 0.167748 |
| enrich_parents | 10 | 0.010789 |
| **Total** | **694** | **30.504402** |

### Cumulative spend vs the ceiling

- Ceiling: **USD 150 cumulative across this node** (not per invocation) — hard stop.
- Spent after slice 1: **USD 30.504402**.
- Remaining: **USD 119.495598**.

### Live tail shape after slice 1 (measured 2026-09-10 ~06:29Z, same queries as T0)

| Metric | T0 (pre-run) | After slice 1 | Delta |
|---|---|---|---|
| Live (non-superseded) | 2,952 | 2,952 | 0 |
| Fully accepted both axes | 2,182 | **2,237** | **+55** |
| Missing `reviewer_score_name` (full live) | 493 | **417** | **−76** |
| Missing `reviewer_score_docs` (full live) | 660 | 620 | −40 |
| No documentation text (full live) | 403 | **292** | **−111** |
| name_stage=exhausted (full live) | 259 | 280 | +21 |
| Run scope (live, non-terminal, not fully accepted) | 509 | **384** | −125 |

Cross-check: the independently measured after-slice-1 figures (fully accepted
2,205; missing name score 420; no docs 307) agree with these to within the
window of late writes from the interrupted run; this node's read is taken
after the process fully stopped and is the authoritative current state.

Run scope after slice 1 (the slice-2 target) by axis:

| name_stage | Count | docs_stage | Count |
|---|---|---|---|
| accepted | 233 | drafted | 202 |
| reviewed | 114 | pending | 133 |
| drafted | 26 | accepted | 18 |
| pending | 11 | reviewed | 27 |
| | | exhausted | 2 |
| | | (null) | 2 |
| **Total** | **384** | **Total** | **384** |

Missing within slice-2 scope: name score 104, docs score 339, no docs text 112.

### Slice 2 (planned)

`sn run --name <384> --skip-global-maintenance --cost-limit 119.495598 --time 15`.
Bounded to one turn; cumulative ceiling 150 enforced by shrinking the cap each
slice. Preflight dry-run passed.

## Spend and pools (post-run)

_Cumulative ledger and per-slice pool tables are recorded in the After-state
sections above; appended as further slices land._
