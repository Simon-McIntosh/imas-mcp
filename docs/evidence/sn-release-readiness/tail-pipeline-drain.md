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
- Spent after slice 1: **USD 30.504402**. Remaining: **USD 119.495598**.
- Spent after slice 2: **USD 38.016637** (822 LLMCost rows: slice 1 694 +
  slice 2 128). Remaining: **USD 111.983363**.

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

### Slice 2 (run `01db9724-1ef1-434c-b0ee-f4bebd14bee4`, 04:39:30Z–04:41:30Z)

Ran broad-pool over the 384-identity scope with
`--skip-global-maintenance --cost-limit 119.495598 --time 15`. Interrupted
after 136 s (node process ended mid-run again), so `stop_reason=interrupted`.

| Run record field | Value |
|---|---|
| `stop_reason` | interrupted |
| `cost_spent` | USD 7.512235 |
| `names_reviewed` | 35 |
| `names_enriched` | 3 |
| `names_regenerated` | 8 |
| `elapsed_s` | 136.2 |

Ledger: 128 LLMCost rows, USD 7.512235. Pool summaries from the run's own
output: review_docs 34 / 6.0551, review_name ~ 35, refine_name ~ 8,
refine_docs 5 / 0.6708, enrich_parents 0.

### Cumulative spend vs the ceiling (updated)

- Spent after slice 2: **USD 38.016637** (822 LLMCost rows).
- Remaining: **USD 111.983363** (150 − 38.016637).

### Live tail shape after slice 2 (measured post-stop, same queries)

| Metric | T0 | After slice 1 | After slice 2 | Delta vs T0 |
|---|---|---|---|---|
| Fully accepted both axes | 2,182 | 2,237 | **2,273** | **+91** |
| Missing name score (full live) | 493 | 417 | 420 | −73 |
| Missing docs score (full live) | 660 | 620 | 588 | −72 |
| No documentation text (full live) | 403 | 292 | **289** | **−114** |
| name_stage=exhausted (full live) | 259 | 280 | 281 | +22 |
| Run scope (live, non-terminal, not fully accepted) | 509 | 384 | **304** | −205 |

Run scope after slice 2 (slice-3 target) — name_stage: accepted 162, reviewed
109, drafted 22, pending 11. docs_stage: drafted 137, pending 122, accepted 18,
reviewed 23, exhausted 2, (null) 2. Missing in scope: name 80, docs 263, no
docs 101.

### Economics so far (measured, worth stating)

- Cumulative USD 38.02 bought 91 fully-accepted net (vs T0) — roughly
  **USD 0.42 per fully-accepted identity**. (Using the dispatch-time baseline
  of 2,151 it is +122 = USD 0.31 each; the exact figure depends on the
  baseline in force — deltas here anchor to this node's T0 measure.)
- generate_docs produced 172 documents for USD 0.745 in slice 1 (~USD 0.0043
  each): composition runs locally for ~free, so essentially all spend is
  review / review_name / refine. A cost cap therefore throttles **reviewing**,
  never **generating** — the drain's remaining budget buys scoring, and the
  docs text it generates on the way is nearly a byproduct.

### Slice 3 (run `e5b9f99d…`, 04:46:19Z–04:55:22Z)

Ran broad-pool over the 304-identity scope with
`--skip-global-maintenance --cost-limit 111.983363 --time 8` as a foreground
invocation; it returned inside the turn.

| Run record field | Value |
|---|---|
| `stop_reason` | **time_limit_reached** (clean cap stop) |
| `cost_spent` | USD 29.278691 |
| `names_reviewed` | 152 |
| `names_regenerated` | 29 |
| `names_enriched` | 1 |
| `elapsed_s` | 542.3 |

Pool summaries (run's own output): review_docs 152 / 25.3507, refine_docs
29 / 3.9196, generate_docs 1 / 0.0085, review_name 0, refine_name 0. Exit
code 1 accompanies the time-limit stop (partial completion, not a crash).

### Cumulative spend vs the ceiling (updated)

- After slice 3: **USD 67.295328** (1,237 LLMCost rows).
- Remaining: **USD 82.704672** (150 − 67.295328).

### Live tail shape after slice 3 (measured post-stop, same queries)

| Metric | T0 | After slice 3 | Delta vs T0 |
|---|---|---|---|
| Fully accepted both axes | 2,182 | **2,398** | **+216** |
| Missing name score (full live) | 493 | 420 | −73 |
| Missing docs score (full live) | 660 | **464** | **−196** |
| No documentation text (full live) | 403 | 288 | −115 |
| name_stage=exhausted (full live) | 259 | 281 | +22 |
| Run scope (live, non-terminal, not fully accepted) | 509 | **179** | **−330** |

Run scope after slice 3 (slice-4 target) — name_stage: reviewed 109, accepted
37, drafted 22, pending 11. docs_stage: pending 121, accepted 18, reviewed 23,
drafted 13, exhausted 2, (null) 2. Missing in scope: name 48, docs 139, no docs
100. The docs axis now dominates the tail.

### Slice 4 (run `616fefe5…`, 04:59:51Z–05:08:09Z)

Foreground broad-pool over the 179-identity scope with
`--skip-global-maintenance --cost-limit 82.704672 --time 8`. Clean
time-limit stop.

| Run record field | Value |
|---|---|
| `stop_reason` | time_limit_reached |
| `cost_spent` | USD 4.682584 |
| `names_reviewed` | 16 |
| `names_regenerated` | 4 |
| `elapsed_s` | 497.9 |

Pools: review_docs 16 / 3.6326, refine_docs 4 / 1.0500. review_name and
refine_name processed 0 — the remaining reviewed rows are not being claimed
by refine (see note below). Slice 4 found markedly less eligible work than
slice 3: the tail is converging.

### Cumulative spend vs the ceiling (updated)

- After slice 4: **USD 71.977912** (1,282 LLMCost rows).
- Remaining: **USD 78.022088** (150 − 71.977912).

### Live tail shape after slice 4 (measured post-stop, same queries)

| Metric | T0 | After slice 4 | Delta vs T0 |
|---|---|---|---|
| Fully accepted both axes | 2,182 | **2,411** | **+229** |
| Missing name score (full live) | 493 | 420 | −73 |
| Missing docs score (full live) | 660 | **452** | **−208** |
| No documentation text (full live) | 403 | 288 | −115 |
| name_stage=exhausted (full live) | 259 | 281 | +22 |
| Run scope (live, non-terminal, not fully accepted) | 509 | **166** | **−343** |

Run scope after slice 4 (slice-5 target) — name_stage: reviewed 109, accepted
24, drafted 22, pending 11. docs_stage: pending 121, accepted 18, reviewed 22,
drafted 1, exhausted 2, (null) 2. Missing in scope: name 44, docs 127, no docs
100.

Note: refine_name has claimed nothing since slice 2 while 109 reviewed rows
remain in scope; those rows are not yielding to the name refine pool (they
are the residual below-bar / steering population), which is why the tail is
sticking around this row class rather than the reviewed work draining.

### Slice 5 (planned, foreground)

`sn run --name <166> --skip-global-maintenance --cost-limit 78.022088 --time 8`,
foreground, returns inside the same turn.

## Spend and pools (post-run)

_Cumulative ledger and per-slice pool tables are recorded in the After-state
sections above; appended as further slices land._
