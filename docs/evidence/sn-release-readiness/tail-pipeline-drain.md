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
| **Run scope (tail minus exhausted)** | **511** |

### Run scope by name_stage

| name_stage | Count |
|---|---|
| accepted | 206 |
| drafted | 169 |
| reviewed | 125 |
| pending | 11 |
| **Total** | **511** |

### Run scope by docs_stage

| docs_stage | Count |
|---|---|
| pending | 335 |
| drafted | 75 |
| accepted | 72 |
| reviewed | 24 |
| exhausted | 3 |
| (null) | 2 |
| **Total** | **511** |

### Missing-score / missing-docs figures

Within the **run scope (511)**:

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

_Filled after the run completes._

## Spend and pools (post-run)

_Filled after the run completes._
