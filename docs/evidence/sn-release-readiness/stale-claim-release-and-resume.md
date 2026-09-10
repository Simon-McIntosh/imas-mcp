# Stale-claim release and drain resume at the authorised ceiling

## Purpose

The capped tail drain (recorded in
[`tail-pipeline-drain.md`](tail-pipeline-drain.md)) stalled: drain worker
processes were killed mid-invocation and left `claim_token`/`claimed_at` on
rows the pools could not re-claim, so the pipeline reported no eligible work
while budget remained. This record captures (a) the confirmed stale-claim
count, (b) the one-shot orphan sweep that released them, and (c) the resumed
drain over the live non-exhausted tail under the same fences as the original
run, against the same **USD 150 cumulative** ceiling.

All counts come from bounded, read-only `GraphClient` queries against the live
graph on 2026-09-10, immediately before the release (and again after the
drain). Definitions mirror `tail-pipeline-drain.md`:

- **Live** = `name_stage <> 'superseded'`.
- **Fully accepted on both axes** = `name_stage='accepted'` **and**
  `docs_stage='accepted'`.
- **Exhausted cohort** = `name_stage='exhausted'`; `contested` is empty.
- **Run scope** = live, name-terminal states excluded (`exhausted`,
  `contested`), catalog-status-terminal rows excluded (`status` in
  `superseded`/`deprecated`), and not fully accepted on both axes.
- **Missing a name score** = `reviewer_score_name IS NULL`; **missing a docs
  score** = `reviewer_score_docs IS NULL`; **no documentation text** =
  `coalesce(trim(documentation),'') = ''`.

## Confirmed stale-claim state (pre-sweep, 2026-09-10)

A bounded read immediately before the sweep counted live claim tokens:

| Measure | Count |
|---|---|
| StandardName rows with `claim_token IS NOT NULL` | **92** |
| oldest `claimed_at` | 2026-09-10T04:20:11Z |
| newest `claimed_at` | 2026-09-10T04:41:07Z |
| StandardNameSource rows with `claim_token IS NOT NULL` | 0 |

All 92 timestamps are hours past the 600 s orphan threshold; the workers that
held them died at 04:20–04:41Z (the interrupted first two slices of the tail
drain). This confirms the coordinator's measurement of 92 rows with the
oldest claim at 2026-09-10T04:20:11Z exactly.

## The one-shot release

Claims were released by the existing orphan-sweep module, **one-shot**, not by
writing `claim_token`/`claimed_at` directly and not by the global maintenance
pass (which carries an unrelated open delete defect):

- **Entry point:** `_orphan_sweep_tick(timeout_s=600)` from
  `imas_codex.standard_names.orphan_sweep` — the module's synchronous
  one-pass function that executes all five `_SWEEP_QUERIES` in separate
  transactions.
- **Threshold:** 600 s, the same age the pipeline's background sweep uses.

| Sweep leg | Released |
|---|---|
| `name_refining` (revert stale `refining` → `reviewed`, clears claim) | 3 |
| `docs_refining` | 0 |
| `stale_token_sn` (non-refining StandardName claims) | 89 |
| `stale_token_source` | 0 |
| `compose_attempt_cap` | 0 |
| **Total claims released** | **92** |

**Post-sweep verification:** a follow-up read returns **0** rows with a claim
token and **0** rows in either `refining` stage — 3 of the 92 claimed rows
were the stuck `name_stage='refining'` cohort (reverted, claim cleared) and
89 were non-refining claims cleared in place.

## Before-state of the drain (post-release, pre-run, 2026-09-10)

### Whole live partition

| Population | Count |
|---|---|
| Total StandardName nodes | 5,103 |
| Superseded (`name_stage='superseded'`) | 2,151 |
| **Live (non-superseded)** | **2,952** |
| Fully accepted on both axes | 2,412 |
| **Run scope** (live, non-terminal, not fully accepted, not catalog-status-terminal) | **248** |
| — of which exhausted cohort (`name_stage='exhausted'`) | 281 (excluded from run) |
| — of which contested | 0 |

The run scope (248) is wider than the 166 recorded as the slice-5 target in
`tail-pipeline-drain.md`; the graph moved between the two reads (the held
claims were themselves half the delta — claimed rows sit inside the scope).
This record's figures are the authoritative state at release time.

### Run scope by name_stage

| name_stage | Count |
|---|---|
| accepted | 64 |
| reviewed | 109 |
| drafted | 61 |
| pending | 11 |
| refining | 3 |
| **Total** | **248** |

(The 3 `refining` rows are the ones the sweep reverted to `reviewed`; they
remain in scope.)

### Run scope by docs_stage

| docs_stage | Count |
|---|---|
| pending | 131 |
| accepted | 48 |
| drafted | 41 |
| reviewed | 23 |
| exhausted | 3 |
| (null) | 2 |
| **Total** | **248** |

### Missing-score / missing-docs figures

Within the run scope (248):

| Measure | Count |
|---|---|
| Missing `reviewer_score_name` | 74 |
| Missing `reviewer_score_docs` | 209 |
| No documentation text | 109 |

Over the full live population (2,952):

| Measure | Count |
|---|---|
| Missing `reviewer_score_name` | 420 |
| Missing `reviewer_score_docs` | 451 |
| No documentation text | 288 |

## Pipeline resume (fences as before)

Invocation per slice (single scoped command, no piping):

```
uv run --no-sync imas-codex sn run \
  --name <current run-scope identities> \
  --skip-global-maintenance \
  --cost-limit <remaining-budget> \
  --time 8
```

- `--skip-global-maintenance` on every invocation: global structural
  maintenance runs a superseded-child cleanup that removes protected parent
  edges without consulting protection — a broad run would exercise the defect.
- `--name` scopes preflight the set atomically and never seed DD sources.
- `--cost-limit` is the remaining cumulative ceiling, computed from the
  ledger, not a fresh allowance. No `--reseed`, no `--force`.
- `--time 8` bounds each slice to 8 minutes so a lost turn costs one slice
  rather than the drain. A time-limit stop is a completed wave.
- Budget is **cumulative across the whole campaign**: USD 72.123238 of the
  150 USD ceiling is already spent, so each slice's `--cost-limit` is 150
  minus the campaign total at that point.

### Campaign spend baseline (computed from the LLMCost ledger)

| Measure | Value |
|---|---|
| Campaign rows (`llm_at >= 2026-09-10T04:00Z`) | 1,284 |
| Campaign spend | **USD 72.123238** |
| Overspend | 0.0 |
| **Remaining under the 150 USD ceiling** | **USD 77.876762** |

(The dispatch-time figure "72.12 already spent" is confirmed: the ledger sums
to 72.123238. This is the cumulative ceiling across both drains — not a fresh
allowance — and each slice is capped at the remaining figure.)

## After-state (recorded as slices land)

### Slice 1 (run `f2562940…`, 07:37Z–07:45Z)

Broad-pool over the 248-identity scope with
`--skip-global-maintenance --cost-limit 77.876762 --time 8`. Clean
time-limit stop.

| Run record field | Value |
|---|---|
| `stop_reason` | time_limit_reached |
| `cost_spent` | USD 12.003563 |
| `cost_limit` | USD 77.876762 |
| `names_reviewed` | 55 |
| `names_enriched` | 2 |
| `names_regenerated` | 8 |
| `elapsed_s` | 542.6 |

Pool summaries (run's own output): review_docs 33 / 8.2212, review_name
22 / 2.4474, refine_docs 7 / 1.0213, refine_name 1 / 0.2924, generate_docs
2 / 0.0212, generate_name and enrich_parents 0.

#### Cumulative spend vs the ceiling (updated)

- After slice 1: **USD 84.126801** (1,449 LLMCost rows) — ledger delta
  matches the run's own figure exactly.
- Remaining: **USD 65.873199** (150 − 84.126801).

#### Live tail shape after slice 1 (measured post-stop, same queries)

| Metric | Release-time | After slice 1 | Delta |
|---|---|---|---|
| Fully accepted both axes | 2,412 | **2,451** | **+39** |
| Live (non-superseded) | 2,952 | 2,952 | 0 |
| Run scope | 248 | **205** | −43 |
| Missing name score (full live) | 420 | 408 | −12 |
| Missing docs score (full live) | 451 | **422** | −29 |
| No documentation text (full live) | 288 | 287 | −1 |
| name_stage=exhausted (full live) | 281 | 284 | +3 |
| Live claim tokens at stop | 0 | **31** | +31 |

Run scope after slice 1 (slice-2 target) — name_stage: reviewed 110,
drafted 43, accepted 38, pending 11, refining 3. docs_stage: pending 126,
accepted 36, reviewed 24, drafted 14, exhausted 3, (null) 2. Missing in
scope: name 57, docs 165, no docs 105.

The 31 claim tokens present at stop are **fresh** (claimed_at 07:36–07:45Z,
the slice's own processing window; `claimed_at` ages are measured seconds to
a few minutes at the post-stop read) — the pipeline's background sweep gates
on 600 s age, so these are normal in-flight state to be re-claimed in later
slices, not evidence of re-accumulation. Recorded as pipeline state rather
than swept again.
