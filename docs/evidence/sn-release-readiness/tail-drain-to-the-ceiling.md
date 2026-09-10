# Tail drain to the authorised ceiling with the orphan sweep running

## Purpose

Resume the capped tail drain recorded in
[`stale-claim-release-and-resume.md`](stale-claim-release-and-resume.md) now
that the orphan sweep starts unconditionally (see
[`stale-claim-release-and-resume.md`](stale-claim-release-and-resume.md)).
That prior drain stalled because each slice left claims the next slice
refused: the sweep was bundled under the same gate as the reconcile writers,
so a scoped run lost the reaper exactly when a dead worker had left claims on
rows, and the age-blind exact-name preflight then refused those rows until a
sweep ran. The repair (this repo, HEAD `563a613d5`) moves the sweep outside
that gate so it starts on every run regardless of
`--skip-global-maintenance`.

This record covers the drained tail: the census and cumulative spend before
and after, the claim state before and after each slice (with timestamps), and
whether the pipeline still reports eligible work when stopped.

All counts come from bounded, read-only `GraphClient` queries against the
live graph on 2026-09-10, immediately before and after each slice.
Definitions mirror `stale-claim-release-and-resume.md`:

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
- **Budget is cumulative across the whole campaign**: every slice's
  `--cost-limit` is 150 minus the campaign total at that point, never a fresh
  allowance.

## Before-state (2026-09-10, ~11:47 local / 09:47Z)

Picked up exactly where `stale-claim-release-and-resume.md` left off: the
graph rents the 31 claims that had blocked the previous slice 2, and the
campaign ledger sat at the same figure that record reported.

| Population | Count |
|---|---|
| Total StandardName nodes | 5,104 |
| Live (non-superseded) | 2,952 |
| **Fully accepted on both axes** | **2,451** |
| **Run scope** (live, non-terminal, not fully accepted) | **215** |
| — name_stage: reviewed 111, drafted 52, accepted 38, pending 11, refining 3 | |
| — docs_stage: pending 136, accepted 36, reviewed 24, drafted 14, exhausted 3, (null) 2 | |
| Missing name score (full live) | 408 |
| Missing docs score (full live) | 422 |
| No documentation text (full live) | 287 |
| Missing in scope: name 66 / docs 175 / no docs 115 | |

### Prior blocker state (the stalled claims)

| Measure | Count |
|---|---|
| Live claims present on entry | **31** |
| oldest `claimed_at` | 2026-09-10T07:36:44Z |
| newest `claimed_at` | 2026-09-10T07:45:42Z |

These are the exact rows the previous drain's slice 1 left when it hit its
time limit at 07:45Z; no worker ever returned for them, and they were what
made the previous slice 2 refusal ("current worker claim", age-blind) — the
wedge that repair exists to remove. They were reaped at this node's start.

### Campaign spend at entry (LLMCost ledger, `llm_at >= 2026-09-10T04:00Z`)

| Measure | Value |
|---|---|
| Campaign rows | 1,449 |
| Campaign spend | **USD 84.126801** |
| Overspend | 0.0 |
| **Remaining under the 150 USD ceiling** | **USD 65.873199** |

## The sweep is running (repair confirmation)

Both invocations started the orphan sweep unconditionally under
`--skip-global-maintenance`. Log line, slice 1 (`sn_sn-compose.log`,
2026-09-10 11:48:40 local):
`run_sn_pools: Orphan sweep loop started (interval=30s, timeout=1800s)` —
the same line appears at slice 2 (12:09:14 local). The bundled mutating
reconcile writers stayed bypassed: `reconcile complete — 0 actions`.
Maintenance-only modes still return before the sweep block.

Stale claims were released via the existing orphan-sweep turn,
`_orphan_sweep_tick(timeout_s=600)` from `imas_codex.standard_names.orphan_sweep`,
the same synchronous one-pass mechanism the pipeline's background loop uses,
not by writing `claim_token`/`claimed_at` directly and not via the global
maintenance pass.

| Sweep (each slice boundary) | Released |
|---|---|
| Pre-slice-1: `name_refining` / `stale_token_sn` | 3 / 28 = **31** (the stalled block) |
| Pre-slice-2: `name_refining` / `stale_token_sn` | 4 / 15 = **19** (slice 1's in-flight stop) |

### Claim count before and after each invocation

| Moment (local) | Live claims | `claimed_at` range |
|---|---|---|
| Before node (stalled block) | 31 | 07:36:44Z–07:45:42Z (hours old, refused by preflight) |
| After slice 1 stopped | 19 | 09:48:40Z–09:57:13Z (slice 1's own window) |
| Before slice 2 (after age-out + sweep) | 0 | — |
| After slice 2 stopped | **12** | 10:09:15Z–10:15:56Z (slice 2's own window) |

Claims no longer accumulate as stranded inventory: every stop leaves only the
just-stopped slice's in-flight rows, all minutes old, and they age past the
600 s orphan threshold a few minutes after the stop. The next slice's
boundary sweep clears them exactly as this node cleared the 31 that had
blocked the previous drain. The repair took.

## Slice 1 (run started 2026-09-10 11:48:38 local / 09:48:38Z)

`--name <215 identities> --skip-global-maintenance --cost-limit 65.87 --time 8`

Run record (`SNRun`, started 09:48:37.955Z): `stop_reason=time_limit_reached`,
`cost_spent=USD 11.424299`, `cost_limit=65.87`, `elapsed_s=542.5`,
`names_reviewed=42`, `names_regenerated=13`.

Post-slice ledger (`10:03Z` read):

| Measure | Value |
|---|---|
| Campaign rows | 1,585 |
| Campaign spend | **USD 95.551100** |
| Remaining under the 150 USD ceiling | **USD 54.448900** |
| Fully accepted both axes | **2,469** (+18) |
| Run scope | 191 (−24) |
| Missing name (full live) | 394 (−14) |
| Missing docs (full live) | 410 (−12) |

The ledger delta (95.551100 − 84.126801 = 11.424299) matches the run record
exactly.

## Slice 2 (run started 2026-09-10 12:09:12 local / 10:09:12Z)

The 15 slice-1 stragglers were under the 600 s orphan age at the boundary
(claimed 09:56:00Z–09:57:13Z), so the pre-slice-1 sweep could not touch them;
they aged out minutes later and the boundary sweep cleared all 15, restoring
the full 191-identity scope.

`--name <191 identities> --skip-global-maintenance --cost-limit 54.45 --time 8`

Run record (`SNRun`, started 10:09:12.581Z): `stop_reason=time_limit_reached`,
`cost_spent=USD 4.275100`, `cost_limit=54.45`, `elapsed_s=542.3`,
`names_reviewed=14`, `names_regenerated=3`.

## After-state (2026-09-10, ~12:19 local / 10:19Z)

| Measure | Entry | Final | Delta |
|---|---|---|---|
| Fully accepted both axes | 2,451 | **2,478** | **+27** |
| Run scope | 215 | **179** | −36 |
| Missing name score (full live) | 408 | 391 | −17 |
| Missing docs score (full live) | 422 | 404 | −18 |
| No documentation text (full live) | 287 | 282 | −5 |
| `name_stage=exhausted` (full live) | 284 | 293 | +9 |
| Live claim tokens at stop | 31 | **12** | −19 |

Run scope composition at stop — name_stage: reviewed 109, drafted 31,
accepted 25, pending 11, refining 3. docs_stage: pending 126, reviewed 24,
accepted 23, exhausted 3, (null) 2, drafted 1. Missing in scope: name 45 /
docs 139 / no docs 105.

### Cumulative spend (final for this record)

| Measure | Value |
|---|---|
| Campaign rows (`llm_at >= 2026-09-10T04:00Z`) | 1,628 |
| Campaign spend | **USD 99.826166** |
| Overspend | 0.0 |
| **Remaining under the 150 USD ceiling** | **USD 50.173834** |
| Spend this node (2 slices) | USD 15.699365 |

### Does the pipeline still report eligible work when you stop?

Yes. At the stopped state the run scope is **179 identities**, of which the
12 slice-2 in-flight rows are the only ones carrying a claim; 167 are
unclaimed and immediately eligible. The pipeline processes real work to a
clean time-limit stop in both directions (2-3 slices would spend the
remaining ~USD 50 at the measured ~USD 4–12 per 9-minute slice of
review-driven burn). The previous wedge — hours-old claims with no worker
process to sweep them — is gone: the repair starts the reaper on every
scoped run, and the boundary sweep returns claims abandoned by a stopped or
dead slice within minutes of their 600 s age.

## Fences respected

- `--skip-global-maintenance` on every invocation (the graph-wide reconcile
  writers carry an unrelated open delete defect), `--name` scopes preflight
  the exact set atomically and never seeds DD sources, `--time 8` bounds each
  slice to eight minutes, `--cost-limit` is the remaining cumulative ceiling
  at each launch. No `--reseed`, no `--force`.
- The exhausted cohort (`name_stage='exhausted'`, refine cap spent) is
  excluded from scope by definition; the run never reached it.
- Live graph work ran on the login node (the Neo4j tunnel is login-node-local);
  every query above is a bounded indexed read. No CLI output was piped or
  redirected; the run's own log (`~/.local/share/imas-codex/logs/sn_sn-compose.log`)
  is the evidence. The CLI's non-zero exit on each slice is the expected
  `time_limit_reached` signal (`_require_terminal_drain` refuses exit 0 until
  `no_eligible_work`), not a crash: both `SNRun` records are
  `stop_reason=time_limit_reached` and the ledger sums match the run-reported
  spend exactly.
- No figure was produced and no image read: this lane is not multimodal.
