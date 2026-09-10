# The orphan sweep is a safety net, not a mutation

Measured 2026-09-10. Code revision before the fix: `cb53d77d9`
(`imas_codex/standard_names/loop.py`). Fix landed in `6e8e0bcbd`. This node
decouples the orphan sweep from the `skip_global_maintenance` flag so that a
scoped run or bounded drain — which correctly wants none of the graph-wide
mutation writers — still gets the sweep that releases claims abandoned by a
dead process.

## The incident: a suppressed safety net locked the pipeline

On 2026-09-10 a bounded drain run passed `--skip-global-maintenance` to bypass
an unrelated open delete defect. The drain's worker processes were killed
mid-invocation by turn-end. Because the sweep was bundled under the same flag
the drain had set, no sweeper ran to release the claims those dead workers
held. The measured consequence:

- 92 `StandardName` rows kept live `claim_token` values;
- the oldest `claimed_at` among them was `2026-09-10T04:20:11Z`;
- a claimed row is ineligible, so the pipeline reported no work while
  approximately 300 non-exhausted rows still needed processing and 78 USD of
  authorised budget remained unused.

The stall was not a budget or capacity problem: it was a claim-ownership
invariant left unrestored because the sweeper that restores it had been
switched off.

## Gating before the fix

At `loop.py` (near the original line 2113) the sweep was coupled to the
maintenance bypass in an if/elif structure:

```python
if not skip_global_maintenance:
    sweep_task = asyncio.create_task(
        run_orphan_sweep_loop(
            interval_s=DEFAULT_ORPHAN_SWEEP_INTERVAL_S,
            timeout_s=DEFAULT_ORPHAN_SWEEP_TIMEOUT_S,
            stop_event=stop_event,
        ),
        name="orphan_sweep",
    )
elif drain_scope_id:
    drain_heartbeat_task = asyncio.create_task(
        run_manifest_drain_heartbeat_loop(...),
        name="manifest_drain_heartbeat",
    )
```

`skip_global_maintenance` is forced true whenever `drain_scope_id` is set
(`loop.py:1283`), so a drain run took the `elif` branch only: lease heartbeat
but no sweep. A scoped run with `--skip-global-maintenance` and no drain took
neither branch.

## The distinction the coupling missed

- **A mutation** — the graph-wide reconciliation `skip_global_maintenance`
  suppresses (`reconcile_standard_name_sources`, `mark_orphaned_standard_name_runs_stale`,
  `release_all_orphan_claims`, the structural fixups, the embed worker, …) —
  rewrites live rows. Suppressing it in a scoped run is the point of the flag.
- **A safety net** — the orphan sweep only reverts rows whose claim is
  *abandoned*: `name_stage`/`docs_stage` stuck in `refining` with a
  `claimed_at` older than `DEFAULT_ORPHAN_SWEEP_TIMEOUT_S` (1800 s) or absent,
  or a stale non-refining `claim_token`. Releasing that claim restores the
  invariant that a claimed row belongs to a live worker; it cannot rewrite a
  row any live worker owns.

Skipping the second must not disable the first: clearing a dead process's
claim is invariant restoration, not maintenance.

## Why always-on rather than a third flag

A separate `--keep-orphan-sweep`-style flag was considered and rejected:

1. **The sweep is never harmful to a scoped run.** Its queries match only
   rows with a stale or absent claim within a 30-minute window — far longer
   than any batch — or a source that has hit a hard `attempt_count` cap. No
   live claim falls in that set.
2. **The failure mode is silent lock.** A run that stops mid-invocation leaves
   claims stranded; the sweeper is the only periodic restorer. Every mode in
   which a worker can die unrecovered — scoped, drain, ordinary — wants it on.
   Opt-in control makes the dangerous case the default again.
3. **Maintenance-only modes are already excluded structurally.**
   `reconcile_only` returns at `loop.py:2064` and `attach_only` at
   `loop.py:1328`, both before the sweep block, so an unconditional sweep can
   never reach a run that performs no operational workers.
4. **A new flag would need CLI plumbing and a default decision** at every call
   site, buying no behaviour the current mode cannot express: the sweep is
   unconditionally safe, so the honest default is unconditionally on.

The bounded-drain case deserves a note: the drain forces the bypass flag, so
the sweep is now *additive* alongside the lease heartbeat. The two protect
different invariants — the heartbeat refreshes `drain_scope_claimed_at`
(scope-lease liveness), the sweep releases stale worker `claim_token`/`claimed_at`
(worker-claim liveness) — and neither conflicts with the other. The drain's own
`recover_manifest_drain_scope` still handles lease-expiry recovery at
finalization.

## Gating after the fix

```python
# The orphan sweep is a safety net, not a mutation ...
sweep_task: asyncio.Task[None] | None = None
sweep_task = asyncio.create_task(
    run_orphan_sweep_loop(
        interval_s=DEFAULT_ORPHAN_SWEEP_INTERVAL_S,
        timeout_s=DEFAULT_ORPHAN_SWEEP_TIMEOUT_S,
        stop_event=stop_event,
    ),
    name="orphan_sweep",
)
if drain_scope_id:
    drain_heartbeat_task = asyncio.create_task(
        run_manifest_drain_heartbeat_loop(...),
        name="manifest_drain_heartbeat",
    )
```

Unchanged by this fix:

- `skip_global_maintenance` still bypasses every mutation writer (the embed
  worker remains under its own `if not skip_global_maintenance:`, as do the
  `_global_maintenance_call` set and the post-drain structural fixups).
- The drain heartbeat behaviour for `drain_scope_id` is intact; it now simply
  starts alongside the sweep instead of instead of it.
- `reconcile_only` and `attach_only` early returns still precede the sweep
  block, so maintenance-only modes never start the sweep.
- No other guard was weakened or removed; `_global_maintenance_call` semantics
  are untouched.

## Tests

`tests/standard_names/test_orphan_sweep.py`, Section 3, drives `run_sn_pools`
over a mocked graph/worker boundary and holds both directions:

- `test_sweep_starts_when_scoped_maintenance_is_bypassed` — with
  `skip_global_maintenance=True` the `run_orphan_sweep_loop` task is still
  created, while the embed worker bundled with the old gate is not.
- `test_global_maintenance_writers_remain_bypassed_while_sweep_runs` — every
  graph-wide mutation writer (`reconcile_standard_name_sources`,
  `mark_orphaned_standard_name_runs_stale`, `release_all_orphan_claims`,
  `resolve_doc_links`, structural fixups, drift refresh, harmonize restamp)
  stays uncalled under the flag while the sweep runs.
- `test_drain_starts_sweep_and_manifest_heartbeat_together` — a bounded drain
  keeps both its lease heartbeat and the sweep, and still skips the mutation
  writers.

Regression fold (pre-existing, outside this node's write scope): two tests in
`tests/standard_names/test_scoped_global_maintenance.py`
(`test_scoped_run_bypasses_complete_global_maintenance_set`,
`test_scoped_idle_completion_refuses_transient_claim_residue`) asserted the
old coupling — that `run_orphan_sweep_loop` must not be called under the
bypass flag. They fail by design after this fix; their fix (drop the sweep
from the not-called set, assert it instead) is recorded as follow-up work.
