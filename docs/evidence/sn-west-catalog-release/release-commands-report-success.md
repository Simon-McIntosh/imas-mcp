# Release commands: no success without the substance behind it

**Base:** `ff1755f6c` — the HEAD the three defects were confirmed against,
before this node changed anything. Three defects recorded 2026-09-07, each a
command that reported success while something it was responsible for did not
happen. Each section first states whether the defect still reproduces at
HEAD, quoting the reproduction or saying plainly that it no longer does, then
records what this node changed — or why the repair was not this node's to
make. Two sibling release defects (the dry run and the ledger link) were found
already fixed by adjacent work; the same "confirm before fixing" discipline is
applied here.

## 1. The compose gate fails the whole run rather than its pool — still reproduces at HEAD

**Defect.** The compose gate treats an unreachable generation endpoint as
fatal to the whole `sn run` rather than to the generation pool, so a wrong
address in one seat stopped every pool — including review, docs and refine
pools that use only remote models. Measured on 2026-09-07: thirteen sources
sat parked at `status='extracted'` awaiting `generate_name` while the model
served normally, because the run had claimed nothing before it exited.

**Confirmation at HEAD (`ff1755f6c`), not a re-statement.** The gate is
`_require_local_compose_ready` in `imas_codex/cli/sn.py:156-168`. When a run
can dispatch generation (`_run_can_generate_names`, `sn.py:126-142`,
returns `True` for an unscoped run) and the effective `sn-compose` seat routes
to a configured local endpoint (`_configured_local_compose_required`,
`sn.py:144-154`), `async_main` (`sn.py:883-887`) calls it *before any pool
claims anything*:

```python
if local_compose_required:
    if service_monitor is None and not drain_scope_id:
        _require_local_compose_ready(compose_model)
    elif service_monitor is not None and not await _await_services_or_stop(
        service_monitor, stop_event
    ):
        return {"summary": None}
```

An unreachable endpoint makes `_require_local_compose_ready` raise
`click.ClickException` — or, under the Rich monitor, block startup
indefinitely — so the run ends before the first claim and the remote-only
pools never get a turn. The defect therefore still reproduces at the whole-run
level for generation-capable runs.

**Why this node did not repair it.** The gate lives in `imas_codex/cli/sn.py`;
a genuine pool-scoped degradation (only the `generate_name` pool fails, while
the other pools continue on their own seats) additionally needs the pool
construction / readiness path in `imas_codex/standard_names/loop.py` and
`pools.py`. None of those files is inside this node's write fence, and
`loop.py` is concurrently held by a peer. Named under **Follow-ons**.

## 2. An unreachable seat's message and its blast radius — message half already fixed, scoping half open

**Defect.** The failure text named the endpoint, sending the reader outward at
a healthy service instead of at the configuration that was wrong; and an
unreachable seat stopped pools that did not use it.

**Message half — NO LONGER REPRODUCES at HEAD.** `_require_local_compose_ready`
(`sn.py:160-167`) names the *configuration*, never the endpoint URL:

> "The configured local Standard Names compose endpoint is required for
> generation but is unavailable ({detail ...}). Restore the Ambix vLLM service
> and retry; no paid provider fallback will be used."

A reader is pointed at the seat and its remedy, not at a healthy service.
Held by `test_failed_local_readiness_blocks_without_paid_fallback` in
`tests/standard_names/test_sn_service_checks.py`.

**Irrelevant-pool half — NO LONGER REPRODUCES for runs that do not use the
seat.** A run that cannot dispatch generation (`--only review`,
`--only review_name`, `--only reconcile`, `--only attach`, `--docs-only`,
`--skip-generate`, `--flush`) never probes the local compose endpoint:
`_run_can_generate_names` returns `False` and the probe is not registered.
Held by `test_non_generation_routes_do_not_probe_local_compose` and
`test_run_capability_identifies_local_compose_dependency` in the same file.

**Residual.** A generation-capable (unscoped) run still gates the whole run on
the compose seat, which is defect 1 and is named in the same **Follow-on**.

## 3. A post-copy divergence must not exit 0 — reproduced at HEAD, fixed in this node

**Defect.** The release printed a post-copy finding — 196 diverged entries on
the 2026-09-07 WEST cut — and exited 0, because the divergence was a
`logger.warning` and nothing landed in the publish report's errors.

**Confirmation at HEAD (`ff1755f6c`), quoted.** `run_publish` in
`imas_codex/standard_names/publish.py:504-516` (pre-change lines) ran the
post-copy check then logged and carried on:

```python
check_result = check_catalog(isnc)
if check_result.diverged:
    logger.warning(
        "Post-copy check found %d diverged entries",
        len(check_result.diverged),
    )
```

No error reaches `PublishReport.errors`, so `sn release` — which raises
`SystemExit(2)` when `report.errors` is non-empty (`imas_codex/cli/sn.py:
4974-4979`) — printed the finding and returned 0. Run against the pre-change
`publish.py`, this node's divergence test reproduces the defect shape exactly:
the log shows `publish.py:511 Post-copy check found 1 diverged entries` while
`report.errors` stays empty and the tree is committed.

**After (this node).** A real divergence now refuses the publish. The post-copy
block appends the refusal to `report.errors` and returns before the git commit,
so the release command exits non-zero and no known-diverged tree is committed.
The refusal message — rendered by the new `CheckResult.describe_divergence` in
`imas_codex/standard_names/catalog_import.py` — names the divergence count and
the first diverged identities. The check stays best-effort only against graph
*availability*: an uncomparable tree (graph unreachable) is skipped, never a
block, so an otherwise-healthy publish is unchanged.

**Tests holding both directions** (in `tests/standard_names/test_catalog_import_divergence.py`):

| Direction | Test | Asserts |
|---|---|---|
| failure | `TestPublishRefusesDivergedTree::test_diverged_tree_is_refused_with_an_error` | a diverged check lands in `report.errors` (message names the count and identity) and the ISNC checkout keeps its pre-publish commit |
| healthy | `TestPublishRefusesDivergedTree::test_in_sync_tree_still_publishes` | an in-sync tree commits exactly as before, empty error list |
| healthy | `TestPublishRefusesDivergedTree::test_uncomparable_tree_still_publishes` | a graph-unavailable check is skipped, not a block — commits exactly as before |
| helper | `TestDescribeDivergence::test_agreement_renders_none` | no divergence renders `None` (silent healthy path) |
| helper | `TestDescribeDivergence::test_single_diverged_identity_is_named` | one diverged identity renders "1 diverged entry" and its name |
| helper | `TestDescribeDivergence::test_many_diverged_identities_named_with_cap` | many render the count, the first five, and "and N more" |

The pre-change defect (warning + exit 0 + commit) is the exact negation of the
failure-direction test: that test fails on the pre-change `publish.py` and
passes on the fixed one, so the signal is not silenced and no exit code was
widened to zero.

## Gate

- Focused suite (publish + catalog-import coverage, incl. the new file):
  `test_publish_transport_only.py`, `test_catalog_import_divergence.py`,
  `test_catalog_entry_discovery.py`, `test_catalog_layout_hierarchy.py`,
  `test_export_sidecar_surface.py`, `test_no_bulk_import.py` — 67 passed,
  exit 0, no graph.
- Full `tests/standard_names/` on an `*_debug` partition: see the node
  manifest `baseline_suite` / `after_suite`, attributed against the stated
  base. Added failures are reported there, not here.

## Follow-ons

1. **Compose-gate scoping (defects 1 + the residual of 2).** A generation-capable
   `sn run` still aborts before claiming when the local compose seat is
   unreachable, even though review/docs/refine use only remote seats. The
   desired behaviour — the gate fails *its own* pool and the run degrades — needs
   `imas_codex/cli/sn.py` (drop the pre-claim raise, or scope it to the
   `generate_name` pool), and for a real pool-scoped degradation the pool
   construction / readiness path in `imas_codex/standard_names/loop.py` (peer-held)
   and `pools.py`. A right-realised treatment should verify that an unscoped run
   with a dead compose seat still completes its review/docs work while the
   generation counter reports the pool down.
2. **Comparison scope for domain-subset publishes.** `check_catalog` compares
   the whole checkout tree (old sibling-domain files plus the freshly copied
   subset) against the whole graph of accepted/approved names. A graph that
   moved on a *non-updated* domain now refuses a domain-subset publish (an
   error) instead of printing a warning and exiting 0. If that fires on healthy
   subset cuts, the comparison should be scoped to the publish scope — the
   "or the comparison is wrong" reading of the plan, now surfacing as a refusal
   rather than a silent zero. Instrumenting the comparison scope is outside this
   node's write fence.
