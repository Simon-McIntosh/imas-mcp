# Release-path residue currency at HEAD

**Base:** `69f24046b`. **Read by the coordinator**, because the worker node dispatched for
this deliverable was abandoned twice over by local-lane saturation and the endpoint is
still saturated; re-dispatching would have cost lane capacity to re-derive a set of greps.
Every verdict below names the file and line that settles it, so each is falsifiable.

## Verdicts

| Residue | Verdict at HEAD | What settles it |
|---|---|---|
| The export report's verdict does not gate publication | **closed** | `catalog_release.py:1804-1815` reads `all_gates_passed is False`, collects the failed gate names, appends an error and `return report` — it stops before publishing |
| Pull-request calls still shell out to the `gh` CLI | **closed** | no `gh` subprocess remains in `imas_codex/standard_names/`; `read_pull_request_body` (`:110`) and `update_pull_request_body` (`:99`) are REST calls on `github_client` |
| A stale `reclassify_domain` is still defined | **closed** | no `def reclassify_domain` anywhere in `imas_codex/`; the token survives only as a `StandardNameChange.operation` value written at `graph_ops.py:186` |
| The deterministic fallback body omits `REVIEWING.md` | **closed** | `release_notes.py:417` emits it as a markdown link with descriptive text. The residual is that `REVIEWING_GUIDE_URL` (`:388`) hardcodes owner and repository — owned by the live node `n-swcr-the-reviewing-guide-address-derives` |
| The post-copy divergence scan walks the catalog's virtual environment | **not reproducible at HEAD** | both traversals are scoped: `publish.py:145` sets `sn_dir = staging_dir / "standard_names"` and `:149` `rglob`s only that subtree; the post-copy check at `:433` uses a non-recursive `isnc_sn_dir.glob("*.yml")`. Neither can reach a `.venv`. The recorded 387 environment-YAML divergences did not come from either, so the original diagnosis pointed elsewhere and the real traversal is unidentified |
| A dry run mints a roster and burns a candidate number | **split: freeze inert, identity still broken** | see below |

## The dry run no longer writes, and the number is still not an identity

The freeze half is genuinely closed. `run_review_release` returns at its `dry_run` branch
(`catalog_release.py:1836-1848`) *before* `_freeze_review_artifact` is reached at `:1856`,
and the surrounding comment states the intent: the repository stays byte-identical and the
counter does not move for a release that never happens.

**The identity half is not closed, and the mechanism is that the number is derived rather
than allocated.** `run_review_release` takes its candidate version from
`compute_next_version(isnc_path, bump, ...)` at `:1780`, which reads the *existing git tags*
in the catalog checkout. A dry run creates no tag. So:

- **two runs with no intervening tag compute the same number**, and the later roster
  overwrites or duplicates the earlier one under one version string; and
- **two runs with a different `bump` compute divergent numbers**, in an order unrelated to
  when they ran.

Both are visible in this repository's own committed artifacts, and that is what raises this
from hygiene to correctness:

- `v0.4.0rc5+west-task-2e.sn_names.yaml` exists **twice with different bytes** — 214 names
  (`minted_at 2026-09-07T10:29:37`, committed `cfb63454d`) and 226 names
  (`minted_at 2026-09-05T08:45:08`, still uncommitted in the
  `n-wcr-the-divergence-scan-names-its-subtrees` worktree). They share 201 names; the 13
  main-only and 25 worktree-only spellings split cleanly along the operator-rendering
  migration, so one was frozen before it and one after.
- `f782e6529` committed `v0.10.0rc1` at 11:47 and `cfb63454d` committed `v0.4.0rc5` at
  12:32 — 45 minutes later, carrying the **lower** version — for **identical** name sets
  (added 0, dropped 0), both with `pr_number: null`.
- `f782e6529`'s message states "196 emitted identities" for an artifact whose `names` list
  holds **214**: the roster freezes the *candidate* set, not the published set, and nothing
  in the artifact says which it is.

So the repair the plan still owes is not about writing less. It is that **a roster must
declare its own identity** — which set it carries and a content hash — and that a freeze
must refuse to write a version that already exists with different bytes. The two surviving
worktrees are the fixture for exactly that test.
