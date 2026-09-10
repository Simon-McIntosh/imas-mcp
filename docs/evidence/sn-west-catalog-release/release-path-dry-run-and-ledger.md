# Release path: dry run writes nothing, and the exclusion-ledger link appears once

**Base:** `cb53d77d9`. Two defects in the review-release path, both gated by the same
node because they share `catalog_release.py`. Each section records the before and
after behaviour and the test names that hold the after state.

## 1. A release dry run is inert

**Defect.** `run_review_release` computed the next RC and called
`_freeze_review_artifact` unconditionally before the `dry_run` branch was consulted,
so a rehearsal wrote a release-shaped roster into
`imas_codex/standard_names/manifests/reviews/`, wrote into the staging directory, and
consumed an RC number that names no release — the origin of the untracked rc2, rc3 and
rc4 rosters. An earlier fix (`a9638457e`) moved the freeze after the `dry_run` branch,
but the export still ran first, so the staging write remained.

**After (this node).** The `dry_run` branch now sits *before* the export. A rehearsal
resolves the focus file, computes the RC it would take, reports the roster path it
would freeze, and returns without creating the staging directory, invoking the
exporter, freezing a roster, creating a branch, or moving the RC counter. The RC
counter is derived (read from existing git tags by `compute_next_version`, which
writes nothing), so an inert rehearsal cannot burn a number.

| Behaviour | Before | After |
|---|---|---|
| Roster in `manifests/reviews/` | written by the freeze | never written |
| Staging directory + export output | created / written | neither created nor touched |
| RC counter | advanced by a release that never happens | unchanged |
| Reported `artifact_path` / `rc_version` | present | still present (the would-be path and candidate) |

**Tests holding it** (in `tests/standard_names/test_review_release.py`):

- `test_review_release_dry_run_no_push_no_pr` — exporter never invoked, staging dir
  never created, no roster written, no branch created, no PR.
- `test_dry_run_writes_no_roster_and_moves_no_candidate` — the reviews directory is
  byte-for-byte unchanged, the staging directory is absent, and the next candidate
  `compute_next_version` would take is unmoved.
- `test_real_run_writes_exactly_one_artifact_and_advances_candidate_once` — the
  paired real run: seeding `v0.1.0rc1+first-batch` and cutting a real release writes
  exactly one `v0.1.0rc2+second-batch` roster and advances the next candidate to
  `v0.1.0rc3+second-batch`, i.e. exactly one artifact and exactly one RC step.

The export-machinery tests that previously leaned on a dry run to exercise the write
path (`test_review_export_restores_approved_entry_bytes`,
`test_review_assembly_reproduces_and_closes_sparse_baseline_failure`) now run as real
cuts, because a rehearsal no longer performs the export.

## 2. The exclusion-ledger link is appended at most once

**Defect.** `body_with_exclusion_ledger_link` appended the ledger blob address
unconditionally, so a body that already linked the ledger — authored inline, or
composed by an earlier pass — carried the same information twice in two shapes. The
ledger-to-repository-relative-path resolution was inlined inside
`exclusion_ledger_blob_url`, giving two computations of one thing.

**After.** The resolution `_ledger_relative_path` is a shared helper used by both
`exclusion_ledger_blob_url` and `body_with_exclusion_ledger_link`, so both callers
derive the same answer. The appender returns the body **unchanged** when the
repository-relative path already appears in it, and otherwise appends exactly one
markdown link with descriptive text (never a bare URL).

| Behaviour | Before | After |
|---|---|---|
| Body already naming the ledger by its relative path | gained a second address | returned byte-identical |
| Body without the ledger | appended one link | still exactly one link and exactly one copy of the URL |

**Tests holding it** (in `tests/standard_names/test_review_body_ledger_link.py`):

- `test_body_already_naming_the_ledger_is_returned_unchanged` — authored body equal to
  the returned body, byte for byte.
- `test_body_lacking_the_ledger_gains_exactly_one_markdown_link` — the descriptive
  link and the address each appear exactly once.

## Gate

`tests/standard_names` on a `_debug` partition at the node's head; added failures
reported against base `cb53d77d9` rather than absolute green.
