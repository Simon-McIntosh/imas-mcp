# Worktree recovery audit — prior-session trees carrying uncommitted content

Nine worktrees from earlier sessions were present at the start of session
`ship-s10-20260907`. Every one of their HEAD commits is reachable from `main`.
Five carried uncommitted content, which the 2026-09-04 drain ledger deliberately
did not reclaim on the grounds that it was "uncommitted content this session did
not author". That is the correct default, and it left an unknown behind. This
audit settles each by content rather than by assumption.

## Verdicts

| Worktree | Uncommitted content | Verdict | Proof |
|---|---|---|---|
| `ship-s10-20260901/n-locus-rule-reads-what-the-path-measures` | `attachment_audit.py` +70/-4; new `test_locus_rule_semantics.py` | **superseded** | its test file is **byte-identical** to `tests/standard_names/test_locus_rule_semantics.py` in `main`; the source change landed at `8e2ed486f` |
| `ship-s10-20260901/n-tense-rule-respects-operator-scope` | `workers.py` +99/-3; new `test_tense_operator_scope.py` | **superseded** | its test file is **byte-identical** to `tests/standard_names/test_tense_operator_scope.py` in `main`; same landing commit `8e2ed486f` |
| `s12hyg-20260825a/n-importwriteguard` | `graph_ops.py` +56; new `test_import_write_guard.py` | **superseded** | it adds `record_catalog_import_provenance`, which landed at `d606bf09` in a different module — `imas_codex/standard_names/catalog_import.py:51` — with an equivalent positive allow-list, and is covered by `tests/standard_names/test_import_write_authority.py` |
| `ship-s10-20260901/n-ledger-link-appears-once` | `catalog_release.py` +21/-10; new `test_review_body_ledger_link.py` | **genuinely open — retained** | neither the guard nor its test exists in `main`; this is live prior art and the node repairing it was pointed at this tree |
| `west-demo-20260901d/n-west-demo-rc` | **modifies a tracked frozen roster**, `v0.3.0rc1+west-task-2e.sn_names.yaml` +15/-13 | **retained for adjudication** | a frozen release artifact is not a file to edit in place; what this change intended is not recoverable from the tree, and destroying it is not a coordinator call |

Two further trees carry only an untracked roster that is already committed to
`main` (`n-west-recut-reduced-surface` holds `v0.4.0rc1`,
`n-wcr-the-divergence-scan-names-its-subtrees` holds a `v0.4.0rc5`) — see the
roster-identity finding on the release plan, because those two copies are the
evidence that the version string does not identify a roster.

## The finding that outlives the cleanup

**Three of five trees held work that had already been re-implemented and landed
under a different module or commit, and nothing recorded that.** Byte-identical
test files are the strongest available signal and they cost one `diff` each to
check. The 2026-09-04 ledger's caution was right in the absence of that check;
what it could not say is that the caution was protecting duplicates. An
inventory that stops at "dirty, not mine" preserves the content and preserves
the unknown with it — so the audit belongs in the same beat as the inventory.
