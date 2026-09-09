# Validation observation writer

## Result

Every non-null validation verdict carried by the rich Standard Name persistence
query now receives a `validated_at` observation in the same Cypher `SET`. A
persist with no incoming verdict preserves the stored timestamp, so it cannot
manufacture an observation merely by re-coalescing an unchanged status.

The implementation was made at base revision
`2dcc80afbf0c03cd5a6b3188825910df9363e8f0` and committed as
`0602ebf1a6a787b4b93e1125cf1d79aa8a93aca1`.

## Pre-landing whole-suite exposure

The candidate source change and both regression tests were first applied to an
archive scratch copy of the base revision. The assigned worktree was unchanged
while this measurement ran. The complete `tests/standard_names` suite ran on
the `all_debug` partition.

| Measurement | Count | Files |
| --- | ---: | --- |
| Passed | 7,314 | package-wide |
| Observed failures | 1 | `tests/standard_names/test_export_determinism.py` |
| Failures exposed by the writer invariant | 0 | none |

The one observed failure was
`TestManifestDeterminism::test_no_commit_uses_stable_unversioned_timestamp`.
An archive scratch copy has no `.git` metadata, so export tests require the
documented `SOURCE_DATE_EPOCH` fallback. That variable supplies the base commit
time, while this particular test deliberately asserts the different no-commit,
no-environment fallback of `1970-01-01T00:00:00Z`. It is therefore a scratch
harness conflict, not a product or fixture failure exposed by validation
stamping. The initial archive run without the fallback is also retained: its 13
failures all stopped at the expected "export provenance timestamp unavailable"
guard before exercising the writer change.

## Producer and persistence topology

The pooled composer records its inline audit result in
`imas_codex/standard_names/workers.py:6330-6358`. Each candidate receives
`validation_status='valid'` or `validation_status='quarantined'` there. That is
the verdict producer; it does not persist the graph property itself.

The shared rich-name writer persists the incoming value in
`imas_codex/standard_names/graph_ops.py:5398-5403`. The change was placed there
because every caller of `write_standard_names` passes through that query. The
query now performs these assignments atomically:

```cypher
sn.validation_status = coalesce(b.validation_status, sn.validation_status),
sn.validated_at = CASE WHEN b.validation_status IS NOT NULL
                  THEN datetime()
                  ELSE sn.validated_at END
```

This reuses the sanctioned behavior already present in
`mark_names_validated` at
`imas_codex/standard_names/graph_ops.py:8139-8145`, where `validated_at` and
`validation_status` are stamped in one `SET`. Centralizing the additional rule
in the persistence query covers all rich-name writer callers without duplicating
clock behavior in the pooled worker.

| Incoming batch value | Persisted status | Persisted observation |
| --- | --- | --- |
| non-null verdict | incoming verdict | fresh `datetime()` in the same write |
| null | existing status retained by `coalesce` | existing `validated_at` retained |

The second row is essential: testing the resulting coalesced status would be
too late, because an existing non-null status does not prove that this persist
made a new validation observation.

## Regression tests

Both regressions live beside the existing writer-query test in
`tests/standard_names/test_validation_status.py`:

- `TestWriteStandardNamesValidationStatus::test_pooled_inline_validation_status_sets_observation`
  supplies the pooled inline-audit verdict `valid` and proves the same write
  conditionally stamps `validated_at`.
- `TestWriteStandardNamesValidationStatus::test_persist_without_incoming_validation_status_preserves_observation`
  omits `validation_status`, proves the normalized batch value is null, and
  proves the query selects the timestamp-preserving branch.

The focused writer class completed with 3 passed and 0 failed, including the
pre-existing coalescing test and the two new cases.

## Landed verification

At commit `0602ebf1a6a787b4b93e1125cf1d79aa8a93aca1`, the complete
`tests/standard_names` suite ran on `all_debug` and completed with:

- 7,316 passed
- 11 skipped
- 323 deselected
- 0 failed
- 0 failures added against the scratch candidate measurement

The suite log is
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260909T132441130606-n-swcr-a-persisted-verdict-must-carry-its-observation/after-suite.log`.

No export admission rule was loosened. In particular,
`validation_observation_missing` remains the signal for a verdict without an
observation. The 99 already-affected live rows were not changed; they require a
separate sanctioned validation rotation after the writer is integrated. This
node made no live-graph write, ran no broad pipeline, and attempted no signed
manifest apply.
