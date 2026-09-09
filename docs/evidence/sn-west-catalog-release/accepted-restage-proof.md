# Accepted-name restage proof

## Diagnosis: the expectation omitted the mutation's timestamp

The rejected live restage was not concurrent graph drift. The compare-and-set
at `edit.py:4287` had already returned all four intended identities, and an
actual-versus-expected instrument then showed the same result for every row:
`name_stage='drafted'` and the deterministic review scope matched the
expectation. The sole differing key was `updated_at`.

| Identity | Actual stage | Expected stage | Actual run id | Expected run id | Actual `updated_at` | Expected `updated_at` |
| --- | --- | --- | --- | --- | --- | --- |
| `atomic_mass` | `drafted` | `drafted` | `sn-review-restage-d8e6de4b1aed2a8cc6d8` | same | `2026-09-09T13:05:00Z` | null |
| `gas_flow` | `drafted` | `drafted` | `sn-review-restage-d8e6de4b1aed2a8cc6d8` | same | `2026-09-09T13:05:00Z` | null |
| `plasma_pressure` | `drafted` | `drafted` | `sn-review-restage-d8e6de4b1aed2a8cc6d8` | same | `2026-09-09T13:05:00Z` | null |
| `wave_phase_of_ion_cyclotron_heating_antenna` | `drafted` | `drafted` | `sn-review-restage-d8e6de4b1aed2a8cc6d8` | same | `2026-09-09T13:05:00Z` | null |

The mutation deliberately writes `updated_at = datetime()`, but
`_accepted_review_expected_post()` previously changed only `name_stage` and
`run_id`. The expected structure therefore retained each pre-write null value,
and strict whole-structure comparison raised the post-state error even though
the transaction had written exactly its intended state. The null
`reviewer_score_name` is not itself a bad expectation: it is required to make
these accepted identities eligible for restage and remains null until ordinary
review writes a score.

## Repair: model the dynamic field without relaxing the proof

The expectation now declares that an accepted restage changes `updated_at`.
The post-state matcher requires that field to be non-null and different from
its pre-write value before it normalizes the server-generated timestamp to the
expected dynamic marker. It still compares every other scalar, every
relationship, and every relationship property exactly.

This is not removal of the proof. The focused test suite includes both needed
halves:

- An accepted row with `reviewer_score_name=null` and a prior timestamp stages
  successfully, preserves the null score, and records a newer timestamp.
- A transaction that also writes an unexpected property still raises
  `RuntimeError: accepted restage post-state proof failed` and rolls back.

## Live exact-cohort result

The repaired exact command completed with `outcome='applied'` and `staged=4`:

```text
imas-codex sn restage-accepted atomic_mass gas_flow plasma_pressure \
  wave_phase_of_ion_cyclotron_heating_antenna --include-accepted --apply
```

Its relationship proof remained intact: 300 relationships before and after,
identical signature
`4ce79058bde8d50cfb178c3b7373ae71de8d2fefcc1308a209569e924640e20d`,
and unchanged binding counts (`HAS_STANDARD_NAME=64`, `HAS_UNIT=4`,
`HAS_COCOS=0`). A bounded live read then returned:

| Identity | Resulting `name_stage` | Resulting `reviewer_score_name` | Name-axis `HAS_REVIEW` count |
| --- | --- | ---: | ---: |
| `atomic_mass` | `drafted` | null | 0 |
| `gas_flow` | `drafted` | null | 6 |
| `plasma_pressure` | `drafted` | null | 0 |
| `wave_phase_of_ion_cyclotron_heating_antenna` | `drafted` | null | 0 |

The four rows are now eligible for the ordinary exact name-review rotation.
`gas_flow` remains distinct: its six existing name-axis reviews mean its null
scalar is a projection inconsistency, not absence of review evidence. The
subsequent review operation must preserve that distinction rather than treating
all four null scores as the same defect.

## Verification

Focused accepted-restage coverage passed **8/8** with one pre-existing pytest
configuration warning. The required `all_debug` run of
`tests/standard_names/` completed with **7,314 passed, 11 skipped, 323
deselected, 34 warnings, and zero failures** in 292.27 seconds. Its empty
failure set means this change adds zero failures against base revision
`f14bc6dab2ceb6529228778c6b45e78d4695560f`.
