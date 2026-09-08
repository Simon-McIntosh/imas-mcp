# A valid verdict carries its observation time

## Decision

The export is now fenced rather than revalidating the historical cohort in
this change. A graph row is eligible only when `validation_status` is `valid`
and its `validated_at` observation is non-null. The population query retains
unstamped rows long enough to record the explicit
`validation_observation_missing` exclusion, so a reduced batch is observable
rather than an unexplained change in its candidate count.

Revalidating the 538 rows would establish real observations and remains the
correct data-repair follow-on. It is not sufficient as the export repair: a
future writer could again leave a `valid` scalar without its accompanying
observation, and revalidating only the current cohort would leave that export
path trusting the next undated verdict. This fence closes the path immediately
without inventing timestamps. In particular, no `validated_at` value was
backfilled with the time of this repair.

## Measurement

The earlier census recorded 538 `valid` rows without `validated_at`; 255 of
them were accepted or approved and not superseded, while 2,058 accepted or
approved valid rows did carry the stamp.

A bounded live read through `GraphClient` on the login node at
2026-09-08T15:16:01Z found 5,048 identities: 538 `valid` rows still lack the
stamp, 261 are currently accepted or approved and not superseded, and 2,058
such rows carry it. The increase from 255 to 261 is live-state drift, not an
effect of this source-only change.

The export fence does not mutate the graph, so the after-census remains 538
unstamped valid rows and 261 currently publishable-but-withheld identities.
Within export selection, the after count is zero identities that have both a
`valid` verdict and no observation time. The 2,058 stamped accepted or
approved valid identities remain eligible subject to the export's other
gates.

No verdict changed: this node did not run `drain_validation_for_ids`, and no
identity from the current 208-name cut was reclassified. The deterministic
revalidation follow-on must report every changed identity and its old and new
verdict before those withheld rows can return to a batch.

## Source binding

`imas_codex/standard_names/export.py` applies the non-null predicate to both
normal and batch candidate queries. Its export-population projection carries
the observation value into eligibility accounting, which gives an unstamped
otherwise-valid row the `validation_observation_missing` reason rather than
silently omitting it.
