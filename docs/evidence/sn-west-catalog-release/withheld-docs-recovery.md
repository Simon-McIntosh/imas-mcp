# Withheld-documentation recovery

## Verdict: the 31-row documentation hold is recovered at zero cost

The WEST cut may treat the `documentation_not_accepted` hold for this 31-name
cohort as cleared. Every affected StandardName retained both its documentation
text and its docs-axis review history. The regression was confined to two
mirrored projections: `docs_stage` became `pending` and
`reviewer_score_docs` became null. The repair therefore re-derived those
projections from the surviving review evidence; it did not generate or review
any documentation again.

The predecessor's independent post-repair export found all 31 emitted, with
the `documentation_not_accepted` exclusion category reduced from 31 to zero.
Its final re-read reports 31 accepted stages, non-null scores for all 31, and
documentation lengths from 840 through 1,988 characters. It also records zero
provider spend for the deterministic repair.

![Withheld rows by exclusion reason before and after recovery](/imas-codex/figures/sn-west-catalog-release/withheld-rows-by-exclusion-reason.png)

The figure is an exact copy of the predecessor's evidence asset (SHA-256
`9a56596515b6fab95c9c0119a9acdaaf9a1a3dc1d35fb001492aa96229102399`).

## Bounded live verification

Two read-only Cypher reads ran on the login node on 2026-09-09. This placement
is required because the Neo4j tunnel is login-node-local. Both reads were
bounded by the 31 identities listed in
[`reminted-cohort.md`](reminted-cohort.md), completed in under one second, and
made no graph change.

| Check | Result |
| --- | --- |
| Named cohort rows returned | 31 / 31 |
| `docs_stage = accepted` | 31 / 31 |
| non-null `reviewer_score_docs` | 31 / 31 |
| newest `StandardName.updated_at` | `2026-09-09T12:00:13.486Z` |
| documentation length across cohort | 840–1,988 characters |
| retained docs-axis `HAS_REVIEW` relationships | 285 |
| newest review-record timestamp (`reviewed_at`) | `2026-09-07T05:23:15.218196Z` |
| docs review records timestamped on 2026-09-09 | 0 |

`StandardNameReview` exposes its review-record time as `reviewed_at`; the
bounded aggregate found no non-null `created_at` values. Thus the persisted
review-record evidence is both older than the 2026-09-09 scalar update and has
zero records timestamped on that date. The 285 retained relationships, their
latest 2026-09-07 timestamp, and the zero 2026-09-09 count prove that the
recovery did not create a new review edge or consume a new review run.

## Scalar recovery by identity

The before values are the cohort snapshot recorded in
[`reminted-cohort.md`](reminted-cohort.md): every row had
`docs_stage = pending` and `reviewer_score_docs = null`. The after values below
are the current bounded live read. The `docs reviews` column is not a new
payload; it counts the surviving docs-axis relationships used as authority.

| StandardName | Before `docs_stage` | Before docs score | After `docs_stage` | After docs score | Docs reviews |
| --- | --- | ---: | --- | ---: | ---: |
| `effective_charge` | `pending` | null | `accepted` | 0.9500 | 4 |
| `electron_density_at_plasma_boundary` | `pending` | null | `accepted` | 1.0000 | 11 |
| `faraday_angle` | `pending` | null | `accepted` | 0.9750 | 11 |
| `frequency_of_ion_cyclotron_heating_antenna` | `pending` | null | `accepted` | 0.9625 | 11 |
| `gap_at_outboard_midplane` | `pending` | null | `accepted` | 0.9750 | 11 |
| `initial_polarization_ellipticity_of_polarimeter_beam` | `pending` | null | `accepted` | 0.9000 | 16 |
| `launched_power_of_lower_hybrid_antenna` | `pending` | null | `accepted` | 0.9375 | 10 |
| `line_integrated_electron_number_density` | `pending` | null | `accepted` | 0.8875 | 4 |
| `magnetic_shear_at_flux_surface` | `pending` | null | `accepted` | 0.9500 | 4 |
| `maximum_magnetic_field_magnitude` | `pending` | null | `accepted` | 0.8750 | 12 |
| `normalized_plasma_internal_inductance` | `pending` | null | `accepted` | 0.9250 | 11 |
| `normalized_toroidal_flux_coordinate_at_measurement_position` | `pending` | null | `accepted` | 0.9375 | 12 |
| `plasma_current` | `pending` | null | `accepted` | 0.9625 | 7 |
| `poloidal_angle_of_flux_surface` | `pending` | null | `accepted` | 0.8875 | 16 |
| `poloidal_angle_of_measurement_position` | `pending` | null | `accepted` | 0.9500 | 10 |
| `poloidal_magnetic_flux_at_flux_surface` | `pending` | null | `accepted` | 0.9500 | 5 |
| `poloidal_magnetic_flux_at_measurement_position` | `pending` | null | `accepted` | 0.9625 | 4 |
| `poloidal_magnetic_flux_of_flux_loop` | `pending` | null | `accepted` | 0.9250 | 9 |
| `radial_coordinate_of_geometric_axis` | `pending` | null | `accepted` | 0.8625 | 8 |
| `radial_coordinate_of_magnetic_axis` | `pending` | null | `accepted` | 0.9375 | 10 |
| `radial_coordinate_of_strike_point` | `pending` | null | `accepted` | 0.9375 | 6 |
| `radial_outline_of_antenna_strap` | `pending` | null | `accepted` | 0.8875 | 11 |
| `safety_factor` | `pending` | null | `accepted` | 0.9375 | 9 |
| `toroidal_beta` | `pending` | null | `accepted` | 0.9250 | 9 |
| `toroidal_magnetic_flux` | `pending` | null | `accepted` | 0.9375 | 12 |
| `toroidal_magnetic_flux_due_to_diamagnetic_drift` | `pending` | null | `accepted` | 0.8750 | 6 |
| `total_power_due_to_ion_cyclotron_heating` | `pending` | null | `accepted` | 0.9500 | 17 |
| `vertical_coordinate_of_camera` | `pending` | null | `accepted` | 0.9625 | 6 |
| `vertical_coordinate_of_geometric_axis` | `pending` | null | `accepted` | 0.9250 | 8 |
| `vertical_coordinate_of_strike_point` | `pending` | null | `accepted` | 0.9000 | 9 |
| `volume_averaged_electron_density` | `pending` | null | `accepted` | 1.0000 | 6 |

The table totals 31 identities and 285 surviving docs-axis review
relationships. It distinguishes a re-derived mirror from a new review: no
assertion or edge has been deleted or weakened, and no documentation content
was replaced.

## Recovery mechanism recorded by the predecessor

The predecessor stream contains 31 textual references to
`reconcile_reviewable_name_stage`, but it does **not** record invocation of
that name-axis function for this recovery. It explicitly distinguishes that
function as the name-axis analogue while determining that no purpose-built
docs-axis re-derivation function was available.

Instead, the stream records a bounded, idempotent `GraphClient` transaction in
`/tmp/docs_reaccept_apply.py`. It selected only these 31 identities when their
accepted name state, valid validation, present documentation, absence of a docs
quorum shortfall, and surviving winning docs review met the acceptance
predicate. It then restored `docs_stage`, `reviewer_score_docs`, the reviewer
model, and the reviewed timestamp from that winning record, while leaving the
documentation and review/revision nodes untouched.

The predecessor's recorded outcome is:

> `updated rows: 31`
>
> `docs_stage after: {'accepted': 31}`

The stream therefore settles the predicates and the actual bounded
transaction, but it does not settle a reusable named function that performed a
docs-axis re-derivation. This record intentionally does not misname
`reconcile_reviewable_name_stage` as that function.

## Remaining gates before another cut

This recovery clears only the 31 documentation-mirror holds. A re-cut still
needs all of the following independent evidence:

- `net_power_due_to_ion_cyclotron_heating` needs a genuine docs-axis review.
  Its `docs_stage` is `accepted`, but it has zero docs-axis review edges, so
  the scalar cannot substitute for review evidence.
- `etendue_of_soft_xray_detector` needs a validation observation.
- Four identities lack `reviewer_score_name`: `atomic_mass`, `gas_flow`,
  `plasma_pressure`, and `wave_phase_of_ion_cyclotron_heating_antenna`.

No signed-manifest apply, pipeline operation, or graph write was issued by this
recording node. The live checks above were read-only confirmation of the
predecessor's completed recovery.
