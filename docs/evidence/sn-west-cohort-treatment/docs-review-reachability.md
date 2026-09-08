# WEST docs-review reachability

## Finding

The nine WEST entries in the export's `never_reviewed` bucket have genuinely
never received a reachable docs-axis review. They are not entries whose review
resolution was dropped.

The bounded live-graph query matched the exact nine `StandardName.id` values,
then optionally matched `(sn)-[:HAS_REVIEW]->(review:StandardNameReview
{review_axis: 'docs'})`. It returned nine rows: every row had
`docs_stage='accepted'`, `docs_review_count=0`, an empty `docs_reviews` list,
and null `reviewer_score_docs`, `reviewer_verdict_docs`, `reviewed_docs_at`,
and `docs_review_quorum_shortfall`. The scalar lifecycle projection therefore
claims acceptance while the relationship that carries docs review evidence is
absent.

An absent docs review and an unrecorded resolution are different predicates:

```text
never reviewed:        NOT EXISTS { (sn)-[:HAS_REVIEW]->(:StandardNameReview
                                   {review_axis: 'docs'}) }
resolution unrecorded: a docs-axis review exists, but no review group has a
                       winning resolution_method
```

The nine meet the first predicate, not the second. A later docs review must be
run for them; there is no stored resolution to surface.

## Why the review cannot be reached

The `docs_stage` scalar is the disagreeing field. The source predicates are:

```text
claim_generate_docs_batch: sn.name_stage = 'accepted'
                           AND sn.docs_stage = 'pending'

claim_review_docs_batch:   sn.docs_stage = 'drafted'
                           OR (drain scope AND sn.docs_stage = 'reviewed' ...)
```

None accepts `docs_stage='accepted'`. Because the nine have no docs review
evidence, they cannot satisfy the evidence-led refinement route either. The
accepted projection therefore makes a genuinely unreviewed name ineligible
for document generation and review: it is a deadlock, not a missing review
resolution.

The export used to repeat that scalar as an additional eligibility predicate:

```text
sn.docs_stage = 'accepted' AND EXISTS { reachable winning docs review }
```

The corrected exporter treats the reachable winning review as the authority
and does not require the scalar projection. Its exclusion classifier checks
docs evidence before `docs_stage`, so a false accepted projection cannot hide
`never_reviewed`, and an existing review without a winning method becomes
`resolution_unrecorded`.

The lifecycle writer that produced the false `docs_stage='accepted'` value is
outside this export-only change. The next repair must correct that projection
through its owning graph lifecycle path, then run the docs rotation; this work
did not set graph data by hand or start that rotation.

## Nine-name evidence

All nine rows have the same docs-axis values: `docs_stage=accepted`, zero
`HAS_REVIEW` docs relationships, and null review scalar fields. The name score
is the stored `reviewer_score_name`; source paths are the WEST batch bindings.

| Identity | Description | Source-path binding | Name score | Docs evidence |
| --- | --- | --- | ---: | --- |
| `derivative_of_area_of_flux_surface_with_respect_to_toroidal_flux_coordinate` | Rate of change of a flux surface's poloidal cross-sectional area with toroidal flux. | `equilibrium/time_slice/profiles_1d/darea_drho_tor` | 1.0 | no docs review; all docs scalars null |
| `derivative_of_volume_of_flux_surface_with_respect_to_poloidal_magnetic_flux_coordinate` | Derivative of a flux surface's enclosed volume with respect to poloidal magnetic flux. | `equilibrium/time_slice/profiles_1d/dvolume_dpsi` | 1.0 | no docs review; all docs scalars null |
| `derivative_of_volume_of_flux_surface_with_respect_to_toroidal_flux_coordinate` | Rate of change of a flux surface's enclosed volume with toroidal flux. | `equilibrium/time_slice/profiles_1d/dvolume_drho_tor` | 1.0 | no docs review; all docs scalars null |
| `flux_surface_averaged_toroidal_current_density` | Flux-surface-averaged toroidal current density. | `equilibrium/time_slice/profiles_1d/j_phi` | 1.0 | no docs review; all docs scalars null |
| `length_variation_of_interferometer_beam` | Plasma-relative change in an interferometer beam's optical path length. | `interferometer/channel/path_length_variation` | 0.9875 | no docs review; all docs scalars null |
| `normalized_toroidal_flux_coordinate_at_minimum_absolute_safety_factor` | Toroidal-flux coordinate at the minimum absolute safety factor. | `equilibrium/time_slice/global_quantities/q_min/rho_tor_norm` | 1.0 | no docs review; all docs scalars null |
| `product_of_poloidal_current_function_and_derivative_of_poloidal_current_function_with_respect_to_poloidal_magnetic_flux_coordinate` | Product of the poloidal current function and its derivative with respect to poloidal flux. | `equilibrium/time_slice/profiles_1d/f_df_dpsi` | 1.0 | no docs review; all docs scalars null |
| `voltage_amplitude_of_ion_cyclotron_heating_antenna` | Radio-frequency voltage amplitude on an ion-cyclotron heating antenna. | `ic_antennas/antenna/module/voltage/amplitude` | 0.98125 | no docs review; all docs scalars null |
| `wave_current_amplitude_of_antenna_strap` | Radio-frequency current amplitude on an antenna strap. | `ic_antennas/antenna/module/current/amplitude` | 0.8875 | no docs review; all docs scalars null |

## Accounting and execution boundary

The retained WEST export census remains exact: `214 - 196 = 18`, with zero
residue. Its largest reason is `never_reviewed=9`; after this correction that
reason means exactly `no docs-axis review is reachable`, rather than a generic
documentation-stage failure.

The live read used the login-node exception because the Neo4j tunnel is local
to that node. It was bounded to these nine indexed identities and did no graph
write or local heavy computation. The query returned nine records; its result
is sufficient to prove the relationship predicate can match the cohort and
that the zero review count is not a missing-property zero-row artifact.
