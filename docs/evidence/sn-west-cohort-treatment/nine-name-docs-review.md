# Nine WEST documentation-review outcomes

## Result

All nine names that had appeared in the WEST export's `never_reviewed` bucket
now have reachable docs-axis review evidence and `docs_stage=accepted`. The
documentation run completed normally after 299.614 seconds with
`stop_reason=no_eligible_work`: it reviewed exactly nine scoped names and did
not spend its time or budget on unrelated sources.

Actual recorded spend was **$1.004414 of the $150.00 ceiling**. The remaining
**$148.995586** was not spent because the exact cohort completed its required
documentation generation and review on its first bounded rotation.

## Repair path

Before the run, each identity had `docs_stage=accepted` but zero docs-axis
`HAS_REVIEW` relationships and null docs review scalars. The disagreement was
repaired through the lifecycle owner, not by setting a graph property: the
dry run of `mark_members_for_regen()` found all nine eligible, and its applied
call reset all nine through `reset_standard_name_docs()` under scope run
`9503c417-2ee8-4eaf-ad74-7283b9344de5`.

`reset_standard_name_docs()` snapshots the current documentation as a
`DocsRevision`, clears docs-axis review state, and sets `docs_stage=pending`.
This reconciles the false acceptance projection with the absent evidence, then
allows the ordinary pipeline claim predicates to select the names normally.

The scoped standard-name command used `--docs-only --flush` and
`--skip-global-maintenance`, with `--cost-limit 150` and `--time 10` (minutes).
The resulting pipeline run was `a086509a-5aeb-417c-bf10-fbcf34d8a140`:
`status=completed`, `names_reviewed=9`, `cost_is_exact=true`, and
`cost_spent=cost_total=1.004414`.

## Per-name review evidence

`docs_stage` was `accepted` before the lifecycle reset for every row, despite
zero docs reviews. It is `accepted` after the ordinary review pipeline for
every row below. All resolved review relationships have tier `outstanding`.

The legacy scalar `reviewer_verdict_docs` remains null. The actual decision is
the relationship-level winning `resolution_method`, which the corrected export
uses as authority: `quorum_consensus` for eight names and
`authoritative_escalation` for one. “Accepted” below therefore means the
recorded resolution method reached an accepted docs stage; it does not invent a
missing scalar verdict.

| Identity | WEST source-path binding | Docs stage before | Docs review decision | Score | Docs stage after | Review evidence |
| --- | --- | --- | --- | ---: | --- | --- |
| `derivative_of_area_of_flux_surface_with_respect_to_toroidal_flux_coordinate` | `equilibrium/time_slice/profiles_1d/darea_drho_tor` | accepted, 0 reviews | accepted through quorum consensus | 0.97500 | accepted | 2 docs reviews; outstanding |
| `derivative_of_volume_of_flux_surface_with_respect_to_poloidal_magnetic_flux_coordinate` | `equilibrium/time_slice/profiles_1d/dvolume_dpsi` | accepted, 0 reviews | accepted through quorum consensus | 0.97500 | accepted | 2 docs reviews; outstanding |
| `derivative_of_volume_of_flux_surface_with_respect_to_toroidal_flux_coordinate` | `equilibrium/time_slice/profiles_1d/dvolume_drho_tor` | accepted, 0 reviews | accepted through quorum consensus | 0.96875 | accepted | 2 docs reviews; outstanding |
| `flux_surface_averaged_toroidal_current_density` | `equilibrium/time_slice/profiles_1d/j_phi` | accepted, 0 reviews | accepted through quorum consensus | 1.00000 | accepted | 2 docs reviews; outstanding |
| `length_variation_of_interferometer_beam` | `interferometer/channel/path_length_variation` | accepted, 0 reviews | accepted through quorum consensus | 0.96250 | accepted | 2 docs reviews; outstanding |
| `normalized_toroidal_flux_coordinate_at_minimum_absolute_safety_factor` | `equilibrium/time_slice/global_quantities/q_min/rho_tor_norm` | accepted, 0 reviews | accepted through authoritative escalation | 1.00000 | accepted | 3 docs reviews; outstanding |
| `product_of_poloidal_current_function_and_derivative_of_poloidal_current_function_with_respect_to_poloidal_magnetic_flux_coordinate` | `equilibrium/time_slice/profiles_1d/f_df_dpsi` | accepted, 0 reviews | accepted through quorum consensus | 0.97500 | accepted | 2 docs reviews; outstanding |
| `voltage_amplitude_of_ion_cyclotron_heating_antenna` | `ic_antennas/antenna/module/voltage/amplitude` | accepted, 0 reviews | accepted through quorum consensus | 0.98750 | accepted | 2 docs reviews; outstanding |
| `wave_current_amplitude_of_antenna_strap` | `ic_antennas/antenna/module/current/amplitude` | accepted, 0 reviews | accepted through quorum consensus | 0.98750 | accepted | 2 docs reviews; outstanding |

None of the nine was forced past a refusal or left without an advancing
outcome. The lowest resolved documentation score was 0.96250, above the run's
0.85 minimum.

## Evidence boundary

Live graph reads used the login-node exception because the Neo4j tunnel is
login-node-local. Every read was restricted to these nine indexed identities
or to the single pipeline run connected to their scope; the longest query took
9.9 seconds, within the ten-second ceiling. No ad-hoc Cypher mutation was used.

The durable execution receipt is
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260908T091827400325-n-swct-the-nine-unreviewed-names-get-their-docs-review/docs-review-receipt.md`.
