# Single-edge produced-name divergence adjudication

## Live cohort and measure

The live graph contains **17** `StandardNameSource` rows for which a real
`produced_sn_id` scalar names one `StandardName`, exactly one
`PRODUCED_NAME` edge names another real `StandardName`, and the two identities
differ. This is unchanged from the **17** recorded in
[`produced-name-adjudication.md`](produced-name-adjudication.md); the cohort has
not moved.

This is not the 100-row multi-target cohort. Each row has one scalar authority
and one edge authority, so there is no target set to reduce. The evidence below
adjudicates the direct disagreement only. It performs no scalar rewrite, edge
creation/deletion, source repair, or pipeline run.

![Verdicts and likely later writer split](/imas-codex/figures/scalar-edge-authority/single-edge-divergence-verdicts.svg)

| Verdict | Rows |
| --- | ---: |
| Scalar intended | 10 |
| Edge intended | 4 |
| Unresolved naming equivalence | 3 |
| **Total** | **17** |

The verdict counts were checked as `10 + 4 + 3 = 17` against the live census.
Ten rows are also **KEEP-AS-DISTINCT** findings: their two candidates denote
different physical quantities or scopes, so resolving the binding does not make
the rejected identity an alias or a deletion candidate.

## Adjudications

`S` means the scalar identifies the intended binding; `E` means the sole edge
does. “Later writer” is a likelihood, not an invented timestamp: the graph has
no relationship creation time and no source identity-repair or attachment event
for any of these 17 rows. The inference is based on candidate creation/generation
times relative to the source's `composed_at` time; the mechanism is stated as a
likely rebind/retarget or scalar refresh accordingly.

| Source id | Scalar identity / unit | Sole edge identity / unit | Deciding source and candidate evidence | Verdict, distinction, and likely later writer |
| --- | --- | --- | --- | --- |
| `dd:bolometer/channel/power` | `power_of_bolometer` / `W` | `radiated_power` / `W` | Source says “Power received on the detector”; scalar says “single bolometer detector channel”, while edge says total plasma power reconstructed from bolometry. | **S** — a detector-channel measurement is distinct from total plasma radiated power (**KEEP-AS-DISTINCT**). Later writer is **indeterminate**: both candidates and the source share the same initial generation time, and no binding event is recorded. |
| `dd:core_profiles/profiles_1d/neutral/state/velocity/parallel` | `parallel_neutral_state_velocity` / `m.s^-1` | `parallel_neutral_state_particle_convection_velocity` / `m.s^-1` | Path ends `neutral/state/velocity/parallel`; its clause is “Parallel component” under parent “Velocity”. The scalar names velocity; the edge names a particle-convection coefficient. | **S** — velocity and convection velocity are distinct transport quantities (**KEEP-AS-DISTINCT**). Edge is likely later: its pipeline candidate was created after the scalar/source composition, consistent with a later over-specific rebind. |
| `dd:distributions/distribution/global_quantities/thermalisation/torque` | `toroidal_thermal_torque_due_to_thermalization` / `N.m` | `toroidal_thermal_torque_due_to_collisions` / `N.m` | Source explicitly says “Torque input … due to the thermalisation of fast particles”; scalar preserves thermalization, edge substitutes Coulomb collisions. | **S** — thermalization and collisions are distinct mechanisms (**KEEP-AS-DISTINCT**). Later writer is **indeterminate**: both candidates have the same initial generation timestamp and source history is empty. |
| `dd:edge_profiles/ggd/ion/state/ionization_potential/values` | `ion_state_potential` / `e` | `ion_state_average_charge` / `1` | Parent clause says “Cumulative and average ionization potential”; scalar description says ionization potential and retains unit `e`; edge says average charge number with unit `1`. | **S** — ionization potential and charge number are distinct quantities (**KEEP-AS-DISTINCT**). Edge is likely later: the average-charge candidate postdates the scalar/source composition, consistent with an incorrect later retarget. |
| `dd:edge_profiles/profiles_1d/neutral/state/velocity/parallel` | `parallel_neutral_state_velocity` / `m.s^-1` | `parallel_neutral_state_particle_convection_velocity` / `m.s^-1` | The source path and “Parallel component” under “Velocity” are the same direct velocity evidence as the core-profiles row. | **S** — velocity and convection velocity remain distinct (**KEEP-AS-DISTINCT**). Edge is likely later for the same candidate-timing and likely rebind mechanism as the core-profiles row. |
| `dd:equilibrium/time_slice/boundary_secondary_separatrix/x_point/r` | `major_radius_of_secondary_x_point` / `m` | `toroidal_angle_of_secondary_x_point` / `m` | Path ends `x_point/r` and source clause says “Major radius”; scalar says major-radius coordinate. The edge identifier says toroidal angle even though its copied description incorrectly repeats major radius. | **S** — radius and toroidal angle are distinct coordinates (**KEEP-AS-DISTINCT**). Edge is likely later: the angle candidate was created after the scalar/source composition, indicating a later identifier misbinding. |
| `dd:focs/spun` | `spun_period_of_fiber_optic_current_sensor` / `m` | `spun_wavelength_of_fiber_optic_current_sensor` / `m` | Source clause is exactly “Spun period”; scalar describes the spatial period of spin modulation, while edge names a wavelength. | **S** — the DD names the period directly. Edge is likely later: the wavelength candidate was created later than the scalar/source composition, consistent with synonym substitution during a later rebind. |
| `dd:gyrokinetics/flux_surface/dc_dr_minor_norm` | `radial_derivative_of_flux_surface_coefficient` / `1` | `radial_derivative_of_coefficient_of_flux_surface` / `1` | Source says derivative of the `c` shape coefficient with respect to normalized minor radius. Both candidate descriptions say the same radial derivative of a flux-surface coefficient and share unit `1`. | **UNRESOLVED** — these are naming-equivalent descriptions, not distinct physical quantities. The edge candidate is likely later by four minutes, but canonical name approval or a binding ledger is required to choose. |
| `dd:gyrokinetics/flux_surface/delongation_dr_minor_norm` | `radial_derivative_of_flux_surface_coefficient` / `1` | `radial_derivative_of_coefficient_of_flux_surface` / `1` | Source says derivative of elongation with respect to normalized minor radius; again the two candidate descriptions and unit `1` are equivalent. | **UNRESOLVED** — same naming-equivalence condition. Edge is likely later by candidate timing; a canonical naming decision or binding history would settle it. |
| `dd:gyrokinetics/flux_surface/ds_dr_minor_norm` | `radial_derivative_of_flux_surface_coefficient` / `1` | `radial_derivative_of_coefficient_of_flux_surface` / `1` | Source says derivative of the `s` shape coefficient with respect to normalized minor radius; both candidates have the same derivative description and unit `1`. | **UNRESOLVED** — same naming-equivalence condition. Edge is likely later by candidate timing; a canonical naming decision or binding history would settle it. |
| `dd:pf_active/vertical_force/force` | `vertical_force` / `N` | `vertical_total_force_of_poloidal_field_coil` / `N` | Parent clause is “Vertical forces on the axisymmetric PF coil system”; edge names total vertical force on all poloidal-field coils, while scalar is generic. | **E** — the edge retains the PF-coil-system scope. Edge is likely later: its pipeline candidate postdates the generic scalar/source composition, consistent with a semantic retarget that made the binding more specific. |
| `dd:plasma_profiles/ggd/vorticity_over_r/toroidal` | `ratio_of_vorticity_to_major_radius` / `m^-1.s^-1` | `ratio_of_plasma_vorticity_to_major_radius` / `m^-1.s^-1` | Source clause says “Toroidal component”; edge says toroidal plasma vorticity, while scalar is generic vorticity over major radius. | **E** — generic and toroidal components are distinct (**KEEP-AS-DISTINCT**). Scalar is likely later: it was generated after the toroidal edge candidate already existed, consistent with a later generic scalar refresh while the specific edge remained. |
| `dd:plasma_profiles/profiles_1d/neutral/state/velocity/parallel` | `parallel_neutral_state_velocity` / `m.s^-1` | `parallel_neutral_state_particle_convection_velocity` / `m.s^-1` | Path ends `neutral/state/velocity/parallel`; source says “Parallel component” under “Velocity”. | **S** — it is velocity, not the convection coefficient (**KEEP-AS-DISTINCT**). Edge is likely later: the convection candidate postdates the scalar/source composition, indicating later over-specific edge retargeting. |
| `dd:summary/local/itb/momentum_tor/value` | `toroidal_plasma_momentum_at_internal_transport_barrier` / `kg.m.s^-1` | `toroidal_total_plasma_momentum_at_internal_transport_barrier` / `kg.m.s^-1` | Parent clause says “Total plasma toroidal momentum, summed over ion species and electrons”; edge retains “total”, scalar omits it. | **E** — the edge retains the source's total-plasma qualifier. Edge is likely later: its candidate was created after scalar/source composition, consistent with a later scope-correcting retarget. |
| `dd:wall/description_ggd/ggd/a_field/toroidal` | `toroidal_magnetic_vector_potential` / `T.m` | `toroidal_vector_potential` / `T.m` | Parent clause is “Magnetic vector potential”; scalar identity retains `magnetic`, while the edge identifier omits it although both descriptions say magnetic vector potential. | **S** — the scalar matches the DD's explicit magnetic-field meaning. Scalar is likely later: it was generated after the edge candidate, consistent with a source-specific scalar refresh that left a shorter edge identifier behind. |
| `dd:waves/coherent_wave/profiles_1d/power_density_n_tor` | `per_toroidal_mode_absorbed_power_density` / `W.m^-3` | `per_toroidal_mode_flux_surface_average_total_absorbed_power_density` / `W.m^-3` | Source clause says “Flux surface averaged absorbed wave power density per toroidal mode number”; edge retains flux-surface averaging, scalar does not. | **E** — flux-surface-averaged and non-averaged power density are distinct (**KEEP-AS-DISTINCT**). Edge is likely later: its candidate postdates scalar/source composition, consistent with an appropriate later retarget. |
| `dd:waves/coherent_wave/profiles_2d/power_density_n_tor` | `per_toroidal_mode_absorbed_power_density` / `W.m^-3` | `per_toroidal_mode_flux_surface_average_total_absorbed_power_density` / `W.m^-3` | Parent clause is “2D profiles in poloidal cross-section”; unlike the 1D path, it contains no flux-surface-average clause. Scalar names absorbed power density per toroidal mode; edge adds the absent average. | **S** — 2D cross-section and flux-surface average are distinct representations (**KEEP-AS-DISTINCT**). Edge is likely later: the same late family retarget that is correct for the 1D source is wrong for this 2D source. |

## Later-writer split

| Likely later authority | Rows | Evidence and mechanism |
| --- | ---: | --- |
| Edge | 13 | The edge candidate was created or generated after the scalar/source composition. With no relationship timestamp, this is a likely later pipeline rebind or retarget, not a claimed direct event. It is semantically correct for four rows and wrong for nine. |
| Scalar | 2 | The scalar candidate was generated after the edge candidate: toroidal vorticity and wall magnetic vector potential. This is consistent with a later scalar refresh/recomposition leaving an older edge in place. |
| Indeterminate | 2 | Bolometer channel power and thermalisation torque have matching candidate-generation times and no source attachment or identity-repair event. A source-binding ledger entry or relationship timestamp is required to order their writers. |
| **Total** | **17** | `13 + 2 + 2 = 17`. |

No source carried a `HAS_IDENTITY_REPAIR`, `HAS_AUTHORITY_RETIREMENT`,
`HAS_SNAPSHOT_CHANGE`, or `HAS_SNAPSHOT_ADOPTION` event. That absence is why
the temporal column reports likelihood rather than treating candidate node
creation time as an edge timestamp.

## KEEP-AS-DISTINCT findings

The following ten rows retire false alias hypotheses. Their rejected candidate
remains a valid identity; only this source binding is wrong or unsupported:

- bolometer channel power versus total plasma radiated power;
- the three neutral velocity rows versus neutral particle-convection velocity;
- thermalisation torque versus collision torque;
- ionization potential versus average charge number;
- secondary-X-point radius versus toroidal angle;
- generic versus toroidal vorticity over major radius; and
- each of the two wave power-density rows, where flux-surface-average and
  poloidal-cross-section representations are distinct.

This gives `1 + 3 + 1 + 1 + 1 + 1 + 2 = 10` distinct-quantity findings.

## Unsettled rows and required evidence

The three gyrokinetic coefficient rows are intentionally not guessed. Their
scalar and edge candidates have interchangeable descriptions and unit `1`; the
DD clauses identify `c`, elongation, or `s`, but neither candidate identity
preserves that distinction. A canonical naming approval, a source-binding
receipt, or a derivation record that chooses one word order would settle all
three. Until then, they remain unadjudicated and no repair may consume either
authority.

## Evidence artifacts

- `single-edge-live-census.json` — the bounded live 17-row cohort with source
  properties, both candidate descriptions/units, and sole-edge proof.
- `single-edge-source-history.json` — source lifecycle and absence of source
  identity/attachment history for all 17 rows.
- `single-edge-candidate-history.json` — candidate timestamps and internal
  change records supporting the explicitly qualified later-writer inference.

All live graph reads ran through `GraphClient` on the login node because the
Neo4j tunnel is login-node-local. Each query was limited to this named cohort
and completed in under ten seconds. No tests or pipeline jobs were run because
this node changes evidence only; merged-head verification belongs to the
separate test node.
