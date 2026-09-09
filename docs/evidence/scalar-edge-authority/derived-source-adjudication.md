# Derived-source produced-name adjudication

This document resolves the 26 multi-target rows that the DD-path pass left open. It is read-only evidence: no `produced_sn_id` value or `PRODUCED_NAME` edge was changed, and no pipeline ran.

![Unresolved-source adjudication by subgroup and verdict](/imas-codex/figures/scalar-edge-authority/derived-source-verdicts.svg)

## Fresh cohort check

The live graph still has **100** multi-target `StandardNameSource` rows. The named unresolved cohort remains exactly **26**: **15** `derived` sources with both `dd_path` and `signal` absent, **9** direction/state cases, **1** RMS-versus-peak current, and **1** state-versus-internal-state momentum flux. No member moved between the earlier and this live re-identification.

| Subgroup | Rows | Evidence instrument | Scalar intended | Edge intended | Still unadjudicated |
|---|---:|---|---:|---:|---:|
| Derived sources | 15 | `source_id` parent plus `HAS_PARENT` child bindings | 1 | 14 | 0 |
| Direction/state meaning | 9 | Candidate descriptions and units | 0 | 0 | 9 |
| Current waveform convention | 1 | Candidate descriptions, documentation, and units | 0 | 0 | 1 |
| State terminology on momentum flux | 1 | Candidate descriptions, documentation, and units | 0 | 0 | 1 |
| **Total** | **26** |  | **1** | **14** | **11** |

The counts are checked directly from the row tables below: `1 + 14 + 11 = 26`.

All graph reads were bounded to the named 26-row cohort and their immediate parent, child, unit, and `PRODUCED_NAME` relationships. They ran on the login node because the Neo4j tunnel is local to that node, and each completed under ten seconds.

## Derived-source provenance verdicts

The structural-parent source factory gives the decisive provenance rule: a source `derived:<parent-id>` stores the parent identity in `source_id`. The intended target is therefore that parent, not whichever competing target happens to be accepted or more common. Each chain below is `derived source → source_id parent → incoming HAS_PARENT child bindings → intended target`.

| Derived source | Scalar / all edge targets | Provenance chain and verdict |
|---|---|---|
| `derived:conductivity` | scalar `plasma_electrical_conductivity`; edges `conductivity`, `plasma_electrical_conductivity` | `derived:conductivity → conductivity → {vertical, toroidal, parallel, poloidal, radial}_conductivity → conductivity`; **E:`conductivity`**. The parent description is directional electrical conductivity coefficients, while the scalar is the broader plasma conductivity tensor. |
| `derived:fast_ion_state_pressure` | scalar `fast_ion_charge_state_pressure`; edges `fast_ion_state_pressure`, `fast_ion_charge_state_pressure` | `derived:fast_ion_state_pressure → fast_ion_state_pressure → parallel_fast_ion_state_pressure → fast_ion_state_pressure`; **E:`fast_ion_state_pressure`**. |
| `derived:fast_neutral_state_pressure` | scalar `parallel_fast_neutral_internal_state_pressure`; edges `fast_neutral_state_pressure`, `parallel_fast_neutral_internal_state_pressure` | `derived:fast_neutral_state_pressure → fast_neutral_state_pressure → parallel_fast_neutral_state_pressure → fast_neutral_state_pressure`; **E:`fast_neutral_state_pressure`**. The scalar adds a parallel projection absent from the parent source identity. |
| `derived:flux_surface_averaged_metric` | scalar `flux_surface_averaged_metric`; edges `flux_surface_averaged_metric`, `flux_surface_normal_contravariant_flux_surface_averaged_metric` | `derived:flux_surface_averaged_metric → flux_surface_averaged_metric → no child binding required → flux_surface_averaged_metric`; **S**. The scalar and parent edge agree; the second edge names a distinct normal-gradient metric. |
| `derived:ion_state_momentum` | scalar `ion_charge_state_momentum_source`; edges `ion_state_momentum`, `ion_charge_state_momentum_source` | `derived:ion_state_momentum → ion_state_momentum → parallel_ion_state_momentum → ion_state_momentum`; **E:`ion_state_momentum`**. The scalar changes momentum into momentum source. |
| `derived:ion_state_momentum_convection_velocity` | scalar `ion_charge_state_momentum_convection_velocity`; edges `ion_state_momentum_convection_velocity`, `ion_charge_state_momentum_convection_velocity` | `derived:ion_state_momentum_convection_velocity → ion_state_momentum_convection_velocity → {poloidal, radial}_ion_state_momentum_convection_velocity → ion_state_momentum_convection_velocity`; **E:`ion_state_momentum_convection_velocity`**. |
| `derived:ion_state_momentum_diffusion_coefficient` | scalar `ion_charge_state_momentum_diffusivity`; edges `ion_state_momentum_diffusion_coefficient`, `ion_charge_state_momentum_diffusivity` | `derived:ion_state_momentum_diffusion_coefficient → ion_state_momentum_diffusion_coefficient → poloidal_ion_state_momentum_diffusion_coefficient → ion_state_momentum_diffusion_coefficient`; **E:`ion_state_momentum_diffusion_coefficient`**. |
| `derived:ion_state_momentum_flux_limiter_coefficient` | scalar `ion_charge_state_momentum_flux_limiter_coefficient`; edges `ion_state_momentum_flux_limiter_coefficient`, `ion_charge_state_momentum_flux_limiter_coefficient` | `derived:ion_state_momentum_flux_limiter_coefficient → ion_state_momentum_flux_limiter_coefficient → {radial, poloidal}_ion_state_momentum_flux_limiter_coefficient → ion_state_momentum_flux_limiter_coefficient`; **E:`ion_state_momentum_flux_limiter_coefficient`**. |
| `derived:neutral_fraction` | scalar `ratio_of_neutral_state_density_to_total_hydrogenic_density`; edges `neutral_fraction`, `ratio_of_neutral_state_density_to_total_hydrogenic_density` | `derived:neutral_fraction → neutral_fraction → {hot, cold}_neutral_fraction → neutral_fraction`; **E:`neutral_fraction`**. The scalar is a state-density ratio, a different fraction from the parent population fraction. |
| `derived:neutral_state_density` | scalar `neutral_internal_state_density`; edges `neutral_state_density`, `neutral_internal_state_density` | `derived:neutral_state_density → neutral_state_density → ratio_of_neutral_state_density_to_total_hydrogenic_density → neutral_state_density`; **E:`neutral_state_density`**. |
| `derived:neutral_state_momentum_convection_velocity` | scalar `effective_neutral_internal_state_momentum_velocity_due_to_convection`; edges `neutral_state_momentum_convection_velocity`, `effective_neutral_internal_state_momentum_velocity_due_to_convection` | `derived:neutral_state_momentum_convection_velocity → neutral_state_momentum_convection_velocity → {toroidal, poloidal}_neutral_state_momentum_convection_velocity → neutral_state_momentum_convection_velocity`; **E:`neutral_state_momentum_convection_velocity`**. |
| `derived:neutral_state_momentum_diffusion_coefficient` | scalar `neutral_internal_state_momentum_diffusion_coefficient`; edges `neutral_state_momentum_diffusion_coefficient`, `neutral_internal_state_momentum_diffusion_coefficient` | `derived:neutral_state_momentum_diffusion_coefficient → neutral_state_momentum_diffusion_coefficient → {radial, parallel}_neutral_state_momentum_diffusion_coefficient → neutral_state_momentum_diffusion_coefficient`; **E:`neutral_state_momentum_diffusion_coefficient`**. |
| `derived:neutral_state_momentum_diffusivity` | scalar `toroidal_neutral_internal_state_momentum_diffusion_coefficient`; edges `neutral_state_momentum_diffusivity`, `toroidal_neutral_internal_state_momentum_diffusion_coefficient` | `derived:neutral_state_momentum_diffusivity → neutral_state_momentum_diffusivity → poloidal_neutral_state_momentum_diffusivity → neutral_state_momentum_diffusivity`; **E:`neutral_state_momentum_diffusivity`**. The scalar changes a parent vector property into a toroidal coefficient. |
| `derived:neutral_state_momentum_flux_limiter_coefficient` | scalar `neutral_internal_state_momentum_flux_limiter_coefficient_over_edge_region`; edges `neutral_state_momentum_flux_limiter_coefficient`, `neutral_internal_state_momentum_flux_limiter_coefficient_over_edge_region` | `derived:neutral_state_momentum_flux_limiter_coefficient → neutral_state_momentum_flux_limiter_coefficient → {parallel, vertical, radial, poloidal}_neutral_state_momentum_flux_limiter_coefficient → neutral_state_momentum_flux_limiter_coefficient`; **E:`neutral_state_momentum_flux_limiter_coefficient`**. |
| `derived:radius_of_ferritic_element` | scalar `radial_coordinate_of_ferritic_element_centroid`; edges `radial_coordinate_of_ferritic_element_centroid`, `radius_of_ferritic_element` | `derived:radius_of_ferritic_element → radius_of_ferritic_element → {inner, outer}_radius_of_ferritic_element → radius_of_ferritic_element`; **E:`radius_of_ferritic_element`**. The scalar narrows a generic geometric point to the centroid. |

## Description and unit verdicts

### Directional vorticity rows (8): still unadjudicated

The candidate units are identical—`m^-1.s^-1`—so units cannot select a target. The decisive description clauses are instead incompatible with each source: the scalar says **“Vorticity divided by the local major radius, evaluated on the GGD grid”**, while the other candidate says **“Toroidal component of plasma vorticity divided by major radius.”** A `parallel`, `poloidal`, `r`, or `radial` DD path is not toroidal, but neither candidate names that path's component. These are not a frequency tie: the two candidate identities denote distinct physical quantities, generic vorticity-over-radius and its toroidal component. That is a **KEEP-AS-DISTINCT** finding; neither existing target can be asserted intended for a non-toroidal component.

| Source id | Scalar / all edge targets | Verdict and evidence required to settle |
|---|---|---|
| `dd:edge_profiles/ggd/vorticity_over_r/parallel` | scalar `ratio_of_vorticity_to_major_radius`; edges `ratio_of_vorticity_to_major_radius`, `ratio_of_plasma_vorticity_to_major_radius` | **?** — DD component metadata identifying whether `parallel` is intentionally represented by the generic parent or requires a distinct parallel identity. |
| `dd:edge_profiles/ggd/vorticity_over_r/poloidal` | same two targets | **?** — DD component metadata or an approved poloidal candidate. |
| `dd:edge_profiles/ggd/vorticity_over_r/r` | same two targets | **?** — DD component metadata or an approved radial candidate. |
| `dd:edge_profiles/ggd/vorticity_over_r/radial` | same two targets | **?** — DD component metadata or an approved radial candidate. |
| `dd:plasma_profiles/ggd/vorticity_over_r/parallel` | same two targets | **?** — DD component metadata identifying generic-parent versus parallel-component intent. |
| `dd:plasma_profiles/ggd/vorticity_over_r/poloidal` | same two targets | **?** — DD component metadata or an approved poloidal candidate. |
| `dd:plasma_profiles/ggd/vorticity_over_r/r` | same two targets | **?** — DD component metadata or an approved radial candidate. |
| `dd:plasma_profiles/ggd/vorticity_over_r/radial` | same two targets | **?** — DD component metadata or an approved radial candidate. |

### Phase curvature (1): still unadjudicated

`dd:ec_launchers/launcher/beam/phase/curvature` carries scalar `wave_curvature_of_beam_tracing_beam` and edges `wave_curvature_of_beam_tracing_beam`, `wave_curvature_of_wave_beam`. Both candidates have unit `m^-1`; their descriptions are respectively **“Inverse curvature radius of the beam phase ellipse”** and **“Inverse curvature radii (1/R) of the wave beam phase ellipse.”** Neither clause distinguishes the source's launcher beam from the two candidate vocabularies. Verdict: **?**. The DD field description or a beam-tracing provenance receipt must state whether the data represent the tracing beam or wave beam.

### Wave current convention (1): still unadjudicated, KEEP-AS-DISTINCT

`dd:ic_antennas/antenna/module/strap/current` carries scalar `root_mean_square_wave_current_of_antenna_strap` and edges `peak_wave_current_of_antenna_strap_amplitude`, `root_mean_square_wave_current_of_antenna_strap`; both use unit `A`. The RMS documentation defines \(I_{RMS}=\sqrt{T_{RF}^{-1}\int i^2(t)dt}\), whereas the competing description says **“Peak amplitude of the radio-frequency current.”** RMS magnitude and peak amplitude are genuinely distinct physical conventions, not alternate spellings. Verdict: **?**. The DD field convention or producer waveform metadata must specify which convention it stores; that is the evidence needed to select a target.

### State terminology on radial momentum flux (1): still unadjudicated

`dd:plasma_transport/model/ggd/neutral/state/momentum/flux/radial` carries scalar `radial_neutral_internal_state_momentum_flux` and edges `radial_neutral_internal_state_momentum_flux`, `radial_neutral_state_momentum_flux`; both use `kg.m^-1.s^-2`. Both descriptions identify the \(R\)-\(R\) momentum-flux tensor component for neutral particles in a specified atomic, molecular, charge, or excitation state. No physical distinction is stated between `state` and `internal_state` here, so this is **not** a KEEP-AS-DISTINCT conclusion. Verdict: **?**. The canonical grammar/rename authority or a derivation/approval receipt that declares the terminology mapping is needed before treating either identity as intended.

## Repair boundary

The 15 derived rows are fully adjudicated from structural provenance: one scalar-intended and fourteen edge-intended. The eleven DD rows remain unchanged because their required evidence is absent; the eight vorticity rows and the RMS pair are additionally recorded as physically distinct alternatives, so a repair must not collapse them on the basis of target frequency. A future mutation node must remeasure this named cohort before using these verdicts.
