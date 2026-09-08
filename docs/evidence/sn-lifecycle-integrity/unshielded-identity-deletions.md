# Unshielded identity deletions and classification

## Result

The archive-backed census does not support a uniform restoration verdict. Of the 78 identities deleted after origin reconciliation:

- **32 are RESTORE**: 28 have direct DD bindings whose 30 source rows all survive live, `density_at_pedestal_top` is a 19-child family whose 21 DD bindings survive, and three additional source-less identities carry catalog `unchanged_ratification`. The fourth ratified identity, `safety_factor_at_pedestal_top`, is already in the direct-DD class.
- **24 are CORRECTLY REMOVED**: 13 are source-equivalent single-child shadows, one is the source-less parent of one of those shadows, and 10 were archived as `superseded` or `exhausted` without an independent DD binding. These are structural or terminal nodes, not active independently sourced quantities.
- **22 are UNDETERMINED**: the archive has only a derived source and either no child or only a source-less derived child. The archive predates the deletion by two days, while `StandardNameChange` did not snapshot the deletion-time child/source topology. A RESTORE or CORRECTLY REMOVED verdict for these rows would therefore be conjectural.

This is the actionable boundary: 32 restorations can be routed now, 24 removals should stand, and 22 must not be published or restored until deletion-time source authority or a catalog receipt is recovered.

## Causal chain and information loss

- At `2026-09-08T11:55:49.265Z`, `reconcile_catalog_edit_origin` examined 2,096 identities carrying false `catalog_edit` origin and moved 1,179 to `pipeline`, 246 to `derived`, and 671 to null.
- Between `11:57:28` and `11:57:36`, 99 to 107 seconds later, `remove_derived_parent` deleted the 78 identities in this census with `origin=pipeline_cleanup` and reason `structural derived parent no longer satisfies lifecycle admission`.
- `_query_derived_parents_for_admission_cleanup` selects only `parent.origin = 'derived'`. The false `catalog_edit` value had therefore shielded these names; the origin repair exposed them to the reaper.
- `_delete_derived_parent_nodes` nulls `produced_sn_id` on mirror sources, deletes derived sources and reviews, and then `DETACH DELETE`s the Standard Name. Its `StandardNameChange` stores only identity and operation metadata, not node properties, source links, child topology, reviews, or documentation. The change record proves that deletion happened but cannot reconstruct what was deleted.

The earlier live-source count was the wrong instrument: zero rows whose `produced_sn_id` equals a deleted identity is guaranteed by the deletion code. The broader live population is 9,900 `StandardNameSource` rows, 4,469 with null `produced_sn_id`, including 4,459 with neither the scalar nor a `PRODUCED_NAME` edge. The archive correlation below uses stable source IDs to identify which of those orphans belonged to this cohort.

## Archive and live-source method

The full `2026-09-06T22:00:12Z` archive was loaded on the `all_debug` partition into a temporary Neo4j data directory on distinct loopback ports. The live store and live database name were never used. The temporary instance was removed after each query.

- Archive census: 1,630,556 nodes and 4,276,396 relationships, including 5,078 `StandardName`, 9,912 `StandardNameSource`, and 27,936 `StandardNameReview` nodes.
- Cohort recovery: all 78 target nodes, 136 target/descendant rows, 214 source-lineage rows, and 111 DD-bound source-lineage rows were read.
- Stable-ID live match: 91 of 167 distinct archived source rows survive; 32 of 110 distinct archived target/descendant identities survive.
- Direct repair substrate: 30 direct DD source rows cover 28 deleted targets. All 30 survive live in `extracted` state with their `FROM_DD_PATH` edge intact and no produced-name edge.
- Returnee control: exactly 48 of the 539 identities ever deleted by this reaper currently exist again. Of those 48, 41 have current children and 42 have current sources; 38 are currently `origin=derived`. This demonstrates that re-derivation works when source or child substrate survives.
- Login-node queries were exact-cohort reads and completed in 0.142 s for source matching, 0.030 s for descendant matching, 1.624 s for the returnee control, and 5.912 s for the separate WEST source check.

## Per-identity evidence

Every row below has the same independently established live facts: the `StandardName` is absent, live reviews are zero, and exactly one cleanup event records the reason quoted above. `Archive reviews` is the number of reviews the node held before deletion. `Archive direct DD binding` names a binding on the target itself; child bindings are shown only when they prove a shadow or family route.

| name | archive direct DD binding or child evidence | archive reviews | verdict | publishability | reason and live re-derivation substrate |
|---|---|---:|---|---|---|
| `absorbed_power_of_neutral_beam_injector` | none | 4 | UNDETERMINED | not publishable | Archive has only a derived source and no child. Need the deletion-time child/source topology or an explicit catalog receipt. |
| `absorbed_power_of_plant_system` | none | 15 | UNDETERMINED | not publishable | Archive has only a derived source and no child. Need the deletion-time child/source topology or an explicit catalog receipt. |
| `angle_of_electron_cyclotron_launcher_mirror` | child `maximum_angle_of_electron_cyclotron_launcher_mirror`: `ec_launchers/mirror/movement/rotation_angle_max` | 12 | CORRECTLY REMOVED | not applicable | Source-equivalent single-child shadow; the independently DD-backed child survives live. |
| `argon_density_at_pedestal_top` | `summary/local/pedestal/n_i/argon/value` | 14 | RESTORE | accepted after recompose | The archived DD source survives live as `extracted`, with its DD edge intact. |
| `atomic_count_of_pellet` | child `core_atomic_count_of_pellet`: `spi/injector/pellet/core/atoms_n` | 14 | CORRECTLY REMOVED | not applicable | Source-equivalent single-child shadow; the independently DD-backed child survives live. |
| `atomic_fraction_of_neutron_detector_converter` | none | 6 | CORRECTLY REMOVED | not applicable | Archived as `exhausted` with no independent DD binding. |
| `atomic_mass_of_wall_material` | none | 7 | UNDETERMINED | not publishable | Archive has only a derived source and no child. Need the deletion-time child/source topology or an explicit catalog receipt. |
| `beryllium_density_at_pedestal_top` | `summary/local/pedestal/n_i/beryllium/value` | 6 | RESTORE | accepted after recompose | The archived DD source survives live as `extracted`, with its DD edge intact. |
| `boron_density_at_pedestal_top` | `summary/local/pedestal/n_i/boron/value` | 9 | RESTORE | accepted after recompose | The archived DD source survives live as `extracted`, with its DD edge intact. |
| `bulk_plasma_velocity_due_to_diamagnetic_drift` | none | 13 | UNDETERMINED | not publishable | Archive has only a derived source and no child. Need the deletion-time child/source topology or an explicit catalog receipt. |
| `carbon_density_at_pedestal_top` | `summary/local/pedestal/n_i/carbon/value` | 8 | RESTORE | accepted after recompose | The archived DD source survives live as `extracted`, with its DD edge intact. |
| `count_of_pellet` | sole child `atomic_count_of_pellet` | 4 | CORRECTLY REMOVED | not applicable | Its sole archived child is itself a source-equivalent shadow; this parent has no independent source. |
| `critical_electric_field` | none | 4 | CORRECTLY REMOVED | not applicable | Archived as `exhausted` with no independent DD binding. |
| `current_density_due_to_viscosity` | none | 16 | UNDETERMINED | not publishable | Archive has only a derived source and no child. Need the deletion-time child/source topology or an explicit catalog receipt. |
| `density_at_pedestal_top` | 21 DD paths across 19 archived children | 30 | RESTORE | accepted after family rebuild | Restore the 19 DD-backed children from surviving source rows, then rerun structural derivation. This family cannot be re-minted parent-first. |
| `density_of_pellet` | none | 3 | CORRECTLY REMOVED | not applicable | Archived as `superseded` with no independent DD binding. |
| `deuterium_density_at_pedestal_top` | `summary/local/pedestal/n_i/deuterium/value` | 8 | RESTORE | accepted after recompose | The archived DD source survives live as `extracted`, with its DD edge intact. |
| `deuterium_deuterium_neutron_flux` | `summary/fusion/neutron_rates/dd/total/value` | 4 | RESTORE | review required | The archived DD source survives live as `extracted`; the archived name was only `drafted`. |
| `deuterium_deuterium_neutron_flux_due_to_beam_thermal_fusion` | none | 7 | UNDETERMINED | not publishable | Archive has only a derived source and no child. Need the deletion-time child/source topology or an explicit catalog receipt. |
| `deuterium_deuterium_neutron_source_rate_due_to_thermal_fusion` | `summary/fusion/neutron_rates/dd/thermal/value` | 6 | RESTORE | accepted after recompose | The archived DD source survives live as `extracted`, with its DD edge intact. |
| `deuterium_tritium_density_at_pedestal_top` | `summary/local/pedestal/n_i/deuterium_tritium/value` | 20 | RESTORE | accepted after recompose | The archived DD source survives live as `extracted`, with its DD edge intact. |
| `deuterium_tritium_neutron_flux_due_to_beam_thermal_fusion` | none | 9 | UNDETERMINED | not publishable | Archive has only a derived source and no child. Need the deletion-time child/source topology or an explicit catalog receipt. |
| `effective_charge_at_pedestal_top` | none | 6 | UNDETERMINED | not publishable | Archive has only a derived source and no child. Need the deletion-time child/source topology or an explicit catalog receipt. |
| `electron_density_at_pedestal_top` | `summary/local/pedestal/n_e/value`; `summary/pedestal_fits/linear/n_e/pedestal_height/value`; `summary/pedestal_fits/mtanh/n_e/pedestal_height/value` | 7 | RESTORE | review required | All three archived DD sources survive live as `extracted`; the archived name was `reviewed`. |
| `electron_power_density_due_to_collisions` | child `thermal_electron_power_density_due_to_collisions`: two `distributions/.../power_thermal` paths | 8 | CORRECTLY REMOVED | not applicable | Source-equivalent single-child shadow; the independently DD-backed child survives live. |
| `energy_convection_velocity` | child `electron_energy_convection_velocity`: three electron-energy transport paths | 3 | CORRECTLY REMOVED | not applicable | Source-equivalent single-child shadow; the independently DD-backed child survives live. |
| `energy_flux_at_wall` | `wall/description_ggd/ggd/power_density/values` | 7 | RESTORE | accepted after recompose | The archived DD source survives live as `extracted`, with its DD edge intact. |
| `flux_due_to_diamagnetic_drift` | child `momentum_flux_due_to_diamagnetic_drift`: two momentum-flux paths | 13 | CORRECTLY REMOVED | not applicable | Source-equivalent single-child shadow; the independently DD-backed child survives live. |
| `gyrocenter_pressure` | child `perturbed_gyrocenter_pressure`, derived only | 21 | UNDETERMINED | not publishable | Neither node nor a DD source in this chain survives. Need deletion-time non-derived source authority or a catalog receipt. |
| `helium_3_density_at_pedestal_top` | `summary/local/pedestal/n_i/helium_3/value` | 6 | RESTORE | accepted after recompose | The archived DD source survives live as `extracted`, with its DD edge intact. |
| `helium_4_density_at_pedestal_top` | `summary/local/pedestal/n_i/helium_4/value` | 6 | RESTORE | accepted after recompose | The archived DD source survives live as `extracted`, with its DD edge intact. |
| `hydrogen_density_at_pedestal_top` | `summary/local/pedestal/n_i/hydrogen/value` | 6 | RESTORE | accepted after recompose | The archived DD source survives live as `extracted`, with its DD edge intact. |
| `ion_momentum` | none | 4 | CORRECTLY REMOVED | not applicable | Archived as `superseded` with no independent DD binding. |
| `ion_power_density` | child `fast_ion_power_density`: two wave power-density paths | 9 | CORRECTLY REMOVED | not applicable | Source-equivalent single-child shadow; the independently DD-backed child survives live. |
| `ion_state_energy_flux` | none | 0 | CORRECTLY REMOVED | not applicable | Archived as `superseded` with no independent DD binding and no review. |
| `ion_state_momentum_flux` | none | 3 | CORRECTLY REMOVED | not applicable | Archived as `superseded` with no independent DD binding. |
| `iron_density_at_pedestal_top` | `summary/local/pedestal/n_i/iron/value` | 8 | RESTORE | accepted after recompose | The archived DD source survives live as `extracted`, with its DD edge intact. |
| `krypton_density_at_pedestal_top` | `summary/local/pedestal/n_i/krypton/value` | 10 | RESTORE | accepted after recompose | The archived DD source survives live as `extracted`, with its DD edge intact. |
| `launched_power_of_electron_cyclotron_launcher` | none | 0 | CORRECTLY REMOVED | not applicable | Archived as `exhausted` with no independent DD binding and no review. |
| `lithium_density_at_pedestal_top` | `summary/local/pedestal/n_i/lithium/value` | 6 | RESTORE | accepted after recompose | The archived DD source survives live as `extracted`, with its DD edge intact. |
| `mass_of_wall_material` | none | 7 | CORRECTLY REMOVED | not applicable | Archived as `exhausted` with no independent DD binding. |
| `motional_stark_photon_radiance_at_spectral_line` | none | 12 | RESTORE | catalog-proven | Catalog `unchanged_ratification` supplies authority; rebuild the archived node under signed repair authority. |
| `neon_density_at_pedestal_top` | `summary/local/pedestal/n_i/neon/value` | 7 | RESTORE | accepted after recompose | The archived DD source survives live as `extracted`, with its DD edge intact. |
| `nitrogen_density_at_pedestal_top` | `summary/local/pedestal/n_i/nitrogen/value` | 17 | RESTORE | accepted after recompose | The archived DD source survives live as `extracted`, with its DD edge intact. |
| `normalized_gyrocenter_perturbed_pressure` | none | 9 | UNDETERMINED | not publishable | Archive has only a derived source and no child. Need the deletion-time child/source topology or an explicit catalog receipt. |
| `normalized_perturbed_vector_potential` | child `parallel_normalized_perturbed_vector_potential`, derived only | 19 | UNDETERMINED | not publishable | Neither node nor a DD source in this algebraic chain survives. Need deletion-time non-derived source authority or a catalog receipt. |
| `oxygen_density_at_pedestal_top` | `summary/local/pedestal/n_i/oxygen/value` | 13 | RESTORE | accepted after recompose | The archived DD source survives live as `extracted`, with its DD edge intact. |
| `parallel_normalized_perturbed_vector_potential` | none | 4 | UNDETERMINED | not publishable | Archive has only a derived source and no child. Need the deletion-time child/source topology or an explicit catalog receipt. |
| `parity_of_gyrokinetic_eigenmode` | none | 10 | UNDETERMINED | not publishable | Archive has only a derived source and no child. Need the deletion-time child/source topology or an explicit catalog receipt. |
| `particle_flux_at_wall` | child `ion_charge_state_particle_flux_at_wall`: emitted and incident wall paths | 9 | CORRECTLY REMOVED | not applicable | Source-equivalent single-child shadow; the independently DD-backed child survives live. |
| `particle_flux_at_wall_due_to_recombination` | child `neutral_particle_flux_at_wall_due_to_recombination`: `wall/description_ggd/ggd/energy_fluxes/recombination/neutral/emitted/values` | 12 | CORRECTLY REMOVED | not applicable | Source-equivalent single-child shadow; the independently DD-backed child survives live. |
| `perturbed_gyrocenter_pressure` | none | 16 | UNDETERMINED | not publishable | Archive has only a derived source and no child. Need the deletion-time child/source topology or an explicit catalog receipt. |
| `perturbed_particle_pressure` | none | 1 | UNDETERMINED | not publishable | Archive has only a derived source and no child. Need the deletion-time child/source topology or an explicit catalog receipt. |
| `perturbed_plasma_mass_density` | none | 11 | UNDETERMINED | not publishable | Archive has only a derived source and no child. Need the deletion-time child/source topology or an explicit catalog receipt. |
| `perturbed_plasma_pressure` | none | 12 | RESTORE | catalog-proven | Catalog `unchanged_ratification` supplies authority; rebuild the archived node under signed repair authority. |
| `perturbed_plasma_temperature` | none | 12 | UNDETERMINED | not publishable | Archive has only a derived source and no child. Need the deletion-time child/source topology or an explicit catalog receipt. |
| `perturbed_pressure` | none | 24 | UNDETERMINED | not publishable | Archive has only a derived source and no child. Need the deletion-time child/source topology or an explicit catalog receipt. |
| `plasma_current_due_to_ohmic_induction` | none | 15 | UNDETERMINED | not publishable | Archive has only a derived source and no child. Need the deletion-time child/source topology or an explicit catalog receipt. |
| `plasma_energy` | child `thermal_plasma_energy`: `summary/global_quantities/energy_thermal/value` | 16 | CORRECTLY REMOVED | not applicable | Source-equivalent single-child shadow; the independently DD-backed child survives live. |
| `poloidal_angle` | seven archived locus children | 7 | CORRECTLY REMOVED | not applicable | Archived as `superseded` with no independent DD binding; the surviving children do not justify reviving a terminal parent. |
| `poloidal_momentum_flux_limiter_coefficient` | `edge_transport/model/ggd/momentum/flux_limiter/poloidal` | 12 | RESTORE | review required | The archived DD source survives live as `extracted`; the archived name was `reviewed`. |
| `power_at_inner_divertor_target` | child `electron_power_at_inner_divertor_target`: `wall/global_quantities/electrons/power_inner_target` | 13 | CORRECTLY REMOVED | not applicable | Source-equivalent single-child shadow; the independently DD-backed child survives live. |
| `power_at_outer_divertor_target` | child `electron_power_at_outer_divertor_target`: `wall/global_quantities/electrons/power_outer_target` | 25 | CORRECTLY REMOVED | not applicable | Source-equivalent single-child shadow; the independently DD-backed child survives live. |
| `power_at_wall_due_to_recombination` | child `plasma_power_at_wall_due_to_recombination`: `wall/global_quantities/power_recombination_plasma` | 9 | CORRECTLY REMOVED | not applicable | Source-equivalent single-child shadow; the independently DD-backed child survives live. |
| `power_due_to_fusion` | none | 8 | UNDETERMINED | not publishable | Archive has only a derived source and no child. Need the deletion-time child/source topology or an explicit catalog receipt. |
| `power_due_to_radiation` | none | 13 | UNDETERMINED | not publishable | Archive has only a derived source and no child. Need the deletion-time child/source topology or an explicit catalog receipt. |
| `radial_momentum_flux_limiter_coefficient` | `edge_transport/model/ggd/momentum/flux_limiter/r` | 8 | RESTORE | review required | The archived DD source survives live as `extracted`; the archived name was `reviewed`. |
| `safety_factor_at_pedestal_top` | `summary/local/pedestal/q/value` | 6 | RESTORE | catalog-proven | The archived DD source survives live as `extracted`, and catalog `unchanged_ratification` independently confirms authority. |
| `spectral_etendue_of_spectrometer_channel` | none | 73 | RESTORE | catalog-proven | Catalog `unchanged_ratification` supplies authority; rebuild the archived node under signed repair authority. |
| `target_atomic_fraction_of_neutron_detector_converter` | child `nuclear_target_atomic_fraction_of_neutron_detector_converter`: `neutron_diagnostic/detector/nuclei_n` | 12 | CORRECTLY REMOVED | not applicable | Source-equivalent single-child shadow; the independently DD-backed child survives live. |
| `temperature_at_midplane` | none | 12 | UNDETERMINED | not publishable | Archive has only a derived source and no child. Need the deletion-time child/source topology or an explicit catalog receipt. |
| `toroidal_neutral_momentum_flux_limiter_coefficient` | `edge_transport/model/ggd/neutral/momentum/flux_limiter/phi` | 11 | RESTORE | accepted after recompose | The archived DD source survives live as `extracted`, with its DD edge intact. |
| `total_ion_density_at_pedestal_top` | `summary/local/pedestal/n_i_total/value` | 10 | RESTORE | accepted after recompose | The archived DD source survives live as `extracted`, with its DD edge intact. |
| `tritium_density_at_pedestal_top` | `summary/local/pedestal/n_i/tritium/value` | 14 | RESTORE | accepted after recompose | The archived DD source survives live as `extracted`, with its DD edge intact. |
| `tritium_tritium_neutron_flux` | `summary/fusion/neutron_rates/tt/total/value` | 0 | RESTORE | review required | The archived DD source survives live as `extracted`; the archived name was only `drafted` and had no review. |
| `tungsten_density_at_pedestal_top` | `summary/local/pedestal/n_i/tungsten/value` | 8 | RESTORE | accepted after recompose | The archived DD source survives live as `extracted`, with its DD edge intact. |
| `voltage_of_neutron_detector` | none | 0 | CORRECTLY REMOVED | not applicable | Archived as `exhausted` with no independent DD binding; its requested-voltage child survives. |
| `xenon_density_at_pedestal_top` | `summary/local/pedestal/n_i/xenon/value` | 11 | RESTORE | accepted after recompose | The archived DD source survives live as `extracted`, with its DD edge intact. |

## Restoration order and feasibility

The RESTORE set is ranked by publishability, not by how physical its spelling sounds:

1. **Catalog-proven:** `motional_stark_photon_radiance_at_spectral_line`, `perturbed_plasma_pressure`, `safety_factor_at_pedestal_top`, and `spectral_etendue_of_spectrometer_channel` carry `unchanged_ratification`. `safety_factor_at_pedestal_top` also has a surviving DD source. These are the first publication candidates after identity reconstruction.
2. **Accepted after recompose:** 22 accepted direct-DD identities have surviving `extracted` source rows. Re-run the DD source compose/attach lifecycle; do not replay rows by hand.
3. **Accepted after family rebuild:** restore the 19 DD-backed species members of the pedestal-top density family first, then rerun structural derivation to recreate `density_at_pedestal_top` from its children.
4. **Review required:** `deuterium_deuterium_neutron_flux`, `electron_density_at_pedestal_top`, `poloidal_momentum_flux_limiter_coefficient`, `radial_momentum_flux_limiter_coefficient`, and `tritium_tritium_neutron_flux` have recoverable DD provenance but were only `drafted` or `reviewed` in the archive.

Feasibility is demonstrated in two independent ways. First, all 30 direct DD sources required by the 28 direct-source targets are live, reset to `extracted`, and still attached to their DD nodes. Second, 48 historical reaper deletions have already returned, with 42 carrying current sources or 41 current children. The source-backed route is therefore an observed pipeline behavior, not a hypothetical repair.

The three ratified names without direct DD sources require selective archive reconstruction: the isolated load returned their full node properties and archived reviews, while live `unchanged_ratification` supplies the authority that the archive alone cannot. A repair node can turn those records into a signed, exact-identity manifest. Replaying the entire archive is prohibited because it predates two days of valid graph work.

## WEST review cut

None of the 78 identities appears in the 208-name WEST review cut. One additional WEST name was deleted by the same reaper outside the origin-reconcile intersection:

- `etendue_of_spectrometer_channel` was already `origin=derived` in the archive, so it had no origin-reconcile event.
- The archive shows an accepted scalar with four reviews, mean score 0.975, a direct DD binding to `soft_x_rays/channel/etendue`, and child `spectral_etendue_of_spectrometer_channel`.
- Its DD source survives live with the DD edge intact and no produced name. It attempted recomposition after deletion and stopped at the compose-attempt cap.
- Verdict: **RESTORE**. Reset the surviving DD source through the governed source lifecycle, recompose the parent, then reconstruct or reattach the ratified spectral child. This is separate from the 32 members of the 78-row cohort.

## Code defects and evidence limit

- `imas_codex/standard_names/graph_ops.py:3608`: origin is used as the safety boundary even though independently DD-backed names can carry `origin=derived`. The WEST parent and 28 direct-DD targets prove that this property is not a sufficient authorship guard.
- `imas_codex/standard_names/graph_ops.py:3633`: deletion destroys source links, reviews, documentation revisions, and node properties while its change record keeps no reconstructive snapshot.
- The archive is from `2026-09-06`, not the deletion instant. For the 22 UNDETERMINED rows, the exact missing fact is the incoming `HAS_PARENT` child set and non-derived source ownership immediately before deletion, or an explicit catalog receipt. Neither the live graph, the archive, nor `StandardNameChange` contains that state. Their verdicts must remain UNDETERMINED until another durable record supplies it.
