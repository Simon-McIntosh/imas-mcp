# Successor migration receipts

## Outcome

**All 47 executable routes of the 48 strict-non-canonical drafted-and-open
successors reached their canonical spellings: 35 folds and 12 staged renames,
each dry-run immediately before its own live transaction, executed one identity
at a time with folds before renames.** Every one carries a per-identity receipt
below recording the identity before, the route taken, the canonical target, and
the state read back from the graph after the transaction committed. The
48th row remains blocked by the fold guard and is reported as a finding, not
forced; the single identity the strict parser cannot resolve is outside the 48
and untouched.

| Route | Planned | Executed | Verified |
|---|---:|---:|---:|
| Fold | 35 | 35 | 35 |
| Staged rename | 12 | 12 | 12 |
| Blocked fold | 1 | 0 | n/a — rechecked, still blocked |
| **Total strict-non-canonical** | **48** | **47** | **47** |
| No canonical target (held separate) | 1 | 0 | n/a — no target exists |

The refusal that stopped the earlier attempt did not recur. That attempt's
first fold rolled back on a postflight exact-state proof whose expectation was
computed before the write while the mutation stamped `updated_at` from the
transaction clock. With both sides taking the stamp from the single instant the
receipt declares, identity 1 — `argon_density_flux_surface_averaged_at_plasma_boundary`
— committed on the first attempt, and so did the other 46.

## Numbered findings

1. **Every executed route matched its planned route.** No identity took a
   different mechanism from the one the census enumerated, and no dry run had
   to be re-run or forced. The route table was read from the committed census
   document rather than from a prior run's log, and its 35/12/1 partition was
   asserted before the first mutation.
2. **The one blocked fold is still blocked under the current fold guard.** The
   guard now reads the current successor from the ordering of change-ledger
   rows rather than from lineage-edge existence, and it returns the same
   verdict:

   ```text
   target 'flux_surface_averaged_deuterium_tritium_density_at_plasma_boundary' is superseded and has successor lineage: flux_surface_averaged_deuterium_tritium_density — fold into the successor instead
   ```

   The dry run took 2.422 seconds, returned `ok: false`, and raised nothing. No
   fold was attempted for this identity and no guard was edited.
3. **Source bindings moved with every identity and none were left behind.**
   Each postflight asserted that the predecessor retains no authoritative
   source binding and that the canonical target's source set is exactly the
   predecessor's prior set. All 47 passed.
4. **The two routes leave different postflight signatures, both as designed.**
   A fold preserves the target's own description, documentation, unit and kind
   while the predecessor becomes `superseded`; a staged rename carries the
   predecessor's description digest and validation qualifier onto a target that
   did not previously exist. Both were asserted per identity.
5. **The one quarantined predecessor kept its qualifier.**
   `toroidal_fast_electron_torque_density_volume_integrated_due_to_collisions`
   was the only staged-rename predecessor with zero attached sources and a
   `quarantined` validation status. Its successor
   `volume_integrated_toroidal_fast_electron_torque_density_due_to_collisions`
   reads back `quarantined`, so successful staging was not mistaken for
   eligibility for review or acceptance.
6. **Live graph access ran on the login node under the declared exception.**
   The Neo4j tunnel is loopback-local to the login node. Every read was bounded
   to one exact identity pair; the slowest of 564 bounded reads took **0.043
   seconds** and none approached the ten-second ceiling. The slowest single
   operation of the run was one live fold transaction at **8.072 seconds**.

## Per-identity receipts — fold, 35

Each row is one transaction: the stored identity as it was read immediately
before the write, the route, the canonical target, and the state read back
afterwards. `sources` is the count of authoritative source bindings that moved
from the predecessor to the target.

| # | Identity before (stage / edit / validation) | Route | Canonical target | State read back after |
|---:|---|---|---|---|
| 1 | `argon_density_flux_surface_averaged_at_plasma_boundary`<br>drafted / open / valid; target `superseded` | fold | `flux_surface_averaged_argon_density_at_plasma_boundary` | predecessor superseded / applied, 0 sources; target reviewed / None / valid, 1 source, lineage 1 |
| 2 | `beryllium_density_flux_surface_averaged_at_plasma_boundary`<br>drafted / open / valid; target `superseded` | fold | `flux_surface_averaged_beryllium_density_at_plasma_boundary` | predecessor superseded / applied, 0 sources; target reviewed / None / valid, 1 source, lineage 1 |
| 3 | `carbon_count_accumulated_due_to_gas_injection`<br>drafted / open / valid; target `superseded` | fold | `accumulated_carbon_count_due_to_gas_injection` | predecessor superseded / applied, 0 sources; target reviewed / None / valid, 1 source, lineage 1 |
| 4 | `carbon_density_flux_surface_averaged_at_plasma_boundary`<br>drafted / open / valid; target `superseded` | fold | `flux_surface_averaged_carbon_density_at_plasma_boundary` | predecessor superseded / applied, 0 sources; target reviewed / None / valid, 1 source, lineage 1 |
| 5 | `deuterated_methane_count_accumulated_due_to_gas_injection`<br>drafted / open / valid; target `superseded` | fold | `accumulated_deuterated_methane_count_due_to_gas_injection` | predecessor superseded / applied, 0 sources; target reviewed / None / valid, 2 sources, lineage 1 |
| 6 | `deuterium_count_accumulated_due_to_gas_injection`<br>drafted / open / valid; target `superseded` | fold | `accumulated_deuterium_count_due_to_gas_injection` | predecessor superseded / applied, 0 sources; target reviewed / None / valid, 1 source, lineage 1 |
| 7 | `deuterium_density_flux_surface_averaged_at_plasma_boundary`<br>drafted / open / valid; target `superseded` | fold | `flux_surface_averaged_deuterium_density_at_plasma_boundary` | predecessor superseded / applied, 0 sources; target reviewed / None / valid, 1 source, lineage 1 |
| 8 | `electron_density_flux_surface_averaged_at_plasma_boundary`<br>drafted / open / valid; target `superseded` | fold | `flux_surface_averaged_electron_density_at_plasma_boundary` | predecessor superseded / applied, 0 sources; target reviewed / None / valid, 1 source, lineage 1 |
| 9 | `ethylene_count_cumulative_due_to_gas_injection`<br>drafted / open / quarantined; target `superseded` | fold | `cumulative_ethylene_count_due_to_gas_injection` | predecessor superseded / applied, 0 sources; target exhausted / None / quarantined, 0 sources, lineage 1 |
| 10 | `helium_3_count_accumulated_due_to_gas_injection`<br>drafted / open / valid; target `superseded` | fold | `accumulated_helium_3_count_due_to_gas_injection` | predecessor superseded / applied, 0 sources; target reviewed / None / valid, 1 source, lineage 1 |
| 11 | `helium_3_density_flux_surface_averaged_at_plasma_boundary`<br>drafted / open / valid; target `superseded` | fold | `flux_surface_averaged_helium_3_density_at_plasma_boundary` | predecessor superseded / applied, 0 sources; target reviewed / None / valid, 1 source, lineage 1 |
| 12 | `helium_3_prefill_count_accumulated_due_to_gas_injection`<br>drafted / open / valid; target `superseded` | fold | `accumulated_helium_3_prefill_count_due_to_gas_injection` | predecessor superseded / applied, 0 sources; target reviewed / None / valid, 1 source, lineage 1 |
| 13 | `helium_4_density_flux_surface_averaged_at_plasma_boundary`<br>drafted / open / valid; target `superseded` | fold | `flux_surface_averaged_helium_4_density_at_plasma_boundary` | predecessor superseded / applied, 0 sources; target reviewed / None / valid, 1 source, lineage 1 |
| 14 | `hydrogen_count_accumulated_due_to_gas_injection`<br>drafted / open / valid; target `superseded` | fold | `accumulated_hydrogen_count_due_to_gas_injection` | predecessor superseded / applied, 0 sources; target reviewed / None / valid, 1 source, lineage 1 |
| 15 | `hydrogen_density_flux_surface_averaged_at_plasma_boundary`<br>drafted / open / valid; target `superseded` | fold | `flux_surface_averaged_hydrogen_density_at_plasma_boundary` | predecessor superseded / applied, 0 sources; target reviewed / None / valid, 1 source, lineage 1 |
| 16 | `iron_density_flux_surface_averaged_at_plasma_boundary`<br>drafted / open / valid; target `superseded` | fold | `flux_surface_averaged_iron_density_at_plasma_boundary` | predecessor superseded / applied, 0 sources; target reviewed / None / valid, 1 source, lineage 1 |
| 17 | `lithium_count_accumulated_due_to_gas_injection`<br>drafted / open / valid; target `superseded` | fold | `accumulated_lithium_count_due_to_gas_injection` | predecessor superseded / applied, 0 sources; target reviewed / None / valid, 1 source, lineage 1 |
| 18 | `lithium_density_flux_surface_averaged_at_plasma_boundary`<br>drafted / open / valid; target `superseded` | fold | `flux_surface_averaged_lithium_density_at_plasma_boundary` | predecessor superseded / applied, 0 sources; target reviewed / None / valid, 1 source, lineage 1 |
| 19 | `lithium_prefill_count_accumulated_due_to_gas_injection`<br>drafted / open / valid; target `superseded` | fold | `accumulated_lithium_prefill_count_due_to_gas_injection` | predecessor superseded / applied, 0 sources; target reviewed / applied / valid, 1 source, lineage 1 |
| 20 | `methane_carbon_13_count_accumulated_due_to_gas_injection`<br>drafted / open / valid; target `superseded` | fold | `accumulated_methane_carbon_13_count_due_to_gas_injection` | predecessor superseded / applied, 0 sources; target reviewed / None / valid, 2 sources, lineage 1 |
| 21 | `neon_density_flux_surface_averaged_at_plasma_boundary`<br>drafted / open / valid; target `superseded` | fold | `flux_surface_averaged_neon_density_at_plasma_boundary` | predecessor superseded / applied, 0 sources; target reviewed / None / valid, 1 source, lineage 1 |
| 22 | `nitrogen_count_accumulated_due_to_gas_injection`<br>drafted / open / valid; target `superseded` | fold | `accumulated_nitrogen_count_due_to_gas_injection` | predecessor superseded / applied, 0 sources; target reviewed / None / valid, 1 source, lineage 1 |
| 23 | `nitrogen_density_flux_surface_averaged_at_plasma_boundary`<br>drafted / open / valid; target `superseded` | fold | `flux_surface_averaged_nitrogen_density_at_plasma_boundary` | predecessor superseded / applied, 0 sources; target reviewed / None / valid, 1 source, lineage 1 |
| 24 | `oxygen_density_flux_surface_averaged_at_plasma_boundary`<br>drafted / open / valid; target `superseded` | fold | `flux_surface_averaged_oxygen_density_at_plasma_boundary` | predecessor superseded / applied, 0 sources; target reviewed / None / valid, 1 source, lineage 1 |
| 25 | `oxygen_prefill_count_accumulated_due_to_gas_injection`<br>drafted / open / valid; target `superseded` | fold | `accumulated_oxygen_prefill_count_due_to_gas_injection` | predecessor superseded / applied, 0 sources; target reviewed / None / valid, 1 source, lineage 1 |
| 26 | `particle_count_accumulated_at_pellet_path_due_to_pellet_injection`<br>drafted / open / valid; target `superseded` | fold | `accumulated_particle_count_at_pellet_path_due_to_pellet_injection` | predecessor superseded / applied, 0 sources; target reviewed / None / valid, 1 source, lineage 1 |
| 27 | `propane_count_accumulated_due_to_gas_injection`<br>drafted / open / valid; target `superseded` | fold | `accumulated_propane_count_due_to_gas_injection` | predecessor superseded / applied, 0 sources; target reviewed / None / valid, 1 source, lineage 1 |
| 28 | `silane_count_accumulated_due_to_gas_injection`<br>drafted / open / valid; target `superseded` | fold | `accumulated_silane_count_due_to_gas_injection` | predecessor superseded / applied, 0 sources; target reviewed / None / valid, 1 source, lineage 1 |
| 29 | `toroidal_total_plasma_momentum_cumulative_inside_flux_surface_at_separatrix`<br>drafted / open / valid; target `superseded` | fold | `toroidal_cumulative_inside_flux_surface_total_plasma_momentum_at_separatrix` | predecessor superseded / applied, 0 sources; target exhausted / None / quarantined, 0 sources, lineage 1 |
| 30 | `total_gas_count_accumulated_at_midplane_due_to_gas_injection`<br>drafted / open / valid; target `superseded` | fold | `accumulated_total_gas_count_at_midplane_due_to_gas_injection` | predecessor superseded / applied, 0 sources; target reviewed / None / valid, 2 sources, lineage 1 |
| 31 | `total_ion_density_flux_surface_averaged_at_plasma_boundary`<br>drafted / open / valid; target `superseded` | fold | `flux_surface_averaged_total_ion_density_at_plasma_boundary` | predecessor superseded / applied, 0 sources; target reviewed / None / valid, 1 source, lineage 1 |
| 32 | `tritium_count_accumulated_due_to_gas_injection`<br>drafted / open / valid; target `superseded` | fold | `accumulated_tritium_count_due_to_gas_injection` | predecessor superseded / applied, 0 sources; target reviewed / None / valid, 1 source, lineage 1 |
| 33 | `tritium_density_flux_surface_averaged_at_plasma_boundary`<br>drafted / open / valid; target `superseded` | fold | `flux_surface_averaged_tritium_density_at_plasma_boundary` | predecessor superseded / applied, 0 sources; target reviewed / None / valid, 1 source, lineage 1 |
| 34 | `xenon_count_accumulated_due_to_gas_injection`<br>drafted / open / valid; target `superseded` | fold | `accumulated_xenon_count_due_to_gas_injection` | predecessor superseded / applied, 0 sources; target reviewed / None / valid, 1 source, lineage 1 |
| 35 | `xenon_density_flux_surface_averaged_at_plasma_boundary`<br>drafted / open / valid; target `superseded` | fold | `flux_surface_averaged_xenon_density_at_plasma_boundary` | predecessor superseded / applied, 0 sources; target reviewed / None / valid, 1 source, lineage 1 |

## Per-identity receipts — staged rename, 12

| # | Identity before (stage / edit / validation) | Route | Canonical target | State read back after |
|---:|---|---|---|---|
| 36 | `toroidal_carbon_velocity_flux_surface_averaged_at_plasma_boundary`<br>drafted / open / valid; target `absent` | staged rename | `flux_surface_averaged_toroidal_carbon_velocity_at_plasma_boundary` | predecessor superseded / applied, 0 sources; target drafted / open / valid, 1 source, lineage 1 |
| 37 | `toroidal_deuterium_tritium_velocity_flux_surface_averaged_at_plasma_boundary`<br>drafted / open / valid; target `absent` | staged rename | `flux_surface_averaged_toroidal_deuterium_tritium_velocity_at_plasma_boundary` | predecessor superseded / applied, 0 sources; target drafted / open / valid, 1 source, lineage 1 |
| 38 | `toroidal_deuterium_velocity_flux_surface_averaged_at_plasma_boundary`<br>drafted / open / valid; target `absent` | staged rename | `flux_surface_averaged_toroidal_deuterium_velocity_at_plasma_boundary` | predecessor superseded / applied, 0 sources; target drafted / open / valid, 1 source, lineage 1 |
| 39 | `toroidal_fast_electron_torque_density_volume_integrated_due_to_collisions`<br>drafted / open / quarantined; target `absent` | staged rename | `volume_integrated_toroidal_fast_electron_torque_density_due_to_collisions` | predecessor superseded / applied, 0 sources; target drafted / open / quarantined, 0 sources, lineage 1 |
| 40 | `toroidal_helium_3_velocity_flux_surface_averaged_at_plasma_boundary`<br>drafted / open / valid; target `absent` | staged rename | `flux_surface_averaged_toroidal_helium_3_velocity_at_plasma_boundary` | predecessor superseded / applied, 0 sources; target drafted / open / valid, 1 source, lineage 1 |
| 41 | `toroidal_hydrogen_velocity_flux_surface_averaged_at_plasma_boundary`<br>drafted / open / valid; target `absent` | staged rename | `flux_surface_averaged_toroidal_hydrogen_velocity_at_plasma_boundary` | predecessor superseded / applied, 0 sources; target drafted / open / valid, 1 source, lineage 1 |
| 42 | `toroidal_ion_velocity_flux_surface_averaged_at_plasma_boundary`<br>drafted / open / valid; target `absent` | staged rename | `flux_surface_averaged_toroidal_ion_velocity_at_plasma_boundary` | predecessor superseded / applied, 0 sources; target drafted / open / valid, 1 source, lineage 1 |
| 43 | `toroidal_iron_velocity_flux_surface_averaged_at_plasma_boundary`<br>drafted / open / valid; target `absent` | staged rename | `flux_surface_averaged_toroidal_iron_velocity_at_plasma_boundary` | predecessor superseded / applied, 0 sources; target drafted / open / valid, 1 source, lineage 1 |
| 44 | `toroidal_neon_velocity_flux_surface_averaged_at_plasma_boundary`<br>drafted / open / valid; target `absent` | staged rename | `flux_surface_averaged_toroidal_neon_velocity_at_plasma_boundary` | predecessor superseded / applied, 0 sources; target drafted / open / valid, 1 source, lineage 1 |
| 45 | `toroidal_nitrogen_velocity_flux_surface_averaged_at_plasma_boundary`<br>drafted / open / valid; target `absent` | staged rename | `flux_surface_averaged_toroidal_nitrogen_velocity_at_plasma_boundary` | predecessor superseded / applied, 0 sources; target drafted / open / valid, 1 source, lineage 1 |
| 46 | `toroidal_tritium_velocity_flux_surface_averaged_at_plasma_boundary`<br>drafted / open / valid; target `absent` | staged rename | `flux_surface_averaged_toroidal_tritium_velocity_at_plasma_boundary` | predecessor superseded / applied, 0 sources; target drafted / open / valid, 1 source, lineage 1 |
| 47 | `toroidal_xenon_velocity_flux_surface_averaged_at_plasma_boundary`<br>drafted / open / valid; target `absent` | staged rename | `flux_surface_averaged_toroidal_xenon_velocity_at_plasma_boundary` | predecessor superseded / applied, 0 sources; target drafted / open / valid, 1 source, lineage 1 |

## Blocked fold — 1, rechecked and still blocked

| Stored spelling | Canonical target | Recheck verdict |
|---|---|---|
| `deuterium_tritium_density_flux_surface_averaged_at_plasma_boundary` | `flux_surface_averaged_deuterium_tritium_density_at_plasma_boundary` | Not foldable: the target is `superseded` with successor lineage to `flux_surface_averaged_deuterium_tritium_density`, so the guard directs the operation to that successor instead |

This is the row the census flagged for recheck if the fold guard changed its
answer. It did not. Folding into the named successor is a different target from
the one the strict parser supplies for this predecessor, so it is a decision
this node was not authorised to take and is left as a follow-on condition.

## No canonical target — 1, held separate and untouched

`inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width`
has no parser-supplied canonical form, so it has no target to migrate toward. It
is outside the 48-row route total and was neither read for a route nor mutated.

## Instrument

The execution instrument is preserved beside the worker manifest as
`execute_successor_routes.py`. It reads the route enumeration from the committed
census document, asserts the 35/12/1 partition before any write, rechecks the
blocked fold, then for each identity in turn reads the exact pair state, runs
that route's own dry run, executes only if the dry run passes, reads the pair
state back, and appends a verified receipt to
`successor-migration-receipts.jsonl` with an `fsync` before the next identity
begins. Any refusal or postflight mismatch raises, which stops the run at that
identity and writes a stop record naming it; no route is retried and no guard is
bypassed.
