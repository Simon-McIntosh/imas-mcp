# Non-canonical successor route census

## Outcome

The live graph contains **105 drafted-and-open successors**. Their validation
partition is unchanged at **94 valid, 8 quarantined, and 3 pending**, while the
strict-parser partition has moved to **56 strict-valid, 48
strict-non-canonical, and 1 generic parse failure**. The earlier flat `49 of
105` figure therefore combines two different problems: 48 rows have a
parser-supplied canonical spelling, while one row cannot be parsed far enough
to supply any target.

The exact routes for the 48 strict-non-canonical rows are:

| Route | Count | Canonical target state | Governing mechanism |
|---|---:|---|---|
| Staged rename | 12 | Absent | `apply_edit(..., dry_run=True)` and every source/unit/attachment preflight passed |
| Fold | 35 | Present but `superseded`; never live | `supersede_into(..., dry_run=True)` passed |
| Blocked fold | 1 | Present but `superseded`; never live | Base `_fold_guard_reason` directs the operation to the target's current successor |
| **Total strict-non-canonical** | **48** | **0 targets occupied by another live identity** | Every row enumerated below |

The sizing result that motivated this census is unexpectedly clean:
**`retarget_standard_name_sources` would refuse 0 of the 48 rows for a
co-bound source**. More precisely, only the 12 absent-target rows take the
staged-rename path through `retarget_standard_name_sources`, and all 12 have a
clean compare-and-set; the other 36 take the fold path. The 2026-09-05
whole-cohort expectation that co-bound sources would dominate does not describe
this current drafted-and-open successor slice.

No edit was staged, no fold was executed, no source or graph data was changed,
and `sn run` was not invoked.

## Numbered findings

1. **The parser population moved without changing the validation census.** The
   graph still reads 105 = 94 valid + 8 quarantined + 3 pending, but the parser
   answer is now 56 strict-valid + 48 strict-non-canonical + 1 generic parse
   failure rather than 56 + 49.
2. **No canonical spelling is occupied by another live identity.** Thirty-six
   spellings exist only as retired `superseded` identities; twelve do not exist.
3. **The co-bound-source refusal count is zero.** The source compare-and-set on
   the 12 staged-renames found no third live binding, stale source, active claim,
   or scalar mismatch.
4. **The compute-shell `uv` wrapper injects `--no-sync`.** The first
   `all_debug` launch called `uv run --no-sync`; the shell function expanded it
   to `command uv run --no-sync --no-sync` and uv refused that duplicate option.
   A compute-node command should write `uv run ...` and let the wrapper provide
   the flag.
5. **Live graph access is login-node-local.** The graph profile resolves to
   `bolt://localhost:17687`. An `all_debug` allocation has no listener on that
   loopback address and cannot SSH out to establish its own tunnel. After the
   lead explicitly authorised this 105-row, read-only, indexed census on the
   login node, the slowest per-row preflight completed in **2.943 seconds**;
   the two bounded cohort queries took **0.039** and **0.074 seconds** in the
   full pass, and **0.031** and **0.025 seconds** in the 12-row correction pass.

## Instrument and code revision

The read-only instrument is preserved beside the worker manifest as
`route_census.py`. It obtains canonical targets only from the strict parser,
checks target identity and lifecycle, evaluates occupied spellings through a
rolled-back `supersede_into` dry-run, and evaluates absent spellings through the
edit dry-run plus the exact live-binding, source-status, claim, scalar, DD-unit,
and attachment predicates. Every query starts from the indexed
`StandardName.id` or a bounded list derived from the 105-row cohort. The
instrument stops if any query or per-row dry-run exceeds ten seconds.

Both governing mechanisms were read from worker base
`23461893c9d2b9818b3f3eb8ba1ccdc2ea3c69f3`:

- `retarget_standard_name_sources` in
  `imas_codex/standard_names/provenance_lifecycle.py`, last touched there by
  `0b79867ecc9b37a95bd1f6376cf7d8e11063fc95`;
- `_fold_guard_reason` in `imas_codex/standard_names/edit.py`, last touched
  there by `a8bd01c90aba331d8efa02c08261a3c982ea992b`.

A concurrent node changes `_fold_guard_reason`, not
`retarget_standard_name_sources`. The **0 co-bound-source refusals** are
therefore stable across that concurrent scope. The one blocked fold verdict
must be rechecked after integration if the fold guard changes its answer.

## Fold route — 35

All 35 canonical spellings already exist as retired
`name_stage='superseded'` identities. Therefore the answer to “occupied by
another live identity” is **No** for every row: the spelling is occupied in the
graph, but not by a live identity. The base-revision
`supersede_into(..., dry_run=True)` preflight passed all 35, so their reachable
route is a fold onto the occupied retired identity.

| Stored spelling | Ordered-grammar spelling | Occupied by another live identity? | Route |
|---|---|---|---|
| `argon_density_flux_surface_averaged_at_plasma_boundary` | `flux_surface_averaged_argon_density_at_plasma_boundary` | No — target is `superseded` | Fold onto the occupied retired identity; base dry-run passed |
| `beryllium_density_flux_surface_averaged_at_plasma_boundary` | `flux_surface_averaged_beryllium_density_at_plasma_boundary` | No — target is `superseded` | Fold onto the occupied retired identity; base dry-run passed |
| `carbon_count_accumulated_due_to_gas_injection` | `accumulated_carbon_count_due_to_gas_injection` | No — target is `superseded` | Fold onto the occupied retired identity; base dry-run passed |
| `carbon_density_flux_surface_averaged_at_plasma_boundary` | `flux_surface_averaged_carbon_density_at_plasma_boundary` | No — target is `superseded` | Fold onto the occupied retired identity; base dry-run passed |
| `deuterated_methane_count_accumulated_due_to_gas_injection` | `accumulated_deuterated_methane_count_due_to_gas_injection` | No — target is `superseded` | Fold onto the occupied retired identity; base dry-run passed |
| `deuterium_count_accumulated_due_to_gas_injection` | `accumulated_deuterium_count_due_to_gas_injection` | No — target is `superseded` | Fold onto the occupied retired identity; base dry-run passed |
| `deuterium_density_flux_surface_averaged_at_plasma_boundary` | `flux_surface_averaged_deuterium_density_at_plasma_boundary` | No — target is `superseded` | Fold onto the occupied retired identity; base dry-run passed |
| `electron_density_flux_surface_averaged_at_plasma_boundary` | `flux_surface_averaged_electron_density_at_plasma_boundary` | No — target is `superseded` | Fold onto the occupied retired identity; base dry-run passed |
| `ethylene_count_cumulative_due_to_gas_injection` | `cumulative_ethylene_count_due_to_gas_injection` | No — target is `superseded` | Fold onto the occupied retired identity; base dry-run passed |
| `helium_3_count_accumulated_due_to_gas_injection` | `accumulated_helium_3_count_due_to_gas_injection` | No — target is `superseded` | Fold onto the occupied retired identity; base dry-run passed |
| `helium_3_density_flux_surface_averaged_at_plasma_boundary` | `flux_surface_averaged_helium_3_density_at_plasma_boundary` | No — target is `superseded` | Fold onto the occupied retired identity; base dry-run passed |
| `helium_3_prefill_count_accumulated_due_to_gas_injection` | `accumulated_helium_3_prefill_count_due_to_gas_injection` | No — target is `superseded` | Fold onto the occupied retired identity; base dry-run passed |
| `helium_4_density_flux_surface_averaged_at_plasma_boundary` | `flux_surface_averaged_helium_4_density_at_plasma_boundary` | No — target is `superseded` | Fold onto the occupied retired identity; base dry-run passed |
| `hydrogen_count_accumulated_due_to_gas_injection` | `accumulated_hydrogen_count_due_to_gas_injection` | No — target is `superseded` | Fold onto the occupied retired identity; base dry-run passed |
| `hydrogen_density_flux_surface_averaged_at_plasma_boundary` | `flux_surface_averaged_hydrogen_density_at_plasma_boundary` | No — target is `superseded` | Fold onto the occupied retired identity; base dry-run passed |
| `iron_density_flux_surface_averaged_at_plasma_boundary` | `flux_surface_averaged_iron_density_at_plasma_boundary` | No — target is `superseded` | Fold onto the occupied retired identity; base dry-run passed |
| `lithium_count_accumulated_due_to_gas_injection` | `accumulated_lithium_count_due_to_gas_injection` | No — target is `superseded` | Fold onto the occupied retired identity; base dry-run passed |
| `lithium_density_flux_surface_averaged_at_plasma_boundary` | `flux_surface_averaged_lithium_density_at_plasma_boundary` | No — target is `superseded` | Fold onto the occupied retired identity; base dry-run passed |
| `lithium_prefill_count_accumulated_due_to_gas_injection` | `accumulated_lithium_prefill_count_due_to_gas_injection` | No — target is `superseded` | Fold onto the occupied retired identity; base dry-run passed |
| `methane_carbon_13_count_accumulated_due_to_gas_injection` | `accumulated_methane_carbon_13_count_due_to_gas_injection` | No — target is `superseded` | Fold onto the occupied retired identity; base dry-run passed |
| `neon_density_flux_surface_averaged_at_plasma_boundary` | `flux_surface_averaged_neon_density_at_plasma_boundary` | No — target is `superseded` | Fold onto the occupied retired identity; base dry-run passed |
| `nitrogen_count_accumulated_due_to_gas_injection` | `accumulated_nitrogen_count_due_to_gas_injection` | No — target is `superseded` | Fold onto the occupied retired identity; base dry-run passed |
| `nitrogen_density_flux_surface_averaged_at_plasma_boundary` | `flux_surface_averaged_nitrogen_density_at_plasma_boundary` | No — target is `superseded` | Fold onto the occupied retired identity; base dry-run passed |
| `oxygen_density_flux_surface_averaged_at_plasma_boundary` | `flux_surface_averaged_oxygen_density_at_plasma_boundary` | No — target is `superseded` | Fold onto the occupied retired identity; base dry-run passed |
| `oxygen_prefill_count_accumulated_due_to_gas_injection` | `accumulated_oxygen_prefill_count_due_to_gas_injection` | No — target is `superseded` | Fold onto the occupied retired identity; base dry-run passed |
| `particle_count_accumulated_at_pellet_path_due_to_pellet_injection` | `accumulated_particle_count_at_pellet_path_due_to_pellet_injection` | No — target is `superseded` | Fold onto the occupied retired identity; base dry-run passed |
| `propane_count_accumulated_due_to_gas_injection` | `accumulated_propane_count_due_to_gas_injection` | No — target is `superseded` | Fold onto the occupied retired identity; base dry-run passed |
| `silane_count_accumulated_due_to_gas_injection` | `accumulated_silane_count_due_to_gas_injection` | No — target is `superseded` | Fold onto the occupied retired identity; base dry-run passed |
| `toroidal_total_plasma_momentum_cumulative_inside_flux_surface_at_separatrix` | `toroidal_cumulative_inside_flux_surface_total_plasma_momentum_at_separatrix` | No — target is `superseded` | Fold onto the occupied retired identity; base dry-run passed |
| `total_gas_count_accumulated_at_midplane_due_to_gas_injection` | `accumulated_total_gas_count_at_midplane_due_to_gas_injection` | No — target is `superseded` | Fold onto the occupied retired identity; base dry-run passed |
| `total_ion_density_flux_surface_averaged_at_plasma_boundary` | `flux_surface_averaged_total_ion_density_at_plasma_boundary` | No — target is `superseded` | Fold onto the occupied retired identity; base dry-run passed |
| `tritium_count_accumulated_due_to_gas_injection` | `accumulated_tritium_count_due_to_gas_injection` | No — target is `superseded` | Fold onto the occupied retired identity; base dry-run passed |
| `tritium_density_flux_surface_averaged_at_plasma_boundary` | `flux_surface_averaged_tritium_density_at_plasma_boundary` | No — target is `superseded` | Fold onto the occupied retired identity; base dry-run passed |
| `xenon_count_accumulated_due_to_gas_injection` | `accumulated_xenon_count_due_to_gas_injection` | No — target is `superseded` | Fold onto the occupied retired identity; base dry-run passed |
| `xenon_density_flux_surface_averaged_at_plasma_boundary` | `flux_surface_averaged_xenon_density_at_plasma_boundary` | No — target is `superseded` | Fold onto the occupied retired identity; base dry-run passed |

## Blocked fold route — 1

| Stored spelling | Ordered-grammar spelling | Occupied by another live identity? | Route |
|---|---|---|---|
| `deuterium_tritium_density_flux_surface_averaged_at_plasma_boundary` | `flux_surface_averaged_deuterium_tritium_density_at_plasma_boundary` | No — target is `superseded` | Blocked by `_fold_guard_reason`: target has successor lineage to `flux_surface_averaged_deuterium_tritium_density`; fold into that successor instead |

This verdict was read from `imas_codex/standard_names/edit.py` as present in
worker base `23461893c9d2b9818b3f3eb8ba1ccdc2ea3c69f3`; the last commit touching
that file at the base is `a8bd01c90aba331d8efa02c08261a3c982ea992b`.

## Staged rename route — 12

All 12 canonical spellings are absent, so none is occupied by another live or
retired identity. The self-scoped edit dry-run, rename-unit authority,
attachment consistency, and exact source compare-and-set all pass. These rows
can therefore be queued with the sanctioned staged-rename route before the
single batch review.

| Stored spelling | Ordered-grammar spelling | Occupied by another live identity? | Route |
|---|---|---|---|
| `toroidal_carbon_velocity_flux_surface_averaged_at_plasma_boundary` | `flux_surface_averaged_toroidal_carbon_velocity_at_plasma_boundary` | No — target is absent | Staged rename; all read-only preflights passed |
| `toroidal_deuterium_tritium_velocity_flux_surface_averaged_at_plasma_boundary` | `flux_surface_averaged_toroidal_deuterium_tritium_velocity_at_plasma_boundary` | No — target is absent | Staged rename; all read-only preflights passed |
| `toroidal_deuterium_velocity_flux_surface_averaged_at_plasma_boundary` | `flux_surface_averaged_toroidal_deuterium_velocity_at_plasma_boundary` | No — target is absent | Staged rename; all read-only preflights passed |
| `toroidal_fast_electron_torque_density_volume_integrated_due_to_collisions` | `volume_integrated_toroidal_fast_electron_torque_density_due_to_collisions` | No — target is absent | Staged rename; all read-only preflights passed |
| `toroidal_helium_3_velocity_flux_surface_averaged_at_plasma_boundary` | `flux_surface_averaged_toroidal_helium_3_velocity_at_plasma_boundary` | No — target is absent | Staged rename; all read-only preflights passed |
| `toroidal_hydrogen_velocity_flux_surface_averaged_at_plasma_boundary` | `flux_surface_averaged_toroidal_hydrogen_velocity_at_plasma_boundary` | No — target is absent | Staged rename; all read-only preflights passed |
| `toroidal_ion_velocity_flux_surface_averaged_at_plasma_boundary` | `flux_surface_averaged_toroidal_ion_velocity_at_plasma_boundary` | No — target is absent | Staged rename; all read-only preflights passed |
| `toroidal_iron_velocity_flux_surface_averaged_at_plasma_boundary` | `flux_surface_averaged_toroidal_iron_velocity_at_plasma_boundary` | No — target is absent | Staged rename; all read-only preflights passed |
| `toroidal_neon_velocity_flux_surface_averaged_at_plasma_boundary` | `flux_surface_averaged_toroidal_neon_velocity_at_plasma_boundary` | No — target is absent | Staged rename; all read-only preflights passed |
| `toroidal_nitrogen_velocity_flux_surface_averaged_at_plasma_boundary` | `flux_surface_averaged_toroidal_nitrogen_velocity_at_plasma_boundary` | No — target is absent | Staged rename; all read-only preflights passed |
| `toroidal_tritium_velocity_flux_surface_averaged_at_plasma_boundary` | `flux_surface_averaged_toroidal_tritium_velocity_at_plasma_boundary` | No — target is absent | Staged rename; all read-only preflights passed |
| `toroidal_xenon_velocity_flux_surface_averaged_at_plasma_boundary` | `flux_surface_averaged_toroidal_xenon_velocity_at_plasma_boundary` | No — target is absent | Staged rename; all read-only preflights passed |

`toroidal_fast_electron_torque_density_volume_integrated_due_to_collisions` is
the only staged-rename predecessor with zero attached sources in this census.
That does not block an edit-origin rename: the atomic persistence path permits
an empty authoritative cohort when `edit_mode` is present, and the self-scoped
dry-run admitted it. Its stored validation status is `quarantined`, so the
migration node must retain that qualifier and must not equate successful
staging with eligibility for review or acceptance.

## No canonical target — 1 separate parser problem

This identity is outside the 48-row route total because strict parsing does not
produce `canonical_form`. It has no staged-rename or fold target to test.

| Stored spelling | Ordered-grammar spelling | Occupied by another live identity? | Route |
|---|---|---|---|
| `inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width` | None — parser supplies no canonical form | No target exists to test | Blocked by the strict ISN parse guard; requires a separately derived spelling, not a migration guess |

The parser reports:

```text
residue 'inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width'
does not match any physical_base or geometry_carrier; nearest candidates:
['normalized_toroidal_flux_coordinate']
```
