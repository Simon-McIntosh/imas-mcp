# The re-minted WEST cohort, measured against the standing candidate

**Node:** re-mint of the WEST batch against the live `codex` graph and the
current tree, 2026-09-09. The re-mint reproduces the release path exactly —
resolve the 355 `west_production_dd_paths` manifest paths to their terminal
identities (`fetch_manifest_source_release_rows`), then export with that
review batch (`run_export`, same call the release orchestrates) — but stops at
the cohort: no freeze, tag, branch or pull request. Login-node run (the Neo4j
bolt endpoint is a login-node-local tunnel), bounded to the 355 paths and the
227-name roster; every read under the ten-second ceiling.

**Standing candidate** = fork request 18 (`review/v0.4.0rc6+west-task-2e`, cut
2026-09-08T10:48Z): 227-name roster, 226 candidates, **208 emitted identities**,
18 exclusions (12 `invalid_validation_status` + 6 `name_not_accepted`),
residue zero. Its emitted list is read from the committed
`.export_report.json` on that branch (`4ba2c0eb`).

## Re-minted cohort size, beside the standing candidate

| Cohort quantity | Standing (rc6, 09-08) | Re-minted (09-09) |
|---|---:|---:|
| Minted roster (review-batch terminal ids) | 227 | 227 |
| Export candidates | 226 | 226 |
| **Emitted identities** | **208** | **171** |
| Accounted exclusions | 18 | 55 |
| Accounting residue | 0 | 0 |
| Failed export gates | 0 | 0 |

The roster is the same size and the same count of candidates, but the emitted
set is **37 smaller** and adds **nothing**: the re-mint's 171 emitted
identities are a strict subset of the standing candidate's 208. The roster
membership differs by exactly two in each direction (below); both new members
are withheld from emission by the export, which is why the emitted set shrinks
rather than staying level.

## Roster membership: the exchange the known edits predict

Standing roster minus minted roster (2) and minted minus standing (2):

| Direction | Name | Mechanism |
|---|---|---|
| standing → out | `power_due_to_ion_cyclotron_heating` | the old spelling is a superseded tombstone (`name_stage=superseded`, `status=superseded`); its two WEST source bindings (`ic_antennas/antenna/power_launched`, `summary/heating_current_drive/ic/power/value`) now resolve to the successor |
| standing → out | `etendue_of_spectrometer_channel` | archived spelling, **absent from the live graph** (no stored identity) |
| → in (minted) | `net_power_due_to_ion_cyclotron_heating` | the successor the two IC-power sources resolve to |
| → in (minted) | `etendue_of_soft_xray_detector` | restored identity bound to `soft_x_rays/channel/etendue` |

## Per-name verdicts for the three known edits (re-minted cohort)

| Name | In minted roster | In emitted set | Verdict and reason |
|---|---|---|---|
| `power_due_to_ion_cyclotron_heating` | **absent** | **absent** | Correctly excluded: superseded tombstone; terminal resolution follows the successor, so the old spelling is not proposed. |
| `net_power_due_to_ion_cyclotron_heating` | **present** | **absent** | In the roster as the successor, but **withheld at emission as `never_reviewed`**: `docs_stage` scalar reads `accepted` yet the name carries **zero docs-axis review edges** (only 9 name-axis reviews), so the corrected exporter refuses — the false-acceptance projection defect (§9a class) on the successor. |
| `etendue_of_soft_xray_detector` | **present** | **absent** | Restored and bound to the manifest path, but **withheld as `validation_observation_missing`**: `validation_status='valid'` with `validated_at IS NULL` — an undated verdict the instrument-authority rule refuses. |
| `spectral_etendue_of_soft_xray_detector` | **absent** | **absent** | Lives in the graph accepted/valid (a qualifier child of `etendue_of_soft_xray_detector`, `origin=pipeline`), but is **not a cohort member**: its only `StandardNameSource` binding is `derived:spectral_etendue_of_spectrometer_channel`, not a dd batch path. The batch resolves manifest dd paths to terminal ids, so a derived-only identity falls outside the cohort even though it is published-eligible in the catalog. |
| archived etendue spellings | — | **absent** | `etendue_of_spectrometer_channel` is gone from the graph; `etendue`, `effective_spectral_etendue_of_spectrometer_channel` and `viewing_etendue_of_spectrometer_channel` are superseded tombstones. None emits. |

Measured truth over the plan's expectation: `etendue_of_soft_xray_detector`
and `spectral_etendue_of_soft_xray_detector` are **both restored identities and
both accepted/valid**, but only the first is a batch member today, and neither
emits (one lacks a `validated_at` observation; the other is derived-only).
The archived spellings are absent as intended.

## Full delta against the standing candidate's 208 emitted identities

**Added: 0.** Nothing emits in the re-mint that was not in the standing
candidate.

**Removed: 37**, every row attributed:

| Cause | Count | Names |
|---|---:|---|
| `documentation_not_accepted` (`docs_stage='pending'`) | 31 | `effective_charge`, `electron_density_at_plasma_boundary`, `faraday_angle`, `frequency_of_ion_cyclotron_heating_antenna`, `gap_at_outboard_midplane`, `initial_polarization_ellipticity_of_polarimeter_beam`, `launched_power_of_lower_hybrid_antenna`, `line_integrated_electron_number_density`, `magnetic_shear_at_flux_surface`, `maximum_magnetic_field_magnitude`, `normalized_plasma_internal_inductance`, `normalized_toroidal_flux_coordinate_at_measurement_position`, `plasma_current`, `poloidal_angle_of_flux_surface`, `poloidal_angle_of_measurement_position`, `poloidal_magnetic_flux_at_flux_surface`, `poloidal_magnetic_flux_at_measurement_position`, `poloidal_magnetic_flux_of_flux_loop`, `radial_coordinate_of_geometric_axis`, `radial_coordinate_of_magnetic_axis`, `radial_coordinate_of_strike_point`, `radial_outline_of_antenna_strap`, `safety_factor`, `toroidal_beta`, `toroidal_magnetic_flux`, `toroidal_magnetic_flux_due_to_diamagnetic_drift`, `total_power_due_to_ion_cyclotron_heating`, `vertical_coordinate_of_camera`, `vertical_coordinate_of_geometric_axis`, `vertical_coordinate_of_strike_point`, `volume_averaged_electron_density` |
| `unreviewed_name` (`reviewer_score_name` missing) | 4 | `atomic_mass`, `gas_flow`, `plasma_pressure`, `wave_phase_of_ion_cyclotron_heating_antenna` |
| absent from the live graph | 1 | `etendue_of_spectrometer_channel` (archived spelling, no stored identity) |
| not a batch candidate | 1 | `power_due_to_ion_cyclotron_heating` (superseded tombstone; replaced by its successor) |
| **Total** | **37** | |

Caveat on the 31: all 31 went `docs_stage='pending'` **today** (updated_at
spans 2026-09-09T07:30:27Z → 10:30:49Z) — a docs-lifecycle reset that is
concurrent restore work in a peer scope, not a settled graph state. The 171
emitted figure is therefore a live snapshot: it will grow as those families'
docs review re-lands, and shrink again if net_power's docs review is added and
accepted. The re-mint *machine* (227 roster, 226 candidates, residue zero, all
gates passed) is stable; only the emitted total is moving.

## Origin truthfulness in the re-minted cohort

Census of `origin` across the 227 minted-roster rows: **217 carry
`origin='pipeline'`; 10 carry a null origin.** So **yes — 10 roster rows still
lack a truthful origin** (an unset field states no provenance at all):

`accumulated_thermal_energy`, `accumulated_total_gas_count`,
`flux_surface_averaged_parallel_current_density`, `gap_at_closest_wall_point`,
`line_integrated_opacity`, `outer_hard_xray_half_width`,
`radial_outline_of_plasma_boundary`, `radial_outline_of_wall`,
`total_energy_of_calorimetry_component`, `total_neutron_rate`.

These are the newly-composed drain names plus two quarantine-held outlines:
the compose path minted them without setting `origin`. None is mislabelled as
a catalog edit; the gap is a null field, which is the "no provenance statement"
rather than a false one — but per the plan's own frame, an unset origin is not
a truthful origin, and it is the field the origin-provenance work must fill
before any of the ten can claim a truthful provenance.

## Method

Re-mint driver `remint_west.py` (mint + export of the 355 paths, dry-run
direction); existing `remint_west.json` and `remint_west.log`; export report
`remint-export-report.json`; the standing emitted list is from the rc6 fork
branch's committed `.export_report.json` (`standing-rc6-emitted.txt`). No
graph write was issued by this census.
