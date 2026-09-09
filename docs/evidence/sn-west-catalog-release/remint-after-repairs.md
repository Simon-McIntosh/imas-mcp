# WEST re-mint after repairs

## Authoritative result

The exact release path now emits **205 identities**. This was measured by
resolving all 355 paths in `west_production_dd_paths.yaml` with
`fetch_manifest_source_release_rows`, deduplicating the resulting 227 terminal
standard-name identities as the review batch, then calling `run_export` with
`force=True`, `skip_gate=True`, that review batch, and the manifest-source
projection. The run stopped after export: it did not freeze an artifact, create
a tag or branch, open a pull request, or apply a signed manifest.

| Measure | Standing candidate | Mid-repair re-mint | Fresh re-mint |
| --- | ---: | ---: | ---: |
| Manifest paths | 355 | 355 | 355 |
| Review-batch roster | 227 | 227 | 227 |
| Export candidates | 226 | 226 | 226 |
| **Emitted identities** | **208** | **171** | **205** |
| Accounted identity exclusions | 18 | 55 | 21 |
| Accounting residue | 0 | 0 | 0 |

![WEST export counts and dispositions](/imas-codex/figures/sn-west-catalog-release/remint-after-repairs.svg)

The fresh figure recovers **34** identities from the mid-repair 171 and is
three below the standing candidate's 208. It is one above the coordinator's
204-clean hypothesis. That hypothesis inferred lifecycle predicates only; the
release exporter additionally validates the emitted catalog entry against the
ISN catalog model. `etendue_of_soft_xray_detector` is accepted on both review
axes, has `validation_status='valid'` with
`validated_at='2026-09-09T13:03:04.778Z'`, and has scores 1.0 (name) and
0.9875 (docs), but fails that final catalog-entry validation. It therefore
forms the one extra hold the inferred read did not see.

All in-export gates passed: `catalog_status`, `identity_token_collision`,
`exclusion_accounting`, and `manifest_source_accounting` each passed with zero
issues and were not skipped. Source accounting independently covered all 355
paths: 314 emitted source rows, 23 excluded source rows, and 18 documented
non-nameable source rows, with 355 accounted.

## Exclusion ledger and dispositions

The export reports 21 excluded identities, fully attributed by the release
ledger: 1 `invalid_catalog_entry`, 12 `invalid_validation_status`, and 8
`name_not_accepted`. Each belongs to exactly one disposition below.

| Export reason | Count | Correct-and-permanent | Needs-real-work | Metadata-only |
| --- | ---: | ---: | ---: | ---: |
| `invalid_catalog_entry` | 1 | 0 | 1 | 0 |
| `invalid_validation_status` | 12 | 0 | 12 | 0 |
| `name_not_accepted` | 8 | 0 | 8 | 0 |
| **Total** | **21** | **0** | **21** | **0** |

There are no correct-and-permanent export exclusions in the terminal review
batch: the superseded IC-power spelling was resolved to its successor during
manifest minting, and the archived spectrometer-etendue spelling is absent from
the graph rather than an export candidate. There are no metadata-only holds:
every current exclusion needs a catalog-entry, validation, or name-lifecycle
repair. No row is unclassified.

| Identity | Export reason | Live evidence | Disposition |
| --- | --- | --- | --- |
| `etendue_of_soft_xray_detector` | `invalid_catalog_entry` | Accepted, valid, both review scores present; emitted entry fails ISN catalog-model validation. | Needs-real-work: correct the catalog-entry defect. |
| `accumulated_total_gas_count` | `invalid_validation_status` | `validation_status='pending'`, `name_stage='drafted'`. | Needs-real-work: complete candidate validation. |
| `distance_of_antenna_strap` | `invalid_validation_status` | `validation_status='pending'`, `name_stage='drafted'`. | Needs-real-work: complete candidate validation. |
| `gap_at_closest_wall_point` | `invalid_validation_status` | `validation_status='pending'`, `name_stage='drafted'`. | Needs-real-work: complete candidate validation. |
| `hard_xray_emissivity` | `invalid_validation_status` | `validation_status='pending'`, `name_stage='drafted'`. | Needs-real-work: complete candidate validation. |
| `neutral_pressure` | `invalid_validation_status` | `validation_status='pending'`, `name_stage='drafted'`. | Needs-real-work: complete candidate validation. |
| `outer_hard_xray_half_width` | `invalid_validation_status` | `validation_status='pending'`, `name_stage='drafted'`. | Needs-real-work: complete candidate validation. |
| `surface_temperature` | `invalid_validation_status` | `validation_status='pending'`, `name_stage='drafted'`. | Needs-real-work: complete candidate validation. |
| `toroidal_width_of_antenna_strap` | `invalid_validation_status` | `validation_status='pending'`, `name_stage='drafted'`. | Needs-real-work: complete candidate validation. |
| `inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width` | `invalid_validation_status` | `validation_status='quarantined'`, `name_stage='drafted'`. | Needs-real-work: resolve genuine quarantine. |
| `radial_outline_of_plasma_boundary` | `invalid_validation_status` | `validation_status='quarantined'`, `name_stage='accepted'`. | Needs-real-work: resolve genuine quarantine. |
| `radial_outline_of_wall` | `invalid_validation_status` | `validation_status='quarantined'`, `name_stage='accepted'`. | Needs-real-work: resolve genuine quarantine. |
| `vertical_outline_of_plasma_boundary` | `invalid_validation_status` | `validation_status='quarantined'`, `name_stage='drafted'`. | Needs-real-work: resolve genuine quarantine. |
| `accumulated_thermal_energy` | `name_not_accepted` | `name_stage='drafted'`; validation observed valid. | Needs-real-work: obtain a name review. |
| `flux_surface_averaged_parallel_current_density` | `name_not_accepted` | `name_stage='drafted'`; validation observed valid. | Needs-real-work: obtain a name review. |
| `gas_flow` | `name_not_accepted` | `name_stage='reviewed'`, score 0.68750. | Needs-real-work: semantic refinement after below-threshold review. |
| `hot_neutral_temperature` | `name_not_accepted` | `name_stage='reviewed'`, score 0.30000 and a name-review quorum shortfall. | Needs-real-work: re-review with a complete quorum and refine. |
| `line_integrated_opacity` | `name_not_accepted` | `name_stage='drafted'`; validation observed valid. | Needs-real-work: obtain a name review. |
| `plasma_pressure` | `name_not_accepted` | `name_stage='reviewed'`, score 0.71875. | Needs-real-work: semantic refinement after below-threshold review. |
| `total_energy_of_calorimetry_component` | `name_not_accepted` | `name_stage='drafted'`; validation observed valid. | Needs-real-work: obtain a name review. |
| `total_neutron_rate` | `name_not_accepted` | `name_stage='drafted'`; validation observed valid. | Needs-real-work: obtain a name review. |

## Measurement boundary

The run used the login node only because the Neo4j tunnel is login-node-local.
Every graph call was bounded to the 355 named manifest paths, its 227-name
terminal roster, or the 21 returned export exclusions. The temporary export
staging directory is not a release artifact. No graph identity changed and no
provider work occurred; total USD spend was **$0.00**.
