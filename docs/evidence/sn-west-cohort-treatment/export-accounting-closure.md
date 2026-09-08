# WEST review export accounting closure

## Result

The real WEST review export was run against the 214-name review batch from
`v0.4.0rc5+west-task-2e.sn_names.yaml`, using the login-node-local graph
tunnel. The post-change report is:

`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260908T070933065747-n-swct-the-export-accounts-for-every-exit/real-export-after/.export_report.json`

The closure is exact:

```text
candidate_count             214
published_count             196
accounted exclusions         18
candidate - published        18
accounting residue             0
```

The report's machine-readable counts are:

| Exclusion bucket | Count | Meaning |
| --- | ---: | --- |
| `invalid_validation_status` | 5 | `validation_status` is `quarantined`; the graph's validation authority refuses publication. |
| `missing_physics_domain` | 2 | `physics_domain` is null, so the entry cannot be assigned to a catalog domain. |
| `name_not_accepted` | 2 | `name_stage` is `reviewed`, not `accepted` or `approved`. |
| `never_reviewed` | 9 | No docs-axis review is reachable, even though the lifecycle projection says `docs_stage: accepted`. |
| **Total** | **18** | **214 - 196 = 18; residue 0.** |

The `exclusion_accounting` gate passed with no issues. The
`manifest_source_accounting` gate also passed: all 355 source rows were
accounted as 290 emitted, 52 excluded, and 13 documented non-nameable. The
source count is a separate source-to-name reconciliation; repeated source
bindings can point to one emitted identity, so it is not expected to equal the
196-name publication count.

## What changed in the exporter

The export population query already deliberately retrieves the complete
review population before domain, lifecycle, validation, quorum, and
documentation predicates are applied. The old report comment incorrectly said
that domain filtering happened upstream and that the domain counter was
therefore always zero. That statement was false for the accounting path: a
scoped export must retain a candidate long enough to record why it was not
published.

`_classify_export_population` now gives an empty or null domain its own
terminal reason, `missing_physics_domain`, before entry serialization. This
prevents a null graph field from being misreported later as a generic
`invalid_catalog_entry`. `ExportReport.record_exclusions` maintains the new
`excluded_missing_domain` counter and a complete `exclusion_by_reason` map;
`ExportReport.to_dict` emits both, together with `accounted_exclusions` and
`accounting_residue`. The identity-bearing `exclusion_ledger` remains the
authoritative list, so adding a new terminal reason cannot silently disappear
from the report merely because it was not added to a fixed counter list.

Before the change, the same real run closed numerically only because the two
null-domain identities were recorded as `invalid_catalog_entry` after the
ISN model rejected the synthetic `unscoped` domain. After the change, the
arithmetic is unchanged but the mechanism is explicit:

```text
before: invalid_catalog_entry       2
after:  missing_physics_domain      2
```

The other 16 identities retain their direct graph-state reasons. No counter
was widened or suppressed to make the gate pass.

## Identity-level census

The table records every one of the 18 identities that was handed to the
export and did not become a catalog entry. Scores are the graph's
`reviewer_score_name`; `—` means the graph field is null. Source paths are the
WEST manifest bindings used to construct the review batch.

### Quarantined validation status: 5

These are not score exclusions. Their terminal field is
`validation_status: quarantined`, and the exporter now reports that exact
field value. The first and fifth rows are also still in drafted/pending
state; the other three demonstrate that accepted lifecycle and high name
scores do not override the validation authority.

| Identity | Description | Source-path binding | Name score | State |
| --- | --- | --- | ---: | --- |
| `inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width` | Inward half-width of the hard X-ray emissivity peak in normalized toroidal flux coordinate. | `hard_x_rays/emissivity_profile_1d/half_width_internal` | — | `name_stage=drafted`, `docs_stage=pending`, `validation_status=quarantined` |
| `radial_outline_of_plasma_boundary` | Major-radius coordinate of each point on the plasma-boundary contour in a poloidal cross-section. | `equilibrium/time_slice/boundary/outline/r` | 0.875 | `name_stage=accepted`, `docs_stage=accepted`, `validation_status=quarantined` |
| `radial_outline_of_wall` | Major-radius coordinate of every point on a wall boundary outline, measured from the machine symmetry axis in the right-handed cylindrical (R, phi, Z) frame. | `wall/description_2d/mobile/unit/outline/r` | 0.86875 | `name_stage=accepted`, `docs_stage=accepted`, `validation_status=quarantined` |
| `vertical_coordinate_of_line_of_sight` | Signed vertical coordinate of a designated point defining a diagnostic line of sight in the right-handed cylindrical (R, phi, Z) frame. | 16 bindings: the `first_point/z`, `second_point/z`, and `third_point/z` line-of-sight paths under `bremsstrahlung_visible`, `camera_x_rays`, `hard_x_rays`, `interferometer`, `polarimeter`, and `spectrometer_visible` | 0.975 | `name_stage=accepted`, `docs_stage=accepted`, `validation_status=quarantined` |
| `vertical_outline_of_plasma_boundary` | Signed vertical outline of the plasma-boundary contour in the right-handed cylindrical (R, phi, Z) frame. | `equilibrium/time_slice/boundary/outline/z` | — | `name_stage=drafted`, `docs_stage=pending`, `validation_status=quarantined` |

The exporter can state the cause for all five because the graph field is
present and non-valid. It does not claim that the quarantine verdict is
current; that is the separate quarantine-instrument question. The export
accounting claim is narrower: these five were withheld because the stored
validation field was `quarantined`.

### Missing physics domain: 2

These are the two confirmed null-domain silent exits. Both are otherwise
strong candidates: accepted, valid, documented, and scored above the export
threshold. The field that would have carried the missing cause is
`physics_domain`; it is empty in both graph records, so no domain file can be
selected. The exporter now records that empty field directly rather than
inventing a domain or allowing the ISN model's later schema error to obscure
the mechanism.

| Identity | Description | Source-path binding | Name score | State |
| --- | --- | --- | ---: | --- |
| `total_electron_count` | Total inventory of free electrons within the plasma, including thermal and fast populations and excluding electrons bound in atoms or molecules. | `interferometer/electrons_n` | 0.96875 | `name_stage=accepted`, `docs_stage=accepted`, `validation_status=valid`, `physics_domain=null` |
| `vertical_coordinate_of_ece_channel` | Signed vertical coordinate of an electron-cyclotron-emission channel's measurement position in the right-handed cylindrical (R, phi, Z) frame. | `ece/channel/position/z` | 0.99375 | `name_stage=accepted`, `docs_stage=accepted`, `validation_status=valid`, `physics_domain=null` |

### Name stage not accepted: 2

These identities are valid graph records but have not reached an exportable
name stage. The terminal field is `name_stage`, not the score. One has a low
name score and the other has a score above the threshold, which is evidence
that the stage gate is doing distinct work from score filtering.

| Identity | Description | Source-path binding | Name score | State |
| --- | --- | --- | ---: | --- |
| `hot_neutral_temperature` | Translational kinetic temperature, expressed as energy per particle, of the energetic neutral-atom component in the plasma edge or scrape-off layer. | `spectrometer_visible/channel/isotope_ratios/isotope/hot_neutrals_temperature` | 0.3 | `name_stage=reviewed`, `docs_stage=accepted`, `validation_status=valid` |
| `normalized_toroidal_beta` | Normalized toroidal beta defined as 100 * beta_tor * a[m] * B0[T] / Ip[MA], a key stability metric. | `equilibrium/time_slice/global_quantities/beta_tor_norm`; `summary/global_quantities/beta_tor_norm_mhd/value` | 0.7875 | `name_stage=reviewed`, `docs_stage=pending`, `validation_status=valid` |

### No reachable docs-axis review: 9

These nine are accepted and valid, with name scores from 0.8875 to 1.0, but
the graph query found no reachable docs-axis review. The report therefore
does not guess a physics or score failure. The field that would have carried
the missing cause is the docs-review relation and its winning resolution
method; the relation is absent, so the exporter records
`never_reviewed: no docs-axis review is reachable`.

| Identity | Description | Source-path binding | Name score |
| --- | --- | --- | ---: |
| `derivative_of_area_of_flux_surface_with_respect_to_toroidal_flux_coordinate` | Rate of change of the poloidal cross-sectional area enclosed by a nested magnetic flux surface as the dimensionful toroidal-flux coordinate varies. | `equilibrium/time_slice/profiles_1d/darea_drho_tor` | 1.0 |
| `derivative_of_volume_of_flux_surface_with_respect_to_poloidal_magnetic_flux_coordinate` | Derivative of the enclosed volume of a nested magnetic flux surface with respect to its signed poloidal magnetic-flux label. | `equilibrium/time_slice/profiles_1d/dvolume_dpsi` | 1.0 |
| `derivative_of_volume_of_flux_surface_with_respect_to_toroidal_flux_coordinate` | Radial rate of change of the volume enclosed by a nested magnetic flux surface as its dimensionful toroidal flux coordinate increases. | `equilibrium/time_slice/profiles_1d/dvolume_drho_tor` | 1.0 |
| `flux_surface_averaged_toroidal_current_density` | Flux-surface-averaged toroidal current density assigned to a closed magnetic-flux surface; the geometry-weighted total toroidal current-density component that acts as the effective toroidal current source for the equilibrium poloidal magnetic-flux distribution. | `equilibrium/time_slice/profiles_1d/j_phi` | 1.0 |
| `length_variation_of_interferometer_beam` | Signed change in the effective optical path length accumulated along an interferometer beam, caused by plasma relative to the corresponding no-plasma reference path. | `interferometer/channel/path_length_variation` | 0.9875 |
| `normalized_toroidal_flux_coordinate_at_minimum_absolute_safety_factor` | Dimensionless toroidal-flux coordinate locating the nested plasma surface where the safety-factor magnitude is globally minimized. | `equilibrium/time_slice/global_quantities/q_min/rho_tor_norm` | 1.0 |
| `product_of_poloidal_current_function_and_derivative_of_poloidal_current_function_with_respect_to_poloidal_magnetic_flux_coordinate` | Grad-Shafranov equilibrium source term equal to the poloidal current function multiplied by its derivative with respect to poloidal magnetic flux. | `equilibrium/time_slice/profiles_1d/f_df_dpsi` | 1.0 |
| `voltage_amplitude_of_ion_cyclotron_heating_antenna` | Peak magnitude of the radio-frequency voltage signal on one transmission-line feed of an ion-cyclotron-heating antenna module, used in feed power and reflection analysis. | `ic_antennas/antenna/module/voltage/amplitude` | 0.98125 |
| `wave_current_amplitude_of_antenna_strap` | Peak magnitude of the radio-frequency current oscillating along an ion-cyclotron-heating antenna strap at a specified location. | `ic_antennas/antenna/module/current/amplitude` | 0.8875 |

## Verification and limits

The pre-change focused baseline was measured at base revision
`b72c99e8b727054eb3d058abcdf5e858ddf46fe0`:

```text
tests/standard_names/test_export_manifest_accounting.py
tests/standard_names/test_export_exclusion_ledger.py
tests/standard_names/test_export_report_gate.py
17 passed, 0 failed, 1 warning in 15.04s
```

The same focused command after the change returned:

```text
17 passed, 0 failed, 1 warning in 13.91s
```

An expanded export-focused selection also returned 41 passed, 0 failed. The
warning is the existing unknown `cache_dir` pytest configuration option; it is
not a test failure. The logs are retained at:

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260908T070933065747-n-swct-the-export-accounts-for-every-exit/baseline-focused-2.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260908T070933065747-n-swct-the-export-accounts-for-every-exit/after-focused-same.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260908T070933065747-n-swct-the-export-accounts-for-every-exit/real-export-after.log`

The real export was run with `skip_gate=True` to avoid launching the broad
pytest gate from the login node. The login-node exception is required because
the Neo4j tunnel is local to the login node. The accounting, catalog-status,
identity-token, and manifest-source gates all ran in the real export and
passed. The broad `tests/standard_names` suite was not claimed by this node;
its merged-head verification belongs to the separately dispatched test node.

The export also reported 210 internal documentation links pruned because their
targets were not in the 196-name published set, and the loaded grammar
checkout identity (`0.9.0`) differed from installed distribution metadata
(`0.8.1.dev32+g12b557363`). Neither fact changed the candidate accounting;
both remain visible in the retained export log.
