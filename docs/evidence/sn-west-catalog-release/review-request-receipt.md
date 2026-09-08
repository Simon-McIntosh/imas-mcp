# WEST catalog review-request receipt

## Review request

The governed WEST catalog review is open as [fork request 18](https://github.com/Simon-McIntosh/imas-standard-names-catalog/pull/18). It targets catalog `main` from `review/v0.4.0rc6+west-task-2e`. The branch and matching annotated tag both peel to catalog commit `4ba2c0ebce88fdf65941a2772a1088ed58822b2b` on the fork.

The organisation catalog received no branch, tag, or pull request. Its `main` remained at `a06e52052d4776b25e94fdfaa22c2bc6651a98eb`. The grammar repository has neither candidate ref. Fork request 16 remains open and unchanged on `review/v0.4.0rc5+west-task-2e`; this operation did not close, update, or reuse it.

Catalog continuous integration is complete: every required build, validation, and review-edit guard passed. The request changes only added files. The [rendered catalog preview](https://Simon-McIntosh.github.io/imas-standard-names-catalog/pr-18/), [reviewing guide](https://github.com/Simon-McIntosh/imas-standard-names-catalog/blob/main/REVIEWING.md), and committed [export accounting report](https://github.com/Simon-McIntosh/imas-standard-names-catalog/blob/review/v0.4.0rc6%2Bwest-task-2e/.export_report.json) each return HTTP 200.

## Catalog-status repair

The first inert preflight stopped at the `catalog_status` gate. Its bounded five-row query proved that every named blocker existed, was `name_stage=drafted` and `validation_status=valid`, and carried a null catalog status.

`reconcile_catalog_status()` was then run through its documented graph lifecycle path. It changed **77 identities**: 25 became `draft`, 52 became `superseded`, and none became `active`, `deprecated`, or `quarantined`. Approval therefore remained the sole route to `active`, and validation status was not changed.

| Identity | Description and WEST source binding | Before | After | Export disposition |
| --- | --- | --- | --- | --- |
| `accumulated_thermal_energy` | Cumulative coolant-extracted thermal energy since pulse start; `calorimetry/group/component/energy_cumulated` | null status; drafted; valid | draft; drafted; valid | withheld as `name_not_accepted` |
| `flux_surface_averaged_parallel_current_density` | Flux-surface average of parallel current density; `equilibrium/time_slice/profiles_1d/j_parallel` | null status; drafted; valid | draft; drafted; valid | withheld as `name_not_accepted` |
| `line_integrated_opacity` | ECE line-of-sight optical depth at the channel measurement position; `ece/channel/optical_depth` | null status; drafted; valid | draft; drafted; valid | withheld as `name_not_accepted` |
| `total_energy_of_calorimetry_component` | Total discharge energy extracted by a calorimetry component; `calorimetry/group/component/energy_total/data` | null status; drafted; valid | draft; drafted; valid | withheld as `name_not_accepted` |
| `total_neutron_rate` | Total neutron emission rate over all fusion reactions; `summary/fusion/neutron_rates/total/value` | null status; drafted; valid | draft; drafted; valid | withheld as `name_not_accepted` |

The five have no review scores because they remain drafted. The repair made their catalog lifecycle explicit; it did not force them through name review or publication.

## Export accounting

The corrected preflight and live cut both passed all eight export gates. The `.export_report.json` file is present in the catalog commit and carries the closed census:

```text
226 candidates - 208 published = 18 withheld
18 withheld = 12 invalid_validation_status + 6 name_not_accepted
accounting residue = 0

355 manifest sources = 317 emitted bindings + 20 excluded + 18 documented non-nameable
source-accounting residue = 0
```

The frozen review roster contains 227 names while the export candidate census contains 226. The single difference is `normalized_toroidal_beta`: the roster preserves it as batch history, while its graph catalog status is `superseded`, so the current exporter correctly does not admit it as a candidate. This explains the release commit's generated `227 - 208 = 19` wording without adding an unexplained export exclusion; governed exclusion accounting remains `226 - 208 = 18` with residue zero.

The frozen roster is `v0.4.0rc6+west-task-2e.sn_names.yaml`, 79,926 bytes with SHA-256 `18e7beb7b77882c5a72ba095aa49340ee8e69ff620269f659302ca05676db68c`. It is backfilled with request 18. The release command generated it outside this node's exclusive source write scope, so it remains an expected untracked artifact for the coordinator to preserve in a separately scoped commit.

## Grammar and published-byte checks

The catalog tooling commit `61c6cc8` pins `imas-standard-names==0.9.1` in both declared dependency locations. PyPI returned HTTP 200 for release `0.9.1`; the fork and upstream grammar remotes expose the same annotated `v0.9.1` tag object `09527b233c7783e1faa835f49d7cc03bb0166a3c`, peeled to commit `2a4584d82537f3339e9c87294048ca4feb658495`.

Direct validation recognizes `event_instant`, `signal_to_noise_ratio`, and `signal_to_noise_level` as physical bases. A bounded query across all 355 batch source paths found no current produced identity containing any of the three, so this WEST cut needs none of them. The catalog sidecar stamps both `grammar_version` and `isn_model_version` as `0.9.1`.

The tag contains 17 domain files and 208 entries. It has zero Unicode escapes, zero line continuations, and zero formatter drift. The raw-entry kind/domain check still documented in the release recipe reports all 208 missing because the current four-field catalog format deliberately moved those machine fields out of entry YAML; that obsolete check should be aligned with the sidecar-aware validator already exercised by the passing export and catalog CI.

## Test gate

The full `tests/standard_names/` suite ran on `all_debug` without an added quiet flag:

```text
== 7259 passed, 11 skipped, 323 deselected, 34 warnings in 291.00s (0:04:51) ===
```

Exit status was zero, so the cut added no test failure.
