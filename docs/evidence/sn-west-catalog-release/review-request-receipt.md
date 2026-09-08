# WEST review-request preflight receipt

## Result

No new catalog review request was opened. The governed batch export refused to
advance, so no branch, tag, or request may be cut from this preflight.

Fork request 16 remains untouched and open on
`review/v0.4.0rc5+west-task-2e`. It predates this preflight and is not a
substitute for the current batch.

## Target resolution

`imas-codex sn release status` resolved the catalog checkout to
`/home/ITER/mcintos/Code/imas-standard-names-catalog` in `rc` state. Its latest
catalog tag is `v0.4.0rc5+west-task-2e`; the fork remote is
`git@github.com:Simon-McIntosh/imas-standard-names-catalog.git` and the
upstream remote is `git@github.com:iterorganization/imas-standard-names-catalog.git`.
The status output offered the next command in that existing catalog series.

## Export gate

The release tool's inert batch preflight used the committed
`west_production_dd_paths` manifest and wrote the current
`/home/ITER/mcintos/.cache/imas-codex/staging/.export_report.json`. It did not
modify either repository worktree or create a catalog ref.

| Report field | Observed value |
| --- | ---: |
| Candidate population | 226 |
| Published entries | 0 |
| Failed gates | 1 |
| `catalog_status` issues | 5 |
| Accounted exclusions | 0 |
| Accounting residue | 226 |

The required arithmetic does not close: `226 - 0 = 226`, while the reported
per-reason exclusion sum is `0`, leaving residue `226`. This is an intentional
early refusal, not an empty release: `source_reconciliation.manifest_size` and
its related rows are zero only because the failed gate returned before source
accounting ran.

The failed `catalog_status` gate names these five graph rows, each carrying a
null catalog status:

- `accumulated_thermal_energy`
- `flux_surface_averaged_parallel_current_density`
- `line_integrated_opacity`
- `total_energy_of_calorimetry_component`
- `total_neutron_rate`

`--skip-gate` was not used. The release recipe requires this condition to be
repaired through its owning graph lifecycle path before a new preflight can
calculate a candidate identity, close its accounting, and create a fork review
request.

## Grammar availability

The catalog tooling commit `61c6cc8` pins `imas-standard-names==0.9.1` in both
declared dependency locations. PyPI answered HTTP 200 for the `0.9.1` package
metadata endpoint, and both grammar remotes expose the same annotated
`v0.9.1` tag object `09527b233c7783e1faa835f49d7cc03bb0166a3c` with peeled
commit `2a4584d82537f3339e9c87294048ca4feb658495`.

Direct parser validation accepts `event_instant`, `signal_to_noise_ratio`, and
`signal_to_noise_level` as physical bases. A bounded live query across all 355
source paths in the committed batch found no produced batch identity containing
any of those three tokens, so none is needed by this blocked export.

The active editable package reports metadata version
`0.8.1.dev32+g12b557363` despite loading the checked-out `v0.9.1` grammar code.
That mismatch did not prevent the three parser validations above, but it should
be reconciled before the next release attempt so runtime version reporting
agrees with the declared catalog tooling pin.

## Required next action

Repair the five null `catalog_status` rows through their owning lifecycle code,
then rerun the release preflight. Only a report with a passing gate and zero
residue may mint the next catalog branch and open its fork review request.
