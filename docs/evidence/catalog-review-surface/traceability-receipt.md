# Catalog traceability receipt

## Result

An accepted catalog content edit now has a complete graph-side receipt. The
approval fold-back records the request and its author, and the successful
catalog publish records the export timestamp for every name in the published
staging manifest. A post-commit write is used for `exported_at`, so an entry
cannot claim that it was exported when the catalog commit did not complete.

The controlled starting census specified by the live plan found all four
provenance fields at zero across 4,666 `StandardName` rows. A fresh bounded
census on 2026-09-08 found 5,126 rows and the same zero values:

| Graph field | Populated before this change | Writer | Populated in controlled read-back |
| --- | ---: | --- | ---: |
| `catalog_approved_at` | 0 | `mark_catalog_name_approved` at `promote.py:1555` | yes |
| `catalog_pr_number` | 0 | `mark_catalog_name_approved` at `promote.py:1551` | yes |
| `catalog_merge_commit_sha` | 0 | `mark_catalog_name_approved` at `promote.py:1553` | yes |
| `exported_at` | 0 | `_record_export_receipt` at `publish.py:281-284`, after the catalog commit | yes |
| `catalog_reviewer_actor` (author) | 0 | `mark_catalog_name_approved` at `promote.py:1554` | yes |

## Code path

The approval path is `sn approve` → `run_approval` →
`mark_catalog_name_approved`. Its graph write matches an existing accepted
name, requires the complete request tuple, and sets the request number, URL,
merge SHA, author login, and approval timestamp in one update:

```text
promote.py:1543-1555
MATCH (sn:StandardName {id: $name})
WHERE sn.name_stage IN ['accepted', 'approved']
  AND sn.docs_stage = 'accepted'
  AND coalesce(sn.status, 'draft') = 'draft'
  AND coalesce(sn.validation_status, 'valid') <> 'quarantined'
SET sn.name_stage = 'approved',
    sn.status = 'active',
    sn.catalog_pr_number = $pr_number,
    sn.catalog_pr_url = $pr_url,
    sn.catalog_merge_commit_sha = $merge_commit,
    sn.catalog_reviewer_actor = $reviewer_actor,
    sn.catalog_approved_at = coalesce(sn.catalog_approved_at, datetime())
```

The actor is taken from the same merged pull-request evidence that supplies
the request number, URL, and merge SHA. It is not inferred from
`edit_origin`. The existing focused approval regression covers that
authoritative tuple.

The export path writes `exported_at` into the staging manifest in
`export.py:2209`. The new publication path copies that timestamp to the graph
only after the ISNC git commit has returned its SHA:

```text
publish.py:566-573
report.graph_receipt_count = _record_export_receipt(
    manifest,
    graph_client=graph_client,
)
```

`_record_export_receipt` at `publish.py:281-284` matches the manifest's named
identity set, sets `sn.exported_at = datetime($exported_at)`, and refuses a
partial match instead of reporting a receipt for only the rows it happened to
find. A dry run never invokes this writer.

## Controlled end-to-end read-back

The new regression test
`tests/standard_names/test_catalog_traceability_fields.py` first writes the
approval tuple, runs `run_publish` through a temporary catalog checkout, and
then reads all fields back from the graph double. It passed as part of a
focused 27-test gate:

```text
27 passed, 1 warning in 13.89s
```

The live graph proof used the accepted, unapproved identity
`vacuum_magnetic_vector_potential`, which was outside the WEST batch. A
synthetic controlled request was used so no real batch identity or request was
changed:

```text
request number:       999001
request URL:          [controlled request 999001](https://github.com/iterorganization/imas-standard-names-catalog/pull/999001)
request merge SHA:    traceability-receipt-test
author:               traceability-test
approval read-back:   pr=999001, actor=traceability-test, merge=traceability-receipt-test, exported=null
export receipt count: 1
publish read-back:    pr=999001, actor=traceability-test,
                      approved=2026-09-08T09:33:23Z,
                      merge=traceability-receipt-test,
                      exported=2026-09-08T10:00:00Z
```

The controlled identity was then demoted through `undo_approval`, and its
`exported_at` was cleared. The final live read-back was
`name_stage=accepted`, `status=draft`, with request, author, approval time,
merge SHA, and export time all null. The internal change record remains as
audit history, which is the intended behavior for an accepted human edit.

The live graph commands ran on the login node because the graph tunnel is
login-local. The census and read-back queries were bounded and each completed
under the ten-second query ceiling; the controlled write/read/restore run
completed in 0.164 seconds.
