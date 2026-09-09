# Etendue validation observation

## Outcome

`etendue_of_soft_xray_detector` now carries a real validation observation and
is admitted by a fresh pass through the export eligibility path. The
deterministic validation gate re-read the identity, returned `valid`, recorded
an empty issue list and stamped
`validated_at=2026-09-09T13:03:04.778000000Z`. No timestamp was written by
hand, no signed manifest was applied, and no other identity was passed to the
validator.

The exact-name pipeline run spent **`$0.000000` against its explicit `$5.00`
ceiling**. The validation gate is deterministic and made no LLM request, so the
total provider spend for this recovery is also `$0.000000`.

## Live state before validation

The first bounded read returned the withheld state verbatim:

```text
validation_status='valid'
validated_at=NULL
```

That pair asserts a verdict but records no time at which the verdict was
observed. The release exporter therefore classified this exact row as
`validation_observation_missing`; the preceding fresh WEST re-mint recorded
the detail `validation_status='valid' has no validated_at observation time`.
The export predicate and exclusion reason are implemented in
`imas_codex/standard_names/export.py:507-508` and
`imas_codex/standard_names/export.py:782-787`.

The rest of the live row was already publication-ready:

| Field | Before |
|---|---|
| identity | `etendue_of_soft_xray_detector` |
| description | Geometric optical throughput of a soft X-ray detector channel, set by the collecting area and accepted solid angle of its optical system. |
| `name_stage` | `accepted` |
| name review score | `1.0` |
| `docs_stage` | `accepted` |
| documentation review score | `0.9875` |
| catalog `status` | `draft` |
| `origin` | `pipeline` |
| producer | `dd:soft_x_rays/channel/etendue` |
| producer type and DD version | `dd`, `4.1.0` |

## Which writer omitted the observation

The authoritative validation completion path is
`mark_names_validated` in
`imas_codex/standard_names/graph_ops.py:8111-8150`. Its decisive statement is
at lines 8139-8142:

```text
SET sn.updated_at = datetime(), sn.validated_at = datetime(),
    sn.validation_issues = b.issues,
    sn.validation_layer_summary = b.summary,
    sn.validation_status = b.validation_status,
```

The verdict and observation time are one token-verified write. That validator
cannot produce `validation_status='valid'` with a null `validated_at`.

The omission instead comes from the pooled generation path. Its inline audit
sets `candidate["validation_status"]` at
`imas_codex/standard_names/workers.py:6330-6358`, and the graph writer persists
that value at `imas_codex/standard_names/graph_ops.py:5398-5400`. Neither
statement supplies `validated_at`. This is therefore the second defect class:
**a validation verdict was recorded without its observation stamp**, not a row
that merely waited in an unvalidated state.

The population measurement confirms that the defect is not unique. Before the
repair, **100** live, nonterminal `StandardName` rows had a non-null
`validation_status` and a null `validated_at`: this target plus **99 other
rows**. The other rows break down as follows:

| Other live rows by stored verdict | Count |
|---|---:|
| `pending` | 18 |
| `quarantined` | 38 |
| `valid` | 43 |
| **Total other rows** | **99** |

After this one-name validation, the same aggregate query still returns those
99 other rows with the same 18/38/43 breakdown. The one-row decrease is this
target leaving the unstamped population; it is also a non-interference check
that the scoped operation did not stamp another identity.

## Sanctioned recovery

The exact-name dry run was gap-only and admitted exactly one existing identity
without writing:

```text
imas-codex sn run --name etendue_of_soft_xray_detector \
  --skip-global-maintenance --dry-run -c 5 -t 2
Exact-name dry run: 1 existing name(s) eligible; no graph writes performed
```

The matching live pipeline invocation used the same one-name scope, no
`--reseed`, and no `--force`:

```text
imas-codex sn run --name etendue_of_soft_xray_detector \
  --skip-global-maintenance -c 5 -t 2
stop_reason=no_eligible_work
cost_spent=0.0
cost_limit=5.0
names_composed=0
names_enriched=0
names_reviewed=0
names_regenerated=0
```

That result is expected: the ordinary pools had no name, documentation, or
parent work left to do. It proved the requested exact-name pipeline scope but
did not invent an observation.

The row then passed through the existing ID-scoped validation path,
`drain_validation_for_ids`, which claims only named rows whose
`validated_at` is null, runs the shared deterministic admission gate, and
finishes through `mark_names_validated`:

```text
requested ids=['etendue_of_soft_xray_detector']
validated=1
quarantined=0
cleared_ids=['etendue_of_soft_xray_detector']
requarantined_ids=[]
```

This is an observed verdict: the validator checked the Pydantic, semantic,
description, structural, and canonical layers and recorded every layer as
passing. It is not a timestamp backfill.

## Read-back and fresh export result

The final exact-row read returned:

| Field | After |
|---|---|
| `validation_status` | `valid` |
| `validated_at` | `2026-09-09T13:03:04.778000000Z` |
| `validation_issues` | `[]` |
| Pydantic layer | passed; 0 errors |
| semantic layer | 0 issues; not skipped |
| description layer | 0 issues |
| structural layer | 0 issues; not skipped |
| canonical layer | 0 issues |
| `name_stage` / `docs_stage` | `accepted` / `accepted` |
| `status` / `origin` | `draft` / `pipeline` |
| DD producer | `dd:soft_x_rays/channel/etendue` |

A fresh call through the export population and eligibility functions returned
one target row, **one eligible identity, and zero exclusions**:

```text
target_rows=1
target_eligible=['etendue_of_soft_xray_detector']
target_excluded=[]
elapsed_s=0.662572
```

So a fresh WEST export would now admit
`etendue_of_soft_xray_detector`, subject to the unchanged batch membership
that already includes its DD producer path.

All live graph work ran on the login node because the authenticated Neo4j
endpoint is login-node-local. Reads were bounded to this exact identity, its
one producer, or the aggregate unstamped-verdict population; every reported
read completed below the ten-second ceiling. The fresh export eligibility read
took 0.663 seconds.
