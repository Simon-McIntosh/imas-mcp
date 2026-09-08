# Etendue parent restore

## Outcome

The surviving DD source was reset through the governed retry command and a
focused compose run created the requested parent identity. The graph now has
`etendue_of_spectrometer_channel`, at `name_stage=accepted` with a valid
three-review quorum and a live `PRODUCED_NAME` binding from
`dd:soft_x_rays/channel/etendue`.

This is a partial restore, not an acceptance-complete recovery. The source
compose initially proposed `etendue_of_soft_xray_detector`; the exact target
identity was then applied through the CLI rename surface. The resulting node
has `origin=null` and `status=null` rather than the required `origin=pipeline`
and `status=draft`, and its live topology is smaller than the archived
topology. The ratified child `spectral_etendue_of_spectrometer_channel` was not
restored by this node.

## Governed operations

| Operation | Result | Evidence |
|---|---:|---|
| `sn retry --failed --dry-run --reason "restore surviving DD source" soft_x_rays/channel/etendue` | eligible 1/1 | exact source was attempt-capped and unbound |
| `sn retry --failed --reason "restore surviving DD source" soft_x_rays/channel/etendue` | retried 1/1 | retry event written; source returned to `extracted` |
| `sn run --focus soft_x_rays/channel/etendue --skip-global-maintenance --dry-run -c 100 -t 10` | 1 focused path, no writes | focused preview passed |
| first bounded focused run | 600 s, 0 names, `$0.000000` | candidate was dropped as a vocabulary gap (`locus` is not a grammar class; `channel` is unregistered); the stale source claim was released with the token-verified recovery helper |
| exact-source `sn source-hint` | set and consumed | hint named the locked target identity and preserved the DD source |
| second bounded focused run | 1 name composed and accepted; interrupted during docs | generated `etendue_of_soft_xray_detector`, then name review accepted it; no docs completion was claimed |
| `sn run --rename "etendue_of_soft_xray_detector:etendue_of_spectrometer_channel" --include-accepted --dry-run` | 1 planned rename, 0 conflicts | exact target correction was previewed |
| same rename without `--dry-run` | applied | source binding remained on the target identity |

No hand-written Cypher created, accepted, or deleted an identity. The stale
generate and documentation claims left by interrupted runs were released only
through their existing token-verified recovery helpers.

## Live identity and source

The bounded live graph read returned:

| Field | Observed value |
|---|---|
| identity | `etendue_of_spectrometer_channel` |
| description | Optical etendue (AΩ) of a soft X-ray channel's optical system, product of detector area and accepted solid angle |
| `name_stage` | `accepted` |
| `docs_stage` | `pending` |
| `validation_status` | `valid` |
| `origin` | **null**; required `pipeline` was not written by the focused route |
| catalog `status` | **null**; required `draft` was not written because global maintenance was deliberately skipped |
| source path | `soft_x_rays/channel/etendue` |
| source ID | `dd:soft_x_rays/channel/etendue` |
| source status | `composed` |
| source `produced_sn_id` | `etendue_of_spectrometer_channel` |
| source DD version | `4.1.0` |
| source unit | `m^2.sr` |
| source claim | cleared after the interrupted docs phase |
| ratified child | `spectral_etendue_of_spectrometer_channel` remains absent |

The recomposed node carries the current pipeline description and decomposition
(`physical_base=etendue`, `object=soft_xray_detector`) while retaining the
locked target ID after the explicit rename. That semantic mismatch is why the
rename is recorded as a correction rather than presented as an untouched
model-generated result.

## Incident-edge comparison

The archive query was run against the cited full dump in an isolated Neo4j
instance on the `all_debug` partition. It counted every relationship incident
to the exact `StandardName` node, regardless of direction. The fresh per-type
query sums to 20 archived edges; the earlier coarse lineage note reported 19,
so this per-type result is the quantitative record used here.

| Relationship type | Archive count | Live count | Difference |
|---|---:|---:|---:|
| `DOCS_REVISION_OF` | 2 | 0 | -2 |
| `HAS_INTERNAL_CHANGE` | 4 | 1 | -3 |
| `HAS_LOCUS` | 1 | 1 | 0 |
| `HAS_PARENT` | 1 | 0 | -1 |
| `HAS_PHYSICS_DOMAIN` | 1 | 0 | -1 |
| `HAS_REVIEW` | 4 | 3 | -1 |
| `HAS_SEGMENT` | 1 | 2 | +1 |
| `HAS_STANDARD_NAME` | 1 | 1 | 0 |
| `HAS_STRUCTURAL_AUTHORITY` | 1 | 0 | -1 |
| `HAS_UNIT` | 1 | 1 | 0 |
| `IN_CLUSTER` | 1 | 1 | 0 |
| `PRODUCED_NAME` | 2 | 1 | -1 |
| **total** | **20** | **12** | **-8** |

The archive returned zero incident edges for these other StandardName roles,
which are therefore named explicitly rather than inferred from absence of a
row: `HAS_AGGREGATION`, `HAS_COCOS`, `HAS_COMPONENT`, `HAS_COORDINATE`,
`HAS_DEVICE`, `HAS_GEOMETRIC_BASE`, `HAS_ORBIT`, `HAS_POPULATION`,
`HAS_POSITION`, `HAS_PROCESS`, `HAS_REGION`, `HAS_SUBJECT`, `REFINED_FROM`,
and `ENTAILED_FROM_CHILD`.

The archive’s two `PRODUCED_NAME` edges are the direct DD source and
`derived:etendue_of_spectrometer_channel`. Its one `HAS_PARENT` edge is the
archived child `spectral_etendue_of_spectrometer_channel` with
`operator=spectral` and `operator_kind=qualifier`. The current direct DD
source is present, but the derived source and the ratified child are not.

## Spend

The removed identity’s pre-existing attribution was 19 `LLMCost` rows totaling
`$0.844849`, apportioned from the incident deletion history. This restore run
used `$0.20672225` of the authorized `$100.00` ceiling for the name review
quorum; name generation itself was zero-cost. The interrupted documentation
phase did not produce a completed docs result. The current node’s scalar
`llm_cost` is `$0.20672225`, but the run-level `LLMCost` query returned zero
persisted rows because interruption occurred during finalization; the scalar is
therefore the provider-spend observation, not a claim that the durable cost
ledger is complete.

## Evidence and blockers

The reset and focused pipeline path were exercised with exact source scope and
the authorized ceiling. The parent identity and direct producer now exist, but
the restore acceptance gate is not met:

1. The required `origin=pipeline` and `status=draft` postconditions are absent.
2. Live incident-edge counts do not equal the archive per type (12 versus 20).
3. The archived derived producer and ratified spectral child remain absent.
4. Documentation remains `pending`, and the interrupted run did not complete
   its docs phase.
5. The current composed properties identify `soft_xray_detector`, while the
   restored identity is the locked `spectrometer_channel` spelling; this needs
   an authority decision or a governed semantic correction before promotion.

The separate child-reconstruction and merged-head verification work remains
outside this node’s write scope.
