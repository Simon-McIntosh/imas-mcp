# Etendue parent restore

## Outcome

The surviving DD source was reset through the governed retry command and
recomposed into `etendue_of_soft_xray_detector`. That composer spelling is the
restored identity. The previously archived
`etendue_of_spectrometer_channel` spelling was applied temporarily, then
reversed through the same governed rename surface after the semantic ruling
identified it as a misnaming.

The final live identity is present at `name_stage=accepted`, with
`status=draft`, `origin=pipeline`, `validation_status=valid`, and
`docs_stage=accepted`. An independent post-rename query counted exactly
**one** `PRODUCED_NAME` edge, from
`dd:soft_x_rays/channel/etendue`; the source's `produced_sn_id` scalar also
equals `etendue_of_soft_xray_detector`. The archived spelling is absent.

The live identity has **16** incident relationships versus **20** on the
archived identity. This comparison is a recorded finding, not an equality
requirement for this node. Reconstructing the missing archived topology and
restoring the spectral child require the separately owned typed archive
adapter.

## Why the composer spelling is authoritative

Three independent inputs agree on the detector identity:

1. The exact DD source describes “Étendue (geometric extent, AΩ) of a soft
   X-ray diagnostic channel's optical system” and says it is the product of
   detector area and solid angle. It does not mention a spectrometer.
2. The live sibling `soft_x_rays/channel/power` is already represented by
   `power_of_soft_xray_detector`.
3. The live hard-X-ray counterpart is
   `etendue_of_hard_xray_detector`.

The generated identity also parses as
`physical_base=etendue`, `object=soft_xray_detector`. The archived
`spectrometer_channel` spelling therefore described an instrument class that
the DD path did not assert. The recomposition exposed the mismatch while the
review cut was closed but not approved, allowing the correction to remain at
catalog `status=draft`.

## Governed operations

| Operation | Result | Evidence |
|---|---:|---|
| `sn retry --failed --dry-run --reason "restore surviving DD source" soft_x_rays/channel/etendue` | eligible 1/1 | exact source was attempt-capped and unbound |
| same retry without `--dry-run` | retried 1/1 | retry event written; source returned to `extracted` |
| `sn run --focus soft_x_rays/channel/etendue --skip-global-maintenance --dry-run -c 100 -t 10` | 1 focused path, no writes | focused preview passed |
| first bounded focused run | 600 s, 0 names, `$0.000000` | candidate was dropped as a vocabulary gap; the stale source claim was released with the token-verified recovery helper |
| exact-source `sn source-hint` | set and consumed | preserved the exact DD source through the second composition attempt |
| second bounded focused run | 1 name composed and accepted; interrupted during docs | generated `etendue_of_soft_xray_detector`; three name reviews accepted it |
| temporary rename to the archived spelling | 1 rename, 0 conflicts | the source binding followed the target, but the later semantic ruling rejected the archived spelling |
| corrective rename dry-run | 1 planned rename, 0 conflicts | `etendue_of_spectrometer_channel` → `etendue_of_soft_xray_detector` |
| corrective rename apply | 1 applied rename | immediate independent query found the old identity absent and exactly one producer on the new identity |
| signed per-name `set_properties` manifest | 1 mutation, 1 receipt, 0 refusals | wrote `origin=pipeline` and `status=draft` in one transaction |
| `sn run --docs-only --name etendue_of_soft_xray_detector --skip-global-maintenance -c 99.79 -t 15` | clean drain; 1 docs generation and 1 two-review quorum | `docs_stage=accepted`, score 0.9875, `$0.098574` spent |

No operation passed `--reseed` or `--force`. No hand-written Cypher created,
accepted, renamed, or deleted an identity. Interrupted claims were released only
through their token-verified recovery helpers.

## Signed property receipt

The property correction followed the same generic signed
`apply_signed_manifest` boundary already exercised on the direct-DD origin
cohort. Its authority contains one exact `StandardName` participant and one
`set_properties` action carrying both scalar assignments.

| Receipt field | Value |
|---|---|
| authority file SHA-256 | `eac364a0e8c7935c6d59a7d2567bbd2beef046545a6ef85126691a13c0d5a420` |
| authority payload SHA-256 | `de353be14c0952db542fb65fc3015c0099c92545f3dd0e11091e1a7eb22343b9` |
| applied manifest SHA-256 | `b73bab7171eb46f5fd95d28c6c37bc74bf05465a529868fea014a635fb1631b9` |
| mutation count | 1 |
| receipt count | 1 |
| persistent writes | 2 |
| refusal count | 0 |
| receipt ID | `sn-change:signed-manifest:b73bab7171eb46f5fd95d28c6c37bc74bf05465a529868fea014a635fb1631b9:dc24269538e0823a7357c6b7` |

The post-apply query found exactly one
`HAS_INTERNAL_CHANGE` edge to that manifest and operation, and read back
`origin=pipeline` and `status=draft` from the same identity.

## Final live identity and source

| Field | Observed value |
|---|---|
| identity | `etendue_of_soft_xray_detector` |
| archived identity | absent |
| description | Geometric optical throughput of a soft X-ray detector channel, set by the collecting area and accepted solid angle of its optical system |
| `name_stage` | `accepted` |
| catalog `status` | `draft` |
| `origin` | `pipeline` |
| `docs_stage` | `accepted` |
| docs review score | 0.9875 |
| `validation_status` | `valid` |
| source ID | `dd:soft_x_rays/channel/etendue` |
| source path | `soft_x_rays/channel/etendue` |
| source status | `composed` |
| source `produced_sn_id` | `etendue_of_soft_xray_detector` |
| `PRODUCED_NAME` count | **1** |
| source DD version | `4.1.0` at extraction; recomposed identity records `4.1.1` |
| source unit | `m^2.sr` |
| source claim | clear |

## Incident-edge comparison

The archive query ran against the cited full dump in an isolated Neo4j instance
on the `all_debug` partition. It counted every relationship incident to the
archived `etendue_of_spectrometer_channel` node, regardless of direction.
The live query used the same incident-edge definition on the corrected
`etendue_of_soft_xray_detector` identity after its rename, signed property
receipt, and documentation completion.

| Relationship type | Archive count | Live count | Live − archive |
|---|---:|---:|---:|
| `DOCS_REVISION_OF` | 2 | 0 | -2 |
| `HAS_INTERNAL_CHANGE` | 4 | 3 | -1 |
| `HAS_LOCUS` | 1 | 1 | 0 |
| `HAS_PARENT` | 1 | 0 | -1 |
| `HAS_PHYSICAL_BASE` | 0 | 1 | +1 |
| `HAS_PHYSICS_DOMAIN` | 1 | 0 | -1 |
| `HAS_REVIEW` | 4 | 5 | +1 |
| `HAS_SEGMENT` | 1 | 2 | +1 |
| `HAS_STANDARD_NAME` | 1 | 1 | 0 |
| `HAS_STRUCTURAL_AUTHORITY` | 1 | 0 | -1 |
| `HAS_UNIT` | 1 | 1 | 0 |
| `IN_CLUSTER` | 1 | 1 | 0 |
| `PRODUCED_NAME` | 2 | 1 | -1 |
| **total** | **20** | **16** | **-4** |

`HAS_PHYSICAL_BASE` is the only live incident relationship type absent from
the archive, and is named explicitly in the table. The archive also had zero
incident edges for these other inspected StandardName relationship roles:
`HAS_AGGREGATION`, `HAS_COCOS`, `HAS_COMPONENT`, `HAS_COORDINATE`,
`HAS_DEVICE`, `HAS_GEOMETRIC_BASE`, `HAS_ORBIT`, `HAS_POPULATION`,
`HAS_POSITION`, `HAS_PROCESS`, `HAS_REGION`, `HAS_SUBJECT`,
`HAS_TRANSFORMATION`, `REFERENCES`, `MAGNITUDE_OF`, `HAS_ERROR`,
`HAS_PREDECESSOR`, `HAS_SUCCESSOR`, `HAS_DOCS_REVIEW_ADMISSION`,
`REFINED_FROM`, and `ENTAILED_FROM_CHILD`.

The archive's second `PRODUCED_NAME` edge was the missing
`derived:etendue_of_spectrometer_channel` source. Its `HAS_PARENT` edge
linked the archived spectral child. Reconstructing that derived producer, the
missing archived reviews, docs revisions, internal changes, physics-domain
edge, structural authority, and child edge is deliberately not attempted here;
the 16-versus-20 result is the input to the separate typed-adapter node.

## Spectral child follow-on

The child is absent from the live graph and is outside this node. Historical
change evidence contains **three** `unchanged_ratification` rows for
`spectral_etendue_of_spectrometer_channel`. Its archive reconstruction must
therefore use the corrected spelling
`spectral_etendue_of_soft_xray_detector`; because the old identity carries
ratification, that future rename is a catalog-visible correction, not an
internal spelling cleanup.

## Spend

The first successful composition and name-review pass spent
`$0.20672225`. The completed documentation-only pass spent `$0.098574`:
`$0.007833` on generation and `$0.090741` on review. Actual provider spend
for this node is therefore **`$0.305296` against the authorized
`$100.00` ceiling**.

The live identity's `llm_cost` scalar is `$0.29746315`, comprising name
review (`$0.20672225`) and docs review (`$0.09074090`). It omits the
generation charge even though the completed run reports it; the provider/run
total above is consequently the spend measure, while the scalar discrepancy is
preserved rather than hidden. Historical attribution to the removed archived
identity remains 19 `LLMCost` rows totaling `$0.844849`; it is separate from
this node's spend.

## Acceptance

This node's own gate is met quantitatively:

- corrected identity present: **1**;
- archived misnaming present: **0**;
- `name_stage=accepted`, `status=draft`, `origin=pipeline`,
  `docs_stage=accepted`: **all four true**;
- `PRODUCED_NAME` edges: **1**, exactly
  `dd:soft_x_rays/channel/etendue`;
- signed property receipts: **1**, with **0** refusals;
- live/archive incident edges: **16/20**, recorded per type without attempting
  equality;
- actual provider spend: **`$0.305296/$100.00`**.

Archive-topology reconstruction, the catalog-visible spectral-child correction,
and merged-head verification remain explicitly separate work.
