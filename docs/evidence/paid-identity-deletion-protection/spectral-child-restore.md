# Spectral child reconstruction

## Outcome

**Applied atomically with zero refusals.** The signed reconstruction transaction
created `spectral_etendue_of_soft_xray_detector`, 77 archived counterpart nodes,
and all 89 incident archive edges. Its receipt reports 167 mutations: 1
Standard Name, 77 counterpart nodes, and 89 relationships.

The live parent is present exactly once:

| identity | live `StandardName` count |
| --- | ---: |
| `etendue_of_soft_xray_detector` | 1 |
| `spectral_etendue_of_soft_xray_detector` | 1 |
| `spectral_etendue_of_spectrometer_channel` | 0 |

The reconstructed child has `origin=pipeline` and `status=draft`. Its required
live edge
`(spectral_etendue_of_soft_xray_detector)-[:HAS_PARENT]->(etendue_of_soft_xray_detector)`
exists exactly once. The parent is also live with `origin=pipeline` and
`status=draft`.

## Corrected identity and retained ratification

The archive source is `spectral_etendue_of_spectrometer_channel`, but the
reconstructed identity is `spectral_etendue_of_soft_xray_detector`. The DD
definition at `soft_x_rays/channel/etendue` describes a detector channel and
never a spectrometer. Its sibling at `soft_x_rays/channel/power` is already
`power_of_soft_xray_detector`, and the hard- and soft-X-ray etendue
documentation strings are byte-identical, so the source supplies no semantic
distinction that would justify the spectrometer spelling. The authority
therefore rekeys the child and its `HAS_PARENT` target together, preserving the
archived semantic body while attaching it to the corrected live identity.

The graph retains three `unchanged_ratification` records for the archived
identity, all with `from_name` and `to_name` equal to
`spectral_etendue_of_spectrometer_channel`:

- `sn-change:5fdaf740-87cf-49e9-8da9-cf7de1a60b22`
- `sn-change:adc11782-99ed-4b5f-b64c-59ffcefeb59b`
- `sn-change:d24dfc17-cb29-4c8a-b9c7-5569701cae60`

Those three retained ratifications make the respelling a catalog-visible
correction. They establish the accepted semantic identity that the corrected
spelling now carries; recreating the archived spelling would conceal rather
than record that correction.

## Archived closure versus live state

Before reconstruction the child was absent, so every one of the following ten
archive relationship types was live-absent: `DOCS_REVISION_OF`,
`ENTAILED_FROM_CHILD`, `HAS_INTERNAL_CHANGE`, `HAS_LOCUS`, `HAS_PARENT`,
`HAS_PHYSICS_DOMAIN`, `HAS_REVIEW`, `HAS_UNIT`, `PRODUCED_NAME`, and
`REFINED_FROM`. After the atomic transaction, every live per-type count equals
its archive count.

| relationship type | archived count | live count | state |
| --- | ---: | ---: | --- |
| `DOCS_REVISION_OF` | 3 | 3 | equal |
| `ENTAILED_FROM_CHILD` | 1 | 1 | equal |
| `HAS_INTERNAL_CHANGE` | 6 | 6 | equal |
| `HAS_LOCUS` | 1 | 1 | equal |
| `HAS_PARENT` | 1 | 1 | equal |
| `HAS_PHYSICS_DOMAIN` | 1 | 1 | equal |
| `HAS_REVIEW` | 73 | 73 | equal |
| `HAS_UNIT` | 1 | 1 | equal |
| `PRODUCED_NAME` | 1 | 1 | equal |
| `REFINED_FROM` | 1 | 1 | equal |

The complete edge count is `3 + 1 + 6 + 1 + 1 + 1 + 73 + 1 + 1 + 1 = 89`.
The actual live relationship census contains exactly these ten types and no
extra type. Equality is therefore established per type, not inferred from the
absence of an exception or from the aggregate total alone.

## Signed authority receipt

The authority reconstructs the single identity
`spectral_etendue_of_soft_xray_detector`, with the archived `HAS_PARENT` target
rewritten to `etendue_of_soft_xray_detector`. The first adapter revision refused
77 missing counterparts. After the signed authority included those 73
`StandardNameReview`, 3 `DocsRevision`, and 1 `StandardNameSource` nodes, the
fresh preview refusal count fell from 77 to zero. Only that fresh digest was
authorized for the atomic apply.

| receipt field | value |
| --- | --- |
| authority schema | `imas-codex.archive-reconstruction.v1` |
| authority file SHA-256 | `9da8f79ad1b55ee774b2dfafbdd4b63055a9c4fd6aefe13fbeca639a9bc94d56` |
| signed payload SHA-256 | `bb01e1c38be2a7e225dcd096f4e5dee1d8e5ddad7f40acaa374a2cdab3314fd0` |
| receipt schema | `imas-codex.signed-repair-receipt.v1` |
| authorized manifest SHA-256 | `26b82b5e0fd4de38ab80db68fa8c87b8442e415c05cdd5464be9624670e22f23` |
| receipt outcome | `applied` |
| authority rows | 1 |
| counterpart rows | 77 |
| admitted rows | 1 |
| refusal count | 0 |
| graph mutations | 167 |

The manifest digest binds the pre-apply node and counterpart state. The apply
re-read that closure inside its transaction and committed only because its
fresh digest still matched the authorized value and all per-type counts matched
before commit.

## Evidence artifacts

The archive extraction, signed preview, applied receipt, and independent
post-apply live verification are captured outside the source tree:

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260909T074045931481-n-pidp-restore-ratified-spectral-child/spectral-restore-inspection.json`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260909T074045931481-n-pidp-restore-ratified-spectral-child/spectral-reconstruction-authority.json`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260909T074045931481-n-pidp-restore-ratified-spectral-child/archive-spectral-child.json`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260909T074045931481-n-pidp-restore-ratified-spectral-child/spectral-restore-apply.json`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260909T074045931481-n-pidp-restore-ratified-spectral-child/spectral-restore-verification.json`

## Execution provenance

The apply and the independent verification imported `signed_manifest.py` from
this worker's merged worktree at commit
`75800a95f9c3960ed50524bf338b9324203edfda`. That `HEAD` contains the typed
counterpart-node registry for `StandardNameReview`, `DocsRevision`, and
`StandardNameSource`; the transaction did not execute code from another
checkout.
