# Spectral child reconstruction attempt

## Outcome

**Blocked without a graph mutation.** The signed reconstruction adapter refused
the complete archive closure, so the live graph still has no
`spectral_etendue_of_soft_xray_detector` node and no replacement for the
archived `spectral_etendue_of_spectrometer_channel` node. This is the correct
fail-closed outcome: creating the child with only its parent relation would
silently discard its archived provenance and review history.

The live parent is present exactly once:

| identity | live `StandardName` count |
| --- | ---: |
| `etendue_of_soft_xray_detector` | 1 |
| `spectral_etendue_of_soft_xray_detector` | 0 |
| `spectral_etendue_of_spectrometer_channel` | 0 |

Consequently, the required live edge
`(spectral_etendue_of_soft_xray_detector)-[:HAS_PARENT]->(etendue_of_soft_xray_detector)`
does not exist yet. The reconstructed child would have used that corrected
parent identity; it would not have recreated the archived parent spelling.

## Corrected identity and retained ratification

The archive source is `spectral_etendue_of_spectrometer_channel`, but the
candidate identity is `spectral_etendue_of_soft_xray_detector`. The archived
spelling describes the former spectrometer-channel locus, while the current
catalog spelling identifies the soft-X-ray detector locus and matches the live
parent `etendue_of_soft_xray_detector`. The authority rekeys the child and its
`HAS_PARENT` target together, preserving the archived semantic body while not
reintroducing the superseded spelling.

The graph retains three `unchanged_ratification` records for the archived
identity, all with `from_name` and `to_name` equal to
`spectral_etendue_of_spectrometer_channel`:

- `sn-change:5fdaf740-87cf-49e9-8da9-cf7de1a60b22`
- `sn-change:adc11782-99ed-4b5f-b64c-59ffcefeb59b`
- `sn-change:d24dfc17-cb29-4c8a-b9c7-5569701cae60`

Those retained records establish that the archived child was ratified; they do
not authorize restoring the old catalog spelling in place of its corrected
identity.

## Archived closure versus live state

The archive supplied 89 incident edges across the following relationship
types. Every listed type is archive-present and live-absent because the
corrected child node is absent; no per-type count is claimed equal until the
closed reconstruction can apply.

| relationship type | archived count | live count | state |
| --- | ---: | ---: | --- |
| `DOCS_REVISION_OF` | 3 | 0 | archive-present, live-absent |
| `ENTAILED_FROM_CHILD` | 1 | 0 | archive-present, live-absent |
| `HAS_INTERNAL_CHANGE` | 6 | 0 | archive-present, live-absent |
| `HAS_LOCUS` | 1 | 0 | archive-present, live-absent |
| `HAS_PARENT` | 1 | 0 | archive-present, live-absent |
| `HAS_PHYSICS_DOMAIN` | 1 | 0 | archive-present, live-absent |
| `HAS_REVIEW` | 73 | 0 | archive-present, live-absent |
| `HAS_UNIT` | 1 | 0 | archive-present, live-absent |
| `PRODUCED_NAME` | 1 | 0 | archive-present, live-absent |
| `REFINED_FROM` | 1 | 0 | archive-present, live-absent |

The complete edge count is `3 + 1 + 6 + 1 + 1 + 1 + 73 + 1 + 1 + 1 = 89`.
The proposed corrected `HAS_PARENT` edge is among that closure, so it cannot be
applied separately without violating the archive-versus-live edge-count guard.

## Signed authority receipt

The dry run constructed a single-identity authority whose `id` was
`spectral_etendue_of_soft_xray_detector`, with the archived `HAS_PARENT` target
rewritten to `etendue_of_soft_xray_detector`.

| receipt field | value |
| --- | --- |
| authority schema | `imas-codex.archive-reconstruction.v1` |
| authority file SHA-256 | `4e3d947ba4a58c966fa89addad45dddc68f365f3b911955ee01c05133fc88936` |
| signed payload SHA-256 | `26b77d8b6e809483a74ded40635aaf99fb512b50ec0559c2f1e8bff011fce70f` |
| receipt schema | `imas-codex.signed-repair-receipt.v1` |
| receipt manifest SHA-256 | `fcde7f2b9ba9340557b6b08369234b45bb349ac49e43d9e0aa5cf3a1ef3931b3` |
| receipt outcome | `refused` |
| authority rows | 1 |
| admitted rows | 0 |
| refusal count | 77 |
| graph mutations | 0 |

The 77 refusals are exact missing-counterpart refusals: 73 `HAS_REVIEW` edges,
3 `DOCS_REVISION_OF` edges, and 1 `PRODUCED_NAME` edge. The remaining 12
archive edges have live counterparts, but admission is all-or-nothing, so the
adapter correctly rolls back the dry-run transaction rather than materializing
an incomplete Standard Name.

## Evidence artifacts

The bounded live-graph query and signed dry run are captured outside the source
tree for recovery and audit:

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260909T074045931481-n-pidp-restore-ratified-spectral-child/spectral-restore-inspection.json`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260909T074045931481-n-pidp-restore-ratified-spectral-child/spectral-reconstruction-authority.json`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260909T074045931481-n-pidp-restore-ratified-spectral-child/archive-spectral-child.json`

## Required follow-on

The reconstruction authority needs a typed way to include or reconstruct the
missing archived `StandardNameReview`, `DocsRevision`, and
`StandardNameSource` counterpart nodes in the same signed transaction. Once
that closure exists, rerun the signed preview, authorize its fresh manifest
digest, apply it atomically, and then confirm all ten per-type live counts equal
the archived counts above.
