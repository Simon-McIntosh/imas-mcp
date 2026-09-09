# Normalized toroidal beta has no live identity

## Verdict

The live graph contains 21 `StandardName` identities whose `id` contains
`beta`. Within that population, the six spellings that have represented
normalized toroidal beta are all `status='superseded'`, all six have
`superseded_by=NULL`, and therefore **zero live catalog identities carry the
quantity**.

The evidence supports one canonical spelling for the total-pressure quantity:
**`normalized_toroidal_beta`**. It is the wording used by the Data Dictionary,
it aligns with the live unnormalized identity `toroidal_beta`, and it is the
only normalized-family row that still owns producer bindings: seven direct DD
sources plus `derived:beta`. The producer set is not homogeneous, however.
`summary/global_quantities/beta_tor_thermal_norm/value` is explicitly
thermal-pressure-only and must be separated from the total-pressure identity
before that identity is made live. The recommended destination for that source
is the existing semantically exact row
`normalized_toroidal_thermal_plasma_beta`, subject to its own lifecycle review.

This is a recommendation only. No name, lifecycle property, source binding, or
lineage edge was changed, and no pipeline, edit command, or signed manifest
apply was run.

## Measurement method

The reads ran on the login node because the Neo4j tunnel is login-node-local.
The discovery read was bounded to `StandardName.id CONTAINS 'beta'` with
`LIMIT 100`; every follow-up read used the returned IDs or the exact normalized
family IDs and limits of 10–200. The slowest live query took 0.168 seconds,
well below the ten-second ceiling. One change-history projection exposed a
large embedded fold receipt, so it was discarded and re-read using exact
change IDs and bounded reason prefixes; no conclusion below relies on truncated
output.

## Complete live `beta` census

`documentation length` is the graph's `size(coalesce(documentation, ''))`.
`—` means null or no `StandardNameSource-[:PRODUCED_NAME]->StandardName`
binding. Producer states are included because a retained edge and an active
source are different facts.

| StandardName id | status | name_stage | docs_stage | documentation length | superseded_by | producer bindings |
| --- | --- | --- | --- | ---: | --- | --- |
| `beta` | draft | accepted | accepted | 1469 | — | — |
| `beta_angle_of_ferritic_element` | superseded | superseded | pending | 0 | — | — |
| `beta_angle_of_passive_loop` | superseded | superseded | pending | 0 | — | — |
| `electron_beta_at_pedestal_top` | superseded | superseded | pending | 0 | — | — |
| `electron_flux_surface_average_beta_at_pedestal_top` | superseded | superseded | pending | 0 | — | — |
| `mhd_beta` | superseded | superseded | pending | 0 | — | — |
| `normalized_beta` | superseded | superseded | pending | 0 | — | — |
| `normalized_thermal_beta` | superseded | superseded | pending | 0 | — | — |
| `normalized_toroidal_beta` | superseded | reviewed | pending | 0 | — | `dd:core_profiles/global_quantities/beta_tor_norm` (attached)<br>`dd:equilibrium/time_slice/global_quantities/beta_normal` (attached)<br>`dd:equilibrium/time_slice/global_quantities/beta_tor_norm` (composed)<br>`dd:plasma_profiles/global_quantities/beta_tor_norm` (attached)<br>`dd:summary/global_quantities/beta_tor_norm/value` (attached)<br>`dd:summary/global_quantities/beta_tor_norm_mhd/value` (attached)<br>`dd:summary/global_quantities/beta_tor_thermal_norm/value` (attached)<br>`derived:beta` (composed) |
| `normalized_toroidal_plasma_beta` | superseded | superseded | accepted | 1504 | — | — |
| `normalized_toroidal_thermal_plasma_beta` | superseded | superseded | pending | 0 | — | — |
| `poloidal_beta` | draft | accepted | accepted | 1306 | — | `dd:core_profiles/global_quantities/beta_pol` (attached)<br>`dd:equilibrium/time_slice/global_quantities/beta_pol` (attached)<br>`dd:equilibrium/time_slice/profiles_1d/beta_pol` (composed)<br>`dd:plasma_profiles/global_quantities/beta_pol` (attached)<br>`dd:summary/global_quantities/beta_pol/value` (attached)<br>`dd:summary/global_quantities/beta_pol_mhd/value` (attached) |
| `poloidal_electron_beta_at_pedestal_top` | superseded | superseded | pending | 0 | — | — |
| `poloidal_electron_beta_at_pedestal_top_high_field_side` | draft | accepted | accepted | 1676 | — | `dd:summary/pedestal_fits/linear/beta_pol_pedestal_top_electron_hfs/value` (composed)<br>`dd:summary/pedestal_fits/mtanh/beta_pol_pedestal_top_electron_hfs/value` (composed) |
| `poloidal_electron_beta_at_pedestal_top_low_field_side` | draft | accepted | accepted | 1652 | — | `dd:summary/pedestal_fits/linear/beta_pol_pedestal_top_electron_lfs/value` (composed) |
| `poloidal_electron_beta_flux_surface_averaged_at_pedestal_top` | draft | reviewed | accepted | 1982 | — | `dd:summary/pedestal_fits/linear/beta_pol_pedestal_top_electron_average/value` (attached)<br>`dd:summary/pedestal_fits/mtanh/beta_pol_pedestal_top_electron_average/value` (composed)<br>`dd:summary/pedestal_fits/mtanh/beta_pol_pedestal_top_electron_lfs/value` (composed) |
| `poloidal_flux_surface_averaged_electron_beta_at_pedestal_top` | superseded | superseded | accepted | 1982 | — | — |
| `reference_beta` | draft | exhausted | pending | 0 | — | `dd:gyrokinetics/species_all/beta_reference` (stale) |
| `toroidal_beta` | draft | accepted | accepted | 1449 | — | `dd:core_profiles/global_quantities/beta_tor` (attached)<br>`dd:equilibrium/time_slice/global_quantities/beta_tor` (attached)<br>`dd:plasma_profiles/global_quantities/beta_tor` (attached)<br>`dd:summary/global_quantities/beta_tor/value` (attached)<br>`dd:summary/global_quantities/beta_tor_mhd/value` (composed) |
| `toroidal_normalized_plasma_beta` | superseded | superseded | pending | 0 | — | — |
| `toroidal_plasma_beta` | superseded | superseded | pending | 0 | — | — |

The measured position in the dispatch is therefore confirmed exactly for the
six normalized spellings: six superseded, zero live, and zero non-null
`superseded_by` values. Its producer count is also exact: eight bindings on
`normalized_toroidal_beta`, partitioned as seven DD and one derived, with no
producer on any of the other five.

The row also shows the lifecycle inconsistency that makes a simple source
census misleading: `normalized_toroidal_beta` is still at pipeline
`name_stage='reviewed'` and owns live producers while its catalog status says
superseded. Conversely, `normalized_toroidal_plasma_beta` has the accepted
document but no producer.

## Refinement lineage and the real cycle

The current direct lineage is:

```text
normalized_toroidal_beta REFINED_FROM
  ├─ toroidal_normalized_plasma_beta
  ├─ normalized_toroidal_thermal_plasma_beta
  ├─ normalized_beta
  └─ normalized_toroidal_plasma_beta

normalized_toroidal_plasma_beta REFINED_FROM
  ├─ normalized_toroidal_beta
  ├─ beta
  └─ normalized_beta
```

Thus `normalized_toroidal_plasma_beta` is simultaneously a predecessor and a
successor of `normalized_toroidal_beta`. The two opposite `REFINED_FROM` edges
exist in Neo4j; this is a real two-node cycle, not a scalar-display artifact.

The change ledger explains how it arose:

1. At `2026-07-23T18:15:15.743932Z`, change
   `sn-change:a6735f8f-d7e0-4067-9df9-0d110e8f19cb` refined
   `normalized_toroidal_beta` into `normalized_toroidal_plasma_beta`. That
   operation created the edge from the latter to the former.
2. At `2026-09-06T11:04:55.554226Z`, change
   `sn-change:9f1ff58e-e126-4679-a54f-a95f7625f7e4` folded
   `normalized_toroidal_plasma_beta` back into `normalized_toroidal_beta` under
   run `sn-fold:f6594df0-b948-4c1d-b310-0b3a1539f282`. The receipt calls the
   mechanism “fold parser-invalid duplicate into accepted authoritative
   identity.” Its `expected_after.lineage` explicitly contains both directions:
   the fold added the old identity as a predecessor of the target while
   retaining the older edge that already made the target a predecessor of the
   old identity.

The cycle was therefore minted by the fold's lineage carry-forward rule. The
fold reversed the identity choice without pruning the now-obsolete inverse
edge. The current grammar accepts both spellings; “parser-invalid” is the
historical fold receipt's stated rationale, not a current parser result.

## Surviving accepted documentation

`normalized_toroidal_plasma_beta` carries 1,504 characters of documentation,
`docs_stage='accepted'`, `reviewer_score_docs=0.93125` (0.93 rounded), and
seven attached docs-axis review rows. `normalized_toroidal_beta` carries zero
documentation characters. The surviving text is:

> Normalized toroidal plasma beta, conventionally denoted $\beta_N$, is the
> ratio of toroidal plasma beta times plasma minor radius and reference
> toroidal magnetic field to total plasma current, including the conventional
> factor 100. It is a whole-plasma equilibrium beta metric based on the
> volume-averaged total perpendicular plasma pressure entering toroidal beta.
>
> The defining relation is:
>
> $$
> \beta_N = 100\,\beta_{\mathrm{tor}}\,\frac{a B_0}{I_p}
> $$
>
> where $\beta_{\mathrm{tor}}$ is
> [beta](name:normalized_toroidal_plasma_beta), the unnormalized toroidal
> plasma beta; $a$ is the plasma minor radius; $B_0$ is the reference toroidal
> magnetic field; and $I_p$ is the total plasma current. The factor $100$ and
> the prescribed numerical normalization of $a$, $B_0$, and $I_p$ are part of
> the definition.
>
> The scope is the whole plasma. The pressure population is the total
> perpendicular plasma pressure represented by $\beta_{\mathrm{tor}}$, with
> the volume averaging inherited from
> [beta](name:normalized_toroidal_plasma_beta); it is not restricted to thermal
> pressure.
>
> This quantity is distinct from
> [beta](name:normalized_toroidal_plasma_beta), which is the unnormalized
> toroidal pressure ratio, and from a thermal-pressure-only normalized toroidal
> beta. Within the global beta-metric family, $\beta_N$ rescales
> [beta](name:normalized_toroidal_plasma_beta) by plasma size, reference-field,
> and current factors, whereas [poloidal_beta](name:poloidal_beta) uses a
> poloidal-field/current normalization.

The document has one defect class, repeated four times: each sentence is
describing the **unnormalized** toroidal beta, but its `beta` link targets the
document's own identity, `name:normalized_toroidal_plasma_beta`. All four are
self-links. Their target should be `name:toroidal_beta`. The
`name:poloidal_beta` link is already correct.

The score and seven reviews establish that this is the strongest surviving
prose, but correcting four link targets changes the reviewed bytes. The review
rows should remain historical evidence; they must not be silently reattached as
approval of the corrected document.

## Do the DD paths denote one quantity?

The DD descriptions distinguish pressure population from evaluation
provenance. `beta_tor` is defined from the volume-averaged **total perpendicular
pressure**, whereas the thermal branch explicitly says “thermal pressure only.”
The resulting disposition is:

| Producer path | DD meaning | Fast-ion / method distinction | Recommended identity |
| --- | --- | --- | --- |
| `core_profiles/global_quantities/beta_tor_norm` | `100 × beta_tor × a × B0 / Ip` | Total perpendicular pressure; includes non-thermal/fast-ion pressure when it contributes to `beta_tor`. | `normalized_toroidal_beta` |
| `equilibrium/time_slice/global_quantities/beta_normal` | Same normalized toroidal-beta formula | Alias spelling in the DD; no thermal-only restriction. | `normalized_toroidal_beta` |
| `equilibrium/time_slice/global_quantities/beta_tor_norm` | Same normalized toroidal-beta formula | Total normalized beta represented on the equilibrium time slice. | `normalized_toroidal_beta` |
| `plasma_profiles/global_quantities/beta_tor_norm` | Same normalized toroidal-beta formula | Total normalized beta represented by plasma profiles. | `normalized_toroidal_beta` |
| `summary/global_quantities/beta_tor_norm/value` | Summary value of the same formula | Total normalized beta; no thermal-only restriction. | `normalized_toroidal_beta` |
| `summary/global_quantities/beta_tor_norm_mhd/value` | Normalized toroidal beta using pressure determined by an equilibrium reconstruction code | The MHD suffix changes the evaluation route, not the physical target. It does not declare a thermal-only population. Under the repository's provenance-collapse rule, reconstructed and profile-derived estimates of one physical quantity share one Standard Name. | `normalized_toroidal_beta` |
| `summary/global_quantities/beta_tor_thermal_norm/value` | `100 × beta_tor_thermal × a × B0 / Ip` | Thermal pressure only; excludes fast-ion/suprathermal pressure, so it can differ physically from total `beta_tor_norm`. | A separate thermal identity; recommend the existing `normalized_toroidal_thermal_plasma_beta` after independent review. |

The MHD row needs no separate Standard Name because “using pressure determined
by an equilibrium reconstruction code” is provenance of the estimate. If the
reconstruction includes a fast-ion contribution, that contribution is part of
its reconstructed total pressure; the DD suffix does not define a separate
population. By contrast, `thermal` changes the numerator itself, so collapsing
that row with total beta loses physics rather than merely provenance.

`derived:beta` is not a DD path. It is a retained derived producer on the same
canonical row and should stay with the total identity unless a separate source
authority audit disproves it; nothing in this census supports moving it to the
thermal branch.

## Single recommended migration path

After the lead approves the live spelling and resurrection, perform one scoped,
signed migration in this order:

1. **Lock `normalized_toroidal_beta` as the only total-βN identity.** Do not
   resurrect `normalized_toroidal_plasma_beta`, `normalized_beta`, or
   `toroidal_normalized_plasma_beta` as competing spellings. The DD vocabulary,
   the existing producer ownership, and the live sibling `toroidal_beta` all
   select the shorter identity.
2. **Split the thermal quantity before making anything live.** Keep the five
   ordinary `beta_tor_norm`/`beta_normal` DD producers, the MHD-evaluated DD
   producer, and `derived:beta` on `normalized_toroidal_beta`. Move only
   `dd:summary/global_quantities/beta_tor_thermal_norm/value` to
   `normalized_toroidal_thermal_plasma_beta`, then revalidate and review that
   thermal identity independently. The total identity should finish with six
   DD producers plus one derived producer; the thermal identity should finish
   with one DD producer.
3. **Carry the accepted prose, not false approval.** Use the 1,504-character
   document as the canonical document input, replace all four self-link targets
   with `name:toroidal_beta`, and run the normal docs review on the corrected
   text. Preserve the seven old docs reviews as provenance of the predecessor;
   do not present them as review receipts for changed bytes.
4. **Break the lineage cycle in the same transaction.** Retain
   `normalized_toroidal_beta-[:REFINED_FROM]->normalized_toroidal_plasma_beta`
   as the fold lineage and remove the obsolete inverse
   `normalized_toroidal_plasma_beta-[:REFINED_FROM]->normalized_toroidal_beta`.
   Record the removal as a `StandardNameChange` tied to the signed migration.
   Preserve the other predecessor edges unless a separate lineage audit finds
   another contradiction.
5. **Re-enter lifecycle through review, not by declaring acceptance.** Clear the
   erroneous superseded catalog state only through the sanctioned migration,
   set the restored catalog identity to `draft`, and let the normal name and
   docs review paths establish their stages. Only upstream catalog approval may
   later move the catalog status to `active`.
6. **Leave the unused spellings as tombstones.** Their lineage should converge
   on the canonical identity without another cycle. Do not populate a new
   `superseded_by` cache while that scalar is itself scheduled for retirement;
   use the relationship lineage and the migration receipt as authority.
7. **Verify the result quantitatively before release.** Require exactly one
   live total normalized-toroidal-beta identity, six DD plus one derived
   producer on it, exactly one thermal DD producer on its separate identity,
   zero opposite-direction refinement cycles, zero bad self-link targets, and
   no producer remaining on a superseded competing spelling.

This order is material. Resurrecting first would make the current seven-DD
conflation live; copying review receipts before fixing the links would assert a
review that never occurred; and deleting lineage before recording the signed
migration would erase the explanation for why the accepted document and the
producers occupy different tombstones today.
