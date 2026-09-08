# Disposition of the fifteen omitted deletions

## Result

The deletion pass at `2026-09-08T11:57Z` removed 93 distinct identities. The
existing census in
`docs/evidence/sn-lifecycle-integrity/unshielded-identity-deletions.md` contains
78 per-identity rows, all of which belong to that pass. Exact set subtraction
leaves the 15 identities below; there are no census identities outside the
pass.

The locked recorded-spend boundary produces a non-uniform disposition:

- **11 RESTORE**, carrying **$2.269733** of apportioned LLM spend across 56 cost
  rows;
- **4 CORRECTLY REMOVED**, each with zero recorded spend; and
- **0 identities without a verdict**.

Only three of the 15 identities exist in the 2026-09-06 archive. The other 12
have no node, source row, or child topology in that dump. This is retained as a
negative archive result rather than filling the gap from spelling or current
graph state. Eight of those 12 archive-absent identities nevertheless carry
recorded spend and are RESTORE under the boundary; the archive absence limits
the recovery route, not the preservation verdict.

## Method and evidence boundary

The pass cohort was read from `StandardNameChange` rows with
`operation='remove_derived_parent'` and `changed_at` in the exact
`2026-09-08T11:57:` minute. The bounded query returned 93 rows in 0.015 seconds.
The per-identity table in the existing census was parsed only between its
`Per-identity evidence` and `Restoration order` headings and contains 78 rows.
Set subtraction returned 15 rows; the reverse difference returned zero.

Archive evidence comes from
`~/.local/share/imas-codex/exports/imas-codex-graph-dev-002bf65-20260906T220012Z.tar.gz`.
The dump was loaded on the `all_debug` partition into a temporary Neo4j data
directory on distinct loopback ports; neither the live store path nor the live
database was used. The exact 15-identity lookup found three nodes, 11 source
lineage rows, and four DD-bound source rows. The temporary instance was stopped
and removed after the read.

Spend is the canonical per-identity apportionment:

```text
sum(LLMCost.llm_cost / size(LLMCost.standard_name_ids))
```

The one exact-cohort live query completed in 0.133 seconds. Amounts retain six
decimal places so a small positive charge cannot display as zero. Any positive
amount yields RESTORE; zero yields CORRECTLY REMOVED when the archive supplies
no independent source or catalog authority. RESTORE says the paid identity
content must be recovered; it does not itself authorize publication.

## Per-identity disposition

| identity | archive source lineage | archive child topology | cost rows | apportioned spend USD | verdict | basis |
|---|---|---|---:|---:|---|---|
| `angle_of_optical_element` | No archive node; therefore no archived source row. | No archive node; no child topology is recoverable from this dump. | 0 | 0.000000 | CORRECTLY REMOVED | Zero recorded spend and no archive source, child, or catalog authority. |
| `charge_at_pedestal_top` | Direct derived source `derived:charge_at_pedestal_top`; its child also has derived source `derived:effective_charge_at_pedestal_top`. No DD-bound source appears in this lineage. | One direct child: `effective_charge_at_pedestal_top` via qualifier `effective`. | 5 | 0.337981 | RESTORE | Positive recorded spend; archive topology is recoverable. |
| `diffusion_coefficient_due_to_diffusion` | No archive node; therefore no archived source row. | No archive node; no child topology is recoverable from this dump. | 0 | 0.000000 | CORRECTLY REMOVED | Zero recorded spend and no archive source, child, or catalog authority. |
| `etendue_of_spectrometer_channel` | Direct DD source `dd:soft_x_rays/channel/etendue` bound to `soft_x_rays/channel/etendue`, plus `derived:etendue_of_spectrometer_channel`; the spectral child has `derived:spectral_etendue_of_spectrometer_channel`. | One direct child: `spectral_etendue_of_spectrometer_channel` via qualifier `spectral`. | 19 | 0.844849 | RESTORE | Positive recorded spend and direct archived DD provenance. |
| `factor_of_spectrometer_channel` | No archive node; therefore no archived source row. | No archive node; no child topology is recoverable from this dump. | 0 | 0.000000 | CORRECTLY REMOVED | Zero recorded spend and no archive source, child, or catalog authority. |
| `flux_at_first_wall` | No archive node; therefore no archived source row. | No archive node; no child topology is recoverable from this dump. | 1 | 0.001137 | RESTORE | Positive recorded spend; another durable recovery source is required because this archive predates the identity. |
| `flux_at_wall` | Direct derived source `derived:flux_at_wall`. Descendant support includes `dd:wall/description_ggd/ggd/power_density/values` for `energy_flux_at_wall` and two emitted/incident `wall/description_ggd/ggd/particle_fluxes/ion/state/.../values` DD sources below `particle_flux_at_wall`. | Two direct children: `energy_flux_at_wall` via qualifier `energy` and `particle_flux_at_wall` via qualifier `particle`; the latter has descendant `ion_charge_state_particle_flux_at_wall`. | 7 | 0.278929 | RESTORE | Positive recorded spend; archive topology and descendant DD support are recoverable. |
| `flux_at_wall_due_to_eddy_current` | No archive node; therefore no archived source row. | No archive node; no child topology is recoverable from this dump. | 9 | 0.450517 | RESTORE | Positive recorded spend; another durable recovery source is required because this archive predates the identity. |
| `flux_at_wall_due_to_pumping` | No archive node; therefore no archived source row. | No archive node; no child topology is recoverable from this dump. | 2 | 0.008069 | RESTORE | Positive recorded spend; another durable recovery source is required because this archive predates the identity. |
| `flux_due_to_pumping` | No archive node; therefore no archived source row. | No archive node; no child topology is recoverable from this dump. | 2 | 0.002760 | RESTORE | Positive recorded spend; another durable recovery source is required because this archive predates the identity. |
| `ion_charge_state_power` | No archive node; therefore no archived source row. | No archive node; no child topology is recoverable from this dump. | 3 | 0.069196 | RESTORE | Positive recorded spend; another durable recovery source is required because this archive predates the identity. |
| `ion_density_at_pedestal_top` | No archive node; therefore no archived source row. | No archive node; no child topology is recoverable from this dump. | 2 | 0.108893 | RESTORE | Positive recorded spend; another durable recovery source is required because this archive predates the identity. |
| `permeability_of_ferritic_element` | No archive node; therefore no archived source row. | No archive node; no child topology is recoverable from this dump. | 3 | 0.078877 | RESTORE | Positive recorded spend; another durable recovery source is required because this archive predates the identity. |
| `plasma_heating_power` | No archive node; therefore no archived source row. | No archive node; no child topology is recoverable from this dump. | 0 | 0.000000 | CORRECTLY REMOVED | Zero recorded spend and no archive source, child, or catalog authority. |
| `width_of_spectrometer_channel` | No archive node; therefore no archived source row. | No archive node; no child topology is recoverable from this dump. | 3 | 0.088525 | RESTORE | Positive recorded spend; another durable recovery source is required because this archive predates the identity. |

## Reconciliation

| population | rows | RESTORE | CORRECTLY REMOVED | without verdict |
|---|---:|---:|---:|---:|
| Existing census | 78 | 67 | 11 | 0 |
| Omitted identities classified here | 15 | 11 | 4 | 0 |
| Full 11:57Z deletion pass | **93** | **78** | **15** | **0** |

The count closes in both directions: **78 + 15 = 93**, and the final verdict
partition is **78 + 15 = 93**. No identity removed in the pass remains silent.

## Recovery implication

The archive can directly seed recovery planning for the three archive-present
RESTORE identities. It cannot seed the eight archive-absent RESTORE identities:
`flux_at_first_wall`, `flux_at_wall_due_to_eddy_current`,
`flux_at_wall_due_to_pumping`, `flux_due_to_pumping`,
`ion_charge_state_power`, `ion_density_at_pedestal_top`,
`permeability_of_ferritic_element`, and `width_of_spectrometer_channel`.
Their positive spend fixes the verdict, but a restore must obtain identity
properties and relationship topology from another durable record or regenerate
them through a governed source path. Treating the 2026-09-06 dump as complete
recovery material for all 93 would silently omit those eight paid identities.
