# WEST re-mint after the grammar fix and the beta restoration

## Authoritative result

The exact release path now emits **212 identities**. This was measured by
resolving all 355 paths in `west_production_dd_paths.yaml` with
`fetch_manifest_source_release_rows`, deduplicating the resulting 225 terminal
standard-name identities as the review batch, then calling `run_export` with
`force=True`, `skip_gate=True`, that review batch, and the manifest-source
projection. The run stopped after export: it did not freeze an artifact, create
a tag or branch, open a pull request, or apply a signed manifest, and changed
no identity.

| Measure | Standing | Mid-repair | After repairs | After grammar + beta |
| --- | ---: | ---: | ---: | ---: |
| Manifest paths | 355 | 355 | 355 | 355 |
| Review-batch roster | 227 | 227 | 227 | 225 |
| Export candidates | 226 | 226 | 226 | 225 |
| **Emitted identities** | **208** | **171** | **205** | **212** |
| Accounted identity exclusions | 18 | 55 | 21 | 13 |
| Accounting residue | 0 | 0 | 0 | 0 |

The fresh measure recovers **7** identities beyond the after-repairs 205 and
sits **4** above the original standing 208. Every in-export gate passed and
was not skipped: `catalog_status`, `identity_token_collision`,
`exclusion_accounting`, and `manifest_source_accounting` each passed with zero
issues, and `all_gates_passed=True` with `gate_failures=0`. `total_candidates`
equals the roster (225) this round because all 225 minted terminals are
catalogue-eligible; both prior rounds carried one non-candidate roster member.

Source accounting independently covered all 355 paths: 322 emitted source rows,
13 excluded source rows, and 20 documented non-nameable source rows, with 355
accounted and 0 excluded-by-source residue. The two documented-non-nameable
growth rows are treated below.

## The dispatched claims, adjudicated by the export's own verdict

| Claim | Export verdict this round | Prior state (after repairs) |
| --- | --- | --- |
| `radial_outline_of_plasma_boundary` admitted after the rule fix | **CONFIRMED — emitted.** `validation_status='valid'`, `name_stage='accepted'`, `docs_stage='accepted'`, issues `[]`, name score 0.875. | Excluded `invalid_validation_status` (quarantined). |
| `radial_outline_of_wall` admitted after the rule fix | **CONFIRMED — emitted.** `validation_status='valid'`, `name_stage='accepted'`, `docs_stage='accepted'`, issues `[]`, name score 0.86875. | Excluded `invalid_validation_status` (quarantined). |
| `normalized_toroidal_beta` restored to a live accepted identity with 8 producers | **CONFIRMED — emitted.** `name_stage='accepted'`, `docs_stage='accepted'`, `validation_status='valid'`, name score 1.0, **8 producers**, issues `[]`. | Superseded; restored to `draft`/accepted/accepted by the signed revival, and now releases. |

The third outline member, `vertical_outline_of_plasma_boundary`, is **not**
emitted: its false quarantine cleared to `valid`, but it is excluded with reason
`name_not_accepted` (`name_stage='drafted'`, `docs_stage='pending'`) and it still
carries the independent `audit:unit_dimension_check` metre-coordinate advisory.
This matches the outline-clearance prediction through the export's own verdict.

## What moved since the after-repairs round

Seven after-repairs exclusions are now emitted, six of them on a plain
lifecycle advance, the seventh carrying the restoration:

- `accumulated_total_gas_count`, `gap_at_closest_wall_point`,
  `hard_xray_emissivity`, `toroidal_width_of_antenna_strap` — were
  `invalid_validation_status` (pending); now `valid` +
  accepted/accepted, emitted.
- `radial_outline_of_plasma_boundary`, `radial_outline_of_wall` — were
  `invalid_validation_status` (quarantined); now emitted after the
  path-qualification rule fix.
- `normalized_toroidal_beta` — was not an after-repairs candidate (superseded);
  now emitted after the signed restoration.

Two after-repairs exclusions left the cohort entirely and now appear in the
source census as documented non-nameable with `cause not recorded`:

- `neutral_pressure` is now `name_stage='exhausted'`, `valid`, with **zero**
  producers anywhere in the graph. Its only plausible WEST producer path is
  `barometry/gauge/pressure`, which is now an unbound extracted source
  (`produced_sn_id=null`, no skip reason, no last error).
- `surface_temperature` is now `name_stage='exhausted'`, `valid`, with **zero**
  producers. Its only plausible WEST producer path is
  `camera_ir/channel/camera/frame/apparent_temperature`, which is likewise an
  unbound extracted source.

Both rows were simply pending validation in the after-repairs round. Their
source binding was detached without a durable non-nameable reason being left on
the source — the `cause not recorded` marker. This is the +2 that takes the
documented-non-nameable census from 18 to 20 and the roster from 227 to 225. The
detach cause is outside this node's scope and is reported as a follow-on.

The remaining 13 exclusions are all rows that were also excluded in the
after-repairs round; no previously-excluded row that is still a candidate flipped
from excluded to emitted besides those seven above, and no previously-emitted
row is newly excluded.

## Exclusion ledger and dispositions

The export reports 13 excluded identities, fully attributed by the release
ledger: 1 `documentation_not_accepted`, 2 `invalid_catalog_entry`, 1
`invalid_validation_status`, and 9 `name_not_accepted`. Each belongs to exactly
one disposition below; all 13 are `needs-real-work`.

| Export reason | Count | Correct-and-permanent | Needs-real-work | Metadata-only |
| --- | ---: | ---: | ---: | ---: |
| `documentation_not_accepted` | 1 | 0 | 1 | 0 |
| `invalid_catalog_entry` | 2 | 0 | 2 | 0 |
| `invalid_validation_status` | 1 | 0 | 1 | 0 |
| `name_not_accepted` | 9 | 0 | 9 | 0 |
| **Total** | **13** | **0** | **13** | **0** |

**The disposition split has not moved in kind** — the prior round was also
0 correct-and-permanent, 0 metadata-only, all-needs-real-work — but the specific
rows and their reasons have moved. Five excluded rows changed exclusion reason
since the after-repairs round (detailed row-by-row below): `total_neutron_rate`
and `flux_surface_averaged_parallel_current_density` advanced on the name axis
and are held by later gates, and `distance_of_antenna_strap`,
`outer_hard_xray_half_width` and `vertical_outline_of_plasma_boundary` cleared
validation and are now held only on `name_not_accepted`. No row changed into
correct-and-permanent or metadata-only, so no part of the split has flipped.

The two `invalid_catalog_entry` rows share one concrete defect, read from the
run log: the strict ISN catalog model rejects
`scalar.physics_domain` — `Value error, physics_domain must be a valid
PhysicsDomain enum value [input_value='unscoped']` — for both
`etendue_of_soft_xray_detector` and `total_neutron_rate`. Both are otherwise
fully accepted (`valid`, accepted/accepted, name scores 1.0 and 0.975). The
graph identity carries a stored `physics_domain='unscoped'` that must be set to
a valid enum before the entry can publish. Classified per the established
convention (the after-repairs round classified `invalid_catalog_entry` as
needs-real-work) as needs-real-work: it is a graph-field repair on the
identity, not a name-spelling or documentation-text matter.

| Identity | Export reason | Live state | Disposition and change since after-repairs |
| --- | --- | --- | --- |
| `etendue_of_soft_xray_detector` | `invalid_catalog_entry` | valid, accepted/accepted, score 1.0 | Needs-real-work: `physics_domain='unscoped'` rejected by the catalog model. Same row and reason as before. |
| `total_neutron_rate` | `invalid_catalog_entry` | valid, accepted/accepted, score 0.975 | Needs-real-work: same `physics_domain='unscoped'` defect. **Changed from `name_not_accepted`** — name axis advanced, now held at catalog-entry validation. |
| `flux_surface_averaged_parallel_current_density` | `documentation_not_accepted` | valid, name accepted, `docs_stage='reviewed'`, score 1.0 | Needs-real-work: complete docs-axis acceptance. **Changed from `name_not_accepted`** — name now accepted, held on docs. |
| `inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width` | `invalid_validation_status` | quarantined, drafted, `parse_error: grammar round-trip failed` | Needs-real-work: genuine quarantine (grammar parse error on the name), same as before. |
| `accumulated_thermal_energy` | `name_not_accepted` | valid, reviewed, score 0.63125 | Needs-real-work: name below the acceptance threshold. Same disposition as before. |
| `distance_of_antenna_strap` | `name_not_accepted` | valid, reviewed, score 0.6875 | Needs-real-work: advance the name axis past `reviewed`. **Changed from `invalid_validation_status`** — validation cleared, now held only on name. |
| `gas_flow` | `name_not_accepted` | valid, reviewed, score 0.6875, 5 producers | Needs-real-work: semantic refinement / name acceptance. Same as before. |
| `hot_neutral_temperature` | `name_not_accepted` | valid, reviewed, score 0.3 | Needs-real-work: name below threshold (quorum shortfall). Same as before. |
| `line_integrated_opacity` | `name_not_accepted` | valid, reviewed, score 0.84375 | Needs-real-work: advance the name axis past `reviewed`. Same as before. |
| `outer_hard_xray_half_width` | `name_not_accepted` | valid, reviewed, score 0.5375 | Needs-real-work: name below threshold. **Changed from `invalid_validation_status`** — validation cleared, now held only on name (unit advisory no longer blocks). |
| `plasma_pressure` | `name_not_accepted` | valid, reviewed, score 0.71875, 4 producers | Needs-real-work: name acceptance. Same as before. |
| `total_energy_of_calorimetry_component` | `name_not_accepted` | valid, reviewed, score 0.66875 | Needs-real-work: name acceptance. Same as before. |
| `vertical_outline_of_plasma_boundary` | `name_not_accepted` | valid, `name_stage='drafted'`, unit advisory | Needs-real-work: obtain name acceptance (docs also pending). **Changed from `invalid_validation_status`** — the false outline quarantine cleared; now held only on the drafted name stage. |

No excluded row's disposition is undetermined: every one is classified above
from its export reason plus a live, bounded id-scoped read. The one part of the
ledger that is not identity-attributable is the `cause not recorded` marker on
the two detached source paths, whose historical binding to
`neutral_pressure`/`surface_temperature` is inferred from path semantics rather
than proven by a retained record; a source-detach readiness record would settle
it.

## Delta versus the standing 208 candidate

8 identities are newly emitted relative to the standing candidate and 4 are no
longer emitted (net +4, matching 208 → 212).

Added: `accumulated_total_gas_count`, `gap_at_closest_wall_point`,
`hard_xray_emissivity`, `net_power_due_to_ion_cyclotron_heating`,
`normalized_toroidal_beta`, `radial_outline_of_plasma_boundary`,
`radial_outline_of_wall`, `toroidal_width_of_antenna_strap`.

Removed:
- `etendue_of_spectrometer_channel` — taken by the structural pass; the archived
  spelling is gone and its successor spelling `etendue_of_soft_xray_detector`
  is in the roster, still held on `invalid_catalog_entry`.
- `power_due_to_ion_cyclotron_heating` — superseded spelling, replaced by its
  live successor `net_power_due_to_ion_cyclotron_heating`.
- `gas_flow`, `plasma_pressure` — present in the standing cut, now excluded on
  `name_not_accepted`; they left the emitted set because their name axis is
  `reviewed`, not `accepted`.

## Measurement boundary

The run used the login node only because the Neo4j tunnel is login-node-local.
Every graph call was bounded to the 355 named manifest paths, the 225-name
terminal roster, the 13 returned exclusions, or the two probed source rows; the
slowest read was far below the ten-second ceiling. The temporary export staging
directory (`/tmp/remint-after-grammar`) is not a release artifact. No graph
identity changed, no freeze/tag/branch/PR was attempted, and no signed manifest
was applied. No provider work occurred; total USD spend was **$0.00**. The
corrected grammar is live in the pinned `imas-standard-names` v0.9.2 (editable
source head `d50f5d6`); this measurement exercised it through the authentic
mint + export code path.

## Reproduction

Scripts and raw outputs: `/tmp/remint_after_grammar.py` and
`/tmp/remint_after_grammar.json` (mint + export + delta; staging report at
`/tmp/remint-after-grammar/.export_report.json`),
`/tmp/probe_excluded_state.py` and `/tmp/excluded_state.json` (live state of the
13 excluded and 7 newly-emitted rows), `/tmp/probe_roster_holes.py` (cohort-hole
evidence for the two detached rows). Gate and validation-rejection detail is in
`/tmp/remint_after_grammar.log`.
