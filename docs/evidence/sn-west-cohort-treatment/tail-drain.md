# WEST actionable tail drain

## Outcome

The current WEST tail was re-read against the live graph and every actionable
row now ends in either a measured pipeline result or a named upstream gate.
The final census covers the two parked sources, the four quarantined batch
names, the rescore target, and the superseded batch identity that was present
in the earlier tail count.

The name-axis rescore of `hot_neutral_temperature` completed one review item at
zero LLM cost. It now carries a fresh review timestamp and
`reviewer_score_name=0.300`; the score is accompanied by the explicit
`semantic_similarity_gate` shortfall (`semantic_sim=0.5363245606422424`), so
the unresolved result is visible rather than being represented as a missing
review. Its `validation_status=valid`, `status=draft`, and
`docs_stage=accepted` remain intact.

All four current quarantined batch names have `validated_at` timestamps and
non-empty validation errors. None was released to `valid`, because each still
has a stated validation failure. The two genuine parked vocabulary sources
also have explicit token-level causes. The third source in the residual
inventory, `soft_x_rays/channel/etendue`, is no longer parked: it is composed
as `etendue_of_soft_xray_detector`.

## Final row census

### Name-axis tail

| Identity | Final state | Score or gate | Action/result |
|---|---|---|---|
| `hot_neutral_temperature` | `name_stage=reviewed`, `validation_status=valid`, `status=draft`, `docs_stage=accepted` | fresh `reviewer_score_name=0.300`; `review_resolution_method=semantic_similarity_gate`; `review_quorum_shortfall` is explicit; `semantic_sim=0.5363245606422424` | exact-name review drain processed 1 item; no model charge; held at the named semantic gate |
| `normalized_toroidal_beta` | `name_stage=reviewed`, `validation_status=valid`, `status=superseded`, `docs_stage=pending` | terminal catalog state | already superseded; no review or documentation action is admissible |

The rescore target's fresh review timestamp is
`2026-09-09T07:33:38.906000000+00:00`. It has nine attached name-review
records after the rescore, including the new non-winning semantic-gate result.

### Quarantined batch names

| Identity | Final validation state | Stated validation error | Disposition |
|---|---|---|---|
| `inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width` | `name_stage=drafted`, `validation_status=quarantined`, `validated_at=2026-09-08T07:48:57.586000000+00:00` | `parse_error: grammar round-trip failed for inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width` | held at the explicit grammar parse error; no documentation claim made |
| `radial_outline_of_plasma_boundary` | `name_stage=accepted`, `validation_status=quarantined`, `validated_at=2026-09-08T07:48:57.586000000+00:00` | semantic error: `'outline' must specify what entity's path/boundary is described` | held at the explicit semantic validation error |
| `radial_outline_of_wall` | `name_stage=accepted`, `validation_status=quarantined`, `validated_at=2026-09-08T07:48:57.586000000+00:00` | semantic error: `'outline' must specify what entity's path/boundary is described` | held at the explicit semantic validation error |
| `vertical_outline_of_plasma_boundary` | `name_stage=drafted`, `validation_status=quarantined`, `validated_at=2026-09-08T07:48:57.586000000+00:00` | semantic error: `'outline' must specify what entity's path/boundary is described`; unit audit also reports `unit='m'` without expected outline terms | held at both explicit validation errors |

These rows were not reset or silently cleared. The current validation
instrument is the authoritative result, and each failure is retained beside
the identity that caused it.

### Parked WEST source paths

| Source path | Final source state | Exact cause or produced identity | Disposition |
|---|---|---|---|
| `calorimetry/group/component/power` | `source_status=extracted`, no `PRODUCED_NAME` | missing device vocabulary token `component`; the graph records that no registered device/object token represents a generic calorimetry component; grammar-gap signature `0bba43219eec8edd` | upstream vocabulary gate; not retried into an invented carrier |
| `equilibrium/time_slice/profiles_1d/darea_dpsi` | `source_status=failed`, no `PRODUCED_NAME` | claim-attempt cap reached; the retained compose receipt identifies the missing physical-base token `rate_of_change_of_area_with_respect_to_poloidal_magnetic_flux`; grammar-gap signature `397fef806e377126` | upstream vocabulary gate; exact missing token recorded |
| `soft_x_rays/channel/etendue` | `source_status=composed` | `PRODUCED_NAME` → `etendue_of_soft_xray_detector`; `produced_sn_id` agrees | terminal composed result; not a residual |

The two vocabulary-held paths are therefore distinguishable from silent
untouched rows: both carry an exact missing-token explanation. The etendue
path has a composed identity and is included here only to close the residual
inventory without double-counting it.

## Governed operations and checks

| Operation | Result |
|---|---|
| General exact-name `review_name` drain before rescore | 0 items, `$0.00`; the existing quorum shortfall correctly excluded the reviewed row |
| `sn rescore hot_neutral_temperature -c 1` | entered the row at `drafted` but exited 1 during unrelated global derived-parent cleanup; no model call was charged |
| Exact-name `review_name` drain with `--skip-global-maintenance -c 1 -t 15` | exit 0; processed 1 item; `$0.00`; recorded the fresh `0.300` semantic-gate result |
| Final bounded graph census | exit 0; one target score, four explicit quarantine errors, two exact vocabulary gates, and one composed etendue source |

No `--reseed` or `--force` option was passed. The successful continuation
used exact identity scope and `--skip-global-maintenance`; it did not seed
other sources or change unrelated identities.

## Spend

The rescore and its successful review continuation consumed **$0.00 of the
$1.00 rescore command cap**. The broader WEST tail receipts record the
previously authorised $150.00 pipeline ceiling, with $0.00 for the composed
source drain and $1.004414 for the separate nine-name documentation rotation.
This node's own new LLM spend is therefore **$0.00**, and no additional model
budget was used because the semantic gate produced a non-winning result without
an LLM request.

## Out-of-scope refusal

The first `sn rescore` invocation reached shared global maintenance and was
refused by the protected derived-parent cleanup for four spend-bearing
identities:

`electron_power_density_due_to_collisions` ($0.254140),
`ion_charge_state_power` ($0.069196),
`ion_power_density` ($0.797604), and
`pressure_at_pedestal_top` ($0.074878).

That refusal is preserved as a follow-on. It is outside this node's
evidence-document scope and was neither triaged nor repaired here. The later
exact-name `--skip-global-maintenance` continuation completed the actionable
tail check without re-entering that cleanup.

## Acceptance measure

This node's evidence gate is met quantitatively:

- `hot_neutral_temperature`: one fresh name-axis score, `0.300`, with the
  semantic gate and observation timestamp recorded;
- quarantined batch names: **4/4** retain a non-empty stated validation error;
- parked source paths: **2/2** retain exact missing grammar-token causes;
- residual etendue source: composed as `etendue_of_soft_xray_detector`;
- new LLM spend: **$0.00/$1.00** rescore cap, with the broader authorised
  pipeline ceiling of **$150.00** also reported;
- no forbidden reset or force flag used.

Merged-result verification belongs to the separately dispatched test node.
