# Direct-DD origin correction receipt

## Outcome

The signed property authority corrected **40** live StandardName identities from `origin='derived'` to `origin='pipeline'`. Each corrected identity had one or more direct `dd:` producers, which contradicts the structural-only meaning of `derived`.

The authority was previewed twice. The first apply rolled back when its collateral proof detected an unrelated concurrent graph change; the second preview admitted all 40 rows with no refusals and its fresh manifest committed successfully. The successful transaction made 40 property mutations and persisted 40 StandardNameChange receipts (80 persistent writes), without generating or spending on any LLM work.

- Authority file SHA-256: `e733141b0140da91dbbe6516965518d634ef3c057d746561da5a34e6d7a6e620`
- Authority payload SHA-256: `689ae0cace4856a46f444b64bf7ad14e30a21d323b0ad2daecbf241520ff192b`
- Applied manifest SHA-256: `536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5`
- Receipt operation: `correct_false_derived_origins`
- Direct DD producers across the corrected rows: **161**

## Per-identity signed receipts

Every table row was a signed `set_properties` action against one exact `StandardName` participant. The `before` value is the signed audit predicate, `derived`; the `after` value and receipt are read back from the committed graph.

| Standard name | Before | After | Direct DD producers | Signed receipt | Published cut |
|---|---|---|---:|---|---|
| `atomic_mass` | `derived` | `pipeline` | 55 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:ae8865ad344c0b458f5daaac` | published cut
| `breakdown_magnetic_field` | `derived` | `pipeline` | 1 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:134c6a7a307a987f31f7b035` | |
| `co_passing_thermal_electron_torque_density_due_to_collisions` | `derived` | `pipeline` | 1 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:9e3d6856c83bccf211074434` | |
| `current_density_due_to_ohmic_current_drive` | `derived` | `pipeline` | 1 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:3b78e280c9db4c4a62bae4e3` | |
| `diamagnetic_momentum_flux` | `derived` | `pipeline` | 1 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:6159157b52890b58ee0af060` | |
| `diamagnetic_momentum_flux_limiter_coefficient` | `derived` | `pipeline` | 2 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:38a1df7ab352e90780f19114` | |
| `effective_electron_diffusivity` | `derived` | `pipeline` | 1 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:41f5f9e178fca2331c4aaa61` | |
| `effective_ion_diffusivity` | `derived` | `pipeline` | 1 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:d96be9ba8c36dbacc2f8cc8e` | |
| `effective_neutral_diffusivity` | `derived` | `pipeline` | 1 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:d5803547bcc0d9b48a0da810` | |
| `electron_absorbed_wave_power` | `derived` | `pipeline` | 1 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:6f818ad62d492cd1a0fc24d2` | |
| `electron_energy` | `derived` | `pipeline` | 1 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:b85675a6c71ed40be2c58a2c` | |
| `electron_energy_diffusion_coefficient` | `derived` | `pipeline` | 1 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:0571a32d244ae64a28c6c800` | |
| `electron_pressure_at_pedestal_top` | `derived` | `pipeline` | 2 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:8421fe0987ff00771872da6c` | |
| `fast_ion_charge_state_power_density_due_to_collisions` | `derived` | `pipeline` | 2 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:1479d65de2b791b904649500` | |
| `gas_flow` | `derived` | `pipeline` | 5 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:c5139a60c77887613dc55683` | published cut
| `heat_decay_length_over_scrape_off_layer` | `derived` | `pipeline` | 3 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:84dea3d2bb15938c7dc1dab6` | |
| `ion_charge` | `derived` | `pipeline` | 40 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:530139d2dcaba3d6e4c5dc38` | |
| `ion_charge_number` | `derived` | `pipeline` | 1 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:8710f4070dbb3c7b223de4e6` | |
| `ion_charge_state_charge_number` | `derived` | `pipeline` | 3 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:09d6ce886b3f9d2b9d9f0fe3` | |
| `ion_charge_state_density` | `derived` | `pipeline` | 7 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:b23992ac09378c0bd90268f7` | |
| `ion_charge_state_power_density` | `derived` | `pipeline` | 2 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:9a262a53372b90451a3744ee` | |
| `ion_particle_diffusivity` | `derived` | `pipeline` | 1 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:7cdfca69ecb642671243ff6d` | |
| `ion_velocity` | `derived` | `pipeline` | 1 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:705cea4b75b938504835c828` | |
| `ion_velocity_due_to_diamagnetic_drift` | `derived` | `pipeline` | 1 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:0d2955be3d102e77473f8e9b` | |
| `length_of_poloidal_field_coil` | `derived` | `pipeline` | 1 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:40728377187a03d96e795ade` | |
| `neutral_internal_state_convection_velocity` | `derived` | `pipeline` | 1 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:d0298e4d6462541bdde580e5` | |
| `neutral_particle_diffusivity` | `derived` | `pipeline` | 1 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:1cc578b5f4aea95ceef1f5c1` | |
| `neutral_state_particle_convection_velocity` | `derived` | `pipeline` | 2 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:0ff60101e0f2037788bdbd9b` | |
| `neutral_temperature` | `derived` | `pipeline` | 6 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:12742df9c0516027dba5b330` | |
| `plasma_pressure` | `derived` | `pipeline` | 3 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:dbfb5dbb2f462fdd7182ca71` | published cut
| `prefill_count` | `derived` | `pipeline` | 1 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:f922c1b1e0a566b814385169` | |
| `radius_of_antenna_strap` | `derived` | `pipeline` | 1 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:f7a8f26b41ce36949c17dcab` | |
| `right_hand_circularly_polarized_wave_electric_field` | `derived` | `pipeline` | 2 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:7feb1216fe459c747c519ece` | |
| `thermal_ion_charge_state_energy_diffusion_coefficient` | `derived` | `pipeline` | 1 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:3449ed350586b0389ac34f7a` | |
| `thermal_ion_charge_state_power_density_due_to_collisions` | `derived` | `pipeline` | 1 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:26449505a1217ee76b2c6806` | |
| `torque_density_due_to_thermalization` | `derived` | `pipeline` | 1 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:f324f1c5b20122b8839a1c1a` | |
| `trapped_fast_ion_charge_state_torque_density_due_to_collisions` | `derived` | `pipeline` | 1 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:b947b347182e19af87df6fc0` | |
| `voltage_of_spectrometer_channel` | `derived` | `pipeline` | 2 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:b9b79df4159f79ec6e05be27` | |
| `wave_phase_of_ion_cyclotron_heating_antenna` | `derived` | `pipeline` | 1 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:3b8eb43a38b2129328fda30d` | published cut
| `wavelength_of_filter` | `derived` | `pipeline` | 1 | `sn-change:signed-manifest:536460046c429656aa8c8d2107532d5b08bf398b29a63c6c8750f637782d4bc5:fc4ac41e86c5600c9bd74e27` | |

## Reproduction check

The audit's direct-producer reproduction query was re-run after commit:

```cypher
MATCH (sn:StandardName {origin: 'derived'})
WHERE coalesce(sn.name_stage, '') <> 'superseded'
MATCH (:StandardNameSource {source_type: 'dd'})-[:PRODUCED_NAME]->(sn)
RETURN count(DISTINCT sn) AS rows
```

Result: **0 rows**. The required postcondition is therefore met: no live non-superseded name simultaneously carries `origin='derived'` and a direct DD producer.

The four names from the 208-name published cut are present in the signed correction set: `atomic_mass`, `gas_flow`, `plasma_pressure`, and `wave_phase_of_ion_cyclotron_heating_antenna`.
