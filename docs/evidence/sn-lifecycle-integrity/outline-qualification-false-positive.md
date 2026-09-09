# Outline qualification audit: confirmed false positive

**Verdict: the path-qualification rule reads the wrong parsed field.** The parser does not lose the `of_…` clause. It preserves the qualified boundary in `geometry`, while `_check_trajectory_path_qualification` in the separate `imas-standard-names` checkout reads only `object`. A qualified outline is consequently rejected exactly as if it were bare.

This was read from `imas-standard-names` at `2a4584d82537f3339e9c87294048ca4feb658495`; no code or graph state was changed. The one live-graph read used the login node because its Neo4j tunnel is login-node-local. Its selection was bounded to the validation-issue phrase and completed in under one second.

## Parser and rule evidence

The parser was invoked with `parse_standard_name` and `model_dump` for the three names below. This is the complete returned field set, including `None` and empty tuple fields, so the qualifier location is auditable rather than inferred from the source spelling.

```python
radial_outline_of_plasma_boundary
{'component': None, 'coordinate': <Coordinate.RADIAL: 'radial'>, 'section_plane': None, 'aggregation': None, 'orbit': None, 'population': None, 'subject': None, 'state': None, 'device': None, 'zone': (), 'qualifier': (), 'channel_qualifier': None, 'channel': None, 'geometry_representation': None, 'geometric_base': <GeometricBase.OUTLINE: 'outline'>, 'physical_base': None, 'object': None, 'geometry': <Position.PLASMA_BOUNDARY: 'plasma_boundary'>, 'position': None, 'position_value': None, 'region': None, 'path': None, 'locus_qualifiers': (), 'process': None, 'transformation': None, 'decomposition': None, 'binary_operator': None, 'secondary_base': None}

radial_outline_of_wall
{'component': None, 'coordinate': <Coordinate.RADIAL: 'radial'>, 'section_plane': None, 'aggregation': None, 'orbit': None, 'population': None, 'subject': None, 'state': None, 'device': None, 'zone': (), 'qualifier': (), 'channel_qualifier': None, 'channel': None, 'geometry_representation': None, 'geometric_base': <GeometricBase.OUTLINE: 'outline'>, 'physical_base': None, 'object': None, 'geometry': <Position.WALL: 'wall'>, 'position': None, 'position_value': None, 'region': None, 'path': None, 'locus_qualifiers': (), 'process': None, 'transformation': None, 'decomposition': None, 'binary_operator': None, 'secondary_base': None}

vertical_outline_of_plasma_boundary
{'component': None, 'coordinate': <Coordinate.VERTICAL: 'vertical'>, 'section_plane': None, 'aggregation': None, 'orbit': None, 'population': None, 'subject': None, 'state': None, 'device': None, 'zone': (), 'qualifier': (), 'channel_qualifier': None, 'channel': None, 'geometry_representation': None, 'geometric_base': <GeometricBase.OUTLINE: 'outline'>, 'physical_base': None, 'object': None, 'geometry': <Position.PLASMA_BOUNDARY: 'plasma_boundary'>, 'position': None, 'position_value': None, 'region': None, 'path': None, 'locus_qualifiers': (), 'process': None, 'transformation': None, 'decomposition': None, 'binary_operator': None, 'secondary_base': None}
```

| Semantic role | Parsed field | Values in the three qualified names | Field read by the audit |
|---|---|---|---|
| Path/boundary base | `geometric_base` | `outline` | `geometric_base` |
| Explicit `of_…` entity | `geometry` | `plasma_boundary`, `wall`, `plasma_boundary` | **not read** |
| Object vocabulary slot | `object` | `None` for all three | **`object` only** |
| Directional projection | `coordinate` | `radial`, `radial`, `vertical` | not used for qualification |

`PATH_BASES` is exactly:

```python
{
    GeometricBase.TRAJECTORY.value,
    GeometricBase.OUTLINE.value,
    GeometricBase.CONTOUR.value,
}
```

At `imas_standard_names/validation/semantic.py:350-371`, the control path parses the name, gets `geometric_base`, tests membership in `PATH_BASES`, then does `obj = getattr(parsed, "object", None)` and raises the path/boundary error when `obj` is falsy. Direct calls to that check returned the same error for all three qualified names:

```text
<name>: ERROR - 'outline' must specify what entity's path/boundary is described.
Example: outline_of_limiter_tile
```

The `radial_` and `vertical_` prefixes do change the parse, but only by setting `coordinate` to `radial` or `vertical`; they do not move, erase, or otherwise affect the `geometry` qualifier. Bare controls make the distinction clear: `outline`, `radial_outline`, and `vertical_outline` all parse with `geometry=None`, and all correctly receive the same error. The fault is therefore a validator field-selection error, not a lost qualifier in the parser.

## Live impact

The exact distinctive phrase, `must specify what entity's path/boundary is described`, currently occurs on **9 live `StandardName` rows**:

```text
radial_outline_of_plasma_boundary
radial_outline_of_limiter
radial_outline
toroidal_outline
vertical_outline
radial_outline_of_wall
radial_outline_of_flux_surface
vertical_outline_of_plasma_boundary
vertical_outline_of_control_surface
```

The requested three still carry the semantic issue live. Their remaining state establishes that this is not merely an isolated parser experiment:

| Name | Unit | Description | Live path-qualification issue | Lifecycle state |
|---|---|---|---|---|
| `radial_outline_of_plasma_boundary` | `m` | Major-radius coordinate of each point on the plasma-boundary contour in a poloidal cross-section. | yes | `validation_status=quarantined`, `name_stage=accepted`, `docs_stage=accepted` |
| `radial_outline_of_wall` | `m` | Major-radius coordinate of every point on a wall boundary outline, measured from the machine symmetry axis in the right-handed cylindrical (R, φ, Z) frame. | yes | `validation_status=quarantined`, `name_stage=accepted`, `docs_stage=accepted` |
| `vertical_outline_of_plasma_boundary` | `m` | Signed vertical outline of the plasma-boundary contour in the right-handed cylindrical (R, φ, Z) frame. | yes | `validation_status=quarantined`, `name_stage=drafted`, `docs_stage=pending` |

The nine-row population mixes genuine bare outlines with qualified names. The proposed test retains rejection of the bare control, so correcting the false positives does not turn the semantic invariant off.

## The independent metre-description finding

`vertical_outline_of_plasma_boundary` also carries:

```text
audit:unit_dimension_check: unit='m' but description lacks expected terms
['circumference', 'coordinate', 'displacement', 'distance', 'elongation']
```

This is a **description-wording repair, not another misfiring rule**. The `unit_dimension_check` implementation in this repository uses a deliberately minimal metre noun set containing `coordinate`, and the live description says only “Signed vertical outline …”. It does not state that the stored value is the vertical coordinate of a contour point. Re-running the audit locally without writing anything produced the issue for the live wording and no issue for:

> Vertical coordinate of each point on the plasma-boundary contour in the right-handed cylindrical (R, φ, Z) frame.

That proposed wording states the physical quantity more precisely and uses an existing expected noun. By contrast, the qualified-path semantic error remains wrong even when the description is perfect because it is computed from the parsed name, not documentation text.

## Minimal upstream repair and proof

The smallest upstream change is in `imas_standard_names/validation/semantic.py`, inside `_check_trajectory_path_qualification`: decide whether an outline is qualified from the slots that can actually hold its `of_…` target, not just the object-vocabulary slot. For the demonstrated grammar, that is:

```python
entity = getattr(parsed, "object", None) or getattr(parsed, "geometry", None)
if not entity:
    # retain the existing error
```

No parser change is indicated. A focused upstream regression test may sit beside the existing semantic-check helpers in `tests/test_semantic_unit_vector_locus.py` (or an equivalently named semantic test module) and should prove both halves:

1. Parameterize `radial_outline_of_plasma_boundary`, `radial_outline_of_wall`, and `vertical_outline_of_plasma_boundary`; each must produce no path-qualification error.
2. Keep `radial_outline` (and preferably bare `outline`) as a control; each must still produce the path/boundary qualification error.

Those assertions prove that the repair admits a qualifier the parser already represents while retaining the refusal for an actually unqualified outline.

## Scope and follow-on

This investigation made no graph write, no pipeline call, and no code change in either repository. The upstream validator fix and its test are a follow-on in `~/Code/imas-standard-names`; the vertical-name description edit is a separate catalog-documentation follow-on.
