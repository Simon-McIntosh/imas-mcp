# Outline quarantine clearance after the corrected path-qualification rule

**Verdict: the falsely quarantined qualified-outline rows have cleared.** The
live count of `StandardName` rows whose `validation_issues` carry the
path/boundary semantic issue phrase fell from **9 to 3** after re-validating
exactly the 9-row cohort through the sanctioned id-scoped audit drain. The 3
remaining rows are bare outlines that the corrected rule **still rightly
rejects**, and each now carries a freshly observed quarantine verdict. The two
WEST batch members `radial_outline_of_plasma_boundary` and
`radial_outline_of_wall` are valid, accepted on both lifecycle axes, and a
fresh export would admit them. `vertical_outline_of_plasma_boundary` is also no
longer quarantined, but a fresh export would still not admit it because
`name_stage=drafted` and `docs_stage=pending`; its separate metre-description
advisory still fires and is reported, not edited, here.

## Setup: the corrected grammar is live

The rule fix landed in the pinned `imas-standard-names` v0.9.2 (editable source
at `/home/ITER/mcintos/Code/imas-standard-names`, head `d50f5d6`):
`_check_trajectory_path_qualification` now treats a qualified `of_…` entity as
satisfying the path/boundary requirement. The earlier rule read only the
`object` vocabulary slot, which the parser leaves `None` for these names while
preserving the entity in `geometry` — so the earlier rule rejected a qualified
outline exactly as if it were bare (recorded in
`docs/evidence/sn-lifecycle-integrity/outline-qualification-false-positive.md`).

A fresh interpreter against this linkmap confirms the corrected rule directly:

```text
qualified radial_outline_of_plasma_boundary: issues=[]
qualified radial_outline_of_wall: issues=[]
qualified vertical_outline_of_plasma_boundary: issues=[]
bare      outline: issues=["outline: ERROR - 'outline' must specify what entity's path/boundary is described. Example: outline_of_limiter_tile"]
bare      trajectory: issues=["trajectory: ERROR - 'trajectory' must specify what entity's path/boundary is described. Example: trajectory_of_neutral_beam"]
```

## The measure: before

A bounded login-node query (Neo4j tunnel is login-node-local; single indexed
`CONTAINS` over the `validation_issues` list, completed in well under the
ten-second ceiling) found **9 live rows** carrying the exact distinctive phrase
`must specify what entity's path/boundary is described` — the same population
the earlier audit measured. They are a mix of six qualified outlines and three
genuinely bare outlines:

| id | pre `validation_status` | pre `name_stage` | pre `docs_stage` | pre `validated_at` |
|---|---|---|---|---|
| `radial_outline_of_plasma_boundary` | quarantined | accepted | accepted | 2026-09-08T07:48:57.586Z |
| `radial_outline_of_wall` | quarantined | accepted | accepted | 2026-09-08T07:48:57.586Z |
| `vertical_outline_of_plasma_boundary` | quarantined | drafted | pending | 2026-09-08T07:48:57.586Z |
| `radial_outline_of_limiter` | quarantined | superseded | accepted | (audits, after) |
| `radial_outline_of_flux_surface` | quarantined | accepted | accepted | (audits, after) |
| `vertical_outline_of_control_surface` | quarantined | accepted | accepted | 2026-09-08T15:42:28.026Z |
| `radial_outline` | valid | superseded | pending | 2026-07-17T06:47:28.689Z |
| `toroidal_outline` | valid | reviewed | pending | 2026-07-17T06:47:39.980Z |
| `vertical_outline` | valid | reviewed | pending | 2026-07-17T06:47:34.757Z |

Pre-state detail for the three WEST members (full `validation_issues`):

- `radial_outline_of_plasma_boundary` — `[semantic] … ERROR - 'outline' must
  specify what entity's path/boundary is described. Example: outline_of_limiter_tile`
- `radial_outline_of_wall` — same single semantic error.
- `vertical_outline_of_plasma_boundary` — same semantic error **plus** the
  independent advisory
  `audit:unit_dimension_check: unit='m' but description lacks expected terms
  ['circumference', 'coordinate', 'displacement', 'distance', 'elongation']`.

## Method: re-validated through the sanctioned drain, cohort-only

No verdict was hand-written. The graph was made to re-observe exactly the nine
named rows through the deterministic id-scoped audit — the same two-stage
mechanism the CLI's `--revalidate` + `--only validate` path uses, scoped down to
the cohort instead of the whole backlog:

1. The `--revalidate` stamp clear (`validated_at = NULL, claimed_at = NULL,
   claim_token = NULL`) applied to **only the 9 cohort ids**,
2. then `drain_validation_for_ids(cohort)` re-claimed them, ran the full
   admission gate (`validate_name_candidate`: ISN grammar round-trip, three
   ISN layers, post-generation audits) on each, and wrote the resulting
   `validation_status` / `validation_issues` / `validation_layer_summary` /
   `validated_at` under a claim token, atomically, LLM-free.

Guards: the cohort set was asserted equal to the live phrase-carrier set before
any write (any mismatch aborts); every cohort row's `description` was asserted
non-null before its stamp was cleared, so no row could be left with
`validated_at` cleared and no re-observation; the drain verified all nine were
re-stamped (`validated: 9`); no row outside the cohort had any validation field
touched; no signed manifest apply and no broad pipeline were run. All graph
queries were bounded id-scoped reads/writes on the login node, each well under
the ten-second ceiling.

## The measure: after

The same census over the live graph now returns **3 rows**, all bare outlines
that the corrected rule continues to reject because they name no entity:

| id | post `validation_status` | why it still carries the issue |
|---|---|---|
| `radial_outline` | quarantined | bare geometric base, no `of_…` entity — true positive; never falsely quarantined |
| `toroidal_outline` | quarantined | bare geometric base, no `of_…` entity — true positive |
| `vertical_outline` | quarantined | bare geometric base, no `of_…` entity — true positive |

Each of these three carried a stored `[semantic] … ERROR -` issue under an
older `valid` verdict from 2026-07-17 — an inconsistent record (an ERROR stored
against a valid verdict). The re-observation reconciles it: under the current
quarantine classifier a semantic ERROR quarantines, so all three now carry a
freshly observed `quarantined` verdict at `2026-09-09T20:25:22.073Z`. None is an
export candidate (`superseded`/`reviewed`), so no export surface is affected.

The six qualified outlines no longer carry the semantic issue. End state across
the cohort: **5 valid, 4 quarantined** (the three bare outlines above, plus
`vertical_outline_of_control_surface`, which is quarantined on an unrelated,
genuine `audit:latex_def_check: symbol $Z=0$ lacks a definition sentence` — its
semantic issue is gone). `radial_outline_of_limiter` (superseded) and
`radial_outline_of_flux_surface` (accepted) both cleared to `valid` with empty
issues; the old `_of_ → _at_` canonical advisories stored on them did not re-fire
in the observational audit.

## The three WEST batch members

| Field | `radial_outline_of_plasma_boundary` | `radial_outline_of_wall` | `vertical_outline_of_plasma_boundary` |
|---|---|---|---|
| `validation_status` before → after | quarantined → **valid** | quarantined → **valid** | quarantined → **valid** |
| `validated_at` before → after | 2026-09-08T07:48:57.586Z → **2026-09-09T20:25:22.073Z** | 2026-09-08T07:48:57.586Z → **2026-09-09T20:25:22.073Z** | 2026-09-08T07:48:57.586Z → **2026-09-09T20:25:22.073Z** |
| `validation_issues` before → after | semantic ERROR → **[]** | semantic ERROR → **[]** | semantic ERROR + unit advisory → **unit advisory only** |
| `name_stage` before → after | accepted → accepted | accepted → accepted | drafted → drafted |
| `docs_stage` before → after | accepted → accepted | accepted → accepted | pending → pending |

**Would a fresh export now admit it?** Evaluated through the authentic code path
(`_fetch_export_population(batch=[…])` + `_classify_export_population`, no domain
filter):

- `radial_outline_of_plasma_boundary` — **yes**: `valid`, observed
  `validated_at`, `name_stage=accepted`, `docs_stage=accepted`, winning docs
  review present, no exclusions.
- `radial_outline_of_wall` — **yes**: identical profile, no exclusions.
- `vertical_outline_of_plasma_boundary` — **no**: excluded with reason
  `name_not_accepted` (`name_stage='drafted'`). It is not in the normal export
  population at all (population starts at accepted/approved), and in an
  additive review-batch export it fails the name-accepted predicate; were that
  resolved, `docs_stage=pending` would still fail `documentation_not_accepted`.

## The second advisory on `vertical_outline_of_plasma_boundary`

`audit:unit_dimension_check: unit='m' but description lacks expected terms
['circumference', 'coordinate', 'displacement', 'distance', 'elongation']`
**still fires** after revalidation — it is now the row's only stored issue. It is
an advisory (not an ERROR), so it does not quarantine the name; it is a
description-wording matter: the description ("Signed vertical outline of the
plasma-boundary contour in the right-handed cylindrical (R, φ, Z) frame.") does
not state that the stored value is a coordinate. Per the fence, the description
is **not edited in this node** — it is reported for the authorized catalog
documentation path.

## Scope and standing

Only the nine measured rows were re-validated; no other `StandardName` node had
a validation field touched, no signed manifest apply was attempted, and no broad
pipeline was run. The merged-result verification of the wider release surface
belongs to a separately dispatched test node; anything found outside this node's
declared scope is reported as a follow-on rather than triaged here.

## Reproduction

All scripts and raw outputs used for this record are under
`/tmp/outline-clear-833912/` (`revalidate.py` + `revalidate.out` hold the full
before/after JSON; `probe_rule.py`, `probe_after.py`, `admission.py` and their
outputs hold the rule check, the final census, and the export-admission probe).
The full before/after JSON is embedded verbatim below so this document stands
alone if the scratch directory is reclaimed.

```json
{
  "cleared_validated_at": 9,
  "drain_totals": {
    "cleared_ids": [
      "radial_outline_of_plasma_boundary",
      "radial_outline_of_limiter",
      "radial_outline_of_wall",
      "radial_outline_of_flux_surface",
      "vertical_outline_of_plasma_boundary"
    ],
    "quarantined": 4,
    "requarantined_ids": [
      "radial_outline",
      "toroidal_outline",
      "vertical_outline",
      "vertical_outline_of_control_surface"
    ],
    "validated": 9
  },
  "post": {
    "radial_outline": {
      "docs_stage": "pending",
      "id": "radial_outline",
      "name_stage": "superseded",
      "validated_at": "2026-09-09T20:25:22.073000000+00:00",
      "validation_issues": [
        "[semantic] radial_outline: ERROR - 'outline' must specify what entity's path/boundary is described. Example: outline_of_limiter_tile"
      ],
      "validation_status": "quarantined"
    },
    "radial_outline_of_flux_surface": {
      "docs_stage": "accepted",
      "id": "radial_outline_of_flux_surface",
      "name_stage": "accepted",
      "validated_at": "2026-09-09T20:25:22.073000000+00:00",
      "validation_issues": [],
      "validation_status": "valid"
    },
    "radial_outline_of_limiter": {
      "docs_stage": "accepted",
      "id": "radial_outline_of_limiter",
      "name_stage": "superseded",
      "validated_at": "2026-09-09T20:25:22.073000000+00:00",
      "validation_issues": [],
      "validation_status": "valid"
    },
    "radial_outline_of_plasma_boundary": {
      "docs_stage": "accepted",
      "id": "radial_outline_of_plasma_boundary",
      "name_stage": "accepted",
      "validated_at": "2026-09-09T20:25:22.073000000+00:00",
      "validation_issues": [],
      "validation_status": "valid"
    },
    "radial_outline_of_wall": {
      "docs_stage": "accepted",
      "id": "radial_outline_of_wall",
      "name_stage": "accepted",
      "validated_at": "2026-09-09T20:25:22.073000000+00:00",
      "validation_issues": [],
      "validation_status": "valid"
    },
    "toroidal_outline": {
      "docs_stage": "pending",
      "id": "toroidal_outline",
      "name_stage": "reviewed",
      "validated_at": "2026-09-09T20:25:22.073000000+00:00",
      "validation_issues": [
        "[semantic] toroidal_outline: ERROR - 'outline' must specify what entity's path/boundary is described. Example: outline_of_limiter_tile"
      ],
      "validation_status": "quarantined"
    },
    "vertical_outline": {
      "docs_stage": "pending",
      "id": "vertical_outline",
      "name_stage": "reviewed",
      "validated_at": "2026-09-09T20:25:22.073000000+00:00",
      "validation_issues": [
        "[semantic] vertical_outline: ERROR - 'outline' must specify what entity's path/boundary is described. Example: outline_of_limiter_tile"
      ],
      "validation_status": "quarantined"
    },
    "vertical_outline_of_control_surface": {
      "docs_stage": "accepted",
      "id": "vertical_outline_of_control_surface",
      "name_stage": "accepted",
      "validated_at": "2026-09-09T20:25:22.073000000+00:00",
      "validation_issues": [
        "audit:latex_def_check: symbol $Z=0$ lacks a definition sentence"
      ],
      "validation_status": "quarantined"
    },
    "vertical_outline_of_plasma_boundary": {
      "docs_stage": "pending",
      "id": "vertical_outline_of_plasma_boundary",
      "name_stage": "drafted",
      "validated_at": "2026-09-09T20:25:22.073000000+00:00",
      "validation_issues": [
        "audit:unit_dimension_check: unit='m' but description lacks expected terms ['circumference', 'coordinate', 'displacement', 'distance', 'elongation']"
      ],
      "validation_status": "valid"
    }
  },
  "post_carrying_phrase": [
    "radial_outline",
    "toroidal_outline",
    "vertical_outline"
  ],
  "pre": {
    "radial_outline": {
      "docs_stage": "pending",
      "id": "radial_outline",
      "name_stage": "superseded",
      "validated_at": "2026-07-17T06:47:28.689000000+00:00",
      "validation_issues": [
        "[semantic] radial_outline: ERROR - 'outline' must specify what entity's path/boundary is described. Example: outline_of_limiter_tile"
      ],
      "validation_status": "valid"
    },
    "radial_outline_of_flux_surface": {
      "docs_stage": "accepted",
      "id": "radial_outline_of_flux_surface",
      "name_stage": "accepted",
      "validated_at": null,
      "validation_issues": [
        "[semantic] radial_outline_of_flux_surface: ERROR - 'outline' must specify what entity's path/boundary is described. Example: outline_of_limiter_tile",
        "[canonical] audit:canonical_locus_check: name 'radial_outline_of_flux_surface' has field-evaluation structure but uses intrinsic-geometry relation '_of_'. Rewrite as 'radial_outline_at_flux_surface'.",
        "audit:canonical_locus_check: name 'radial_outline_of_flux_surface' has field-evaluation structure but uses intrinsic-geometry relation '_of_'. Rewrite as 'radial_outline_at_flux_surface'."
      ],
      "validation_status": "quarantined"
    },
    "radial_outline_of_limiter": {
      "docs_stage": "accepted",
      "id": "radial_outline_of_limiter",
      "name_stage": "superseded",
      "validated_at": null,
      "validation_issues": [
        "[semantic] radial_outline_of_limiter: ERROR - 'outline' must specify what entity's path/boundary is described. Example: outline_of_limiter_tile",
        "[canonical] audit:canonical_locus_check: name 'radial_outline_of_limiter' has field-evaluation structure but uses intrinsic-geometry relation '_of_'. Rewrite as 'radial_outline_at_limiter'.",
        "audit:canonical_locus_check: name 'radial_outline_of_limiter' has field-evaluation structure but uses intrinsic-geometry relation '_of_'. Rewrite as 'radial_outline_at_limiter'."
      ],
      "validation_status": "quarantined"
    },
    "radial_outline_of_plasma_boundary": {
      "docs_stage": "accepted",
      "id": "radial_outline_of_plasma_boundary",
      "name_stage": "accepted",
      "validated_at": "2026-09-08T07:48:57.586000000+00:00",
      "validation_issues": [
        "[semantic] radial_outline_of_plasma_boundary: ERROR - 'outline' must specify what entity's path/boundary is described. Example: outline_of_limiter_tile"
      ],
      "validation_status": "quarantined"
    },
    "radial_outline_of_wall": {
      "docs_stage": "accepted",
      "id": "radial_outline_of_wall",
      "name_stage": "accepted",
      "validated_at": "2026-09-08T07:48:57.586000000+00:00",
      "validation_issues": [
        "[semantic] radial_outline_of_wall: ERROR - 'outline' must specify what entity's path/boundary is described. Example: outline_of_limiter_tile"
      ],
      "validation_status": "quarantined"
    },
    "toroidal_outline": {
      "docs_stage": "pending",
      "id": "toroidal_outline",
      "name_stage": "reviewed",
      "validated_at": "2026-07-17T06:47:39.980000000+00:00",
      "validation_issues": [
        "[semantic] toroidal_outline: ERROR - 'outline' must specify what entity's path/boundary is described. Example: outline_of_limiter_tile"
      ],
      "validation_status": "valid"
    },
    "vertical_outline": {
      "docs_stage": "pending",
      "id": "vertical_outline",
      "name_stage": "reviewed",
      "validated_at": "2026-07-17T06:47:34.757000000+00:00",
      "validation_issues": [
        "[semantic] vertical_outline: ERROR - 'outline' must specify what entity's path/boundary is described. Example: outline_of_limiter_tile"
      ],
      "validation_status": "valid"
    },
    "vertical_outline_of_control_surface": {
      "docs_stage": "accepted",
      "id": "vertical_outline_of_control_surface",
      "name_stage": "accepted",
      "validated_at": "2026-09-08T15:42:28.026000000+00:00",
      "validation_issues": [
        "[semantic] vertical_outline_of_control_surface: ERROR - 'outline' must specify what entity's path/boundary is described. Example: outline_of_limiter_tile",
        "audit:latex_def_check: symbol $Z=0$ lacks a definition sentence"
      ],
      "validation_status": "quarantined"
    },
    "vertical_outline_of_plasma_boundary": {
      "docs_stage": "pending",
      "id": "vertical_outline_of_plasma_boundary",
      "name_stage": "drafted",
      "validated_at": "2026-09-08T07:48:57.586000000+00:00",
      "validation_issues": [
        "[semantic] vertical_outline_of_plasma_boundary: ERROR - 'outline' must specify what entity's path/boundary is described. Example: outline_of_limiter_tile",
        "audit:unit_dimension_check: unit='m' but description lacks expected terms ['circumference', 'coordinate', 'displacement', 'distance', 'elongation']"
      ],
      "validation_status": "quarantined"
    }
  },
  "pre_live_count": 9
}
```
