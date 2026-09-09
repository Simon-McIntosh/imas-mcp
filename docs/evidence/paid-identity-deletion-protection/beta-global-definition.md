# Beta now names the global pressure ratio

`beta` is the global beta: the dimensionless ratio of volume-averaged total
perpendicular plasma pressure to the magnetic pressure of the combined toroidal
and poloidal fields. It is no longer described as a generic family of possible
field normalizations.

## Governed edit

The documentation steer was staged through the Standard Names edit boundary,
with an explicit self scope because `beta` has descendants:

```text
imas-codex sn edit beta \
  --hint 'Define beta as global beta: the ratio of volume-averaged perpendicular plasma pressure to total magnetic pressure. State B^2 = B_t^2 + B_p^2 and therefore 1/beta = 1/toroidal_beta + 1/poloidal_beta. Do not describe beta as a generic family, and distinguish it from normalized_toroidal_beta.' \
  --axis docs \
  --scope self \
  --reason 'The accepted beta identity must name the total-field quantity: a common volume-averaged perpendicular pressure with toroidal and poloidal magnetic fields adding in quadrature gives the reciprocal relation to toroidal_beta and poloidal_beta.' \
  --dry-run
```

The preview reported exactly one target:

```text
DRY RUN sn edit: beta  mode=hint axis=docs scope=only_self entry=generate
Actions:
  - hint attached to 'beta' (axis=docs)
  -  no writes performed
```

There was no subtree cascade. The only affected identity was `beta`; the three
children were not placed in this edit's run scope.

The initial inline review staged the correctly scoped edit but stopped when an
unrelated global structural-maintenance operation refused protected identities.
The staged row remained `beta` only, at run
`sn-edit-20260909T104321Z`, with `edit_scope=only_self`, `edit_status=open`,
and `docs_stage=pending`. The recovery rotation deliberately retained ordinary
docs generation and review while bypassing only that unrelated global work:

```text
imas-codex sn run \
  --scope-run-id sn-edit-20260909T104321Z \
  --docs-only \
  --skip-global-maintenance \
  --cost-limit 1.0
```

Its zero-write preview stated that normal worker pools would run only for that
exact run ID. The completed rotation processed one `generate_docs` item and one
`review_docs` item, processed no refinement item, stopped with
`no_eligible_work`, and spent `$0.099287` against the `$1.00` ceiling.

## Live outcome

After the review rotation, the live `beta` row has:

| Field | Value |
| --- | --- |
| `name_stage` | `accepted` |
| `docs_stage` | `accepted` |
| `edit_status` | `applied` |
| `edit_scope` | `only_self` |
| `reviewer_score_docs` | `0.99375` |
| docs review quorum shortfall | none |

Its accepted description is:

> Dimensionless global beta formed from volume-averaged total perpendicular
> plasma pressure and the magnetic pressure of the combined toroidal and
> poloidal fields.

The accepted documentation fixes both the definition and its component relation:

```text
Global beta is the dimensionless ratio of total plasma pressure, averaged over
the plasma volume, to the magnetic pressure of the combined toroidal and
poloidal magnetic fields.

β = 2μ₀ <p⊥>_V / B²

B² = B_t² + B_p²

1/β = 1/β_tor + 1/β_pol
```

The full accepted text defines `<p⊥>_V` as the volume average of total
perpendicular pressure, distinguishes the toroidal and poloidal component
normalizations, and identifies `normalized_toroidal_beta` as a separate
size/field/current normalization rather than the total-field ratio.

## Descendant containment

The following descriptions were read immediately before the edit and again
after the accepted review. They are byte-for-byte unchanged:

| Identity | Before | After |
| --- | --- | --- |
| `toroidal_beta` | “Toroidal beta is a dimensionless equilibrium ratio of volume-averaged total perpendicular plasma pressure to the magnetic pressure of a reference toroidal field.” | “Toroidal beta is a dimensionless equilibrium ratio of volume-averaged total perpendicular plasma pressure to the magnetic pressure of a reference toroidal field.” |
| `poloidal_beta` | “Poloidal beta is a dimensionless measure of total plasma pressure relative to the magnetic-pressure scale of the plasma-current-generated poloidal field.” | “Poloidal beta is a dimensionless measure of total plasma pressure relative to the magnetic-pressure scale of the plasma-current-generated poloidal field.” |
| `normalized_toroidal_beta` | “Normalized toroidal beta defined as 100 * beta_tor * a[m] * B0[T] / Ip[MA], a key stability metric.” | “Normalized toroidal beta defined as 100 * beta_tor * a[m] * B0[T] / Ip[MA], a key stability metric.” |

The three unchanged descriptions, together with the exact run ID and
`edit_scope=only_self`, demonstrate that the global-beta repair changed no
component or normalized-beta wording.

## Separate provenance work

`beta` remains a known sourceless identity. This documentation edit deliberately
did not create a producer; the missing canonical `derived:beta` provenance is a
separate repair.
