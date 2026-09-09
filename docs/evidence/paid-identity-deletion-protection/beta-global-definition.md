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

## First review outcome and the missing description clause

The first review rotation returned the live `beta` row to:

| Field | Value |
| --- | --- |
| `name_stage` | `accepted` |
| `docs_stage` | `accepted` |
| `edit_status` | `applied` |
| `edit_scope` | `only_self` |
| `reviewer_score_docs` | `0.99375` |
| docs review quorum shortfall | none |

Its accepted description was:

> Dimensionless global beta formed from volume-averaged total perpendicular
> plasma pressure and the magnetic pressure of the combined toroidal and
> poloidal fields.

The long documentation carried the reciprocal relation, but the short catalog
description did not. That left the defining harmonic relation implicit and did
not meet the two-clause description gate.

## Corrective self-scoped review

The corrective hint required the relation and its physical basis in the short
description itself while retaining both parts of the accepted definition:

```text
imas-codex sn edit beta \
  --hint 'Revise the short description itself. Keep its existing statement that global beta uses volume-averaged total perpendicular plasma pressure and the magnetic pressure of the combined toroidal and poloidal fields, then state explicitly: 1/beta = 1/beta_toroidal + 1/beta_poloidal. Briefly say in the description that this reciprocal relation follows because the toroidal and poloidal magnetic fields add in quadrature over the same pressure. The formula and quadrature explanation must appear in the short description, not only in the long documentation.' \
  --axis docs \
  --scope self \
  --reason 'For a common volume-averaged total perpendicular pressure, the total magnetic field satisfies B^2 = B_toroidal^2 + B_poloidal^2, so the inverse global beta is the sum of the inverse toroidal and poloidal betas and the total beta is their harmonic combination.' \
  --dry-run
```

The corrective preview again reported exactly one target:

```text
DRY RUN sn edit: beta  mode=hint axis=docs scope=only_self entry=generate
Actions:
  - hint attached to 'beta' (axis=docs)
  -  no writes performed
```

The same command with `--stage-only` in place of `--dry-run` created run
`sn-edit-20260909T105935Z`. A live read proved that run contained exactly one
identity, `beta`, before the review rotation. The exact-run rotation was:

```text
imas-codex sn run \
  --scope-run-id sn-edit-20260909T105935Z \
  --docs-only \
  --skip-global-maintenance \
  --cost-limit 1.0
```

It processed one `generate_docs` item and one `review_docs` item, processed no
refinement item, stopped with `no_eligible_work`, and spent `$0.093954` against
the `$1.00` ceiling. The two rotations together spent `$0.193241`.

## Final live outcome

After the corrective review rotation, the live `beta` row has:

| Field | Value |
| --- | --- |
| `name_stage` | `accepted` |
| `docs_stage` | `accepted` |
| `edit_status` | `applied` |
| `edit_scope` | `only_self` |
| `run_id` | `sn-edit-20260909T105935Z` |
| `reviewer_score_docs` | `1.0` |
| docs review quorum shortfall | none |

The final accepted catalog description is:

> Global beta compares volume-averaged total perpendicular plasma pressure with
> combined magnetic pressure; for a common pressure, 1/beta =
> 1/beta_toroidal + 1/beta_poloidal because B^2 = B_toroidal^2 +
> B_poloidal^2.

The final description therefore states both required facts directly: what
global beta is, and why it is the harmonic combination of the toroidal and
poloidal component betas rather than their sum.

The accepted long documentation gives the corresponding mathematical form:

```text
For a common volume-averaged pressure and orthogonal toroidal and poloidal
magnetic-field components:

1/β = 1/β_tor + 1/β_pol
```

It distinguishes the component normalizations and identifies
`normalized_toroidal_beta` as a separately scaled toroidal-beta variant rather
than a term in the reciprocal global-beta relation.

## Descendant containment

The following descriptions were read immediately before the corrective edit and
again after its accepted review. They are byte-for-byte unchanged:

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
