# Partial normalized-beta identity revival

## Result

`normalized_toroidal_beta` is no longer an orphaned terminal identity. A
one-row signed transaction changed only its catalog `status`, from
`superseded` to `draft`, and wrote one persistent receipt. Its name axis then
passed ordinary quorum review at score 1.0, backed by 12 name-axis review edges.
An independent live read found all eight producers unchanged and no worker
claim. The document migration is blocked by the docs-edit precondition because
the target has no prior document and is therefore still `docs_stage='pending'`.
The thermal-only split and inverse-lineage removal have not run.

The two redesign routes documented for superseded names were both tested first
and both refused because the target spelling already exists. The signed
fallback therefore proved necessary for this already-materialised orphan. No
hand-written Cypher property update was used.

All live graph work ran on the login node because the Neo4j tunnel is local to
that node. Every read was bounded to the four named identities or to one exact
claim token and completed in under ten seconds.

## Live state before the work

| Identity | status | name_stage | docs_stage | producer count | docs length | docs score | docs review edges |
|---|---|---|---|---:|---:|---:|---:|
| `normalized_toroidal_beta` | `superseded` | `reviewed` | `pending` | 8 | 0 | null | 0 |
| `normalized_toroidal_plasma_beta` | `superseded` | `superseded` | `accepted` | 0 | 1,504 | 0.93125 | 7 |
| `normalized_toroidal_thermal_plasma_beta` | `superseded` | `superseded` | `pending` | 0 | 0 | null | 0 |
| `toroidal_beta` | `draft` | `accepted` | `accepted` | 5 | 1,449 | 0.925 | 9 |

The 1,504-character source document contains four textual links whose label
describes unnormalised toroidal beta but whose target is its own
`normalized_toroidal_plasma_beta` identity. The prepared 1,468-character
candidate differs only in those four references:

| Occurrence | Before | Prepared after |
|---:|---|---|
| 1 | `[beta](name:normalized_toroidal_plasma_beta)` | `[toroidal_beta](name:toroidal_beta)` |
| 2 | `[beta](name:normalized_toroidal_plasma_beta)` | `[toroidal_beta](name:toroidal_beta)` |
| 3 | `[beta](name:normalized_toroidal_plasma_beta)` | `[toroidal_beta](name:toroidal_beta)` |
| 4 | `[beta](name:normalized_toroidal_plasma_beta)` | `[toroidal_beta](name:toroidal_beta)` |

The source also carries one correct
`[poloidal_beta](name:poloidal_beta)` reference; it is unchanged. The candidate
has zero old self-targeting references, but it was not applied because the
governed dry-run refused before mutation.

## Initial sanctioned-path deadlock

The first documentation dry-run protected the review gate:

```text
BLOCKED
target name_stage='reviewed' — docs edits require an accepted name
(name_stage='accepted')
Actions considered:
- target name_stage='reviewed' — docs edits require an accepted name
(name_stage='accepted')
```

`sn supersede normalized_toroidal_plasma_beta --into
normalized_toroidal_beta --dry-run` likewise refused because the target name
stage was `reviewed`, not `accepted`.

`sn rescore normalized_toroidal_beta --dry-run` admitted the row. The live
rescore then moved the name stage from `reviewed` to `drafted`, cleared the old
name score, assigned `run_id=sn-rescore-20260909T172147Z`, and revalidated the
row. It retained `status='superseded'`. The exact review continuation therefore
refused:

```text
Error: normalized_toroidal_beta: terminal StandardName lifecycle
```

The `LLMCost` read for that rescore run returned zero rows and USD 0.00.

## Redesign-route dry-runs

The same-spelling redesign command carried this physics reason:

> This is the canonical total-pressure normalized toroidal beta identity: it
> retains the WEST roster membership and all eight producers, including the
> MHD estimator facet, while the thermal-pressure-only source is a physically
> distinct quantity.

Its complete CLI verdict was:

```text
╭────────────────────── sn edit normalized_toroidal_beta ──────────────────────╮
│ BLOCKED                                                                      │
│ a StandardName 'normalized_toroidal_beta' already exists                     │
╰──────────────────────────────────────────────────────────────────────────────╯

Actions considered:
  - a StandardName 'normalized_toroidal_beta' already exists
```

The document-bearing orphan was then aimed at the elected spelling with this
physics reason:

> The source document defines the total-pressure normalized toroidal beta,
> whose canonical family spelling omits the redundant plasma segment; its four
> unnormalised-beta references must resolve to toroidal_beta, while thermal-only
> pressure remains a separate quantity.

Its complete CLI verdict was the same collision refusal:

```text
╭────────────────── sn edit normalized_toroidal_plasma_beta ───────────────────╮
│ BLOCKED                                                                      │
│ a StandardName 'normalized_toroidal_beta' already exists                     │
╰──────────────────────────────────────────────────────────────────────────────╯

Actions considered:
  - a StandardName 'normalized_toroidal_beta' already exists
```

The redesign route can therefore resurrect into an absent spelling, but these
dry-runs show that it cannot revive an existing orphan under its own spelling
or collide a second orphan into that row.

## Signed status transition

The fallback authority contained one `StandardName` participant and one closed
`set_properties` mutation:

```text
normalized_toroidal_beta: {status: draft}
```

It changed no stage, score, claim, documentation, source binding, or lineage
property. The physics reason carried by the authority was:

> Restore the total-pressure normalized toroidal beta identity to draft so its
> eight bound producers can receive independent name and documentation review.

The preview and apply receipts were:

| Measure | Preview | Apply |
|---|---:|---:|
| Authority rows | 1 | 1 |
| Admitted / refused | 1 / 0 | 1 / 0 |
| Outcome | `would_apply` | `applied` |
| Would change / changed | 1 | 1 |
| Receipt rows | — | 1 |
| Persistent writes | 0 | 2 |

- Authority file SHA-256:
  `fbfe19d039588a7d79ce974089d7c98db849bf4d4b5b76c6e101c5e3c9543933`
- Authority payload SHA-256:
  `c1315713597e8e39af62fcbf122622fec78fb0a23dda538035512805f1cbdeac`
- Previewed and applied manifest SHA-256:
  `735bda186280d8d979073f0bd2c61ee5e397afc9534305519c6f9b13e7a9a8ca`

The full authority, preview, and apply receipt are preserved beside the worker
manifest as `status-revival-authority.json`, `status-revival-preview.json`, and
`status-revival-apply.json`. The apply receipt, rather than an absent exception,
is the evidence that the mutation committed.

## Name review and current graph state

After the successful status transition, the first exact review dry-run reached
a different guard:

```text
Error: normalized_toroidal_beta: current worker claim
```

The exact token matched only `normalized_toroidal_beta`; no live process held
it. Its `claimed_at=2026-09-09T17:26:47.868Z` was later than the completed
validation observation at `validated_at=2026-09-09T17:21:57.433Z`. The
repository's token-and-stage-verified `release_review_names_claims` helper was
called for that one identity, that exact token, and expected
`name_stage='drafted'`; it returned `released=1`.

After the user clarified that these were two different repaired causes rather
than contention retries, the corrected-precondition review preview returned:

```text
Exact-name dry run: 1 existing name(s) eligible; no graph writes performed
```

The live exact-scope run used `--only review_name`,
`--skip-global-maintenance`, and a USD 3.00 command ceiling. It processed one
identity, wrote two new `StandardNameReview` nodes, and accepted the name at
score 1.0 by quorum consensus. The run reported USD 0.091405 spent. The
identity now has 12 name-axis review edges in total and zero docs-axis review
edges. The final live read confirms:

| Identity | status | name_stage | docs_stage | name score | docs score | docs length | producer count | claim token |
|---|---|---|---|---:|---:|---:|---:|---|
| `normalized_toroidal_beta` | `draft` | `accepted` | `pending` | 1.0 | null | 0 | 8 | null |

The eight producers remain:

- `dd:summary/global_quantities/beta_tor_norm_mhd/value`
- `dd:equilibrium/time_slice/global_quantities/beta_tor_norm`
- `dd:summary/global_quantities/beta_tor_thermal_norm/value`
- `dd:equilibrium/time_slice/global_quantities/beta_normal`
- `dd:summary/global_quantities/beta_tor_norm/value`
- `derived:beta`
- `dd:core_profiles/global_quantities/beta_tor_norm`
- `dd:plasma_profiles/global_quantities/beta_tor_norm`

The exact docs candidate was then dry-run with `sn edit --docs --scope self`.
The CLI refused before mutation:

```text
╭────────────────────── sn edit normalized_toroidal_beta ──────────────────────╮
│ BLOCKED                                                                      │
│ target docs_stage='pending' — docs edits require the docs axis to have       │
│ settled (docs_stage in accepted/exhausted)                                   │
╰──────────────────────────────────────────────────────────────────────────────╯

Actions considered:
  - target docs_stage='pending' — docs edits require the docs axis to have
settled (docs_stage in accepted/exhausted)
```

This is a new sanctioned-path gap. The `sn-edit` runbook says exact `--docs`
text skips generation and enters directly at docs review, while the
implementation refuses a live accepted name that has no prior document because
its docs axis has not settled. Total provider spend is USD 0.091405 of the USD
15.00 ceiling.

## Remaining work

Two routes remain. The direct route is to extend `sn edit --docs` so exact docs
text can enter review when a live accepted identity is at
`docs_stage='pending'`; that requires code outside this node's write scope. The
existing but wasteful route is to run docs generation and review first, then
replace that newly accepted document through the requested exact docs edit.
Directly setting `docs_stage='accepted'` would create the false-acceptance
projection this work is meant to remove and is not an acceptable route.

After the corrected document has its own accepted docs review, move only
`dd:summary/global_quantities/beta_tor_thermal_norm/value` to
`normalized_toroidal_thermal_plasma_beta`, review that distinct thermal-only
quantity on both axes, leave the MHD estimator on the total identity, and
remove the obsolete inverse `REFINED_FROM` edge. The two lineage directions
remain present now; no lineage mutation was attempted.
