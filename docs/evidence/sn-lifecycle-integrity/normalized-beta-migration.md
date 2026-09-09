# Partial normalized-beta identity revival

## Result

`normalized_toroidal_beta` is now a live, reviewed total-pressure identity. It
ends at `status='draft'` with both review axes accepted, scores 1.0 and
0.9375, 12 name reviews, seven docs reviews, the recovered 1,468-character
document, four links to `toroidal_beta`, zero old self-links, and all eight
producers intact. The defining $\beta_N$ relation is explicit.

The thermal identity's authorized status transition also applied, but exposed
a separate terminal-stage trap: `normalized_toroidal_thermal_plasma_beta` is
now `status='draft'` while `name_stage='superseded'`. The ordinary exact review
still refuses it as terminal. The thermal source therefore remains safely on
the total identity, and the inverse-lineage removal has not run.

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

## Preferred hint route and scope refusal

The docs-axis hint carried the recovered physics explicitly: normalized
toroidal beta is $\beta_N = 100\,\beta_{\mathrm{tor}}\,aB_0/I_p$, a
whole-plasma equilibrium metric built on the volume-averaged total
perpendicular pressure entering toroidal beta. It required the defining
relation and all four links to `toroidal_beta`. Its provenance reason named the
accepted 1,504-character predecessor document, its 0.93125 score, and its seven
docs-axis reviews.

The complete dry-run result was:

```text
DRY RUN sn edit: normalized_toroidal_beta  mode=hint axis=docs scope=only_self
entry=generate

Actions:
  - hint attached to 'normalized_toroidal_beta' (axis=docs)
  -  no writes performed
```

The live attach succeeded and stamped `run_id=sn-edit-20260909T181732Z`,
`edit_status='open'`, `edit_scope='only_self'`, and the exact hint and reason.
The CLI then launched its default inline `run_sn_pools` continuation. Unlike
`sn run`, the `sn edit` surface exposes no `--skip-global-maintenance` option,
and its inline helper does not forward that control. The runner entered global
maintenance before processing the scoped edit. Its output named sourceless-name
reconciliation, attachment-consistency reconciliation, the global source
ledger, and 122 missing derived-parent targets outside this node's cohort. The
runner was interrupted rather than allowed to continue outside the fence.

A bounded post-read of `normalized_toroidal_beta` found the hint intact and
resumable, with `status='draft'`, `name_stage='accepted'`,
`docs_stage='pending'`, `edit_status='open'`, zero document characters, zero
docs review edges, and no claim. No additional LLM cost was
recorded. Whether the global startup maintenance committed collateral changes
before interruption is unverified because this node is not authorized to scan
or mutate identities outside the named cohort.

This is not an isolated observation. The same inline-scope escalation was
recorded earlier the same day by the beta global-definition worker at
`run_id=sn-edit-20260909T104321Z`. The present observation at
`run_id=sn-edit-20260909T181732Z` makes two occurrences in one day. Until the
inline wrapper forwards the maintenance-skip control, `sn edit` cannot safely
complete an inline review inside an exclusive graph fence; the edit must be
attached without review and followed by a manually scoped `sn run`.

## Documentation completion

The attached hint was resumed exactly once with:

```text
sn run --scope-run-id sn-edit-20260909T181732Z \
  --docs-only --skip-global-maintenance
```

The runner reported that it was bypassing global maintenance, generated one
document, wrote two docs reviews, and accepted it at score 0.9375. That
generated text carried the defining relation but only two links to
`toroidal_beta`, so it failed the four-link content gate and activated the
authorized exact fallback.

The settled axis then admitted the prepared 1,468-character redesign:

```text
DRY RUN sn edit: normalized_toroidal_beta  mode=docs axis=docs scope=only_self
entry=review_docs

Actions:
  - docs replacement queued for 'normalized_toroidal_beta'
  -  no writes performed
```

The exact document was attached with `--stage-only` to avoid the inline-scope
defect and reviewed with an exact scope run plus `--skip-global-maintenance`.
An intermediate attachment carried a malformed predecessor-score token in its
reason; it was not hand-edited. After that exact text settled, the identical
text was reattached through `sn edit --docs --stage-only` with the correct
provenance and reviewed again. The surviving applied edit reason states the
predecessor score as 0.93125 across seven docs reviews.

The final total-pressure identity reads:

| Field | Final value |
|---|---|
| `status` | `draft` |
| `name_stage` / score / review edges | `accepted` / 1.0 / 12 |
| `docs_stage` / score / review edges | `accepted` / 0.9375 / 7 |
| Documentation length | 1,468 characters |
| `edit_status` / run | `applied` / `sn-edit-20260909T184657Z` |
| Producers | 8 |
| Claims | none |

The four source-to-final link corrections are:

| Occurrence | Before on predecessor | After on live total identity |
|---:|---|---|
| 1 | `[beta](name:normalized_toroidal_plasma_beta)` | `[toroidal_beta](name:toroidal_beta)` |
| 2 | `[beta](name:normalized_toroidal_plasma_beta)` | `[toroidal_beta](name:toroidal_beta)` |
| 3 | `[beta](name:normalized_toroidal_plasma_beta)` | `[toroidal_beta](name:toroidal_beta)` |
| 4 | `[beta](name:normalized_toroidal_plasma_beta)` | `[toroidal_beta](name:toroidal_beta)` |

The live text contains zero links to `normalized_toroidal_plasma_beta` and
retains the separate `[poloidal_beta](name:poloidal_beta)` reference. Total
provider spend through this point is USD 0.796979 of the USD 15.00 ceiling.

## Thermal status transition and terminal-stage refusal

The thermal authority contained one participant and one property mutation:

```text
normalized_toroidal_thermal_plasma_beta: {status: draft}
```

Its preview admitted one of one rows, refused zero, and reported
`would_change=1`. The exact apply returned `outcome='applied'`, `changed=1`,
one receipt row, and two persistent writes.

- Authority file SHA-256:
  `ac23e513fd10919a25551f3e3a38385ceada4085877dcc0420aa73f3d0a39db9`
- Authority payload SHA-256:
  `8ff55ce963f82cfa03f7c4a723d0b2275e787e646c6e3353ef56ff865365693f`
- Previewed and applied manifest SHA-256:
  `f279a370599d8618a2fab02deef9985d7fe0c57127bbc0b99f992e88849075e8`

The independent post-read found `status='draft'`, but the status-only mutation
correctly left `name_stage='superseded'`, its historical score 0.5875, zero
producers, `docs_stage='pending'`, and no claim. The ordinary review dry-run
then refused:

```text
Error: normalized_toroidal_thermal_plasma_beta: terminal StandardName lifecycle
```

No detach was attempted. The thermal path remains among the total identity's
eight producers, as does the MHD estimator path.

## Remaining work

The thermal identity needs a sanctioned transition of its independent
name-axis state from `superseded` to a reviewable state. The authorized status
transition alone cannot supply it: exact review rejects the terminal name
condition, `sn attach` only permits stable binding lifecycle values, and a name hint has no
producer to regenerate from, and the redesign-to-self route has already been
shown to collide with an existing identity. Moving the source before this is
resolved would create an avoidable unbound interval and cannot complete the
attach.

After the name axis can be reviewed, move only
`dd:summary/global_quantities/beta_tor_thermal_norm/value` from the total
identity to `normalized_toroidal_thermal_plasma_beta`, review its name and docs
axes independently, and prove both scores have nonzero review-edge counts.
Leave `dd:summary/global_quantities/beta_tor_norm_mhd/value` on
`normalized_toroidal_beta`.

Only after that source split should the obsolete
`normalized_toroidal_plasma_beta REFINED_FROM normalized_toroidal_beta`
direction be removed. The intended historical direction,
`normalized_toroidal_beta REFINED_FROM normalized_toroidal_plasma_beta`, stays.
Both directions remain present now; no lineage mutation was attempted.
