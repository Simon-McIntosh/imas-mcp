# Review-prompt grammar audit

## Result

All thirteen review-stage prompt files were inspected against the installed
ISN grammar context, the Standard Names lifecycle guidance, and the repository
review rules. Thirteen files changed. The name-review routes receive the full
dynamic closed vocabulary through `sn/_grammar_reference.md`: 20 registries,
including 177 `physical_base` tokens, plus the complete operator registry.
They do not contain an exhaustive copied base list, so an installed grammar
release refreshes the prompt without a prompt edit. In particular, this audit
does not hard-code either pending vocabulary admission.

The changes remove three authoritative-rule contradictions, make each scoring
rubric use anchored bands instead of taste-level point differences, and state
where a specialized reviewer deliberately lacks grammar or parent evidence.

## Authority comparisons

| Reviewer assertion before this audit | Governing authority | Correction |
|---|---|---|
| `review.md` and `review_names.md` required every prefix operator to use `_of_`. | The grammar reference says an operator's join is grammar-specific and explicitly gives `per_toroidal_mode_X` as a registered bare transformation. | Reviewers now require the installed grammar's canonical join: `_of_` for scoped operators and bare form where registered. |
| `review_names_system.md` called `poloidal_flux` a valid registered compound. | The grammar reference says `poloidal_flux` is not a name and must not be emitted or endorsed. | The prompt now rejects `poloidal_flux` and directs reviewers to the dynamic projection and complete registry. |
| The legacy full and docs-only rubrics required variables to be defined "with units" while their own prose rule prohibited units in documentation. | Unit is DD-authoritative structured metadata; documentation guidance prohibits restating it in prose. | The rubrics now require a variable's physical identity, not a prose unit. |
| The ordinary docs rubric said not to re-litigate a previously accepted name, then instructed empty-neighbour cases to score grammar. | The docs stage follows accepted name review; derived parents skip name review and are structurally accepted. | Ordinary docs review now evaluates supplied evidence, physics, and prose only; the parent route marks missing children as unscoped rather than re-reviewing the grammar peel. |

## What each reviewer actually receives

`Yes` means the rendered model message contains the input when the stated
template condition is met. `No — out of scope` is intentional and is now
explicit in the prompt so the reviewer does not invent missing evidence.

| File | Full closed vocabulary | Unit | Enriched source description | Parent context | Grammar error text | Finding and change |
|---|---|---|---|---|---|---|
| `review.md` | Yes, shared grammar include | Yes | Yes, when candidate carries DD sources | No | Yes, `validation_issues` | Corrected operator-join rule, provenance/error boundaries, unit contradiction, and score calibration. |
| `review_names.md` | Yes, shared grammar include | Yes | Yes, when candidate carries DD sources | No | Yes, `validation_issues` | Corrected atomic-compound claim, operator-join rule, provenance/error boundaries, and score calibration. |
| `review_names_system.md` | Yes, shared grammar include | Paired user message | Paired user message | Paired user message when available | Paired user message | Corrected illegal `poloidal_flux` admission, clarified sign/provenance metadata, and anchored bands. |
| `review_names_user.md` | Paired system message | Yes | Yes: pinned DD definition first, enriched description separately | Yes, exact parent-array context when present | Yes, relabeled as deterministic grammar error text | Already supplied the strongest source grounding; made the edge-only provenance rule explicit. |
| `review_docs.md` | No — docs-only | Yes | Yes, DD definitions when present | Yes for derived children when present | Yes, but not a name-review input | Corrected unit contradiction and added anchored bands; name grammar remains out of scope. |
| `review_docs_system.md` | No — docs-only | Paired user message | Paired user message | Paired user message only through family context | No | Removed grammar re-litigation after accepted name review, removed the conflicting unit-prose comparison, and anchored bands. |
| `review_docs_user.md` | No — docs-only | Yes | Yes, DD description/documentation when present | No; the parent route owns it | No | Made those limits explicit, preventing neighbour context from being mistaken for a parent obligation. |
| `review_docs_parent_system.md` | No — parent docs-only | Paired user message | Child descriptions, not DD enrichment | Yes, live children are primary | No | Parent handling was already structurally correct; added anchored bands that do not penalize a correct abstraction for missing leaf detail. |
| `review_docs_parent_user.md` | No — parent docs-only | Yes | Child descriptions | Yes, live children | No | Replaced the fallback grammar review with an explicit unscoped-parent signal. |
| `review_description_system.md` | No — description-only | Paired user message | No | No | No | Made missing evidence explicit and anchored bands so a source path cannot become invented physics or grammar criticism. |
| `review_description_user.md` | No — description-only | Yes | No | No | No | Made all omitted review inputs explicit; it continues to score only description-to-name consistency. |
| `judge_physics_correctness_system.md` | No — physics-only | Paired user message | Source documentation, not enriched description | No | No | Declared the deliberate grammar-evidence boundary and restricted `valid` to an obvious malformed surface form. |
| `judge_physics_correctness_user.md` | No — physics-only | Yes | No | No | No | Made missing grammar and parent evidence explicit so an unfamiliar legal base is not rejected by taste. |

## Calibration and parent conclusions

Seven numeric score rubrics now use the same evidence bands: 20 for fully
supported, 15 for a minor identified deficiency, 10 for a material bounded
deficiency, 5 for a wrong observable or grammar, and 0 for a fundamentally
unusable result. Explicit caps retain precedence. This prevents a single
reviewer from turning an unsupported preference into a threshold-changing
one-point difference.

Derived parents remain a separate documentation route. Their name is a
structurally accepted grammar peel, their children are the reference for
generalization, and absent child context is an unscoped review condition — not
a reason to judge the parent as though it were a leaf name.

## Remaining boundary

The physics-only judge and the description-only reviewer intentionally do not
receive a full vocabulary or parser result. They no longer claim to perform
that task. Grammar review remains with the name-review route, which receives
the complete grammar reference and the candidate's parser projection and error
text when available.
