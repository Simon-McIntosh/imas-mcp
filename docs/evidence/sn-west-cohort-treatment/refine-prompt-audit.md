# Refine, self-refine, parent-enrichment and notes prompts — audit against their authorities

Scope: the twelve prompt files listed below, assessed against, in precedence
order, (1) the live ISN grammar and controlled vocabulary read from the
installed `imas-standard-names` package, (2) `imas_codex/core/AGENTS.md`,
(3) the root `AGENTS.md`. Where a prompt asserted a rule the grammar does not
hold, the grammar won.

Grammar read at audit time: `imas_standard_names 0.8.1.dev32+g12b557363`,
via `build_compose_context()`. 177 `physical_base` tokens, 35
`geometric_base`, 114 `qualifier`, 82 `process`, 105 `device`, 74 `geometry`,
74 `path`, 56 `subject`, 23 `component`, 13 `zone`, 7 `region`, 7
`population`, 5 `orbit`, 4 `channel`, 2 `aggregation`, 2 `state`, 1
`section_plane`, 1 `geometry_representation`. IR groups: `operators`,
`projection`, `qualifiers`, `base`, `locus`, `mechanism`. Locus relation
matrix: `entity → {of}`, `position → {along, at, of}`, `region → {over}`,
`geometry → {of}`.

Every prompt below was re-rendered after editing and produces text
(byte counts in the manifest); no template raises.

---

## The four questions, answered per file

### `refine_name_system.md` — corrected

**Correctness — two asserted rules the grammar contradicts.**

The prompt named six tokens as unregistered and told the refiner to surface a
gap for them. Four of the six are registered right now:

| Prompt asserted | Live grammar |
|---|---|
| "an unregistered `angle` … base" | `angle` ∈ `physical_base` |
| "an unregistered `phase` … base" | `phase` ∈ `physical_base` |
| "an unregistered `length` … base" | `length` ∈ `physical_base` |
| "an unregistered `extent` base" | `extent` ∈ `geometric_base` |
| "`phase_shift`" | unregistered — correct |
| "an unregistered qualifier such as `perturbation`" | unregistered — correct |

Surfacing a gap for a registered token is the exact failure the shared grammar
reference calls "the most common pipeline error is reporting false vocabulary
gaps", so the prompt was instructing the refiner to commit it. The second
assertion listed non-nameable concepts by token — "a time coordinate /
timestamp (`time`, simulation begin/end time), signal-chain timing (`latency`,
`delay`, acquisition period)". `time`, `delay`, `period` and `duration` are all
registered `physical_base` tokens; only `latency` and `timestamp` are not.

Both were hard-coded token lists inside a prompt, which is the drift hazard
`imas_codex/core/AGENTS.md` bans for `.py` files ("A codex `.py` file must
never hold a literal list/set/dict of ISN grammar tokens — that duplicate
silently drifts the instant ISN adds/renames/removes one"). The same argument
holds for a prompt: `signal_to_noise_ratio`, one of the two admissions the
plan records as in flight, **is already registered** in the installed grammar,
so a hard-coded list is stale within one release.

Fixed by (a) restating the non-nameable rule as a test on the **role of the
source** — a bare time or index axis, a counter, a record identifier, pure
metadata — rather than on a token, and noting explicitly that measured
elapsed times, delays, periods and durations are ordinary quantities with
registered bases; and (b) replacing the "unregistered" list with an
instruction to look the token up in the registry the prompt itself renders,
because that registry is generated from the installed grammar on every call
and the vocabulary grows. No token is now hard-coded, so the event-instant
admission (`instant` and `event` are both unregistered today) will appear on
its own when it lands.

The prompt's one other grammar claim was checked and **holds**: the
lexicalised-geometry exception naming `first_local_tangential_coordinate` /
`second_local_tangential_coordinate` as atomic is correct — both are
registered `geometry_carriers`, and neither `first_local_tangential` nor
`second_local_tangential` exists as a projection axis, so decomposing would
produce an unregistered axis exactly as the prompt says.

**Drift — was unbounded, now bounded.** The prompt contained no statement that
a refinement must preserve the quantity. The free self-refine prompt has that
rule ("Never change the physical meaning of the quantity — you are refining
the *label*") but the paid refine, which rewrites from scratch under reviewer
pressure, did not. Added: the source path and unit are fixed, the successor
must denote the same physical quantity, and a successor naming a different
quantity, owner, carrier or locus is drift that silently detaches the name
from its source.

**Idempotence — was churn-positive, now has a no-op.** Nothing licensed
returning a name unchanged. Added: when the reviewer's objection identifies no
defect in the name — it lands on the description, on documentation, or on a
dimension a name cannot carry — the name is already right; say so in `reason`
rather than rotating a synonym to look responsive.

**Output identity — the measured failure mode was unaddressed.** Added a
section stating what constrains the successor's identity: it must be free (an
occupied identity cannot be taken, because migrating sources between names is
a fold decision, not a rewrite), it must not be a name already in the chain,
and it must parse and re-compose to canonical form. See the collision finding
under `refine_name_user.md`, which is the same defect on the data side.

### `refine_name_user.md` — corrected

**Coverage — the enriched source description never reached the prompt.** This
is the largest single finding. `_enrich_dd_path_context`
(`imas_codex/standard_names/workers.py`) populates four fields on the item
before this template renders: `dd_description` (the rich enriched description),
`dd_documentation` (the terse DD-XML string), `dd_units`, and
`dd_parent_description`. **The template rendered none of them.** What it
rendered as "Description" was `item.description` — which
`claim_refine_name_batch` returns as `sn.description`, the *candidate's own*
one-line description, not the source's meaning.

So the paid refiner was rewriting a name while being shown only the previous
attempt's own gloss of it, with the authoritative source meaning fetched,
attached to the item, and discarded at render. Fixed: a "Source meaning"
section now renders `dd_description` first and falls back to
`dd_documentation` only when no enriched description exists, with the terse
string shown separately and labelled secondary when both are present — the
rich-first ordering the design fact requires. `dd_units` now takes precedence
in the unit line and `dd_parent_description` renders as parent context.

**Coverage — the parent-context block was dead code.** The template guarded on
`item.parent_path` and rendered `item.parent_description`. Neither name is ever
set: the claim returns no `parent_path`, and the enricher writes
`dd_parent_description`. The block could not fire on any call. Replaced with
the field that is actually populated.

**Coverage — a placeholder was rendered as content.** `item.description` was
rendered unguarded, so a derived parent still holding
`DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER` — the literal string
`(deterministic parent — description pending LLM enrichment)` — was presented
to the refiner as the quantity's description. Now suppressed, and the line is
relabelled so it cannot be mistaken for the source's meaning.

**Output identity — the collision loop was in the prompt.** `RefineStopReason`
documents `successor_collision` as: "the proposal is reproduced on every
attempt, across models, so the remaining budget is not spent", with "the
occupied identity … recorded on `refine_collision_name`". The claim returns
both `refine_stop_reason` and `refine_collision_name`
(`graph_ops.py:17408-17409`) and the worker sends `refine_stop_reason` to a log
line only (`workers.py:6856`); **neither reached the prompt.** The refiner was
never told which identity it had just been refused, which is a sufficient
explanation for a proposal reproducing across models. Added a block that names
the refused identity, forbids re-proposing it or a cosmetic variant, and states
the correct outcome when that identity really is the only right name: report it
in `reason` as a fold decision for an operator.

**Correctness — the locus rule was narrower than the matrix.** The prompt gave
two relations, `of` and `at`. The live matrix admits four across four locus
types, including `along` for `position` and `over` for `region`, and the shared
decomposition checklist already requires `over` for a `<base>_<region>` form.
Replaced the hard-coded pair with a rendering of
`grammar.locus_relation_matrix`, so the prompt cannot drift from the matrix
again, plus one sentence of semantics for choosing among the relations a type
admits.

**Idempotence — "do not repeat" forced motion.** The rule "Do not **not**
repeat any name that appears in the refinement history" admitted no exception,
so a refiner facing a correct name and an objection it cannot act on had to
emit something new. That is a synonym rotation mandated by the prompt, and it
contradicted the system prompt's own instruction not to rotate near-synonyms.
Reconciled: the rule stands, with one stated exception for deliberately
returning a correct name and saying so.

### `refine_docs_system.md` — corrected

Fifteen lines, all of them format guidance; it asserted no grammar rule, so
nothing to correct for correctness. Three gaps added, each from a settled
design fact the file did not carry: what refinement may not change (the name
and the quantity are fixed; documentation that drifts onto a neighbouring
quantity is worse than the text it replaced); the grounding rule (ground on the
enriched source description, treat a terse DD string as a secondary check, and
treat a pending placeholder as absent rather than paraphrasing it); and the
idempotence rule (address the objection and carry untouched what already
scored well — reworking sound prose to show effort loses dimensions the entry
had already won).

### `refine_docs_user.md` — corrected

**Coverage — a HARD rule cited context the template never showed.** The
locus-defining cross-link rule says to use "the defining quantity the injected
locus context supplies" and — correctly — forbids guessing a mapping. The
worker does inject it (`prompt_context["locus_context"] = locus_context_for(sn_id)`,
`workers.py`), and `generate_docs_system.md` renders it, but neither
`refine_docs_system.md` nor `refine_docs_user.md` did. A rule that requires
unrendered context and forbids guessing cannot be satisfied, so the cross-link
could never be added on the refine path. Added the block, worded to match the
generate path so the two do not diverge.

**Coverage — the same placeholder leak.** `prompt_context["description"] =
item.get("description", "")` is unguarded, so a derived parent at the
placeholder rendered it as its one-line description. Now suppressed.

**Drift and idempotence.** The task line asked for documentation addressing the
lowest-scoring dimensions with no statement that the subject is fixed. Added:
same quantity, change what the objection reaches, carry the rest through
unchanged. The existing "Do not regress on dimensions that already scored
well" was kept and is now stated at the task level too, where it binds the
whole rewrite rather than one output field.

The derived-parent branch was checked and is **correct as written**: it is a
distinct parent-aware path, it grounds on the children, and the children query
already maps a placeholder child description to null before it reaches the
template.

### `self_refine_system.md` — corrected

**Correctness — the hard-coded segment order was wrong in three ways.** The
prompt stated the canonical order as
`[subject_][physical_base|geometric_base][_component][_position][_process][_object]`.
The grammar's order is
`[operator_of_] [component_] [qualifier]* [subject_] physical_base [_at_position | _of_object | _over_region] [_due_to_process]`:

| Prompt | Live grammar |
|---|---|
| `_component` **after** the base | `component_` is a **prefix** — axiom 4 "Prefix projection", and `ion_rotation_frequency_toroidal → toroidal_ion_rotation_frequency` is listed as a rejected trailing-component form |
| `_process` before `_object` | locus (`_at_`/`_of_`/`_over_`) precedes `_due_to_process` |
| omits `operator`, `qualifier`, `region` | all three are grammar segments; `qualifier` alone carries 114 tokens |

A self-refine step whose stated canonical order puts the component in a
position the grammar rejects can only move a name away from canonical form.
Fixed by deleting the hard-coded order and including the shared
`sn/_grammar_reference.md`, which renders the live order.

**Coverage — no closed vocabulary was supplied.** The prompt asserted the
grammar is closed on every segment and then asked the model to use "tokens that
already appear in the original name's segments or are obviously-registered
siblings" — i.e. to guess the registry from memory. The worker already passes
`build_compose_context()` into both renders, so `closed_vocab_full`,
`operators_full` and `grammar` were available and unused. The include now
renders the full registry (verified: `signal_to_noise_ratio` appears in the
rendered text), so no token is asserted from memory and a newly admitted base
appears without a prompt edit.

**Drift.** Already correct and preserved verbatim: "Never change the physical
meaning of the quantity — you are refining the *label*, not re-deciding *what*
is measured. The source DD path and unit are authoritative and fixed."

**Improve-only, restated as the settled design fact.** The prompt said it never
rejects, quarantines or blanks — true but incomplete. Added that this step
casts no vote, emits no score and counts toward nothing, that acceptance is
decided later by an independent quorum which has not seen the candidate, and
that the model should not reason about whether its own output would pass: a
judgement of one's own output is not evidence about its quality, which is the
reason the step is confined to improving the label.

**Idempotence.** The existing "return both unchanged … Do not churn on cosmetic
preferences" was kept.

### `self_refine_user.md` — corrected

**Coverage — terse-first grounding.** The source block read
`{% if dd_context.documentation %}` first and fell back to
`dd_context.description` only when the terse DD string was absent — precisely
inverted against the design fact that refine grounds on the rich enriched
description and never on the terse DD documentation string. Both fields are
present on the `source_item` the worker passes as `dd_context`. Now the
enriched description is primary, the terse string renders separately and
labelled secondary when it differs, and the terse-only case is labelled as
such. Verified by render order in the emitted text.

**Coverage — placeholder guard** added on `dd_context.description` for the same
reason as the two refine prompts.

**Idempotence.** Added one sentence making `changed: false` the expected answer
for a sound candidate, and naming the failure mode explicitly: do not look for
something to change because you were asked to look, since an unnecessary
rewrite is both churn and a risk to a correct name.

### `enrich_parent_system.md` — corrected

Correctness: asserts no grammar rule; grounds on children, which is right.
Drift: already bounded ("Do not change the name, unit, kind, or any identity
field"). Idempotence: not applicable — the worker claims only parents still
holding the placeholder, so the step is create-only.

Three additions. First, the settled design fact that a derived parent is
accepted **structurally** rather than name-reviewed, with the reason (a name
quorum systematically penalises an abstraction for being less specific than its
children) and the consequence that nothing written here is a verdict — the
prompt is the sanctioned structural-accept path and did not say so. Second, the
handling of a child with no description: `fetch_derived_parent_children` maps a
placeholder child description to null, so a child can legitimately arrive as a
bare name, and a missing description must not be read as a statement about the
physics. Third, two prose rules the docs authority applies to every other
description-writing prompt and this one lacked: no units in prose (the unit is
shown to the model for consistency, not for restatement) and no storage-shape
words.

The file's `sn/_grammar_reference.md` include was left in place: it writes only
a description, so the registry is not strictly needed, but its purpose is
establishable — grounding parent prose in the segment vocabulary the parent
name is built from — and deleting it is not this node's call.

### `enrich_parent_user.md` — corrected

One correction, matching the system prompt: state that a child shown without a
description contributes its name only, and add the no-units-in-prose rule. The
existing empty-children fallback was left in place; the worker releases a
childless parent before rendering, so the branch is defensive rather than dead.

### `approval_notes_system.md` / `approval_notes_user.md` — corrected

These write a git tag body, so the governing authority is the root `AGENTS.md`
rules on any message an agent authors. Two rules were missing and are now
stated: no tool self-attribution in any form (no "generated with", no
co-author trailer, no model or assistant name, no robot-emoji credit — the
authorship is the maintainer's), and no plan, sprint, phase, milestone, task or
ticket identifiers, which a reader outside the session cannot resolve and which
rot when the tracker entry closes. Also added the markdown-link rule: a bare
URL wraps mid-path where this text is read.

Grounding, honesty and concision rules were already correct and are unchanged.
The user prompt needed no change — it carries the PR description, the
conversation, the commit messages and the review delta, which is the full
evidence set the system prompt requires.

### `release_notes_system.md` / `release_notes_user.md` — corrected

Same two authority gaps as the approval notes, and they matter more here: this
prompt writes a **pull-request title and body**, the exact surface the root
`AGENTS.md` names when it records that a `🤖 Generated with [Claude Code]`
footer once required a history rewrite to scrub. The same three rules added —
no self-attribution, no internal identifiers, markdown links rather than bare
URLs.

The existing title contract (exact supplied title, facility scope, a physics
domain only for a single-domain batch, never a version, count, entry name or
enumeration) was checked against the user prompt's variables and is consistent;
the release identifier is already supplied as provenance and excluded from the
title. Unchanged.

---

## Vocabulary admissions in flight

The plan records two admissions belonging to the grammar repository: an
event-instant base and `signal_to_noise_ratio`. Measured against the installed
grammar:

- `signal_to_noise_ratio` **is already a registered `physical_base`.**
- `instant` and `event` are both unregistered in every segment.

No prompt in this scope now enumerates bases, so neither token is hard-coded
and neither needs one. Before this change, `refine_name_system.md` was the one
file that named specific tokens as unregistered — four of the six wrongly —
which is why that list was replaced with a registry lookup rather than
updated. The remaining enumeration is the rendered registry in
`sn/_grammar_reference.md` (outside this scope), which is generated from the
installed package on every call and therefore self-updating.

## Settled facts not re-opened

Provenance (`measured` / `reconstructed` / `reference`) as controlled
vocabulary and an edge property rather than a name segment; estimator facets
collapsing to one base-quantity name; surface-explicit shape parameters; a
universal error modifier rather than per-error names. No prompt in this scope
asserts anything against these, and none was edited toward them.
