# Compose-stage prompt audit

## Scope and authorities

This audit covers exactly nine prompt files under `imas_codex/llm/prompts/sn/`.
The live package pin is `imas-standard-names==0.9.0`; the editable package used
for the audit was at `2ffba62512c07b7cc09d3ca0c1f435d9f108199b` with a clean tree.
Authorities were applied in this order:

1. `imas_standard_names/grammar/specification.yml`, its vocabulary YAML files,
   and `imas_standard_names/value_provenance.py` in the pinned package.
2. `imas_codex/core/AGENTS.md`.
3. `imas_codex/graph/AGENTS.md`.
4. The repository root `AGENTS.md`.

The governing grammar is closed and data-driven. The shared
`sn/_grammar_reference.md` renders `closed_vocab_full` and the full operator
registry from `get_grammar_context()`; it is not a second vocabulary.

Quantitative result: **9 of 9 files assessed; 8 corrected; 1 unchanged**.
No prompt hard-codes either vocabulary admission currently in flight. Prompts
that include `sn/_grammar_reference.md` do enumerate the installed vocabulary
exhaustively at render time, so they need a dependency/context refresh after
the event-instant and `signal_to_noise_ratio` admissions are released. Adding
either token directly to these prompts would create a competing vocabulary and
was deliberately not done.

## Cross-cutting disagreements and authority evidence

| Prompt assertion before correction | Higher-precedence authority | Disposition |
|---|---|---|
| `Provenance qualifiers (measured, reconstructed, simulated) may appear ONLY when they distinguish genuinely different physical quantities` (`generate_name_system.md`) | `value_provenance.py:6-12`: estimator provenance is source-to-name relationship metadata; measured, reconstructed, and reference values collapse to the base name. | Replaced with an unconditional collapse rule for the controlled provenance vocabulary. |
| `Generic fitting/uncertainty quantities (chi_squared, fitting_weight, residual) are standalone standard names` (`generate_name_system.md`) | `graph/AGENTS.md:34-36` defines `fit_artifact` as a DD node category; the compose eligibility policy excludes pure fit roles. `operators.yml:248-266` defines deterministic uncertainty wrappers for error companions. | Pure fit roles now skip; error companions use registered wrappers around the parent identity. |
| `passive_loop_current` is the correction for `current_from_passive_loop` (`generate_name_dd.md`, `generate_name_dd_names.md`) | `specification.yml:41-59`: new hardware/entity authoring uses postfix `of_<entity>`; the device-prefix form remains parse compatibility only. | Corrected to `current_of_passive_loop`. |
| A rate name may use `rate_of_change_of_` (`generate_name_dd.md`, `generate_name_dd_names.md`) | The injected operator registry contains `tendency`, `time_derivative`, and `change_in`; `generate_name_system.md` already states that `rate_of_change_of_` is not an ISN operator. | Removed the unregistered form from the allowed choices. |
| `physical_base` is a free-form snake-case token (`generate_name_signals.md`) | `core/AGENTS.md:171-209` requires all segments and tokens to come from `get_grammar_context()`; `specification.yml:85-101` defines the canonical closed-segment pattern. | Replaced the obsolete surface with current IR fields and injected the live grammar reference. |
| Shape parameters may be bare or treated like averages in examples | `specification.yml:131-140`: elongation and the other shape parameters require an explicit reference surface. | Every compact variant now states that triangularity, elongation, and squareness require a surface locus. |
| Error fields should get independently invented uncertainty names | `operators.yml:248-266`: upper/lower uncertainty and uncertainty index are registered deterministic operators applied outside the parent name. | Prompts now say to wrap the parent identity and never coin a per-error base. |

## Per-file assessment

### `generate_name_system.md`

- **Correctness.** The dynamic grammar reference, segment IR, source-axis
  fidelity, generic-base qualification, and surface-explicit shape rule agree
  with the package. Four contradictions were corrected: conditional use of
  estimator provenance, standalone naming of fit roles, prose units for mixed
  coordinate components, and a worked example whose description said “energy
  flux” while its IR composed only `ion_energy_due_to_collisions`. The example
  now composes `ion_energy_flux_due_to_collisions` from channel `energy` plus
  base `flux`.
- **Coverage.** Complete closed vocabulary and operators arrive through
  `sn/_grammar_reference.md`; DD unit and enriched source data arrive in the
  paired user prompt. No static token list in this file is authoritative.
- **Grounding.** This static prompt holds rules only. It explicitly delegates
  per-source grounding to the exact authoritative binding; the paired DD or
  signal prompt supplies the rich description and unit.
- **Self-descriptiveness.** Already explicit: generic bases require a subject,
  component, or locus, and source-stated carrier/surface distinctions cannot be
  dropped. No structural rewrite was needed.

### `generate_name_dd.md`

- **Correctness.** Corrected the passive-loop authoring form, bare process
  adjectives, the unregistered rate form, `diamagnetic` incorrectly listed as
  a projection, a compound `minor_radius` IR example, and vague
  `measurement_position` guidance. Error companions now cite the three
  registered uncertainty operators. The existing measured/reconstructed/
  reference estimator block already matched `value_provenance.py` and was
  retained.
- **Coverage.** The paired system prompt supplies the complete grammar; each
  item supplies unit, rich description, terse source documentation, ancestors,
  neighbours, identifiers, prior review, and related paths when available.
- **Grounding.** `item.description` is produced rich-first in
  `workers._enrich_batch_items`; the prompt now labels it **PRIMARY GROUNDING**
  and labels `item.documentation` as the secondary terse DD clause.
- **Self-descriptiveness.** Added the explicit rule that a generic DD leaf is
  provenance, not permission for a generic Standard Name, and directs the model
  to recover carrier, surface, subject, and process from the enriched meaning.

### `generate_name_dd_names.md`

- **Correctness.** The prompt falsely claimed that name-only mode intentionally
  omitted rich per-item context even though its own Jinja blocks render that
  context. It also demonstrated suffix projection names, device-prefix
  authoring, and `rate_of_change_of_`. Those statements were corrected without
  changing the batching structure. Estimator collapse was already correct.
- **Coverage.** The installed vocabulary comes from the paired system prompt;
  unit, enriched description, neighbours, identifiers, clusters, review
  feedback, and reference exemplars remain available. The prompt no longer
  tells the model that later review can recover missing identity semantics.
- **Grounding.** The rich `item.description` is now labeled primary. Terse DD
  text is not rendered as a competing description in this mode.
- **Self-descriptiveness.** Added the same generic-leaf prohibition as the full
  DD prompt and preserved the carrier/owner rules already present.

### `generate_name_dd_tool_calling.md`

- **Correctness.** This research variant previously said only “follow
  controlled vocabulary” and supplied examples, but no authoritative registry.
  It now includes the live grammar reference and states current IR,
  provenance-collapse, surface-shape, and error-wrapper rules.
- **Coverage.** The prompt contract now requires every `paths_block` entry to
  carry both DD-authoritative unit and the rich enriched description. The three
  optional lookup tools supplement that evidence; they do not replace it.
- **Grounding.** Added rich-description-first precedence, terse-DD and
  deterministic-placeholder prohibitions, and a fail-visible instruction when
  required grounding is absent.
- **Self-descriptiveness.** Added the generic path/leaf prohibition. This file
  remains a research-harness variant; its output envelope was not restructured.

### `generate_name_signals.md`

- **Correctness.** Replaced the obsolete free-form `physical_base` grammar and
  old flat-field examples with the current IR and authoritative composer. Added
  the settled provenance, surface-shape, and error-wrapper rules.
- **Coverage.** The full live registry is now included. Each signal supplies an
  authoritative unit, enriched signal description, domain, existing-name
  candidates, and related DD evidence where available.
- **Grounding.** The enriched signal description is explicitly primary;
  similar DD paths are supporting evidence and cannot override it.
- **Self-descriptiveness.** A generic facility signal identifier no longer
  licenses a generic name. The prompt requires every stated carrier, surface,
  subject, projection, and process to survive into identity.

### `generate_docs_system.md`

- **Correctness.** Replaced “DD path documentation” as the primary source with
  the rich enriched source description. Pure fit artifacts no longer receive a
  documentation template; estimator facets document their shared base
  quantity. The COCOS guard now keys on supplied transformation metadata rather
  than pretending COCOS is a name segment.
- **Coverage.** The system prompt receives the live grammar, documentation
  format, examples, coordinate conventions, and the user prompt’s unit, family,
  source, ancestor, and peer context.
- **Grounding.** Rich source meaning plus established physics is primary; terse
  DD text is fallback. A deterministic-parent placeholder is explicitly not
  content, and a derived parent grounds only on real accepted children.
- **Self-descriptiveness.** The accepted Standard Name is fixed input and its
  explicit semantic segments outrank a generic source path; documentation may
  deepen that identity but may not silently generalize it.

### `generate_docs_user.md`

- **Correctness.** Existing fixed-name, DD-authoritative unit, family, and COCOS
  instructions agree with the higher authorities. The output schema’s
  contradictory “1-3 sentences, 500 chars” example was aligned with the actual
  two-sentence, 250-character constraint below it.
- **Coverage.** Supplies accepted name, unit, kind, domain, reviewers, family,
  derived children, source paths, rich source descriptions, ancestors,
  neighbours, clusters, and existing names.
- **Grounding.** Renamed the source block to “Enriched Source Descriptions” and
  records how `dd_source_docs.documentation` is populated rich-first. Added an
  explicit ban on copying the deterministic-parent placeholder.
- **Self-descriptiveness.** Added that a generic leaf or parent label cannot
  generalize the accepted identity; the accepted name’s segments and enriched
  source meaning control.

### `fanout_propose.md`

- **Correctness.** It asserts no grammar production rules and its four named
  functions match the typed fan-out catalog. No function or envelope changed.
- **Coverage.** This prompt selects context queries rather than proposing a
  name. Its user message supplies candidate name, path, description, domain,
  scope, and reviewer excerpt; unit and full vocabulary are not inputs needed
  to choose a bounded evidence query.
- **Grounding.** Added description-first query guidance and an explicit rule to
  ignore deterministic-parent placeholder text.
- **Self-descriptiveness.** Added that generic path text cannot drive a generic
  query; queries should seek the stated carrier, surface, subject, or process.

### `fanout_evidence_block.md`

- **Correctness.** No name or documentation rule is asserted. It only renders
  function id, arguments, hit label, and optional score, matching the Python
  renderer’s informational shape.
- **Coverage.** Not an LLM task prompt and asks for no name; unit, vocabulary,
  and source-description requirements are therefore not applicable.
- **Grounding.** It faithfully renders the bounded evidence returned by the
  typed runner and introduces no DD interpretation.
- **Self-descriptiveness.** Not applicable to this renderer. **No change.**

## Resulting invariant

All name-producing variants now receive or require the installed closed
vocabulary, authoritative unit, and rich source meaning; none permits a generic
path to erase source semantics. Measured/reconstructed/reference estimator
facets collapse to one identity, shape parameters retain their surface, and
error companions derive from the parent with registered uncertainty operators.
Documentation prompts use the same rich-first grounding and never treat a
deterministic-parent placeholder as physics.
