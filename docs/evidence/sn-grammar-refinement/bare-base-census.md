# Bare-base census — accepted live identities that are single bare tokens

**Node:** `n-sgr-how-many-accepted-names-are-bare-bases`
**Plan:** imas-codex `sn-grammar-refinement` §9, followup `f-sgr-anchor-cannot-clear-the-name-gate`
**Question:** count the accepted live identities whose name is a bare single-token base, and state for each whether it passed `review_name`'s `semantic_similarity_check` or reached accepted by another route.
**Base:** `69f24046b`; measured against the live graph 2026-09-07. **Report-only — no source file changed.**

## Method

Population: every `StandardName` node with `name_stage='accepted'` (2,360 at base). A name counts as a
*bare single-token base* when it parses (strict) to exactly one bare `physical_base` token and carries no
operator, qualifier, projection, locus or mechanism segment — the same criterion `semantic_gate_name_text`
(workers.py:8175) uses to decide whether the gate substitutes the vocabulary definition for the identifier.
Filtering first by `NOT id CONTAINS '_'` (a bare base is one token, so any multi-token spelling is excluded
by construction) leaves exactly the three names below; all three parse as `single-base` bare, i.e. the
underscore filter and the parse filter agree on the same three. There is no wider population hiding behind
a parse difference.

## Result — exactly three accepted bare single-token bases

| name | name_stage | origin | source_types | reviewer_model_name | reviewer_score_name | resolution_method | stored semantic_sim | HAS_STRUCTURAL_AUTHORITY | producers | live HAS_PARENT children |
|---|---|---|---|---|---|---|---|---|---|---|
| `beta` | accepted | derived | `['dd']` | `structural-inheritance` | 0.49375 | quorum_consensus | 0.8295 | yes | 0 | 3 |
| `momentum` | accepted | derived | null | `structural-inheritance` | null | null | null | yes | 1 (composed derived) | 2 |
| `vorticity` | accepted | catalog_edit | `['catalog']` | `openrouter/x-ai/grok-4.5` | 0.8625 | authoritative_escalation | 0.69865 | no | 1 | 6 |

Live children by name:

- `beta` → `normalized_toroidal_beta` (reviewed), `poloidal_beta` (accepted), `toroidal_beta` (accepted)
- `momentum` → `plasma_momentum` (accepted), `radial_momentum` (reviewed)
- `vorticity` → `parallel_vorticity`, `poloidal_vorticity`, `radial_vorticity`, `toroidal_vorticity`, `vertical_vorticity` (all accepted) and `ratio_of_vorticity_to_major_radius` (reviewed)

## Route per name — did `semantic_similarity_check` admit it?

The gate under study: `review_name`'s `process_review_name_batch` computes
`semantic_similarity_check(semantic_gate_name_text(sn_id), description)` and, when the score is below
`SEMANTIC_SIM_CRITICAL = 0.55` (defaults.py:27), skips the LLM chain and persists a synthetic
`SEMANTIC_SIM_SYNTHETIC_SCORE = 0.30` (defaults.py:35) with `review_quorum_shortfall`, resolution method
`semantic_similarity_gate` (defaults.py:24). Acceptance therefore requires either the gate to clear and a
winning LLM quorum, or a non-review route.

### `vorticity` — PASSED the gate on merit

- No `semantic_similarity_gate` row exists in its `HAS_REVIEW` history; the stored
  `semantic_sim = 0.69865` (> 0.55) is a genuine gate-clear.
- Real LLM review on the name axis (2026-08-23/24): grok-4.5, gpt-5.6-luna, sonnet-5 and gpt-5.5 groups,
  final winning resolution `authoritative_escalation`, stored `reviewer_score_name = 0.8625`,
  `review_quorum_shortfall = null`.
- `origin = catalog_edit`, DD-backed by `dd:mhd/ggd/vorticity/values`. No structural authority.
- **Conclusion: admitted through the gate itself — the counter-example to "the gate rejects every bare
  token by construction."** A single-token name that is a *defined, sourced, self-describing* quantity
  cleared the name-vs-description cosine even at the pre-definition-coupling gate.

### `beta` — NOT passed; accepted by the structural derived-parent route

- Five consecutive pre-definition-coupling review runs each produced a synthetic gate row that withheld
  the chain: `sim=0.504` (2026-07-22), `sim=0.524` (2026-07-23), then `sim=0.488` on 2026-08-11,
  2026-08-25, 2026-09-02 and 2026-09-05T14:13:20. (Two earlier real groups on 2026-07-22/23 — sonnet-5
  and grok-4.5 — landed at quorum scores 0.575/0.4875, still near or below floor.)
- After the definitions-coupled gate landed (2026-09-05T20:24) the 2026-09-05T21:22 review produced NO
  gate row and two real reviewers — grok-4.5 0.4875 (primary) and gpt-5.6-luna 0.5 (quorum_consensus),
  i.e. `reviewer_score_name = 0.49375`, **below the 0.55 floor and below acceptance**.
- It is now `name_stage='accepted'` with `reviewer_model_name='structural-inheritance'` and a
  `HAS_STRUCTURAL_AUTHORITY` edge: admitted by `structural_accept_derived_parents`
  (graph_ops.py, `_structural_accept_route` → `source_free`: zero producers, origin absent at admission,
  live children) which stamps `origin='derived'` and accepts a parent without ever scoring the name.
- Stored `semantic_sim = 0.8295` is the *definitions-coupled* score (measured 0.4877 → 0.8295 by the
  definition-coupling gate node); it would now clear 0.55 if re-scored, but that is not the route by which
  `beta` became accepted.
- **Conclusion: the gate did NOT admit `beta`; the pre-existing structural-parent route did, overriding a
  sub-floor real quorum name score.**

### `momentum` — NOT passed; accepted by the structural derived-parent route

- No `reviewer_score_name`, no `semantic_sim`, no `review_resolution_method`, no gate row, no real review —
  the name was never name-reviewed at all.
- `reviewer_model_name = 'structural-inheritance'`, `HAS_STRUCTURAL_AUTHORITY`, `origin = derived`,
  sole producer is the composed derived source `derived:momentum` (`status=composed`).
- **Conclusion: reached accepted with the gate never run — structural inheritance with zero name-axis
  scoring.**

## The deterministic-parent skip, and whether a bare base with children qualifies under it today

The follow-up's "deterministic-parent skip already at workers.py:8197-8206" is the `is_derived` branch of
`process_review_name_batch` — `is_derived = item.get("origin") == "derived"` (workers.py:8368 in the
current tree; the branch spans ~8368-8410). Its mechanism: when `origin == 'derived'`, the review replaces
the general name-vs-description `semantic_similarity_check` (0.55 floor) with the dedicated
desc-name-sim gate (`desc_name_similarity`) that routes to `REFINE_DOCS` rather than withholding the
name. Derived parents are additionally never chosen for name review at all: `seed_parent_sources`
(graph_ops.py:4430) writes `name_stage='accepted'` directly and stamps `origin='deterministic'`.

**Does a bare base carrying `HAS_PARENT` children qualify under it today?** Only if its origin scalar is
`'derived'`. `HAS_PARENT` children are necessary but not sufficient — the discriminator is the origin
scalar, not the edge. `beta` and `momentum` carry `origin='derived'` today and therefore qualify;
`vorticity` (`origin='catalog_edit'`) does not. Beta's origin was **not** derived at the time of the
follow-up (2026-09-05: null, with `source_types=['dd']`); it became `'derived'` only when the structural
`source_free` promotion admitted it on 2026-09-06. So a bare base with children but an origin of null /
`dd` / `catalog_edit` is excluded from the skip today, exactly as `beta` itself was on 2026-09-05.

## What this decides

- **"Many accepted bare bases?"** No — three of 2,360 accepted identities. Not a large gate-bypassed
  population.
- **"Is the gate already bypassed for them?"** For two of the three (`beta`, `momentum`) yes, but by the
  *existing* structural derived-parent route, which is a deliberate design: a derived/source-free parent
  has nothing to be name-reviewed against, so it is accepted structurally and the gate is never run.
  `vorticity` proves the gate *can* admit a bare base on merit and does so without any bypass.
- **`beta` is therefore NOT the first accepted bare base** — it is the third, and it entered through the
  same structural route that already admitted `momentum`. An exemption keyed on "bare base carrying
  HAS_PARENT children" would be a widening of an existing mechanism, not a genuinely new admission path.
- **One hazard in the widening:** children-counting alone would also capture `vorticity` (six children,
  gate-passed on merit, producer-backed, `catalog_edit` origin) — a name that needs no exemption. Any
  family-anchor skip must inherit the structural route's discriminator (no live producing source /
  source-free, or the derived origin) rather than keying on HAS_PARENT children alone, or it exempts names
  that would and do pass the gate on merit.

## Evidence inputs for writeback

- Census population: 2,360 accepted live names; exactly 3 bare single-token bases (`beta`, `momentum`,
  `vorticity`) by both underscore- and parse-based filters.
- Gate constants: `SEMANTIC_SIM_CRITICAL = 0.55`, `SEMANTIC_SIM_SYNTHETIC_SCORE = 0.30`,
  `SEMANTIC_SIM_GATE_RESOLUTION_METHOD = 'semantic_similarity_gate'` (defaults.py:24/27/35).
- Definitions-coupling substitution: `semantic_gate_name_text` (workers.py:8175); deterministic-parent
  skip: `is_derived` (workers.py:8368); structural accept: `structural_accept_derived_parents` with
  `_structural_accept_route` `source_free` (graph_ops.py:25632, 25650).
- `beta` stored `semantic_sim` 0.8295 is the definitions-coupled score; its last real quorum name score
  (0.49375) is below the 0.55 floor; it was accepted structurally.
- Query artifacts retained under the run directory (`census2.py`, `census3.py`, `reviews.py`, per-name
  review records and authority/source dumps).

## Follow-ons (out of scope)

- None beyond the plan's existing decision surface: whether to widen the family-anchor skip keyed on the
  structural route's source-free discriminator; the beta family provenance/docs cleanup already owned by
  the §9 beta series.
