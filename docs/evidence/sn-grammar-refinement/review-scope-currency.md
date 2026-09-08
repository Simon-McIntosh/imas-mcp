# Does `sn run --only review --edits` scope the run?

Measured against the live graph at worktree base
`69f24046b143dce1048f82f02fd25878b2849126`, installed grammar resolved from the
editable `imas-standard-names` checkout carrying the three merged changes
(domain reduction leads; `accumulated` / `root_mean_square` lead joiner-free;
five measurement-destination loci registered).

## Headline

**The `--edits` flag DOES scope the run's pool claims. The two prior
invocations did not exit 1 because the flag failed to scope — they exited 1
because a review batch over the open edits cannot validate under the current
grammar: 49 of the 105 drafted/open edit successors carry spellings the grammar
now rejects as non-canonical, and the review path throws on the first one it
touches.**

## 1. The scope is wired, and it gates the claims

The flag's path to the graph is complete:

`--edits` (CLI) → `edits_only` → `run_sn_pools(edits_only=True)` →
`_build_pool_specs` → `_scope_kwargs["edits_only"] = True` (loop.py:603) →
every `claim_*_batch` seed/expand step ANDs
`coalesce(sn.edit_status, '') = 'open'` (graph_ops.py:15322 for StandardName,
15624–15625 for StandardNameSource). The review claim — `claim_review_name_batch`
— receives that gate. This is not the historical absence: the wiring has been
present since `449d5d75a` (2026-07-15).

Read-only confirmation — `sn run --only review --edits --dry-run` at HEAD exits
0 and prints the edits-scoped pending plan:

```
Pools: review_name (pending=94), refine_name (pending=1), generate_docs
(pending=0), review_docs (pending=17), refine_docs (pending=0), enrich_parents (pending=0)
```

The pending figures come from `_compute_pool_progress(edits_only=True)`, which
mirrors the claim predicates, so 94 is the edit-scoped review_name backlog the
flag selects.

### One scoping gap: the global maintenance is never scoped

`--skip-global-maintenance` is rejected when combined with `--edits`
(loop.py:1284 — it requires `scope_run_id` or `drain_scope_id`, and `edits_only`
is not one of them). So an edits run unconditionally executes the whole-graph
startup and post-drain maintenance — source reconcile, VocabGap reconcile,
unit-skip revival, structural-edge re-derivation, parent/doc-link normalization
— regardless of the edit scope. That is what the migration worker saw as "the
whole pool machinery rather than a scoped review": the *claims* were scoped; the
*maintenance* was not. The maintenance half-scoping is a design gap worth a
decision, not the exit-1 cause.

## 2. What the two prior invocations hit

The two invocations (`sn run --only review --edits -t 9`, then `-q -t 5`,
2026-09-05, per `cohort-migration-first-fifty.md`) exited 1 and landed no
rotation. Their traceback is genuinely lost — the CLI file log
(`sn_sn.log.1`) carries only unrelated peer noise, and stdout truncation hid the
terminal traceback. The observable mechanism today:

**49 of the 105 drafted/open successors fail strict parse under the installed
grammar.** They are old spellings the grammar no longer accepts:

| Category | Count | Example (`->` canonical rendering) |
|---|---|---|
| trailing `flux_surface_averaged` | 28 | `argon_density_flux_surface_averaged_at_plasma_boundary` → `flux_surface_averaged_argon_density_at_plasma_boundary` |
| trailing `accumulated`/`cumulative` | 19 | `carbon_count_accumulated_due_to_gas_injection` → `accumulated_carbon_count_due_to_gas_injection` |
| other non-canonical / unparseable | 2 | — |
| **strict-valid** | **56** | reviewable as-is |

On the first non-canonical name a review batch touches, the validation raises
`_NonCanonicalParseError` ("name is not canonical: flat segment order renders
as …"), which propagates out of the worker and terminates the run — exit 1, no
rotation. 56 of the 105 are valid and would review; the 49 poisoned names make
the batch as a whole fail. This is the mechanism the 09-05 runs hit; it still
stands at HEAD.

## 3. The live scope this flag must select

`edit_status = 'open'` successors in the graph, by `name_stage`:

| name_stage | count |
|---|---|
| drafted | 105 |
| reviewed | 48 |
| accepted | 25 |
| exhausted | 22 |
| superseded | 13 |
| **total open** | **213** |

**Drafted + open = 105** (the brief expected ~137; the migration's 137 has
shrunk — seeds the narrowly-scoped target at 105, of which 90 remain
edits-scoped review material after the 15 invalid-name remediation noted below).
Of the 105, the review_name claim admits 94 (score/rotation/grammar gates
subtract 11).

## 4. What unblocks the batch, and what is out of scope

The scoped batch cannot land until the 49 non-canonical drafted/open successors
are re-rendered to the spelling the grammar composes (the canonical-migration
work), or the review claim is taught to skip/repair them. That remediation is a
separate node's lane. This node changed no source and wrote no graph row; the
measurements above are the deliverable.

Method notes: every parse was strict (`parse(name, strict=True)`); the cohort
counts came from live `GraphClient` read queries; the dry-run was read-only
(graph writes 0, claims 0, LLM calls 0).
