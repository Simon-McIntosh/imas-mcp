# Scoped review exception reproduction

## Result

The review path does **not** propagate its strict-grammar exception. The real
exception is
`imas_standard_names.grammar.parser._NonCanonicalParseError`, a subclass of
`imas_standard_names.ParseError` and therefore of `ValueError`. It is raised by
the editable grammar package at `imas_standard_names/grammar/parser.py:2006`,
not by the bare `ValueError` fallback in Codex's
`imas_codex/standard_names/grammar_adapter.py`.

The direct review pre-step calls `strict_review_grammar_context()` and catches
`ValueError`, so this exception is caught before it can terminate
`process_review_name_batch`. A scoped, one-item `review_name` claim was made
with `edits_only=True`, reached that pre-step for the non-canonical
`carbon_count_accumulated_due_to_gas_injection`, and returned normally with no
strict grammar context. Its single temporary claim was released immediately;
there was no LLM call, review persistence, stage transition, rename, or
remaining claim.

## Live census

A fresh live-graph query counted **105** `StandardName` identities at
`name_stage='drafted'` and `edit_status='open'`. Strict parsing found **49**
non-canonical stored spellings and **56** strict-valid spellings. The 105 rows
divide by validation status into **94 valid**, **8 quarantined**, and **3
pending**. The 49 are the strict-parser census, not a substring proxy; their
operator tokens occur mid-name, between base and locus, rather than as trailing
tokens.

The scoped review claim selected `carbon_count_accumulated_due_to_gas_injection`
from the open edits. Its stored spelling is rejected in favor of
`accumulated_carbon_count_due_to_gas_injection`.

## Direct scoped-review exercise

The following direct invocation used the real review claim surface while
avoiding a paid reviewer call:

```python
items = claim_review_name_batch(edits_only=True, batch_size=1)
token = items[0]["claim_token"]
try:
    _enrich_name_review_items(items)
    assert items[0]["id"] == "carbon_count_accumulated_due_to_gas_injection"
    assert "grammar_round_trip" not in items[0]
finally:
    assert release_review_claims(token) == 1
```

Observed result:

```text
{'claimed': ['carbon_count_accumulated_due_to_gas_injection'],
 'edit_scoped': True,
 'strict_context_present': [False],
 'exception_propagated': False}
{'released_claims': 1}
```

`_enrich_name_review_items()` contains the actual review-path boundary:

```python
try:
    item.update(strict_review_grammar_context(item["id"]))
except (KeyError, TypeError, ValueError):
    logger.debug(...)
```

Because `_NonCanonicalParseError` has MRO
`_NonCanonicalParseError -> ParseError -> ValueError -> Exception`, this catch
is effective. The batch continues after dropping only the optional strict
grammar context; it does not raise from this site.

## Verbatim traceback of the underlying raise

Calling the exact helper used by that review pre-step directly against its
claimed open-edit successor reproduces the raise:

```text
Traceback (most recent call last):
  File "<stdin>", line 5, in <module>
  File "/home/ITER/mcintos/Code/.reckon-worktrees/imas-codex-c994bf55fb01/ship-s10-20260907/n-sgr-the-review-throw-is-reproduced-and-named/imas_codex/standard_names/graph_ops.py", line 451, in strict_review_grammar_context
    parsed = parse_canonical_name(name)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ITER/mcintos/Code/.reckon-worktrees/imas-codex-c994bf55fb01/ship-s10-20260907/n-sgr-the-review-throw-is-reproduced-and-named/imas_codex/standard_names/grammar_adapter.py", line 119, in parse_canonical_name
    result = parse(name, strict=True)
             ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ITER/mcintos/Code/imas-standard-names/imas_standard_names/grammar/parser.py", line 2250, in parse
    result = _parse_uncached(name, v, strict=strict)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ITER/mcintos/Code/imas-standard-names/imas_standard_names/grammar/parser.py", line 2205, in _parse_uncached
    _strict_validate(name, result.ir, v)
  File "/home/ITER/mcintos/Code/imas-standard-names/imas_standard_names/grammar/parser.py", line 2056, in _strict_validate
    _strict_flat_segment_semantics(name, ir)
  File "/home/ITER/mcintos/Code/imas-standard-names/imas_standard_names/grammar/parser.py", line 2006, in _strict_flat_segment_semantics
    raise _NonCanonicalParseError(name, canonical)
imas_standard_names.grammar.parser._NonCanonicalParseError: name is not canonical: flat segment order renders as 'accumulated_carbon_count_due_to_gas_injection'
```

The direct source inspection used only to identify the type's provenance
confirms the class is defined at `parser.py:380`; the traceback identifies the
actual raise site at `parser.py:2006`.

## What stopped the prior CLI runs

This measure disproves the claimed causal chain: a non-canonical edit successor
does reach strict parsing during review enrichment, but that exception is
handled and cannot itself be the uncaught error that made the earlier
`sn run --only review --edits` invocations exit 1. The prior CLI traceback was
lost, so this report does not assign a different exit-1 cause. The remaining
candidate is an error in a later review step or in the unscoped global
maintenance that still runs around the scoped claim.

## Reproduction commands

All commands used the worktree source through the shared project environment:

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv \
PYTHONPATH=$PWD uv run --no-sync python -
```

The direct parser exercise completed in 11.6 seconds. The claim/enrich/release
exercise completed in 17.9 seconds. Both accessed the live graph; the claim
exercise left it unchanged apart from the temporary claim timestamps that were
cleared by its verified release.

## Follow-on

The uncaught exit-1 source remains unproved. A separately scoped diagnostic
should run the full review loop with paid work disabled or intercepted after the
enrichment step, while preserving its first uncaught traceback and keeping the
open-edit cohort unchanged.
