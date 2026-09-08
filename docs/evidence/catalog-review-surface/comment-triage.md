# Catalog review comment triage

## Routed input surface

`imas_codex.standard_names.review_triage` reads both GitHub surfaces that a
catalog reviewer can use: line comments from the pull-request review endpoint
and request-level comments from the issue-comment endpoint. A line annotation
is resolved to the top-level YAML entry spanning its file and line. A request
comment deliberately has no single standard-name identity: it bears on the
batch and is sent to batch-membership adjudication.

Each comment receives one of five dispositions:

| Disposition | Meaning | Existing route |
| --- | --- | --- |
| `proposal` | An exact name or wording replacement on a resolved entry | `apply_edit` with `origin="human"` and `refine=False` |
| `contested` | A resolved objection to the quantity's physics | Existing contested lifecycle |
| `batch_adjudication` | A question about whether an entry belongs in the review set | Human batch-membership decision |
| `closed` | An acknowledgement requiring no action | No graph write |
| `adjudication` | An unresolved identity, unknown request, or non-exact wording | Human adjudication; never approval |

The proposal reason retains the comment id, author, full text, and comment URL.
That makes the existing edit provenance identify the reviewer remark that
introduced the candidate. The adapter does not alter the proposed text and
never enables refinement of a human's wording.

## Exercised census

The focused test uses both GitHub payload shapes and six representative comments:
two explicit proposals (one name, one wording), one physics objection, one
batch-scope question, one acknowledgement, and one deliberately unresolvable
comment. The resulting count is **proposal 2, contested 1,
batch-adjudication 1, closed 1, adjudication 1**. The unresolvable comment is
the positive control: it remains visible as adjudication rather than being
absorbed as approval.

## Verification

- Focused gate: `tests/standard_names/test_review_comment_triage.py` — **5
  passed**.
- Full gate on `all_debug`: `tests/standard_names/` — baseline **7,264 passed,
  11 skipped, 323 deselected, 0 failed** in 294.35 s; after-change **7,264
  passed, 11 skipped, 323 deselected, 0 failed** in 283.80 s. Added failures:
  **0**.

The required live fork rehearsal is not represented here as completed: opening
and later deleting its purpose-made branch would write outside this node's
exclusive paths. The implementation and its deterministic comment-shape
evidence are complete; a coordinator holding that external scope can perform
the isolated request exercise and record its request number separately.
